"""
Skin Condition Classification - LLM Inference
Using local Ollama model, specifically gemma4:e4b

Task: one image as input, output one label of either Eczema, Insect Bite, Folliculitis
Input: same image that the DL approach and human baseline saw (image_1_path). image is the only
input (no case IDs or labels or file names go in the prompt)
Output: JSON {"answer": <label>} from the schema (the model only able to choose one of the labels)
"""

import json
import os
import sys
import requests
import pandas as pd
import ollama

MODEL = "gemma4:e4b"
OPTIONS = {
    "temperature": 0,  # always choose the most likely next token, so runs are repeatable
    "seed": 0,
    "num_ctx": 8192,   # the context window, in tokens
    "num_predict": 600 # cap on output tokens so a run cannot hang
}
THINK = False # gemma4 can spend its token cap on hidden thinking, which cuts off the answer
LABELS = ["Eczema", "Insect Bite", "Folliculitis"] # order: label_id 0, 1, 2

# The JSON structure the model must answer in: exactly one of the labels.
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": LABELS}},
    "required": ["answer"]
}

TEST_CSV = "data/test_set.csv"
IMAGE_COL = "image_1_path"
ID_COL = "case_id"
# the data sample from your test set shown in our progress report (expected output: Eczema)
SAMPLE_CASE_ID = "5100815424449862591"

BEST_PROMPT = """<task>
Classify the skin condition present in the image.
</task>

<labels>
Eczema 
Insect Bite 
Folliculitis 
</labels>

Respond with exactly one label."""

def local_image(path, cache_dir="data/images"):
    """Return a local file for the image, downloading it from the public bucket if needed."""
    if os.path.exists(path):
        return path
    if not path.startswith("gs://"):
        path = "gs://dx-scin-public-data/" + path.lstrip("/")
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, path.rsplit("/", 1)[-1])
    if not os.path.exists(local):
        r = requests.get(path.replace("gs://", "https://storage.googleapis.com/"), timeout=60)
        r.raise_for_status()
        with open(local, "wb") as f:
            f.write(r.content)
    return local


def predict(image_path):
    """One image as input, one label as output (or 'INVALID' if the answer is not one of the labels)."""
    resp = ollama.chat(model=MODEL,
                       messages=[{"role": "user", "content": BEST_PROMPT, "images": [local_image(image_path)]}],
                       format=SCHEMA, options=OPTIONS, think=THINK)
    try:
        label = json.loads(resp["message"]["content"])["answer"]
        return label if label in LABELS else "INVALID"
    except (json.JSONDecodeError, KeyError, TypeError):
        return "INVALID"


if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_CASE_ID
    test_set = pd.read_csv(TEST_CSV, dtype=str)             # dtype=str keeps the long case IDs intact
    row = test_set[test_set[ID_COL] == case_id]
    assert len(row) == 1, f"case id {case_id} not found in {TEST_CSV}"
    prediction = predict(row.iloc[0][IMAGE_COL])
    print(f"Input: Case ID: {case_id}")
    print(f"LLM Prediction: {prediction}")