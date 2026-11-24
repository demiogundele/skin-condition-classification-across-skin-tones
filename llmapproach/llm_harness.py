"""
Skin Condition Classification - LLM Approach Harness
Using local Ollama model, specifically gemma4:e4b

Task: one image as input, output one label of either Eczema, Insect Bite, Folliculitis
Input: same image that the DL approach and human baseline saw (image_1_path). image is the only
input (no case IDs or labels or file names go in the prompt)
Output: JSON {"answer": <label>} from the schema (the model only able to choose one of the labels)

python llm_harness.py           develop: score every prompt in PROMPTS on a fixed training subset
python llm_harness.py val v0 v?  compare your best few prompts on the validation set
python llm_harness.py test v?  run the prompt "v?" once on the test set, and write eval/llm.csv
"""
import json
import random
import sys
import time
import os
import requests
import pandas as pd
import ollama

# -------------------------- Setup
MODEL = "gemma4:e4b"
OPTIONS = {
    "temperature": 0,  # always choose the most likely next token, so runs are repeatable
    "seed": 0,
    "num_ctx": 8192,   # the context window, in tokens
    "num_predict": 600 # cap on output tokens so a run cannot hang
}
THINK = False  # gemma4 can spend its token cap on hidden thinking which cuts off the answer
LABELS = ["Eczema", "Insect Bite", "Folliculitis"] # order: label_id 0, 1, 2

DATA_DIR = "data"
TRAIN_CSV = f"{DATA_DIR}/training_pool.csv"
TEST_CSV = f"{DATA_DIR}/test_set.csv"
IMG_DIR = f"{DATA_DIR}/images"
RUN_DIR = "runs"
BUCKET = "gs://dx-scin-public-data/"

DEV_PER_CLASS = 15 # development subset: 45 cases
VAL_PER_CLASS = 10 # validation set: 30 cases, not included in dev (folliculitis only has 25 cases in the pool)

SCHEMA_WITH_REASONING = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"},
                  "answer": {"type": "string","enum": LABELS}},
    "required": ["reasoning", "answer"]
} # reasoning comes first so it comes before answer

# The JSON structure the model must answer in: exactly one of the labels.
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": LABELS}},
    "required": ["answer"]
}

REASONING_VERSION = {"v4"}

# -------------------------- Prompts

# One entry per prompt version.
# Write the change and the hypothesis BEFORE running a new version.
PROMPTS = {
    # baseline
    "v0": """<task>
Classify the skin condition present in the image.
</task>

<labels>
Eczema 
Insect Bite 
Folliculitis 
</labels>

Respond with exactly one label.""",

    # change: this is our control. the prompt means the same thing, just reworded and labels
    #         put in a reverse order.
    # hypothesis: no real changes anticipated. the score's movement for this version will be our noise floor.
    "v1": """<task>
Identify the skin condition shown in the image.
</task>

<labels>
Folliculitis  
Insect Bite
Eczema 
</labels>

Respond with exactly one of the labels.""",

    # change: adding in short descriptions of each condition
    # hypothesis: the descriptions will aid the two rarer labels, Insect Bite and Folliculitis,
    # that our deep learning model confuses with Eczema
    "v2": """<task>
Classify the skin condition present in the image. 
Eczema: thickened, leathery, dry skin with discoloration, rash or blisters 
Insect Bite: red bump or bumps often having a centre puncture, can be in a group, line, or singular 
Folliculitis: small red bumps or pimples with pus-filled blisters and crust formation
</task>

<labels>
Eczema 
Insect Bite
Folliculitis
</labels>

Respond with exactly one label.""",

    # change: adding in short descriptions of each condition and mentioning that all three conditions
    #         are equally common to appear
    # hypothesis: highlighting the equal label chances may reduce a bias towards over-selecting one label
    "v3": """<task>
Classify the skin condition present in the image. 
Eczema: thickened, leathery, dry skin with discoloration, rash or blisters 
Insect Bite: red bump or bumps often having a centre puncture, can be in a group, line, or singular 
Folliculitis: small red bumps or pimples with pus-filled blisters and crust formation
All three conditions are equally common.
</task>

<labels>
Eczema 
Insect Bite
Folliculitis
</labels>

Respond with exactly one label.""",

    # change: adding in short descriptions of each condition and chain of thought reasoning
    # hypothesis: the model describing its reasoning and what it sees can improve the recall for the rarer classes; this
    #             may be slower per image
    "v4": """<task>
Classify the skin condition present in the image. 
Eczema: thickened, leathery, dry skin with discoloration, rash or blisters 
Insect Bite: red bump or bumps often having a centre puncture, can be in a group, line, or singular 
Folliculitis: small red bumps or pimples with pus-filled blisters and crust formation
Before responding, describe the visible condition in one to three sentences in "reasoning", then give exactly one 
label in "answer".
</task>

<labels>
Eczema 
Insect Bite
Folliculitis
</labels>

Respond with exactly one label.""",
}

# -------------------------- Input Conversion: downloading each image once
def to_gs(p):
    return p if p.startswith("gs://") else BUCKET + p.lstrip("/")

def get_image(gs_path, case_id):
    """
    Download one image from the Google SCIN bucket to images/. Returns the local path.
    """
    os.makedirs(IMG_DIR, exist_ok=True)
    ext = os.path.splitext(gs_path)[1] or ".png"
    local = os.path.join(IMG_DIR, f"{case_id}{ext}")
    if not os.path.exists(local):
        r = requests.get(gs_path.replace("gs://", "https://storage.googleapis.com/"), timeout=60)
        r.raise_for_status()
        with open(local, "wb") as f:
            f.write(r.content)
    return local

def get_labels(df):
    """Return the label NAME for every row, whichever label column training_pool.csv has."""
    if "label_id" in df.columns:
        return df["label_id"].astype(int).map(lambda i: LABELS[i])
    lookup = {l.lower(): l for l in LABELS}
    for col in ["dominant_condition", "label", "condition", "skin_condition_label"]:
        if col in df.columns:
            names = df[col].str.strip().str.lower().map(lookup)
            assert names.notna().all(), f"unrecognised label values in '{col}': {sorted(df[col].unique())}"
            return names
    raise KeyError(f"no label column found; columns are {list(df.columns)}")

def load_rows(split):
    """Return a list of dicts {id, label, image}. dev and val are fixed, disjoint, stratified samples of the pool."""
    if split == "test":
        df = pd.read_csv(TEST_CSV, dtype=str)
        df["label"] = None   # the harness never reads test labels
    else:
        df = pd.read_csv(TRAIN_CSV, dtype=str)
        df["label"] = get_labels(df)
        rng = random.Random(0) # same cases every run
        keep = []
        for lab in LABELS:
            ids = sorted(df.loc[df["label"] == lab, "case_id"])
            rng.shuffle(ids)
            keep += ids[:DEV_PER_CLASS] if split == "dev" else ids[DEV_PER_CLASS:DEV_PER_CLASS + VAL_PER_CLASS]
        df = df[df["case_id"].isin(keep)]
    rows = []
    for _, r in df.iterrows():
        rows.append({"id": r["case_id"], "label": r["label"] if isinstance(r["label"], str) else None,
                     "image": get_image(to_gs(r["image_1_path"]), r["case_id"])})
    return rows

def predict(version, image_path, use_image=True):
    """
    Ask the model about one image. Returns a dict with the raw output, prediction,
    seconds, and prompt tokens. The prediction is None if the output is not one of the labels.
    """
    schema = SCHEMA_WITH_REASONING if version in REASONING_VERSION else SCHEMA
    msg = {"role": "user", "content": PROMPTS[version]}
    if use_image:
        msg["images"] = [image_path] # image goes in the images field, never by name in the prompt
    t0 = time.time()
    resp = ollama.chat(model=MODEL, messages=[msg], format=schema, options=OPTIONS, think=THINK)
    secs = round(time.time() - t0, 2)
    raw = resp["message"]["content"]
    try:
        pred = json.loads(raw)["answer"]
        if pred not in LABELS:
            pred = None
    except (json.JSONDecodeError, KeyError, TypeError):
        pred = None# invalid: counted, never dropped
    return {"raw": raw, "pred": pred, "secs": secs, "prompt_tokens": resp.get("prompt_eval_count")}

# running one version on one set - saves and no repeated work
def run(version, split, use_image=True):
    rows = load_rows(split)
    setup = {"model": MODEL, "options": OPTIONS, "think": THINK, "prompt": PROMPTS[version], "use_image": use_image,
             "schema": SCHEMA_WITH_REASONING if version in REASONING_VERSION else SCHEMA, "ids": [r["id"] for r in rows]}
    os.makedirs(RUN_DIR, exist_ok=True)
    path = f"{RUN_DIR}/{version}_{split}{'' if use_image else '_noimage'}.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        if lines[0] == setup and len(lines) == len(rows) + 1:
            print(f"[{version} {split}] same setup as saved run: loaded {path}")
            return lines[1:]
    print(f"\n===== EXACT PROMPT ({version}, {split}) =====\n{PROMPTS[version]}\n"
          f"(+ 1 image attached: {use_image})\n===============================")
    out = []
    with open(path, "w") as f:
        f.write(json.dumps(setup) + "\n")
        for k, r in enumerate(rows, 1):
            res = predict(version, r["image"], use_image)
            rec = {"id": r["id"], "label": r["label"], **res}
            f.write(json.dumps(rec) + "\n")
            out.append(rec)
            print(f"{k}/{len(rows)} {r['id']} true={r['label']} pred={res['pred']} {res['secs']}s")
    return out


# Our Metrics
def metrics(recs):
    """Macro recall (mean of per-class recall), invalid count, time per example, amd a confusion matrix.
    An invalid answer (pred None) counts as wrong. On a balanced set, macro recall equals the accuracy."""
    names = LABELS + ["INVALID"]
    cm = pd.DataFrame(0, index=LABELS, columns=names)
    for r in recs:
        cm.loc[r["label"], r["pred"] if r["pred"] else "INVALID"] += 1
    recall = {lab: cm.loc[lab, lab] / max(cm.loc[lab].sum(), 1) for lab in LABELS}
    return {"macro_recall": sum(recall.values()) / len(LABELS), "recall": recall,
            "invalid": sum(r["pred"] is None for r in recs),
            "secs": sum(r["secs"] for r in recs) / len(recs), "cm": cm}

def compare(split, versions, use_image=True):
    unknown = [v for v in versions if v not in PROMPTS]
    if unknown:
        sys.exit(f"No such prompt(s) in PROMPTS: {unknown}")
    results = {v: run(v, split, use_image) for v in versions}
    base = versions[0]
    for v in versions:
        m = metrics(results[v])
        print(f"\n--- {v} on {split}: macro recall {m['macro_recall']:.3f} | "
              f"recall { {k: round(float(x), 2) for k, x in m['recall'].items()} } | "
              f"invalid {m['invalid']} | {m['secs']:.1f}s/example")
        print(m["cm"])
        if v != base:  # fixes and breaks versus the first version
            b = {r["id"]: r["pred"] == r["label"] for r in results[base]}
            n = {r["id"]: r["pred"] == r["label"] for r in results[v]}
            fixes = [i for i in b if not b[i] and n[i]]
            breaks = [i for i in b if b[i] and not n[i]]
            print(f"vs {base}: {len(fixes)} fixes, {len(breaks)} breaks")


# Running once on the test set
def test(version):
    """Run ONCE, with the version chosen on the validation set. The test labels are never read."""
    if version not in PROMPTS:
        sys.exit(f"No such prompt in PROMPTS: {version}")
    out_path = "../eval/llm.csv"
    assert not os.path.exists(out_path), "eval/llm.csv exists - the test set is run once, do not overwrite"
    recs = run(version, "test")
    ids = [r["id"] for r in recs]
    assert len(recs) == 180 and len(set(ids)) == 180, "expected 180 unique test cases"
    os.makedirs("../eval", exist_ok=True)
    # writing eval/llm.csv
    pd.DataFrame({"Example ID": ids, "LLM Prediction": [r["pred"] or "INVALID" for r in recs]}).to_csv(out_path, index=False)
    print("Saved", out_path, "| invalid answers:", sum(r["pred"] is None for r in recs))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        compare("dev", list(PROMPTS))
    elif args[0] == "val":
        compare("val", args[1:])
    elif args[0] == "noimage":
        compare("dev", args[1:], use_image=False)
    elif args[0] == "test":
        test(args[1])
    else:
        print(__doc__)


