# ---- ONE-TIME test export (no scoring here) ----
import io, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf   # only used to read images from GCS
import torch
import torch.nn as nn
import torchvision.transforms as T

from google.colab import drive
drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/CSC211 Course Project/Training Data - CSC211 Course Project')
print(os.getcwd())
print(os.listdir())

LABELS = ["Eczema", "Insect Bite", "Folliculitis"]
device = "cuda" if torch.cuda.is_available() else "cpu"

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def load_pil(path):
    raw = tf.io.read_file(path).numpy()
    return Image.open(io.BytesIO(raw)).convert("RGB")

@torch.no_grad()
def embed(paths, bs=16):
    feats = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([preprocess(load_pil(p)) for p in paths[i:i + bs]]).to(device)
        feats.append(dino(batch).float().cpu().numpy())   # CLS embedding, 768-d
    return np.concatenate(feats)

def show_test_predictions(df, preds_df, n=6, label_filter=None):
    merged = df.merge(preds_df, left_on="case_id", right_on="Example ID", how="inner")
    if label_filter is not None:
        merged = merged[merged["Deep Learning Prediction"] == label_filter]
    sample = merged.head(n)
    if sample.empty:
        print(f"No rows found for filter: {label_filter}")
        return

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = load_pil(row["image_1_path"])
        ax.imshow(img)
        ax.set_title(f"ID: {row['case_id']}\nPred: {row['Deep Learning Prediction']}")
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dir = "/content/drive/MyDrive/CSC211 Course Project/Testing Data - CSC211 Course Project/"
    out_path = test_dir + "deep_learning.csv"
    assert not os.path.exists(out_path), "deep_learning.csv already exists - the test set is run once, do not overwrite"

    weights_path = "/content/drive/MyDrive/CSC211 Course Project/Training Data - CSC211 Course Project/dinov2_pytorch_head.pt"
    assert os.path.exists(weights_path), "run ml_train.py first"

    test_df = pd.read_csv(test_dir + "test_set.csv", dtype=str)
    test_df["image_1_path"] = test_df["image_1_path"].apply(
        lambda p: p if p.startswith("gs://") else f"gs://dx-scin-public-data/{p.lstrip('/')}")

    X_test = embed(test_df["image_1_path"].values)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    head = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 3)).to(device)
    head.load_state_dict(torch.load(weights_path, map_location=device))
    head.eval()
    with torch.no_grad():
        pred_ids = head(torch.tensor(X_test, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
    out = pd.DataFrame({"Example ID": test_df["case_id"],
                        "Deep Learning Prediction": [LABELS[i] for i in pred_ids]})

    # format checks only (IDs and label names), no scoring
    gt = pd.read_csv(test_dir + "ground_truth.csv", dtype=str)
    hum = pd.read_csv(test_dir + "human_eval.csv", dtype=str)
    assert len(out) == 180 and out["Example ID"].is_unique
    assert set(out["Example ID"]) == set(gt["Example ID"])
    assert set(hum["Example ID"]) <= set(out["Example ID"])
    assert out["Deep Learning Prediction"].isin(LABELS).all()

    out.to_csv(out_path, index=False)
    print("Saved", out_path)
    print(out["Deep Learning Prediction"].value_counts())

    chk = pd.read_csv(test_dir + "deep_learning.csv", dtype=str)
    print(chk.shape, list(chk.columns))          # expect (180, 2) ['Example ID', 'Deep Learning Prediction']
    print(chk["Deep Learning Prediction"].value_counts())
    print(chk.head(1))

    show_test_predictions(test_df, chk, n=1)