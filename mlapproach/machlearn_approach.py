# ============================================================
# CSC211 - Skin condition classifier
# Frozen DINOv2 embeddings + logistic regression, 5-fold stratified CV
# ============================================================

# ---- Imports, Drive, data loading ----
import io, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf   # only used to read images from GCS
from tensorflow import keras
from PIL import Image
import joblib

from google.colab import drive
drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/CSC211 Course Project/Training Data - CSC211 Course Project')
print(os.getcwd())
print(os.listdir())

keras.utils.set_random_seed(42)

# Labels are in a fixed order and we use these everywhere
LABELS = ["Eczema", "Insect Bite", "Folliculitis"]
label_to_id = {label: idx for idx, label in enumerate(LABELS)}

df_pool = pd.read_csv("training_pool.csv")
df_pool["image_1_path"] = df_pool["image_1_path"].apply(
    lambda p: p if str(p).startswith("gs://") else f"gs://dx-scin-public-data/{str(p).lstrip('/')}")
df_pool["label_id"] = df_pool["dominant_condition"].map(label_to_id)
assert df_pool["label_id"].notna().all()

print(df_pool["dominant_condition"].value_counts())

# quick check that a GCS image can be read
p = df_pool["image_1_path"].iloc[0]
print(p, len(tf.io.read_file(p).numpy()), "bytes")
