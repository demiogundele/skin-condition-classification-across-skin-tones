# Demi Part

# Core Imports
import cv2
import numpy as np
import pandas as pd

# Visualization Imports
import matplotlib.pyplot as plt

# Scikit-image imports
from skimage import color, filters, measure, morphology, exposure
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops
)
from skimage.measure import regionprops, label

# Imports for scientific computing
from scipy import ndimage
from scipy.spatial.distance import cdist

# Imports for Statistics
from collections import Counter
import math
import os

import io, sys
from google.cloud import storage
from sklearn.metrics import roc_auc_score


# Image generation
def load_image_from_bucket(bucket, image_path):
  """Loads an image from Google Cloud Storage as an RGB numpy array."""
  blob = bucket.blob(image_path)
  raw_bytes = blob.download_as_bytes()
  arr = np.frombuffer(raw_bytes, np.uint8)
  img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  return img_rgb


# Skin segmentation
def segment_skin(img_rgb):
  """Segments skin region using the YCbCr (Luma, Chroma Blue, Chroma Red) color space
  - more robust across skin tones that plain RGB thresholding, since it separates luminance
  from chrominance."""
  img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
  lower = np.array([0, 133, 77], dtype = np.uint8)
  upper = np.array([255, 173, 127], dtype=np.uint8)
  mask = cv2.inRange(img_ycrcb, lower, upper)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  return mask


# Redness detection
def extract_redness_features(img_rgb, skin_mask):
  """Extracts the redness features by using red intensity, max redness/red-to-green
  ratio, % of inflamed pixels to detect the redness"""
  r, g, b = img_rgb[..., 0].astype(float), img_rgb[..., 1].astype(float), img_rgb[..., 2].astype(float)
  skin_pixels = skin_mask > 0
  if skin_pixels.sum() == 0:
    return {"mean_redness": 0, "red_green_ratio": 0, "pct_inflamed": 0}

  mean_r = r[skin_pixels].mean()
  mean_g = g[skin_pixels].mean()
  red_green_ratio = mean_r / (mean_g + 1e-6)


  # "Inflamed" means the red channel notably dominant over green, within the skin region
  inflamed_mask = (r > g * 1.15) & skin_pixels
  pct_inflamed = inflamed_mask.sum() / skin_pixels.sum()

  return {
        "mean_redness": mean_r / 255.0,
        "red_green_ratio": red_green_ratio,
        "pct_inflamed": pct_inflamed,
    }


# Blob/circle detection
def extract_blob_features(img_rgb, skin_mask):
  """Extracts the circle features by using number of circles, average radius,
  radius distribution, clustering. For the blob features, lesion count,
  mean blob area, blob density"""
  gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
  gray_masked = cv2.bitwise_and(gray, gray, mask=skin_mask)

  blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
  circles = cv2.HoughCircles(
      blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
      param1=50, param2=25, minRadius=2, maxRadius=30
  )

  if circles is None:
    return {"lesion_count": 0, "avg_radius": 0, "radius_std": 0}


  radii = circles[0][:, 2]
  return {
      "lesion_count": len(radii),
      "avg_radius": radii.mean(),
      "radius_std": radii.std(),
  }



# Edge detection
def extract_edge_features(img_rgb, skin_mask):
  """Extracts the edge features by using  edge density,
  perimeter, and boundary sharpness"""
  gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
  edges = cv2.Canny(gray, 50, 150)
  edges_masked = cv2.bitwise_and(edges, edges, mask=skin_mask)
  skin_area = max((skin_mask > 0).sum(), 1)
  edge_density = (edges_masked > 0).sum() / skin_area
  return {"edge_density": edge_density}



# Tilosana Part

# Contour extraction
def extract_contour_features(img_rgb, skin_mask):
  """extracts area, perimeter, circularity, solidity, and aspect
  ration interpretation - around 1.0 for round lesion, < 0.5 for
  irregular rash"""
  gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
  contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return {"area": 0, "perimeter": 0, "circularity": 0, "solidity": 0, "aspect_ratio": 0, "has_puncture": 0}
  largest = max(contours, key=cv2.contourArea)
  area = cv2.contourArea(largest)
  perimeter = cv2.arcLength(largest, True)
  circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)  # 1.0 = perfect circle
  hull = cv2.convexHull(largest)
  hull_area = cv2.contourArea(hull)
  solidity = area / (hull_area + 1e-6)
  x, y, w, h = cv2.boundingRect(largest)
  aspect_ratio = w / (h + 1e-6)
  has_puncture = int(has_dark_center(gray, largest))

  return {
      "area": area, "perimeter": perimeter, "circularity": circularity,
      "solidity": solidity, "aspect_ratio": aspect_ratio,
      "has_puncture": has_puncture,
  }

#Puncture Detection
def has_dark_center(gray_image, contour):
  """Detecting any punctures to help distinguish Insect Bite"""
  M = cv2.moments(contour)
  if M["m00"] == 0:
    return False

  center_x = int(M["m10"] / M["m00"])
  center_y = int(M["m01"] / M["m00"])

  center_brightness = gray_image[center_y, center_x]
  y0, x0 = max(center_y - 15, 0), max(center_x - 15, 0)
  nearby_pixels = gray_image[y0:center_y + 15, x0:center_x + 15]
  nearby_average = nearby_pixels.mean()

  return (nearby_average - center_brightness) > 25


# Texture analysis
def extract_texture_features(img_rgb, skin_mask):
  """Extracts texture features by using Local Binary Patterns Histogram,
   entropy, uniformity"""
  gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

  # Local Binary Patterns histogram - "a visual summary and feature vector that counts the frequency of different local texture patterns across an image"
  lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
  lbp_masked = lbp[skin_mask > 0]
  if lbp_masked.size == 0:
      lbp_entropy, lbp_uniformity = 0, 0
  else:
      hist, _ = np.histogram(lbp_masked, bins=10, density=True)
      hist = hist[hist > 0]
      lbp_entropy = -np.sum(hist * np.log2(hist))
      lbp_uniformity = np.sum(hist ** 2)

  # Grey-Level Co-occurrence Matrix - extracts contrast, homogeneity, energy, and correlation
  glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
  contrast = graycoprops(glcm, "contrast")[0, 0]
  homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
  energy = graycoprops(glcm, "energy")[0, 0]
  correlation = graycoprops(glcm, "correlation")[0, 0]

  # Texture roughness as a single composite score — higher = rougher/scalier
  texture_roughness = contrast * (1 - homogeneity)

  return {
      "lbp_entropy": lbp_entropy, "lbp_uniformity": lbp_uniformity,
     "glcm_contrast": contrast, "glcm_homogeneity": homogeneity,
     "glcm_energy": energy, "glcm_correlation": correlation,
     "texture_roughness": texture_roughness,
  }


# Color distribution analysis
def extract_color_features(img_rgb, skin_mask):
  """Extracts colour features using hue histogram, saturation histogram, value histogram"""
  hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
  mask_bool = skin_mask > 0
  if mask_bool.sum() == 0:
    return {"hue_mean": 0, "hue_std": 0, "sat_mean": 0, "val_mean": 0}
  h, s, v = hsv[..., 0][mask_bool], hsv[..., 1][mask_bool], hsv[..., 2][mask_bool]
  return {
        "hue_mean": h.mean(), "hue_std": h.std(),
        "sat_mean": s.mean(), "val_mean": v.mean(),
        }

# Feature vector
def extract_feature_vector(img_rgb):
  """Creates a vector of all the extracted features"""
  skin_mask = segment_skin(img_rgb)
  features = {}
  features.update(extract_redness_features(img_rgb, skin_mask))
  features.update(extract_blob_features(img_rgb, skin_mask))
  features.update(extract_edge_features(img_rgb, skin_mask))
  features.update(extract_contour_features(img_rgb, skin_mask))
  features.update(extract_texture_features(img_rgb, skin_mask))
  features.update(extract_color_features(img_rgb, skin_mask))
  return features


# Rule engine
def classify(features):
  """Rule-based engine that classify the condition"""
    # Folliculitis: stricter correlation threshold to reduce false positives
  if features.get("glcm_correlation", 1.0) < 0.993:
      return "Folliculitis"

  # Insect Bite: stricter uniformity threshold so it doesn't swallow Eczema cases
  elif features.get("lbp_uniformity", 0.0) > 0.165:
      return "Insect Bite"

  # Eczema: default fallback for everything else
  else:
      return "Eczema"


# Probable condition categories
PROBABLE_CONDITIONS = ["Eczema", "Insect Bite", "Folliculitis"]

#Helper functions
def resize_for_features(img_rgb, target_width=512):
  """Resizes so every image has the same width. Several features (Hough radii,
  the 15 px window in has_dark_center) depend on pixel scale, so thresholds
  only mean something if all images are the same size."""
  h, w = img_rgb.shape[:2]
  scale = target_width / w
  return cv2.resize(img_rgb, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def predict_image(img_rgb):
  """RGB array in, one label from PROBABLE_CONDITIONS out."""
  img_rgb = resize_for_features(img_rgb)
  return classify(extract_feature_vector(img_rgb))


def run_on_csv(bucket, input_csv, out_csv, id_col="case_id", path_col="image_1_path"):
  """Reads input_csv, predicts one label per row, writes [id, Algorithmic Prediction] to out_csv.
  IDs and row order are copied from input_csv so they match it exactly."""
  df = pd.read_csv(input_csv, dtype={id_col: str})
  preds = []
  failures = 0
  for _, row in df.iterrows():
    try:
      img = load_image_from_bucket(bucket, row[path_col])
      preds.append(predict_image(img))
    except Exception as e:
      # Never drop an example: keep the row and use the rule's fallback class
      failures += 1
      print(f"Failed on {row[id_col]}: {e}", file=sys.stderr)
      preds.append("Eczema")
  print(f"{failures} of {len(df)} images failed and got the fallback label")
  out = pd.DataFrame({"Example ID": df[id_col], "Algorithmic Prediction": preds})
  out.to_csv(out_csv, index=False)
  return out

# ---------- Threshold exploration (TRAINING DATA ONLY) ----------
def build_feature_table(bucket, input_csv, out_csv, label_col):
  """Extracts the feature vector for every image in input_csv and saves it (plus the true label)
  to out_csv, so the slow image processing only has to happen once."""
  df = pd.read_csv(input_csv, dtype={"case_id": str})
  rows = []
  for _, r in df.iterrows():
    try:
      img = resize_for_features(load_image_from_bucket(bucket, r["image_1_path"]))
      feats = extract_feature_vector(img)
      feats["case_id"] = r["case_id"]
      feats["label"] = r[label_col]
      rows.append(feats)
    except Exception as e:
      print("Failed:", r["case_id"], e)
  feat_df = pd.DataFrame(rows)
  feat_df.to_csv(out_csv, index=False)
  print(feat_df.shape)
  return feat_df


def analyze_features(features_csv):
  """Prints per-class feature statistics and a single-feature AUC ranking per class."""
  feat_df = pd.read_csv(features_csv, dtype={"case_id": str})
  feature_cols = [c for c in feat_df.columns if c not in ("case_id", "label")]

  key = ["texture_roughness", "solidity", "circularity", "has_puncture",
         "lesion_count", "avg_radius", "pct_inflamed", "edge_density"]
  print(feat_df.groupby("label")[key].describe().T.to_string())

  # 0.5 = feature can't separate the class from the rest, near 0 or 1 = separates well
  for cls in PROBABLE_CONDITIONS:
    y = (feat_df["label"] == cls).astype(int)
    scores = {c: roc_auc_score(y, feat_df[c].fillna(0)) for c in feature_cols}
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)[:6]
    print(f"\n{cls}:", [(k, round(v, 2)) for k, v in ranked])


def evaluate_cached(features_csv):
  """Applies classify() to the saved training features (no image downloads), so you can
  edit thresholds and re-score in seconds."""
  from sklearn.metrics import classification_report
  feat_df = pd.read_csv(features_csv, dtype={"case_id": str})
  preds = feat_df.apply(lambda row: classify(row.to_dict()), axis=1)
  print("Train accuracy:", (preds == feat_df["label"]).mean())
  print(pd.crosstab(feat_df["label"], preds))
  print(classification_report(feat_df["label"], preds, zero_division=0))

def find_thresholds(features_csv, top_n=5):
  """For each class vs the rest, finds the best single-feature cut (and direction) by balanced
  accuracy on the TRAINING features. 0.5 = no better than chance."""
  feat_df = pd.read_csv(features_csv, dtype={"case_id": str})
  feature_cols = [c for c in feat_df.columns if c not in ("case_id", "label")]

  for cls in PROBABLE_CONDITIONS:
    y = (feat_df["label"] == cls).values
    results = []
    for c in feature_cols:
      x = feat_df[c].fillna(0).values
      for t in np.unique(np.quantile(x, np.linspace(0.05, 0.95, 37))):
        for direction in (">", "<"):
          pred = (x > t) if direction == ">" else (x < t)
          recall_pos = pred[y].mean()
          recall_neg = (~pred)[~y].mean()
          results.append((c, direction, round(float(t), 4), round((recall_pos + recall_neg) / 2, 3)))
    results.sort(key=lambda r: r[3], reverse=True)
    print(f"\n{cls} (best single-feature rules):")
    for r in results[:top_n]:
      print(f"  {r[0]} {r[1]} {r[2]}  balanced_acc={r[3]}")

def show_test_predictions(bucket, test_csv, preds_df, n=6, label_filter=None, id_col="case_id", path_col="image_1_path"):
  """Displays test set images alongside their Algorithmic Prediction."""
  df = pd.read_csv(test_csv, dtype={id_col: str})
  merged = df.merge(preds_df, left_on=id_col, right_on="Example ID", how="inner")
  if label_filter is not None:
    merged = merged[merged["Algorithmic Prediction"] == label_filter]
  sample = merged.head(n)
  if sample.empty:
    print(f"No rows found for filter: {label_filter}")
    return

  cols = 3
  rows = (len(sample) + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
  axes = np.atleast_1d(axes).flatten()

  for ax, (_, row) in zip(axes, sample.iterrows()):
    try:
      img = resize_for_features(load_image_from_bucket(bucket, row[path_col]))
      ax.imshow(img)
      ax.set_title(f"ID: {row[id_col]}\nPred: {row['Algorithmic Prediction']}")
    except Exception as e:
      ax.set_title(f"ID: {row[id_col]}\nError: {e}")
    ax.axis("off")

  for ax in axes[len(sample):]:
    ax.axis("off")

  plt.tight_layout()
  plt.show()

# ---------- CONFIG ----------
# "features" : extract + save training features (slow, run once)
# "analyze"  : print per-class stats and AUCs from the saved features
# "thresholds" : finds the best single-feature cut
# "tune"     : score classify() on the saved training features (fast, repeat while editing thresholds)
# "train"    : full pass over training images using run_on_csv
# "test"     : final single run on the test set, ONLY ONCE at the very end
MODE = "test"
TEST_DIR  = "/content/drive/MyDrive/CSC211 Course Project/Testing Data - CSC211 Course Project"
TEST_CSV  = f"{TEST_DIR}/test_set.csv"
OUT_CSV   = "symbolic_eval.csv"
TRAIN_DIR  = "/content/drive/MyDrive/CSC211 Course Project/Training Data - CSC211 Course Project"
TRAIN_CSV = f"{TRAIN_DIR}/training_pool.csv"
FEATURES_CSV = f"{TRAIN_DIR}/train_features.csv"
LABEL_COL = "dominant_condition"

if __name__ == "__main__":
  from google.colab import drive, auth
  drive.mount("/content/drive")
  auth.authenticate_user()
  bucket = storage.Client("dx-scin-public").bucket("dx-scin-public-data")

  if MODE == "features":
    build_feature_table(bucket, TRAIN_CSV, FEATURES_CSV, LABEL_COL)

  elif MODE == "analyze":
    analyze_features(FEATURES_CSV)

  elif MODE == "thresholds":
    find_thresholds(FEATURES_CSV)

  elif MODE == "tune":
    evaluate_cached(FEATURES_CSV)

  elif MODE == "train":
    out = run_on_csv(bucket, TRAIN_CSV, "train_predictions.csv")
    truth = pd.read_csv(TRAIN_CSV, dtype={"case_id": str})[LABEL_COL]
    print(out.head())
    print("Train accuracy:", (out["Algorithmic Prediction"].values == truth.values).mean())
    print(pd.crosstab(truth.values, out["Algorithmic Prediction"].values))

  elif MODE == "test":
    if os.path.exists(OUT_CSV):
      raise SystemExit(f"{OUT_CSV} already exists; the test set was already run.")
    out = run_on_csv(bucket, TEST_CSV, OUT_CSV)
    first = out.iloc[0]
    print(f"Sample {first['Example ID']} -> {first['Algorithmic Prediction']}")
    print(f"Wrote {len(out)} predictions to {OUT_CSV}")

    show_test_predictions(bucket, TEST_CSV, out, n=1)

  else:
    raise ValueError(f"Unknown MODE: {MODE}")