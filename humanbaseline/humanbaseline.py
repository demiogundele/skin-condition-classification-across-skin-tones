# Imports
from google.colab import drive, auth
import pandas as pd
import io
from google.cloud import storage
from PIL import Image
import matplotlib.pyplot as plt

# Connecting to Google Drive folders
drive.mount('/content/drive')
auth.authenticate_user()

TEST_DIR = '/content/drive/MyDrive/CSC211 Course Project/Testing Data - CSC211 Course Project'
human_baseline_subset = pd.read_csv(f'{TEST_DIR}/human_baseline_subset.csv', dtype={'case_id': str})

# comments
dataset = storage.Client("dx-scin-public")
bucket = dataset.bucket("dx-scin-public-data")

# comments
def display_data(bucket, image_path):
  blob = bucket.blob(image_path)
  image = Image.open(io.BytesIO(blob.download_as_bytes()))
  plt.figure(figsize=(4, 4))
  plt.imshow(image)
  plt.axis('off')
  plt.show()

# comments
for idx, row in human_baseline_subset.iterrows():
  print(f"Case ID: {row['case_id']}")
  display_data(bucket, row["image_1_path"])
  print("---")

# to check if every row in our human_baseline_subset has image_1_path present always (image 2 and 3 are sometimes missing)
print(human_baseline_subset['image_1_path'].isna().sum())