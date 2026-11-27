# Imports
from google.colab import drive, auth
import pandas as pd
import io
from google.cloud import storage

drive.mount('/content/drive')
auth.authenticate_user()

BASE_DIR = '/content/drive/MyDrive/CSC211 Course Project'
RAW_DIR = f'{BASE_DIR}/Raw Dataset - CSC211 Course Project'
TEST_DIR = f'{BASE_DIR}/Testing Data - CSC211 Course Project'
TRAIN_DIR = f'{BASE_DIR}/Training Data - CSC211 Course Project'

# comments

dataset = storage.Client("dx-scin-public")
bucket = dataset.bucket("dx-scin-public-data")

def load_csv_from_bucket(bucket, path):
  blob = bucket.blob(path)
  raw_bytes = blob.download_as_bytes()
  return pd.read_csv(io.BytesIO(raw_bytes), dtype={'case_id': str})

cases_df = load_csv_from_bucket(bucket, 'dataset/scin_cases.csv')
labels_df = load_csv_from_bucket(bucket, 'dataset/scin_labels.csv')
df = pd.merge(cases_df, labels_df, on='case_id')
print(f"Total merged cases: {len(df)}")


# comment
PROJECT_CONDITIONS = ['Eczema', 'Insect Bite', 'Folliculitis'] # comment
MIN_CONFIDENCE = 0.55 # set the confidence threshold for dominant-label assignment at 0.55
# this is the minimum value that yeilded at least 60 usable cases per target condition, given the imbalance in the SCIN labels
# for Eczema, Insect Bite, and Folliculitis
SEED = 42 # comment

graded = df[
    (df['dermatologist_gradable_for_skin_condition_1'] == 'DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT') |
    (df['dermatologist_gradable_for_skin_condition_2'] == 'DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT') |
    (df['dermatologist_gradable_for_skin_condition_3'] == 'DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT')
].copy()

# comment
def get_dominant_condition(str_label, min_confidence = MIN_CONFIDENCE):
  if pd.isna(str_label):
    return None
  label_dict = eval(str_label)
  if not label_dict:
    return None
  top_condition, top_weight = max(label_dict.items(), key=lambda x: x[1])
  return top_condition if top_weight >= min_confidence else None

graded['dominant_condition'] = graded['weighted_skin_condition_label'].apply(get_dominant_condition)
filtered = graded[graded['dominant_condition'].isin(PROJECT_CONDITIONS)].copy()

print(filtered['dominant_condition'].value_counts())

# comment
TEST_PER_CLASS = 60
HUMAN_SUBSET_PER_CLASS = 15

test_set = (
    filtered.groupby('dominant_condition', group_keys=False).apply(lambda x: x.sample(n=TEST_PER_CLASS, random_state=SEED)
    )
)

training_pool = filtered.drop(test_set.index)

human_baseline_subset = (
    test_set.groupby('dominant_condition', group_keys=False)
    .apply(lambda x: x.sample(n=HUMAN_SUBSET_PER_CLASS, random_state=SEED))
)

print(f"Test set size: {len(test_set)}")
print(test_set['dominant_condition'].value_counts())
print(f"\nHuman Baseline Subset: {len(human_baseline_subset)}")
print(f"\nTraining Pool: {len(training_pool)}")

#comment

filtered.to_csv(f'{RAW_DIR}/filtered_cases.csv', index=False)
test_set.to_csv(f'{TEST_DIR}/test_set.csv', index=False)
human_baseline_subset.to_csv(f'{TEST_DIR}/human_baseline_subset.csv', index=False)
training_pool.to_csv(f'{TRAIN_DIR}/training_pool.csv', index=False)