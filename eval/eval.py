# Imports
import pandas as pd
import numpy as np
from statistics import NormalDist

"""
For running on Google Colab:

from google.colab import drive
drive.mount('/content/drive')

import os

os.chdir('/content/drive/MyDrive/CSC211 Course Project/Testing Data - CSC211 Course Project')
print(os.getcwd())
print(os.listdir())
"""

# Labels are in a fixed order and we use these everywhere
LABELS = ["Eczema", "Insect Bite", "Folliculitis"]
# using variables to hold paths since not everyone has access to our drive folders
HUM_PATH = "human_eval.csv"
GT_PATH = "ground_truth.csv"
SYM_PATH = "symbolic_eval.csv"


def load(path):
    # read csv with dtype=str # keeps the long case IDs exactly as they are
    read_file = pd.read_csv(path, dtype=str)
    # strip the whitespace on all string columns
    read_file = read_file.apply(lambda col: col.str.strip())
    return read_file


def validate(gt, sym, hum):
    # assert IDs unique in each file
    assert len(gt['Example ID'].unique()) == len(
        gt)  # if number of unique ids = number of rows then no ids are repeated
    assert len(sym['Example ID'].unique()) == len(sym)
    assert len(hum['Example ID'].unique()) == len(hum)
    # assert set(sym IDs) == set(gt IDs) # full test set
    assert set(sym['Example ID']) == set(gt['Example ID'])
    # assert set(hum IDs) <= set(gt IDs) # human subset inside test set
    assert set(hum['Example ID']) <= set(gt['Example ID'])
    # assert no missing predictions
    assert sym['Algorithmic Prediction'].notnull().all()
    # assert every label in LABELS # catches the issue of "Eczema" vs "Eczema "
    assert set(sym['Algorithmic Prediction'].unique()) <= set(LABELS)
    assert set(gt['Ground Truth'].unique()) <= set(LABELS)
    cols_hum = [c for c in hum.columns if c not in ("Example ID",)]
    for col in cols_hum:
        assert hum[col].notnull().all()
        assert set(hum[col].unique()) <= set(LABELS)


def confusion_matrix(y_true, y_pred):
    # rows = true, columns = predicted
    # pd.crosstab=(...) then reindex both axes to LABELS, fill_value=0
    cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
    cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)
    return cm


def metrics(cm):
    # accuracy = diagonal sum / total
    accuracy = np.trace(cm.values) / cm.values.sum()
    # using a series instead of a loop to run the metrics on entire series at once
    # series are one-dimensional labeled array/column of values that have labels for each value - here it is our skin conditions
    # np.diag returns the numbers with no labels so putting it in a series puts the condition names back with the numbers
    diagonal = pd.Series(np.diag(cm.values), index=cm.index)
    # dividing the two series - pandas matches the numbers the labels and then divides each of the pair so loop not needed
    # recall - here sum(axis=1) adds across each row (actual counts so recall)
    recall = diagonal / cm.sum(axis=1)
    # precision - here axis=0 adds down each column (predicted counts for precision
    precision = diagonal / cm.sum(axis=0)
    # pandas provides NaN for 0/0 divisions so NaN means it was never predicted, not a wrong prediction
    # return accuracy, recall, precision
    return accuracy, recall, precision


def evaluate(name, y_true, y_pred):
    # creating the confusion matrix used for evaluation
    cm = confusion_matrix(y_true, y_pred)
    # running the metrics on the confusion matrix to get accuracy, recall and precision
    acc, rec, prec = metrics(cm)
    # accuracy sanity check - for a balanced set, accuracy should equal the average of the three recalls
    assert np.isclose(acc, rec.mean()), f"accuracy and mean recall differ for {name}"
    # returns dictionary - one row of the table of results per approach
    print("\n", name)
    print("\n", cm)
    return {
        "Approach": name,
        "n": len(y_true),
        "Accuracy": round(acc, 3),
        "Recall Eczema": round(rec["Eczema"], 3),
        "Recall Insect Bite": round(rec["Insect Bite"], 3),
        "Recall Folliculitis": round(rec["Folliculitis"], 3),
        "Precision Eczema": round(prec["Eczema"], 3),
        "Precision Insect Bite": round(prec["Insect Bite"], 3),
        "Precision Folliculitis": round(prec["Folliculitis"], 3),
    }


# margin of error calculation for classification accuracy - can do so since both sets are balanced
def calculate_moe(acc, n, confidence=0.95):
    z_score = NormalDist().inv_cdf((1 + confidence) / 2)  # for 95%, 1.96
    margin = z_score * np.sqrt((acc * (1 - acc)) / n)
    return margin


def main():
    # gt, sym, hum = load all three
    gt = load(GT_PATH)
    hum_eval = load(HUM_PATH)
    sym_eval = load(SYM_PATH)
    # validate
    validate(gt, sym_eval, hum_eval)
    # full test set - gt joined with sym on example ID
    full_test_set = pd.merge(gt, sym_eval, on="Example ID")
    print(full_test_set.shape)
    # human subset - gt joined with hum on example ID
    human_subset = pd.merge(gt, hum_eval, on="Example ID").merge(sym_eval, on="Example ID")
    print(human_subset.shape)
    # displaying results
    results = []
    results.append(evaluate("Symbolic Approach, full test set(n=180)", full_test_set["Ground Truth"],
                            full_test_set["Algorithmic Prediction"]))
    results.append(evaluate("Symbolic Approach, human subset(n=45)", human_subset["Ground Truth"],
                            human_subset["Algorithmic Prediction"]))
    for col in ["Demi - Human Prediction", "Tilosana - Human Prediction"]:
        person = col.replace(" - Human Prediction", "")
        results.append(evaluate(f"Human Baseline, {person}(n=45)", human_subset["Ground Truth"], human_subset[col]))
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv("results.csv", index=False)

    # the 0.333 expected from guessing
    chance = 1 / len(LABELS)
    print(f"Chance level: {chance}")

    # margin of error outputs/calculation
    for _, row in table.iterrows():
        margin = calculate_moe(row['Accuracy'], row['n'])
        lower, upper = row['Accuracy'] - margin, row['Accuracy'] + margin
        print(f"{row['Approach']}: accuracy {row['Accuracy']:.3f}"
              f"(+/- {margin:.3f}, 95% interval [{lower:.3f}, {upper:.3f}]), "
              f"{row['Accuracy'] - chance:.3f} against chance")


if __name__ == "__main__":
    main()
