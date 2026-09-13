from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_CSV = BASE_DIR / "features.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =========================================================
# Load features
# =========================================================

df = pd.read_csv(FEATURES_CSV)

print("Dataset loaded.")
print("Shape:", df.shape)

print("\nEmotion distribution:")
print(df["emotion"].value_counts())

# =========================================================
# Extract original song ID
# =========================================================
# Example:
# Q2_--2B4d4lQhs_0.wav
#              ↓
# --2B4d4lQhs
#
# This prevents clips from the same original song
# appearing in both train and test folds.
# =========================================================

def get_song_id(filename):
    stem = Path(filename).stem

    match = re.match(r"^Q\d+_(.+)_\d+$", stem)

    if match:
        return match.group(1)

    return stem


df["song_id"] = df["file"].apply(get_song_id)

print("\nNumber of clips:", len(df))
print("Number of original songs:", df["song_id"].nunique())

print("\nExample grouping:")
print(df[["file", "song_id", "emotion"]].head(10))

# =========================================================
# Prepare X / y / groups
# =========================================================

# Do not use these as numerical features
exclude_columns = [
    "file",
    "emotion",
    "song_id"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

X = df[feature_columns]
y = df["emotion"]
groups = df["song_id"]

print("\nNumber of acoustic features:", len(feature_columns))

# Fixed order for reports and confusion matrix
emotion_order = [
    "Happy",
    "Angry",
    "Calm",
    "Sad"
]

# =========================================================
# 5-fold Stratified Group Cross-Validation
# =========================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_true = []
all_pred = []

fold_results = []

print("\n====================================")
print("5-FOLD CROSS-VALIDATION")
print("====================================")

for fold, (train_index, test_index) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])

    # Safety check:
    # same song must never appear in both train and test
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Data leakage detected in fold {fold}!"
        )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    fold_results.append({
        "fold": fold,
        "train_clips": len(train_index),
        "test_clips": len(test_index),
        "train_songs": len(train_groups),
        "test_songs": len(test_groups),
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    all_true.extend(y_test.tolist())
    all_pred.extend(predictions.tolist())

    print(
        f"Fold {fold}: "
        f"Accuracy = {accuracy:.3f} | "
        f"Macro-F1 = {macro_f1:.3f}"
    )

# =========================================================
# Cross-validation summary
# =========================================================

fold_df = pd.DataFrame(fold_results)

mean_accuracy = fold_df["accuracy"].mean()
std_accuracy = fold_df["accuracy"].std()

mean_f1 = fold_df["macro_f1"].mean()
std_f1 = fold_df["macro_f1"].std()

print("\n====================================")
print("CROSS-VALIDATION SUMMARY")
print("====================================")

print(
    f"Mean Accuracy: "
    f"{mean_accuracy:.3f} "
    f"(± {std_accuracy:.3f})"
)

print(
    f"Mean Macro-F1: "
    f"{mean_f1:.3f} "
    f"(± {std_f1:.3f})"
)

print("\nChance level for 4 classes = 0.250")

fold_df.to_csv(
    RESULTS_DIR / "cross_validation_results.csv",
    index=False
)

# =========================================================
# Classification report
# =========================================================

report = classification_report(
    all_true,
    all_pred,
    labels=emotion_order,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    RESULTS_DIR / "classification_report.csv"
)

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        all_true,
        all_pred,
        labels=emotion_order,
        zero_division=0
    )
)

# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=emotion_order
)

fig, ax = plt.subplots(figsize=(7, 6))

ax.imshow(cm)

ax.set_xticks(
    range(len(emotion_order)),
    labels=emotion_order
)

ax.set_yticks(
    range(len(emotion_order)),
    labels=emotion_order
)

ax.set_xlabel("Predicted Emotion")
ax.set_ylabel("True Emotion")

ax.set_title(
    "EMOPIA Random Forest Confusion Matrix\n"
    "5-fold Stratified Group Cross-Validation"
)

for i in range(len(emotion_order)):
    for j in range(len(emotion_order)):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=220,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# Train final model using ALL EMOPIA data
# =========================================================
# Cross-validation above is used for evaluation.
# This final model will later classify MusicGen WAV files.
# =========================================================

print("\nTraining final model on all EMOPIA clips...")

final_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

final_model.fit(X, y)

joblib.dump(
    final_model,
    MODELS_DIR / "random_forest_emopia.joblib"
)

joblib.dump(
    feature_columns,
    MODELS_DIR / "feature_columns.joblib"
)

print("Final classifier saved.")

# =========================================================
# Feature Importance
# =========================================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 15 most important features:")
print(importance_df.head(15))

print("\n====================================")
print("DONE")
print("====================================")

print("Results saved to:")
print(RESULTS_DIR)

print("\nModel saved to:")
print(MODELS_DIR)
