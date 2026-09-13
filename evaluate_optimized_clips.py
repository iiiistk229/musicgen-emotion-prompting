from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "dataset" / "output_optimized"

MODELS_DIR = BASE_DIR / "classifier" / "models"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


MODEL_FILE = (
    MODELS_DIR
    / "random_forest_emopia.joblib"
)

FEATURE_COLUMNS_FILE = (
    MODELS_DIR
    / "feature_columns.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

EMOTION_ORDER = [
    "Happy",
    "Angry",
    "Calm",
    "Sad",
]


# ============================================================
# GET TRUE EMOTION FROM FILE NAME
# ============================================================

def get_true_emotion(wav_path):

    filename = wav_path.name.lower()

    if filename.startswith("happy"):
        return "Happy"

    elif filename.startswith("angry"):
        return "Angry"

    elif filename.startswith("calm"):
        return "Calm"

    elif filename.startswith("sad"):
        return "Sad"

    return "Unknown"


# ============================================================
# FEATURE EXTRACTION
# Must match original training pipeline
# ============================================================

def extract_features(wav_path):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True,
    )

    # ---------- Basic features ----------

    rms = librosa.feature.rms(
        y=y
    )

    zcr = (
        librosa.feature
        .zero_crossing_rate(y)
    )

    spectral_centroid = (
        librosa.feature
        .spectral_centroid(
            y=y,
            sr=sr,
        )
    )

    spectral_rolloff = (
        librosa.feature
        .spectral_rolloff(
            y=y,
            sr=sr,
        )
    )

    chroma = (
        librosa.feature
        .chroma_stft(
            y=y,
            sr=sr,
        )
    )

    # ---------- Tempo ----------

    tempo, _ = (
        librosa.beat.beat_track(
            y=y,
            sr=sr,
        )
    )

    tempo = float(
        np.asarray(
            tempo
        ).squeeze()
    )

    # ---------- MFCC ----------

    mfcc = (
        librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
        )
    )


    # ========================================================
    # BUILD FEATURE DICTIONARY
    # ========================================================

    row = {

        "tempo":
            tempo,

        "rms_mean":
            float(
                np.mean(rms)
            ),

        "rms_std":
            float(
                np.std(rms)
            ),

        "zcr_mean":
            float(
                np.mean(zcr)
            ),

        "spectral_centroid_mean":
            float(
                np.mean(
                    spectral_centroid
                )
            ),

        "spectral_rolloff_mean":
            float(
                np.mean(
                    spectral_rolloff
                )
            ),
    }


    # ---------- Chroma 12 dimensions ----------

    for j in range(12):

        row[
            f"chroma_{j+1}_mean"
        ] = float(
            np.mean(
                chroma[j]
            )
        )


    # ---------- MFCC 13 dimensions ----------

    for j in range(13):

        row[
            f"mfcc_{j+1}_mean"
        ] = float(
            np.mean(
                mfcc[j]
            )
        )

        row[
            f"mfcc_{j+1}_std"
        ] = float(
            np.std(
                mfcc[j]
            )
        )


    return row


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\n========================================"
)

print(
    "OPTIMISED MUSICGEN EVALUATION"
)

print(
    "========================================"
)


print(
    "\nLoading classifier..."
)

model = joblib.load(
    MODEL_FILE
)

feature_columns = joblib.load(
    FEATURE_COLUMNS_FILE
)

print(
    "Classifier loaded."
)

print(
    f"Number of expected features: "
    f"{len(feature_columns)}"
)


# ============================================================
# FIND AUDIO FILES
# ============================================================

wav_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

print(
    f"\nFound {len(wav_files)} WAV files."
)


if len(wav_files) == 0:

    raise RuntimeError(
        "No WAV files found in "
        f"{AUDIO_DIR}"
    )


# ============================================================
# PROCESS ALL 60 CLIPS
# ============================================================

rows = []


for i, wav_path in enumerate(
    wav_files,
    start=1,
):

    print(
        f"[{i}/{len(wav_files)}] "
        f"Processing: "
        f"{wav_path.name}"
    )


    true_emotion = (
        get_true_emotion(
            wav_path
        )
    )


    features = (
        extract_features(
            wav_path
        )
    )


    feature_row = (
        pd.DataFrame(
            [features]
        )
    )


    # ========================================================
    # MATCH ORIGINAL TRAINING FEATURE ORDER
    # ========================================================

    missing_features = [
        col
        for col in feature_columns
        if col
        not in feature_row.columns
    ]


    if missing_features:

        raise RuntimeError(
            "Missing features: "
            f"{missing_features}"
        )


    X_clip = (
        feature_row[
            feature_columns
        ]
    )


    # ========================================================
    # PREDICT
    # ========================================================

    predicted_emotion = (
        model.predict(
            X_clip
        )[0]
    )


    # Prediction probabilities

    probabilities = (
        model.predict_proba(
            X_clip
        )[0]
    )


    probability_map = dict(
        zip(
            model.classes_,
            probabilities,
        )
    )


    row = {

        "file":
            wav_path.name,

        "true_emotion":
            true_emotion,

        "predicted_emotion":
            predicted_emotion,

        "correct":
            true_emotion
            == predicted_emotion,
    }


    # Save probabilities

    for emotion in EMOTION_ORDER:

        row[
            f"prob_{emotion.lower()}"
        ] = (
            probability_map.get(
                emotion,
                0.0
            )
        )


    rows.append(
        row
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    rows
)


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

predictions_file = (
    RESULTS_DIR
    / "optimized_predictions.csv"
)

results_df.to_csv(
    predictions_file,
    index=False,
)


# ============================================================
# OVERALL METRICS
# ============================================================

y_true = (
    results_df[
        "true_emotion"
    ]
)

y_pred = (
    results_df[
        "predicted_emotion"
    ]
)


accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=EMOTION_ORDER,
    average="macro",
    zero_division=0,
)


print(
    "\n========================================"
)

print(
    "OVERALL RESULTS"
)

print(
    "========================================"
)

print(
    f"Accuracy: "
    f"{accuracy:.3f}"
)

print(
    f"Macro-F1: "
    f"{macro_f1:.3f}"
)

print(
    "\nChance level: 0.250"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print(
    "\n========================================"
)

print(
    "PREDICTION DISTRIBUTION"
)

print(
    "========================================"
)

print(
    results_df[
        "predicted_emotion"
    ]
    .value_counts()
    .reindex(
        EMOTION_ORDER,
        fill_value=0,
    )
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print(
    "\n========================================"
)

print(
    "PER-CLASS ACCURACY"
)

print(
    "========================================"
)


for emotion in EMOTION_ORDER:

    subset = results_df[
        results_df[
            "true_emotion"
        ]
        == emotion
    ]

    class_accuracy = (
        subset[
            "correct"
        ]
        .mean()
    )

    print(
        f"{emotion}: "
        f"{class_accuracy:.3f} "
        f"("
        f"{subset['correct'].sum()}"
        f"/"
        f"{len(subset)}"
        f")"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=EMOTION_ORDER,
    output_dict=True,
    zero_division=0,
)

report_df = (
    pd.DataFrame(
        report
    )
    .transpose()
)

report_file = (
    RESULTS_DIR
    / "optimized_classification_report.csv"
)

report_df.to_csv(
    report_file
)


print(
    "\n========================================"
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "========================================"
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=EMOTION_ORDER,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=EMOTION_ORDER,
)


print(
    "\n========================================"
)

print(
    "CONFUSION MATRIX"
)

print(
    "========================================"
)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.imshow(
    cm
)

ax.set_xticks(
    range(
        len(
            EMOTION_ORDER
        )
    ),
    labels=EMOTION_ORDER,
)

ax.set_yticks(
    range(
        len(
            EMOTION_ORDER
        )
    ),
    labels=EMOTION_ORDER,
)

ax.set_xlabel(
    "Predicted Emotion"
)

ax.set_ylabel(
    "True Emotion"
)

ax.set_title(
    "Optimised MusicGen Clips\n"
    "EMOPIA Random Forest Classifier"
)


for i in range(
    len(
        EMOTION_ORDER
    )
):

    for j in range(
        len(
            EMOTION_ORDER
        )
    ):

        ax.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center",
        )


fig.tight_layout()


confusion_matrix_file = (
    RESULTS_DIR
    / "optimized_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_file,
    dpi=220,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# DONE
# ============================================================

print(
    "\n========================================"
)

print(
    "DONE"
)

print(
    "========================================"
)

print(
    "\nPredictions saved to:"
)

print(
    predictions_file
)

print(
    "\nClassification report saved to:"
)

print(
    report_file
)

print(
    "\nConfusion matrix saved to:"
)

print(
    confusion_matrix_file
)