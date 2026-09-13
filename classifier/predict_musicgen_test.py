from pathlib import Path
import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = PROJECT_DIR / "classifier"
MUSICGEN_DIR = PROJECT_DIR / "dataset" / "output"

MODEL_PATH = (
    CLASSIFIER_DIR
    / "models"
    / "random_forest_emopia.joblib"
)

FEATURE_COLUMNS_PATH = (
    CLASSIFIER_DIR
    / "models"
    / "feature_columns.joblib"
)

RESULTS_DIR = CLASSIFIER_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading EMOPIA classifier...")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("Model loaded successfully.")
print("Expected number of features:", len(feature_columns))


# =========================================================
# FEATURE EXTRACTION
# Must match EMOPIA training exactly
# =========================================================

def extract_features(wav_path):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True
    )

    rms = librosa.feature.rms(y=y)

    zcr = librosa.feature.zero_crossing_rate(y)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    tempo = float(
        np.asarray(tempo).squeeze()
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    row = {
        "tempo": tempo,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean":
            float(np.mean(spectral_centroid)),
        "spectral_rolloff_mean":
            float(np.mean(spectral_rolloff)),
    }

    for j in range(12):
        row[f"chroma_{j+1}_mean"] = float(
            np.mean(chroma[j])
        )

    for j in range(13):

        row[f"mfcc_{j+1}_mean"] = float(
            np.mean(mfcc[j])
        )

        row[f"mfcc_{j+1}_std"] = float(
            np.std(mfcc[j])
        )

    return row


# =========================================================
# TARGET EMOTION FROM FILENAME
# =========================================================

def get_target_emotion(filename):

    name = filename.lower()

    if name.startswith("happy_"):
        return "Happy"

    elif name.startswith("angry_"):
        return "Angry"

    elif name.startswith("calm_"):
        return "Calm"

    elif name.startswith("sad_"):
        return "Sad"

    return None


# =========================================================
# FIND ALL MUSICGEN WAV FILES
# =========================================================

wav_files = sorted(
    MUSICGEN_DIR.glob("*.wav")
)

print("\n======================================")
print("MUSICGEN FULL DATASET")
print("======================================")

print("WAV files found:", len(wav_files))

if len(wav_files) == 0:
    raise FileNotFoundError(
        f"No WAV files found in {MUSICGEN_DIR}"
    )


# =========================================================
# RUN PREDICTIONS
# =========================================================

results = []

for i, wav_path in enumerate(
    wav_files,
    start=1
):

    target = get_target_emotion(
        wav_path.name
    )

    if target is None:

        print(
            f"[{i}/{len(wav_files)}] "
            f"Skipping unknown filename: "
            f"{wav_path.name}"
        )

        continue

    print(
        f"[{i}/{len(wav_files)}] "
        f"Processing: {wav_path.name}"
    )

    features = extract_features(
        wav_path
    )

    X = pd.DataFrame(
        [features]
    )

    X = X[
        feature_columns
    ]

    prediction = model.predict(
        X
    )[0]

    probabilities = model.predict_proba(
        X
    )[0]

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    match = (
        prediction == target
    )

    results.append({
        "file": wav_path.name,
        "musicgen_target": target,
        "classifier_prediction":
            prediction,
        "match": match,

        "prob_angry":
            probability_dict.get(
                "Angry", 0
            ),

        "prob_calm":
            probability_dict.get(
                "Calm", 0
            ),

        "prob_happy":
            probability_dict.get(
                "Happy", 0
            ),

        "prob_sad":
            probability_dict.get(
                "Sad", 0
            ),
    })


# =========================================================
# SAVE RAW PREDICTIONS
# =========================================================

results_df = pd.DataFrame(
    results
)

predictions_csv = (
    RESULTS_DIR
    / "musicgen_full_predictions.csv"
)

results_df.to_csv(
    predictions_csv,
    index=False
)


# =========================================================
# OVERALL AGREEMENT
# =========================================================

total = len(results_df)
correct = int(
    results_df["match"].sum()
)

overall_accuracy = (
    correct / total
)

print("\n======================================")
print("OVERALL RESULTS")
print("======================================")

print(
    f"Correct matches: "
    f"{correct}/{total}"
)

print(
    f"Overall agreement: "
    f"{overall_accuracy * 100:.1f}%"
)

print(
    "Chance level for 4 classes: 25.0%"
)


# =========================================================
# ACCURACY BY TARGET EMOTION
# =========================================================

emotion_order = [
    "Happy",
    "Angry",
    "Calm",
    "Sad"
]

summary_rows = []

print("\n======================================")
print("ACCURACY BY EMOTION")
print("======================================")

for emotion in emotion_order:

    subset = results_df[
        results_df["musicgen_target"]
        == emotion
    ]

    n = len(subset)

    n_correct = int(
        subset["match"].sum()
    )

    accuracy = (
        n_correct / n
        if n > 0
        else 0
    )

    summary_rows.append({
        "emotion": emotion,
        "n": n,
        "correct": n_correct,
        "accuracy": accuracy
    })

    print(
        f"{emotion}: "
        f"{n_correct}/{n} "
        f"= {accuracy * 100:.1f}%"
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    RESULTS_DIR
    / "musicgen_accuracy_by_emotion.csv",
    index=False
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    results_df[
        "musicgen_target"
    ],
    results_df[
        "classifier_prediction"
    ],
    labels=emotion_order,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    RESULTS_DIR
    / "musicgen_classification_report.csv"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    results_df[
        "musicgen_target"
    ],
    results_df[
        "classifier_prediction"
    ],
    labels=emotion_order
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.imshow(cm)

ax.set_xticks(
    range(
        len(emotion_order)
    ),
    labels=emotion_order
)

ax.set_yticks(
    range(
        len(emotion_order)
    ),
    labels=emotion_order
)

ax.set_xlabel(
    "Classifier Prediction"
)

ax.set_ylabel(
    "MusicGen Intended Emotion"
)

ax.set_title(
    "MusicGen Emotion Classification\n"
    "EMOPIA-trained Random Forest"
)

for i in range(
    len(emotion_order)
):
    for j in range(
        len(emotion_order)
    ):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.tight_layout()

confusion_path = (
    RESULTS_DIR
    / "musicgen_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=220,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

distribution = (
    results_df[
        "classifier_prediction"
    ]
    .value_counts()
    .reindex(
        emotion_order,
        fill_value=0
    )
)

distribution.to_csv(
    RESULTS_DIR
    / "musicgen_prediction_distribution.csv"
)

print("\n======================================")
print("PREDICTION DISTRIBUTION")
print("======================================")

print(distribution)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n======================================")
print("DONE")
print("======================================")

print("\nPredictions saved to:")
print(predictions_csv)

print("\nConfusion matrix saved to:")
print(confusion_path)

print("\nAll results saved in:")
print(RESULTS_DIR)