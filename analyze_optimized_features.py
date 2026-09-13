from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "dataset" / "output_optimized"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "optimized_acoustic_features.csv"


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
# GET EMOTION FROM FILE NAME
# ============================================================

def get_emotion(wav_path):

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
# EXTRACT FEATURES
# ============================================================

def extract_features(wav_path):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True,
    )

    # RMS = loudness / energy
    rms = librosa.feature.rms(y=y)

    # Spectral centroid = brightness
    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
        )
    )

    # Tempo
    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr,
    )

    tempo = float(
        np.asarray(tempo).squeeze()
    )

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
    )

    # Pitch-class concentration
    chroma_mean = np.mean(chroma, axis=1)

    return {
        "tempo": tempo,
        "rms_mean": float(np.mean(rms)),
        "spectral_centroid_mean":
            float(np.mean(spectral_centroid)),
        "chroma_peak":
            float(np.max(chroma_mean)),
    }


# ============================================================
# PROCESS FILES
# ============================================================

wav_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

print(
    f"Found {len(wav_files)} WAV files."
)

rows = []


for i, wav_path in enumerate(
    wav_files,
    start=1,
):

    print(
        f"[{i}/{len(wav_files)}] "
        f"{wav_path.name}"
    )

    emotion = get_emotion(
        wav_path
    )

    features = extract_features(
        wav_path
    )

    row = {
        "file": wav_path.name,
        "emotion": emotion,
        **features,
    }

    rows.append(row)


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(
    "\nSaved feature table to:"
)

print(
    OUTPUT_CSV
)


# ============================================================
# GROUP SUMMARY
# ============================================================

summary = (
    df.groupby("emotion")[
        [
            "tempo",
            "rms_mean",
            "spectral_centroid_mean",
            "chroma_peak",
        ]
    ]
    .agg(
        ["mean", "std"]
    )
)

print(
    "\n========================================"
)

print(
    "GROUP SUMMARY"
)

print(
    "========================================"
)

print(
    summary
)

summary.to_csv(
    RESULTS_DIR
    / "optimized_feature_summary.csv"
)


# ============================================================
# BOXPLOT FUNCTION
# ============================================================

def make_boxplot(
    column,
    ylabel,
    title,
    filename,
):

    data = []

    for emotion in EMOTION_ORDER:

        subset = df[
            df["emotion"] == emotion
        ]

        data.append(
            subset[column].values
        )

    plt.figure(
        figsize=(7, 5)
    )

    plt.boxplot(
        data,
        labels=EMOTION_ORDER,
    )

    plt.xlabel(
        "Target Emotion"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# TEMPO
# ============================================================

make_boxplot(
    column="tempo",
    ylabel="Tempo (BPM)",
    title="Tempo Distribution by Target Emotion",
    filename="optimized_tempo_boxplot.png",
)


# ============================================================
# RMS / ENERGY
# ============================================================

make_boxplot(
    column="rms_mean",
    ylabel="Mean RMS Energy",
    title="Energy Distribution by Target Emotion",
    filename="optimized_rms_boxplot.png",
)


# ============================================================
# SPECTRAL CENTROID / BRIGHTNESS
# ============================================================

make_boxplot(
    column="spectral_centroid_mean",
    ylabel="Spectral Centroid (Hz)",
    title="Spectral Brightness by Target Emotion",
    filename="optimized_spectral_centroid_boxplot.png",
)


print(
    "\n========================================"
)

print(
    "DONE"
)

print(
    "========================================"
)