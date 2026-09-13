from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = PROJECT_DIR / "classifier"

EMOPIA_FEATURES = CLASSIFIER_DIR / "features.csv"
MUSICGEN_DIR = PROJECT_DIR / "dataset" / "output"

RESULTS_DIR = CLASSIFIER_DIR / "results" / "domain_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FEATURE EXTRACTION
# Must match training exactly
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
        "file": wav_path.name,
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
# LOAD EMOPIA FEATURES
# =========================================================

print("Loading EMOPIA features...")

emopia_df = pd.read_csv(
    EMOPIA_FEATURES
)

print(
    "EMOPIA samples:",
    len(emopia_df)
)


# =========================================================
# EXTRACT MUSICGEN FEATURES
# =========================================================

wav_files = sorted(
    MUSICGEN_DIR.glob("*.wav")
)

print(
    "MusicGen WAV files:",
    len(wav_files)
)

musicgen_rows = []

for i, wav_path in enumerate(
    wav_files,
    start=1
):

    print(
        f"[{i}/{len(wav_files)}] "
        f"{wav_path.name}"
    )

    row = extract_features(
        wav_path
    )

    musicgen_rows.append(
        row
    )

musicgen_df = pd.DataFrame(
    musicgen_rows
)

musicgen_csv = (
    RESULTS_DIR
    / "musicgen_features.csv"
)

musicgen_df.to_csv(
    musicgen_csv,
    index=False
)


# =========================================================
# KEY FEATURES TO COMPARE
# =========================================================

features_to_compare = [
    "tempo",
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "mfcc_1_mean",
]


# =========================================================
# SUMMARY TABLE
# =========================================================

summary_rows = []

for feature in features_to_compare:

    summary_rows.append({
        "feature": feature,

        "emopia_mean":
            emopia_df[feature].mean(),

        "emopia_std":
            emopia_df[feature].std(),

        "musicgen_mean":
            musicgen_df[feature].mean(),

        "musicgen_std":
            musicgen_df[feature].std(),
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_csv = (
    RESULTS_DIR
    / "domain_feature_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print("\n======================================")
print("DOMAIN SUMMARY")
print("======================================")

print(summary_df)


# =========================================================
# DISTRIBUTION PLOTS
# =========================================================

for feature in features_to_compare:

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        emopia_df[feature],
        bins=30,
        alpha=0.6,
        density=True,
        label="EMOPIA"
    )

    ax.hist(
        musicgen_df[feature],
        bins=20,
        alpha=0.6,
        density=True,
        label="MusicGen"
    )

    ax.set_title(
        f"Domain Comparison: {feature}"
    )

    ax.set_xlabel(
        feature
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        RESULTS_DIR
        / f"{feature}_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# STANDARDIZED MEAN DIFFERENCE
# =========================================================
# Rough measure of how far apart the two domains are
# in standard-deviation units.
# =========================================================

shift_rows = []

for feature in features_to_compare:

    emopia_mean = (
        emopia_df[feature].mean()
    )

    musicgen_mean = (
        musicgen_df[feature].mean()
    )

    pooled_std = np.sqrt(
        (
            emopia_df[feature].var()
            +
            musicgen_df[feature].var()
        )
        / 2
    )

    if pooled_std == 0:
        shift = 0
    else:
        shift = (
            musicgen_mean
            -
            emopia_mean
        ) / pooled_std

    shift_rows.append({
        "feature": feature,
        "standardized_shift":
            shift
    })

shift_df = pd.DataFrame(
    shift_rows
)

shift_df = shift_df.reindex(
    shift_df[
        "standardized_shift"
    ]
    .abs()
    .sort_values(
        ascending=False
    )
    .index
)

shift_csv = (
    RESULTS_DIR
    / "domain_shift_scores.csv"
)

shift_df.to_csv(
    shift_csv,
    index=False
)

print("\n======================================")
print("DOMAIN SHIFT SCORES")
print("======================================")

print(shift_df)

print("\n======================================")
print("DONE")
print("======================================")

print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)