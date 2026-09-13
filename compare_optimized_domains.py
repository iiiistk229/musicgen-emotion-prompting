from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Original EMOPIA features used to train the classifier
EMOPIA_FEATURES = (
    BASE_DIR
    / "classifier"
    / "features.csv"
)

# New optimised MusicGen audio
MUSICGEN_DIR = (
    BASE_DIR
    / "dataset"
    / "output_optimized"
)

# Saved classifier information
MODELS_DIR = (
    BASE_DIR
    / "classifier"
    / "models"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
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
# GET MUSICGEN EMOTION
# ============================================================

def get_emotion(wav_path):

    filename = wav_path.name.lower()

    if filename.startswith("happy"):
        return "Happy"

    if filename.startswith("angry"):
        return "Angry"

    if filename.startswith("calm"):
        return "Calm"

    if filename.startswith("sad"):
        return "Sad"

    return "Unknown"


# ============================================================
# EXTRACT MUSICGEN FEATURES
# Same extraction pipeline used for EMOPIA
# ============================================================

def extract_features(wav_path):

    y, sr = librosa.load(
        wav_path,
        sr=22050,
        mono=True,
    )

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

    tempo, _ = (
        librosa.beat.beat_track(
            y=y,
            sr=sr,
        )
    )

    tempo = float(
        np.asarray(tempo).squeeze()
    )

    mfcc = (
        librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
        )
    )


    row = {

        "tempo":
            tempo,

        "rms_mean":
            float(np.mean(rms)),

        "rms_std":
            float(np.std(rms)),

        "zcr_mean":
            float(np.mean(zcr)),

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


    # Chroma
    for j in range(12):

        row[
            f"chroma_{j+1}_mean"
        ] = float(
            np.mean(
                chroma[j]
            )
        )


    # MFCC
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
# LOAD EMOPIA
# ============================================================

print(
    "\n========================================"
)

print(
    "EMOPIA vs OPTIMISED MUSICGEN"
)

print(
    "========================================"
)


print(
    "\nLoading EMOPIA features..."
)

emopia_df = pd.read_csv(
    EMOPIA_FEATURES
)

print(
    f"EMOPIA clips: {len(emopia_df)}"
)

print(
    "\nEMOPIA emotion distribution:"
)

print(
    emopia_df[
        "emotion"
    ].value_counts()
)


# ============================================================
# EXTRACT OPTIMISED MUSICGEN FEATURES
# ============================================================

wav_files = sorted(
    MUSICGEN_DIR.glob("*.wav")
)

print(
    f"\nOptimised MusicGen clips: "
    f"{len(wav_files)}"
)


musicgen_rows = []


for i, wav_path in enumerate(
    wav_files,
    start=1,
):

    print(
        f"[{i}/{len(wav_files)}] "
        f"{wav_path.name}"
    )

    features = extract_features(
        wav_path
    )

    features[
        "emotion"
    ] = get_emotion(
        wav_path
    )

    features[
        "file"
    ] = wav_path.name

    musicgen_rows.append(
        features
    )


musicgen_df = pd.DataFrame(
    musicgen_rows
)


musicgen_df.to_csv(
    RESULTS_DIR
    / "optimized_musicgen_features_full.csv",
    index=False,
)


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

model = joblib.load(
    MODELS_DIR
    / "random_forest_emopia.joblib"
)

feature_columns = joblib.load(
    MODELS_DIR
    / "feature_columns.joblib"
)


importance_df = pd.DataFrame({

    "feature":
        feature_columns,

    "importance":
        model.feature_importances_,
})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
)


print(
    "\n========================================"
)

print(
    "TOP 10 CLASSIFIER FEATURES"
)

print(
    "========================================"
)

print(
    importance_df.head(10)
)


importance_df.to_csv(
    RESULTS_DIR
    / "domain_comparison_feature_importance.csv",
    index=False,
)


# ============================================================
# SELECT TOP 6 FEATURES
# ============================================================

top_features = (
    importance_df[
        "feature"
    ]
    .head(6)
    .tolist()
)


print(
    "\nFeatures selected for comparison:"
)

for feature in top_features:

    print(
        "-",
        feature
    )


# ============================================================
# DOMAIN SUMMARY
# ============================================================

summary_rows = []


for feature in top_features:

    for emotion in EMOTION_ORDER:

        # EMOPIA
        emopia_values = (
            emopia_df[
                emopia_df[
                    "emotion"
                ]
                == emotion
            ][feature]
        )

        # MusicGen
        musicgen_values = (
            musicgen_df[
                musicgen_df[
                    "emotion"
                ]
                == emotion
            ][feature]
        )


        summary_rows.append({

            "feature":
                feature,

            "emotion":
                emotion,

            "emopia_mean":
                emopia_values.mean(),

            "emopia_std":
                emopia_values.std(),

            "musicgen_mean":
                musicgen_values.mean(),

            "musicgen_std":
                musicgen_values.std(),
        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    RESULTS_DIR
    / "domain_comparison_summary.csv",
    index=False,
)


# ============================================================
# STANDARDISE USING EMOPIA DISTRIBUTION
#
# Important:
# EMOPIA is the classifier's training domain.
# Therefore MusicGen is expressed relative to the
# EMOPIA feature distribution.
# ============================================================

comparison_rows = []


for feature in top_features:

    emopia_mean = (
        emopia_df[
            feature
        ].mean()
    )

    emopia_std = (
        emopia_df[
            feature
        ].std()
    )


    if emopia_std == 0:

        continue


    # EMOPIA
    for emotion in EMOTION_ORDER:

        values = (
            emopia_df[
                emopia_df[
                    "emotion"
                ]
                == emotion
            ][feature]
        )

        z_values = (
            values
            - emopia_mean
        ) / emopia_std


        for value in z_values:

            comparison_rows.append({

                "feature":
                    feature,

                "domain_emotion":
                    f"EMOPIA {emotion}",

                "value":
                    value,
            })


    # MusicGen
    for emotion in EMOTION_ORDER:

        values = (
            musicgen_df[
                musicgen_df[
                    "emotion"
                ]
                == emotion
            ][feature]
        )

        z_values = (
            values
            - emopia_mean
        ) / emopia_std


        for value in z_values:

            comparison_rows.append({

                "feature":
                    feature,

                "domain_emotion":
                    f"MusicGen {emotion}",

                "value":
                    value,
            })


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    RESULTS_DIR
    / "domain_comparison_standardized.csv",
    index=False,
)


# ============================================================
# CREATE ONE FIGURE PER TOP FEATURE
# ============================================================

group_order = [

    "EMOPIA Happy",
    "EMOPIA Angry",
    "EMOPIA Calm",
    "EMOPIA Sad",

    "MusicGen Happy",
    "MusicGen Angry",
    "MusicGen Calm",
    "MusicGen Sad",
]


for feature in top_features:

    subset = comparison_df[
        comparison_df[
            "feature"
        ]
        == feature
    ]


    data = []

    for group in group_order:

        values = subset[
            subset[
                "domain_emotion"
            ]
            == group
        ][
            "value"
        ].values

        data.append(
            values
        )


    plt.figure(
        figsize=(12, 6)
    )

    plt.boxplot(
        data,
        labels=group_order,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.ylabel(
        "Standardised Feature Value\n"
        "(relative to EMOPIA training distribution)"
    )

    plt.xlabel(
        "Domain and Target Emotion"
    )

    plt.title(
        f"Domain Comparison: {feature}"
    )

    plt.tight_layout()


    safe_name = (
        feature
        .replace("/", "_")
        .replace(" ", "_")
    )


    plt.savefig(
        RESULTS_DIR
        / f"domain_{safe_name}.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.show()


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
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)