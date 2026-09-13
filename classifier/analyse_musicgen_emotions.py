from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = PROJECT_DIR / "classifier"

FEATURES_CSV = (
    CLASSIFIER_DIR
    / "results"
    / "domain_comparison"
    / "musicgen_features.csv"
)

RESULTS_DIR = (
    CLASSIFIER_DIR
    / "results"
    / "musicgen_emotion_analysis"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD FEATURES
# =========================================================

df = pd.read_csv(
    FEATURES_CSV
)

print("MusicGen feature file loaded.")
print("Number of samples:", len(df))


# =========================================================
# GET TARGET EMOTION FROM FILENAME
# =========================================================

def get_emotion(filename):

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


df["emotion"] = df["file"].apply(
    get_emotion
)

df = df[
    df["emotion"].notna()
].copy()

emotion_order = [
    "Happy",
    "Angry",
    "Calm",
    "Sad"
]


print("\nEmotion counts:")
print(
    df["emotion"].value_counts()
)


# =========================================================
# FEATURES TO ANALYSE
# =========================================================

features = [
    "tempo",
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "mfcc_1_mean",
]


# =========================================================
# SUMMARY STATISTICS
# =========================================================

summary_rows = []

for feature in features:

    for emotion in emotion_order:

        subset = df[
            df["emotion"] == emotion
        ][feature]

        summary_rows.append({
            "feature": feature,
            "emotion": emotion,
            "n": len(subset),
            "mean": subset.mean(),
            "std": subset.std(),
            "median": subset.median(),
            "min": subset.min(),
            "max": subset.max(),
        })


summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    RESULTS_DIR
    / "musicgen_emotion_feature_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("\n======================================")
print("MEAN FEATURE VALUES")
print("======================================")

pivot = summary_df.pivot(
    index="feature",
    columns="emotion",
    values="mean"
)

pivot = pivot[
    emotion_order
]

print(pivot)


# =========================================================
# KRUSKAL-WALLIS TEST
# =========================================================
# Tests whether the four emotion groups differ.
# H0 = the four groups come from the same distribution.
# p < 0.05 = statistically significant difference.
# =========================================================

test_rows = []

print("\n======================================")
print("KRUSKAL-WALLIS TESTS")
print("======================================")

for feature in features:

    groups = []

    for emotion in emotion_order:

        values = df[
            df["emotion"] == emotion
        ][feature].dropna()

        groups.append(
            values
        )

    H, p = kruskal(
        *groups
    )

    significant = (
        p < 0.05
    )

    test_rows.append({
        "feature": feature,
        "H_statistic": H,
        "p_value": p,
        "significant_p_lt_0.05":
            significant
    })

    print(
        f"{feature}: "
        f"H = {H:.3f}, "
        f"p = {p:.5f}, "
        f"significant = {significant}"
    )


test_df = pd.DataFrame(
    test_rows
)

test_path = (
    RESULTS_DIR
    / "kruskal_wallis_results.csv"
)

test_df.to_csv(
    test_path,
    index=False
)


# =========================================================
# BAR CHARTS OF MEAN VALUES
# =========================================================

for feature in features:

    feature_summary = (
        summary_df[
            summary_df["feature"]
            == feature
        ]
        .set_index("emotion")
        .loc[emotion_order]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        emotion_order,
        feature_summary["mean"]
    )

    ax.set_title(
        f"MusicGen: {feature} by Emotion"
    )

    ax.set_xlabel(
        "Target Emotion"
    )

    ax.set_ylabel(
        feature
    )

    for bar, value in zip(
        bars,
        feature_summary["mean"]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    output_path = (
        RESULTS_DIR
        / f"{feature}_mean_by_emotion.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# BOX PLOTS
# =========================================================

for feature in features:

    data = []

    for emotion in emotion_order:

        values = df[
            df["emotion"] == emotion
        ][feature].dropna()

        data.append(
            values
        )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        data,
        labels=emotion_order
    )

    ax.set_title(
        f"MusicGen Distribution: {feature}"
    )

    ax.set_xlabel(
        "Target Emotion"
    )

    ax.set_ylabel(
        feature
    )

    fig.tight_layout()

    output_path = (
        RESULTS_DIR
        / f"{feature}_boxplot.png"
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# OVERALL STANDARDIZED SEPARATION
# =========================================================
# For each feature, compare how far apart
# the four emotion means are relative to
# the overall standard deviation.
# =========================================================

separation_rows = []

for feature in features:

    means = []

    for emotion in emotion_order:

        mean_value = df[
            df["emotion"] == emotion
        ][feature].mean()

        means.append(
            mean_value
        )

    overall_std = df[
        feature
    ].std()

    mean_range = (
        max(means) - min(means)
    )

    if overall_std == 0:
        separation = 0
    else:
        separation = (
            mean_range / overall_std
        )

    separation_rows.append({
        "feature": feature,
        "mean_range": mean_range,
        "overall_std": overall_std,
        "standardized_separation":
            separation
    })


separation_df = pd.DataFrame(
    separation_rows
)

separation_df = separation_df.sort_values(
    "standardized_separation",
    ascending=False
)

separation_path = (
    RESULTS_DIR
    / "feature_separation_scores.csv"
)

separation_df.to_csv(
    separation_path,
    index=False
)


print("\n======================================")
print("FEATURE SEPARATION SCORES")
print("======================================")

print(
    separation_df
)


# =========================================================
# DONE
# =========================================================

print("\n======================================")
print("DONE")
print("======================================")

print("\nResults saved to:")
print(RESULTS_DIR)

print("\nFiles created:")
print("- musicgen_emotion_feature_summary.csv")
print("- kruskal_wallis_results.csv")
print("- feature_separation_scores.csv")
print("- mean bar charts")
print("- box plots")

# =========================================================
# POST-HOC PAIRWISE TESTS
# =========================================================
# Only analyse features that were significant
# in the Kruskal-Wallis test.
#
# Mann-Whitney U = compare two emotion groups
# Holm correction = correct for multiple comparisons
# =========================================================

from scipy.stats import mannwhitneyu
from itertools import combinations


significant_features = [
    "rms_mean",
    "rms_std",
    "spectral_rolloff_mean"
]


posthoc_rows = []


def holm_correction(p_values):
    """
    Holm-Bonferroni correction for multiple comparisons.
    Returns adjusted p-values in the original order.
    """

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted_sorted = np.empty(n)

    running_max = 0

    for i, p in enumerate(sorted_p):

        adjusted = (n - i) * p

        running_max = max(
            running_max,
            adjusted
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0
        )

    adjusted = np.empty(n)

    adjusted[order] = adjusted_sorted

    return adjusted


print("\n")
print("======================================")
print("POST-HOC PAIRWISE TESTS")
print("Mann-Whitney U + Holm correction")
print("======================================")


for feature in significant_features:

    print(f"\nFEATURE: {feature}")
    print("--------------------------------------")

    pair_results = []

    for emotion1, emotion2 in combinations(
        emotion_order,
        2
    ):

        group1 = df[
            df["emotion"] == emotion1
        ][feature].dropna()

        group2 = df[
            df["emotion"] == emotion2
        ][feature].dropna()

        U, p = mannwhitneyu(
            group1,
            group2,
            alternative="two-sided"
        )

        pair_results.append({
            "feature": feature,
            "emotion_1": emotion1,
            "emotion_2": emotion2,
            "U_statistic": U,
            "raw_p": p
        })


    # Holm correction
    raw_p_values = [
        result["raw_p"]
        for result in pair_results
    ]

    adjusted_p_values = holm_correction(
        raw_p_values
    )


    for result, adjusted_p in zip(
        pair_results,
        adjusted_p_values
    ):

        result["holm_p"] = adjusted_p

        result["significant"] = (
            adjusted_p < 0.05
        )

        posthoc_rows.append(
            result
        )

        print(
            f'{result["emotion_1"]} vs '
            f'{result["emotion_2"]}: '
            f'U = {result["U_statistic"]:.1f}, '
            f'raw p = {result["raw_p"]:.5f}, '
            f'Holm p = {adjusted_p:.5f}, '
            f'significant = '
            f'{result["significant"]}'
        )


# =========================================================
# SAVE RESULTS
# =========================================================

posthoc_df = pd.DataFrame(
    posthoc_rows
)

posthoc_path = (
    RESULTS_DIR
    / "posthoc_pairwise_results.csv"
)

posthoc_df.to_csv(
    posthoc_path,
    index=False
)


print("\n======================================")
print("POST-HOC DONE")
print("======================================")

print(
    "Saved to:",
    posthoc_path
)