from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


# =========================================================
# 1. 路径
# 本文件应放在 classifier 文件夹里面
# =========================================================

CLASSIFIER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CLASSIFIER_DIR.parent

# 支持 lyria_wav 放在项目根目录或 classifier 里面
LYRIA_DIR = PROJECT_DIR / "lyria_wav"

if not LYRIA_DIR.is_dir():
    LYRIA_DIR = CLASSIFIER_DIR / "lyria_wav"

MODEL_PATH = (
    CLASSIFIER_DIR / "models" / "random_forest_emopia.joblib"
)

FEATURE_COLUMNS_PATH = (
    CLASSIFIER_DIR / "models" / "feature_columns.joblib"
)

RESULTS_DIR = CLASSIFIER_DIR / "results" / "lyria"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

emotion_order = ["Happy", "Angry", "Calm", "Sad"]


# =========================================================
# 2. 加载已有模型
# =========================================================

print("Loading EMOPIA classifier...")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("Model loaded successfully.")
print("Expected number of features:", len(feature_columns))

if set(model.classes_) != set(emotion_order):
    raise ValueError(
        f"模型类别与预期不一致：{model.classes_}"
    )

print("Audio folder:", LYRIA_DIR)


# =========================================================
# 3. 特征提取
# 与你原来的 MusicGen 分类代码保持一致
# 自动转换为 22050 Hz、单声道
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

    tempo = float(np.asarray(tempo).squeeze())

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
        "spectral_centroid_mean": float(
            np.mean(spectral_centroid)
        ),
        "spectral_rolloff_mean": float(
            np.mean(spectral_rolloff)
        ),
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
# 4. 从文件名读取指定的目标情绪
# 例如 lyria_happy_01.wav → Happy
# =========================================================

def get_target_emotion(filename):
    name = filename.lower()

    if name.startswith("lyria_"):
        name = name[len("lyria_"):]

    for emotion in emotion_order:
        if name.startswith(emotion.lower() + "_"):
            return emotion

    return None


# =========================================================
# 5. 查找音频
# =========================================================

wav_files = sorted(LYRIA_DIR.glob("lyria_*.wav"))

print("\n======================================")
print("LYRIA AUDIO SEGMENTS")
print("======================================")
print("WAV files found:", len(wav_files))

if not wav_files:
    raise FileNotFoundError(
        f"没有找到音频，请检查文件夹：{LYRIA_DIR}"
    )


# =========================================================
# 6. 逐个预测
# =========================================================

results = []

for i, wav_path in enumerate(wav_files, start=1):
    target = get_target_emotion(wav_path.name)

    if target is None:
        raise ValueError(
            f"无法从文件名识别目标情绪：{wav_path.name}"
        )

    print(
        f"\n[{i}/{len(wav_files)}] "
        f"Processing: {wav_path.name}"
    )

    features = extract_features(wav_path)

    X = pd.DataFrame([features])
    X = X[feature_columns]

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    probability_dict = dict(
        zip(model.classes_, probabilities)
    )

    match = prediction == target

    results.append({
        "file": wav_path.name,
        "lyria_target": target,
        "classifier_prediction": prediction,
        "match": match,
        "prob_angry": probability_dict["Angry"],
        "prob_calm": probability_dict["Calm"],
        "prob_happy": probability_dict["Happy"],
        "prob_sad": probability_dict["Sad"],
    })

    print(
        f"Target: {target} | "
        f"Prediction: {prediction} | "
        f"Match: {match}"
    )


# =========================================================
# 7. 保存每个音频的预测结果
# =========================================================

results_df = pd.DataFrame(results)

predictions_csv = (
    RESULTS_DIR / "lyria_full_predictions.csv"
)

results_df.to_csv(
    predictions_csv,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. 总体匹配率
# 此处比较的是预测与文件名标签的一致性
# =========================================================

total = len(results_df)
correct = int(results_df["match"].sum())
overall_agreement = correct / total

print("\n======================================")
print("OVERALL RESULTS")
print("======================================")
print(f"Matches: {correct}/{total}")
print(f"Overall agreement: {overall_agreement:.1%}")
print("Uniform random baseline for 4 classes: 25.0%")


# =========================================================
# 9. 每种情绪的匹配率
# =========================================================

summary_rows = []

print("\nAGREEMENT BY EMOTION")

for emotion in emotion_order:
    subset = results_df[
        results_df["lyria_target"] == emotion
    ]

    n = len(subset)
    n_correct = int(subset["match"].sum())
    agreement = n_correct / n if n else np.nan

    summary_rows.append({
        "emotion": emotion,
        "n": n,
        "matches": n_correct,
        "agreement": agreement,
    })

    print(
        f"{emotion}: "
        f"{n_correct}/{n} = {agreement:.1%}"
    )

pd.DataFrame(summary_rows).to_csv(
    RESULTS_DIR / "lyria_agreement_by_emotion.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 10. 分类报告
# 以文件名指定标签作为比较对象
# =========================================================

report = classification_report(
    results_df["lyria_target"],
    results_df["classifier_prediction"],
    labels=emotion_order,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR / "lyria_classification_report.csv",
    encoding="utf-8-sig"
)


# =========================================================
# 11. 混淆矩阵
# =========================================================

cm = confusion_matrix(
    results_df["lyria_target"],
    results_df["classifier_prediction"],
    labels=emotion_order
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")

ax.set_xticks(
    range(len(emotion_order)),
    labels=emotion_order
)

ax.set_yticks(
    range(len(emotion_order)),
    labels=emotion_order
)

ax.set_xlabel("Classifier Prediction")
ax.set_ylabel("Assigned Lyria Target")
ax.set_title(
    "Lyria Segment Classification\n"
    "EMOPIA-trained Random Forest"
)

threshold = cm.max() / 2

for i in range(len(emotion_order)):
    for j in range(len(emotion_order)):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

fig.tight_layout()

confusion_path = (
    RESULTS_DIR / "lyria_confusion_matrix.png"
)

fig.savefig(
    confusion_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 12. 预测类别分布
# =========================================================

distribution = (
    results_df["classifier_prediction"]
    .value_counts()
    .reindex(emotion_order, fill_value=0)
)

distribution.to_csv(
    RESULTS_DIR / "lyria_prediction_distribution.csv",
    encoding="utf-8-sig"
)

print("\nPREDICTION DISTRIBUTION")
print(distribution)


# =========================================================
# 13. 完成
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

print(
    "\n说明：这 8 个音频来自同一条生成音频的估计分段。"
    "这里的匹配率表示预测与指定标签的一致性，"
    "不代表经过人工验证的情绪准确率。"
)