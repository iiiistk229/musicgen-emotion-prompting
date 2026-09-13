from pathlib import Path
import pandas as pd
import numpy as np
import librosa

BASE_DIR = Path(__file__).resolve().parent
WAV_DIR = BASE_DIR / "wav"
OUTPUT_CSV = BASE_DIR / "features.csv"

rows = []

# 只测试前 5 首
wav_files = list(WAV_DIR.rglob("*.wav"))

print("Found test files:", len(wav_files))

for i, wav_path in enumerate(wav_files, start=1):
    print(f"[{i}/{len(wav_files)}] Processing: {wav_path.name}")

    # 文件夹名就是 emotion label
    emotion = wav_path.parent.name

    # 读取音频
    y, sr = librosa.load(wav_path, sr=22050, mono=True)

    # ---------- 基础特征 ----------
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).squeeze())

    # MFCC 13维
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    row = {
        "file": wav_path.name,
        "emotion": emotion,
        "tempo": tempo,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
    }

    # Chroma 12维：均值
    for j in range(12):
        row[f"chroma_{j+1}_mean"] = float(np.mean(chroma[j]))

    # MFCC 13维：均值 + 标准差
    for j in range(13):
        row[f"mfcc_{j+1}_mean"] = float(np.mean(mfcc[j]))
        row[f"mfcc_{j+1}_std"] = float(np.std(mfcc[j]))

    rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

print("\nDone!")
print("Saved to:", OUTPUT_CSV)
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())