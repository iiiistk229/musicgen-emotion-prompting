import pandas as pd
from pathlib import Path

# EMOPIA 根目录
EMOPIA_ROOT = Path("classifier/data/emopia/EMOPIA_2.2")

# 读取标签
label_file = EMOPIA_ROOT / "label.csv"
df = pd.read_csv(label_file)

# 统一列名，避免有空格
df.columns = [col.strip() for col in df.columns]

# 四象限映射
emotion_map = {
    1: "Happy",   # High Valence, High Arousal
    2: "Angry",   # Low Valence, High Arousal
    3: "Sad",     # Low Valence, Low Arousal
    4: "Calm"     # High Valence, Low Arousal
}

df["Emotion"] = df["4Q"].map(emotion_map)

# 保存整理后的标签
output_file = EMOPIA_ROOT / "emotion_labels.csv"
df.to_csv(output_file, index=False)

print("Saved:", output_file)
print("\nEmotion counts:")
print(df["Emotion"].value_counts())

print("\nFirst 10 rows:")
print(df.head(10))

# -----------------------------
# Check MIDI files
# -----------------------------

midi_dir = EMOPIA_ROOT / "midis"

midi_files = list(midi_dir.rglob("*.mid"))

print("\n----------------------")
print("MIDI CHECK")
print("----------------------")

print("Number of MIDI files:", len(midi_files))

# Get MIDI filenames without .mid
midi_ids = {file.stem for file in midi_files}

# Get IDs from label.csv
label_ids = set(df["ID"].astype(str))

matched_ids = label_ids & midi_ids
missing_midi = label_ids - midi_ids
extra_midi = midi_ids - label_ids

print("Number of labels:", len(label_ids))
print("Matched:", len(matched_ids))
print("Labels without MIDI:", len(missing_midi))
print("MIDI without label:", len(extra_midi))

if missing_midi:
    print("\nExample labels without MIDI:")
    print(list(missing_midi)[:10])

if extra_midi:
    print("\nExample MIDI files without label:")
    print(list(extra_midi)[:10])

# -----------------------------
# Create final usable dataset
# -----------------------------

usable_df = df[df["ID"].isin(midi_ids)].copy()

# Add MIDI path
midi_path_map = {
    file.stem: str(file)
    for file in midi_files
}

usable_df["midi_path"] = usable_df["ID"].map(midi_path_map)

# Keep useful columns
usable_df = usable_df[
    ["ID", "4Q", "Emotion", "annotator", "midi_path"]
]

output_dataset = EMOPIA_ROOT / "emopia_usable_dataset.csv"

usable_df.to_csv(output_dataset, index=False)

print("\n----------------------")
print("FINAL DATASET")
print("----------------------")

print("Usable samples:", len(usable_df))

print("\nEmotion distribution:")
print(usable_df["Emotion"].value_counts())

print("\nSaved to:")
print(output_dataset)