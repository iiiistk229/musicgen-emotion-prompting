from pathlib import Path
import pandas as pd
import subprocess

# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

EMOPIA_DIR = BASE_DIR / "data" / "emopia" / "EMOPIA_2.2"
MIDI_DIR = EMOPIA_DIR / "midis"

CSV_PATH = EMOPIA_DIR / "emopia_usable_dataset.csv"

OUTPUT_DIR = BASE_DIR / "wav"
OUTPUT_DIR.mkdir(exist_ok=True)

FLUIDSYNTH = Path(
    r"C:\fluidsynth-v2.6.0-win10-x64-cpp11"
    r"\fluidsynth-v2.6.0-win10-x64-cpp11"
    r"\bin\fluidsynth.exe"
)

SOUNDFONT = Path(
    r"C:\YDP-GrandPiano-SF2-20160804"
    r"\YDP-GrandPiano-20160804.sf2"
)


# =========================
# Check paths
# =========================

print("Checking files...")

print("FluidSynth:", FLUIDSYNTH.exists())
print("SoundFont:", SOUNDFONT.exists())
print("MIDI folder:", MIDI_DIR.exists())
print("Dataset CSV:", CSV_PATH.exists())

if not FLUIDSYNTH.exists():
    raise FileNotFoundError("FluidSynth not found.")

if not SOUNDFONT.exists():
    raise FileNotFoundError("SoundFont not found.")

if not MIDI_DIR.exists():
    raise FileNotFoundError("MIDI folder not found.")

if not CSV_PATH.exists():
    raise FileNotFoundError("Dataset CSV not found.")


# =========================
# Load dataset
# =========================

df = pd.read_csv(CSV_PATH)

print("\nDataset loaded.")
print("Number of samples:", len(df))

print("\nEmotion distribution:")
print(df["Emotion"].value_counts())


# =========================
# Render MIDI → WAV
# =========================

success = 0
failed = []

for i, row in df.iterrows():

    midi_id = str(row["ID"])
    emotion = str(row["Emotion"])

    midi_path = MIDI_DIR / f"{midi_id}.mid"

    # Separate WAVs by emotion
    emotion_dir = OUTPUT_DIR / emotion
    emotion_dir.mkdir(exist_ok=True)

    wav_path = emotion_dir / f"{midi_id}.wav"

    # Skip files already rendered
    if wav_path.exists():
        success += 1
        continue

    command = [
        str(FLUIDSYNTH),
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        "44100",
        str(SOUNDFONT),
        str(midi_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        success += 1

        print(
            f"[{i + 1}/{len(df)}] "
            f"{emotion}: {midi_id} ✓"
        )

    except subprocess.CalledProcessError:
        failed.append(midi_id)

        print(
            f"[{i + 1}/{len(df)}] "
            f"{emotion}: {midi_id} FAILED"
        )


# =========================
# Summary
# =========================

print("\n=========================")
print("Rendering finished")
print("=========================")

print("Successful:", success)
print("Failed:", len(failed))

if failed:
    print("\nFailed MIDI files:")
    for item in failed:
        print(item)

print("\nWAV files saved to:")
print(OUTPUT_DIR)