from pathlib import Path

import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "output_optimized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = "facebook/musicgen-small"

CLIPS_PER_EMOTION = 15

BASE_SEED = 2026

MAX_NEW_TOKENS = 1500


# ============================================================
# OPTIMISED PROMPTS
# ============================================================

PROMPTS = {

    "Happy": (
        "A happy solo piano piece in a major key, "
        "with a fast tempo, bright upper register, "
        "lively rhythmic patterns, consonant harmony, "
        "and energetic dynamics. "
        "Clear and joyful emotional expression."
    ),

    "Sad": (
        "A sad solo piano piece in a minor key, "
        "with a slow tempo, low to middle register, "
        "gentle rhythm, soft dynamics, "
        "and descending melodic phrases. "
        "Clear and melancholic emotional expression."
    ),

    "Calm": (
        "A calm solo piano piece with a slow tempo, "
        "middle register, consonant harmony, "
        "smooth and steady rhythm, soft dynamics, "
        "and low intensity. "
        "Clear and peaceful emotional expression."
    ),

    "Angry": (
        "An angry solo piano piece with a fast tempo, "
        "low register, dissonant harmony, "
        "strong rhythmic accents, forceful dynamics, "
        "and high intensity. "
        "Clear and aggressive emotional expression."
    ),
}


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

else:

    DEVICE = "cpu"


print(
    f"\nUsing device: {DEVICE}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading MusicGen-small..."
)

processor = (
    AutoProcessor.from_pretrained(
        MODEL_NAME
    )
)

model = (
    MusicgenForConditionalGeneration
    .from_pretrained(
        MODEL_NAME
    )
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# SAMPLE RATE
# ============================================================

SAMPLE_RATE = (
    model.config.audio_encoder.sampling_rate
)

print(
    f"Sample rate: {SAMPLE_RATE} Hz"
)


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_clip(
    emotion,
    prompt,
    clip_number,
    seed,
):

    print(
        "\n----------------------------------------"
    )

    print(
        f"Emotion: {emotion}"
    )

    print(
        f"Clip: {clip_number:02d}"
    )

    print(
        f"Seed: {seed}"
    )

    print(
        f"Prompt: {prompt}"
    )


    # --------------------------------------------------------
    # SET RANDOM SEED
    # --------------------------------------------------------

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    # --------------------------------------------------------
    # PROCESS TEXT PROMPT
    # --------------------------------------------------------

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )


    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }


    # --------------------------------------------------------
    # GENERATE MUSIC
    # --------------------------------------------------------

    with torch.no_grad():

        audio_values = model.generate(
            **inputs,

            do_sample=True,

            guidance_scale=3.0,

            max_new_tokens=MAX_NEW_TOKENS,
        )


    # --------------------------------------------------------
    # CONVERT AUDIO
    # --------------------------------------------------------

    audio = (
        audio_values[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # NORMALISE TO INT16 WAV
    # --------------------------------------------------------

    max_value = abs(
        audio
    ).max()

    if max_value > 0:

        audio = (
            audio
            / max_value
        )


    audio_int16 = (
        audio
        * 32767
    ).astype(
        "int16"
    )


    # --------------------------------------------------------
    # FILE NAME
    # --------------------------------------------------------

    filename = (
        f"{emotion.lower()}_"
        f"{clip_number:02d}.wav"
    )


    output_path = (
        OUTPUT_DIR
        / filename
    )


    # --------------------------------------------------------
    # SAVE WAV
    # --------------------------------------------------------

    scipy.io.wavfile.write(
        output_path,
        SAMPLE_RATE,
        audio_int16,
    )


    duration = (
        len(audio_int16)
        / SAMPLE_RATE
    )


    print(
        f"Saved: {output_path}"
    )

    print(
        f"Duration: "
        f"{duration:.2f} seconds"
    )


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "MUSICGEN OPTIMISED DATASET GENERATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nOutput folder:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nEmotions: {len(PROMPTS)}"
    )

    print(
        f"Clips per emotion: "
        f"{CLIPS_PER_EMOTION}"
    )

    print(
        f"Total clips: "
        f"{len(PROMPTS) * CLIPS_PER_EMOTION}"
    )


    seed_counter = (
        BASE_SEED
    )


    for emotion, prompt in PROMPTS.items():

        print(
            "\n\n========================================"
        )

        print(
            f"GENERATING {emotion.upper()}"
        )

        print(
            "========================================"
        )


        for clip_number in range(
            1,
            CLIPS_PER_EMOTION + 1,
        ):

            generate_clip(
                emotion=emotion,
                prompt=prompt,
                clip_number=clip_number,
                seed=seed_counter,
            )

            seed_counter += 1


    print(
        "\n\n========================================"
    )

    print(
        "GENERATION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\n60 clips saved to:"
        f"\n{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()