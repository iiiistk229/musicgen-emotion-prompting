from pathlib import Path
import random
import csv
from datetime import datetime
import threading
import http.server
import socketserver

import gradio as gr


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_ROOT / "dataset" / "output"

RESULTS_DIR = PROJECT_ROOT / "dataset" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "responses.csv"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EMOTIONS = [
    "Happy",
    "Sad",
    "Calm",
    "Angry",
]

CLIPS_PER_EMOTION = 3

TOTAL_CLIPS = CLIPS_PER_EMOTION * len(EMOTIONS)


# ============================================================
# LOCAL AUDIO SERVER SETTINGS
# ============================================================

AUDIO_SERVER_PORT = 8765

AUDIO_BASE_URL = (
    f"http://127.0.0.1:{AUDIO_SERVER_PORT}"
)


# ============================================================
# START STATIC AUDIO SERVER
# ============================================================

def start_audio_server():

    class AudioRequestHandler(
        http.server.SimpleHTTPRequestHandler
    ):

        def __init__(
            self,
            *args,
            **kwargs
        ):

            super().__init__(
                *args,
                directory=str(AUDIO_DIR),
                **kwargs
            )

        def log_message(
            self,
            format,
            *args
        ):
            pass


    socketserver.TCPServer.allow_reuse_address = True

    server = socketserver.TCPServer(
        ("127.0.0.1", AUDIO_SERVER_PORT),
        AudioRequestHandler
    )

    print(
        f"Audio server running at: "
        f"{AUDIO_BASE_URL}"
    )

    server.serve_forever()


# Start audio server in background thread
audio_thread = threading.Thread(
    target=start_audio_server,
    daemon=True
)

audio_thread.start()


# ============================================================
# LOAD AUDIO FILES
# ============================================================

def get_audio_files():

    return list(
        AUDIO_DIR.glob("*.wav")
    )


# ============================================================
# GET TARGET EMOTION
# ============================================================

def get_target_emotion(
    file_path
):

    filename = Path(
        file_path
    ).name.lower()

    if filename.startswith("happy"):
        return "Happy"

    elif filename.startswith("sad"):
        return "Sad"

    elif filename.startswith("calm"):
        return "Calm"

    elif filename.startswith("angry"):
        return "Angry"

    return "Unknown"


# ============================================================
# CREATE TEST SET
# ============================================================

def create_test_set():

    files = get_audio_files()

    emotion_groups = {
        "Happy": [],
        "Sad": [],
        "Calm": [],
        "Angry": [],
    }


    for file in files:

        emotion = get_target_emotion(
            file
        )

        if emotion in emotion_groups:

            emotion_groups[
                emotion
            ].append(file)


    # Check enough clips
    for emotion in EMOTIONS:

        if (
            len(
                emotion_groups[emotion]
            )
            < CLIPS_PER_EMOTION
        ):

            print(
                f"Not enough {emotion} clips."
            )

            return None


    selected = []


    # Select 3 from each emotion
    for emotion in EMOTIONS:

        chosen = random.sample(
            emotion_groups[
                emotion
            ],
            CLIPS_PER_EMOTION
        )

        selected.extend(
            chosen
        )


    # Random order
    random.shuffle(
        selected
    )


    selected = [
        str(file.resolve())
        for file in selected
    ]


    print(
        "\nSelected clips:"
    )

    for file in selected:

        print(
            Path(file).name
        )


    return selected


# ============================================================
# CREATE HTML AUDIO PLAYER
# ============================================================

def create_audio_html(
    file_path
):

    if not file_path:

        return ""


    filename = Path(
        file_path
    ).name


    audio_url = (
        f"{AUDIO_BASE_URL}/"
        f"{filename}"
    )


    html = f"""
    <div style="
        padding: 20px;
        background: #222;
        border-radius: 8px;
        margin-bottom: 15px;
    ">

        <p style="
            font-size: 16px;
            margin-bottom: 10px;
        ">
            Music Clip
        </p>

        <audio
            controls
            preload="auto"
            style="
                width: 100%;
            "
        >

            <source
                src="{audio_url}"
                type="audio/wav"
            >

            Your browser does not support audio playback.

        </audio>

    </div>
    """

    return html


# ============================================================
# SAVE RESPONSE
# ============================================================

def save_response(
    participant_id,
    age_group,
    music_training,
    file_path,
    clip_number,
    selected_emotion,
    valence,
    arousal,
    confidence,
):

    file_exists = (
        RESULTS_FILE.exists()
    )


    target_emotion = (
        get_target_emotion(
            file_path
        )
    )


    timestamp = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )


    with open(
        RESULTS_FILE,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csvfile:


        writer = csv.writer(
            csvfile
        )


        if not file_exists:

            writer.writerow([
                "timestamp",
                "participant_id",
                "age_group",
                "music_training",
                "clip_number",
                "filename",
                "target_emotion",
                "selected_emotion",
                "valence",
                "arousal",
                "confidence",
            ])


        writer.writerow([
            timestamp,
            participant_id,
            age_group,
            music_training,
            clip_number,
            Path(
                file_path
            ).name,
            target_emotion,
            selected_emotion,
            valence,
            arousal,
            confidence,
        ])


# ============================================================
# START EXPERIMENT
# ============================================================

def start_experiment(
    participant_id,
    age_group,
    music_training,
    consent,
):

    participant_id = (
        participant_id.strip()
    )


    if not participant_id:

        return (
            gr.update(),
            gr.update(),
            "",
            "Please enter a Participant ID.",
            [],
            0,
            f"### Clip 1 of {TOTAL_CLIPS}",
        )


    if age_group is None:

        return (
            gr.update(),
            gr.update(),
            "",
            "Please select your age group.",
            [],
            0,
            f"### Clip 1 of {TOTAL_CLIPS}",
        )


    if music_training is None:

        return (
            gr.update(),
            gr.update(),
            "",
            "Please answer the music training question.",
            [],
            0,
            f"### Clip 1 of {TOTAL_CLIPS}",
        )


    if consent is not True:

        return (
            gr.update(),
            gr.update(),
            "",
            (
                "You must provide consent "
                "before starting the study."
            ),
            [],
            0,
            f"### Clip 1 of {TOTAL_CLIPS}",
        )


    test_set = (
        create_test_set()
    )


    if test_set is None:

        return (
            gr.update(),
            gr.update(),
            "",
            (
                "Not enough audio clips "
                "are available yet."
            ),
            [],
            0,
            f"### Clip 1 of {TOTAL_CLIPS}",
        )


    first_file = (
        test_set[0]
    )


    audio_html = (
        create_audio_html(
            first_file
        )
    )


    return (
        gr.update(
            visible=False
        ),

        gr.update(
            visible=True
        ),

        audio_html,

        "",

        test_set,

        0,

        f"### Clip 1 of {len(test_set)}",
    )


# ============================================================
# NEXT CLIP
# ============================================================

def next_clip(
    participant_id,
    age_group,
    music_training,
    test_set,
    current_index,
    selected_emotion,
    valence,
    arousal,
    confidence,
):


    if selected_emotion is None:

        return (
            gr.update(),
            gr.update(),
            test_set,
            current_index,
            (
                "Please select an emotion "
                "before continuing."
            ),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


    current_file = (
        test_set[
            current_index
        ]
    )


    clip_number = (
        current_index + 1
    )


    save_response(
        participant_id,
        age_group,
        music_training,
        current_file,
        clip_number,
        selected_emotion,
        valence,
        arousal,
        confidence,
    )


    next_index = (
        current_index + 1
    )


    # Finished
    if (
        next_index
        >= len(test_set)
    ):

        return (
            "",
            "",
            test_set,
            next_index,
            "",
            None,
            4,
            4,
            4,

            gr.update(
                visible=False
            ),

            gr.update(
                visible=True
            ),
        )


    next_file = (
        test_set[
            next_index
        ]
    )


    audio_html = (
        create_audio_html(
            next_file
        )
    )


    progress_text = (
        f"### Clip "
        f"{next_index + 1} "
        f"of {len(test_set)}"
    )


    return (
        audio_html,
        progress_text,
        test_set,
        next_index,
        "",
        None,
        4,
        4,
        4,

        gr.update(
            visible=True
        ),

        gr.update(
            visible=False
        ),
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="AI Music Emotion Study"
) as demo:


    # ========================================================
    # PAGE 1
    # ========================================================

    with gr.Group(
        visible=True
    ) as intro_page:


        gr.Markdown(
            """
# AI Music Emotion Study

## Participant Information

This study investigates how people perceive emotion in short musical excerpts.

You will listen to **12 short piano clips**.

For each clip, you will be asked to:

- choose the emotion that best matches the music;
- rate how positive or negative the music feels;
- rate how calm or energetic the music feels;
- rate how confident you are in your emotion judgement.

Each clip is approximately **30 seconds long**.

Please use headphones if possible.

Please respond according to your own perception.

Your responses will be recorded for research purposes.

Please do not enter your real name.
            """
        )


        participant_id = (
            gr.Textbox(
                label="Participant ID",
                placeholder="Example: P001",
            )
        )


        age_group = (
            gr.Radio(
                choices=[
                    "18–24",
                    "25–34",
                    "35–44",
                    "45–54",
                    "55–64",
                    "65+",
                ],

                label="Age Group",
            )
        )


        music_training = (
            gr.Radio(
                choices=[
                    "No formal music training",
                    "Some music training",
                    (
                        "Currently studying / "
                        "practising music"
                    ),
                    (
                        "Professional or advanced "
                        "music training"
                    ),
                ],

                label=(
                    "Music Training Background"
                ),
            )
        )


        consent = (
            gr.Checkbox(
                label=(
                    "I have read the information "
                    "above and consent to participate "
                    "in this study."
                )
            )
        )


        start_message = (
            gr.Markdown()
        )


        start_button = (
            gr.Button(
                "Start Experiment",
                variant="primary",
            )
        )


    # ========================================================
    # PAGE 2
    # ========================================================

    with gr.Group(
        visible=False
    ) as experiment_page:


        gr.Markdown(
            """
# Listening Test

Listen carefully to the music before answering.

There are no right or wrong answers.

Please respond according to your own perception.
            """
        )


        progress = (
            gr.Markdown(
                f"### Clip 1 of {TOTAL_CLIPS}"
            )
        )


        # Browser-native audio player
        audio_player = (
            gr.HTML()
        )


        emotion_rating = (
            gr.Radio(
                choices=[
                    "Happy",
                    "Sad",
                    "Calm",
                    "Angry",
                ],

                label=(
                    "Which emotion best "
                    "matches this music?"
                ),
            )
        )


        valence = (
            gr.Slider(
                minimum=1,
                maximum=7,
                step=1,
                value=4,

                label=(
                    "Valence "
                    "(1 = Very Negative, "
                    "7 = Very Positive)"
                ),
            )
        )


        arousal = (
            gr.Slider(
                minimum=1,
                maximum=7,
                step=1,
                value=4,

                label=(
                    "Arousal "
                    "(1 = Very Calm, "
                    "7 = Very Energetic)"
                ),
            )
        )


        confidence = (
            gr.Slider(
                minimum=1,
                maximum=7,
                step=1,
                value=4,

                label=(
                    "Confidence "
                    "(1 = Not Confident, "
                    "7 = Very Confident)"
                ),
            )
        )


        experiment_message = (
            gr.Markdown()
        )


        next_button = (
            gr.Button(
                "Submit Response & Next Clip",
                variant="primary",
            )
        )


    # ========================================================
    # PAGE 3
    # ========================================================

    with gr.Group(
        visible=False
    ) as completion_page:


        gr.Markdown(
            """
# Thank You

You have completed the listening experiment.

Your responses have been recorded successfully.

Thank you for taking part in this study.
            """
        )


    # ========================================================
    # STATES
    # ========================================================

    test_set_state = (
        gr.State([])
    )

    current_index_state = (
        gr.State(0)
    )


    # ========================================================
    # START EVENT
    # ========================================================

    start_button.click(

        fn=start_experiment,

        inputs=[
            participant_id,
            age_group,
            music_training,
            consent,
        ],

        outputs=[
            intro_page,
            experiment_page,
            audio_player,
            start_message,
            test_set_state,
            current_index_state,
            progress,
        ],

        queue=False,
    )


    # ========================================================
    # NEXT EVENT
    # ========================================================

    next_button.click(

        fn=next_clip,

        inputs=[
            participant_id,
            age_group,
            music_training,
            test_set_state,
            current_index_state,
            emotion_rating,
            valence,
            arousal,
            confidence,
        ],

        outputs=[
            audio_player,
            progress,
            test_set_state,
            current_index_state,
            experiment_message,
            emotion_rating,
            valence,
            arousal,
            confidence,
            experiment_page,
            completion_page,
        ],

        queue=False,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "AI MUSIC EMOTION STUDY"
    )

    print(
        "========================================"
    )

    print(
        f"Audio folder: {AUDIO_DIR}"
    )

    print(
        f"Available WAV files: "
        f"{len(get_audio_files())}"
    )

    print(
        f"Audio server: "
        f"{AUDIO_BASE_URL}"
    )

    print(
        "Starting experiment website...\n"
    )


    demo.launch()