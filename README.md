# Emotion Prompting in MusicGen-Generated Solo Piano: An Exploratory Study

Research project by Xingyuan Song, MSc Artificial Intelligence for Media, Bournemouth University (2026).

This repository preserves the research scripts, generated audio, classifier models, features and result figures. It distinguishes the original MusicGen experiment from two computational follow-ups.

## Study scope

- Original MusicGen: 60 solo-piano clips, 15 per target emotion (Happy, Angry, Calm, Sad). Six listeners provided 72 judgements; reported recognition was 30.56%.
- EMOPIA classifier: 1,071 usable clips, approximately 62.3% grouped cross-validation accuracy. On the original MusicGen set, predictions were 59 Angry and 1 Happy.
- Optimised prompts: 60 additional MusicGen clips. The same classifier again predicted 59 Angry and 1 Happy. No additional human listening study was performed.
- Lyria: eight estimated segments of **one** approximately three-minute generation, two assigned targets per emotion. Predictions were 7 Angry and 1 Happy, with 3/8 assigned-label matches (37.5%). This is not Lyria emotion accuracy or eight independent generations. No human listening study was performed for this follow-up.

The results concern this exploratory setup and classifier transfer limitations. They do not establish overall MusicGen or Lyria quality. Existing report filenames and target-label column names are preserved; classifier matches to prompt targets should not be interpreted as validated perceptual accuracy.

## Files

- `experiment_app.py`: listening interface.
- `scripts/generate_music.py`: historical filename; the current file contains a listening interface and local audio server, not the original MusicGen generation pipeline.
- `generate_optimized_60.py`: optimised-prompt generation with `facebook/musicgen-small`.
- `analyze_optimized_features.py`, `evaluate_optimized_clips.py`, `compare_optimized_domains.py`: follow-up analyses.
- `dataset/output/`, `dataset/output_optimized/`: generated audio.
- `dataset/metadata/`: available generation metadata.
- `lyria_wav/`: the eight Lyria segments.
- `pilot_samples/`, `output/`: additional saved experimental outputs.
- `classifier/`: EMOPIA preparation, rendering, feature extraction, classifier training and prediction scripts.
- `classifier/models/`: saved classifier artifacts; package-version compatibility may be needed when loading.
- `classifier/results/`, `results/`: saved tables and plots.

## Setup and reproduction notes

`requirements.txt` lists direct third-party imports found in the scripts. It is not a tested or version-pinned environment. Original execution environments were not reconstructed during this upload. Scripts were archived without rerunning generation, training or listening experiments.

Run scripts from the project root after installing suitable dependencies. Review each script's inputs and output paths before running: some scripts write over existing result files. The optimised generator specifies `facebook/musicgen-small`, a base seed of 2026 and 1,500 maximum new tokens. GPU resources and model downloads may be required.

EMOPIA download and rendered audio are **not included** in Git:

1. `python classifier/get_emopia.py` downloads EMOPIA via MusPy.
2. Inspect and run `classifier/prepare_emopia.py` for the usable dataset mapping.
3. Configure the FluidSynth executable and Yamaha Disklavier piano soundfont paths in `classifier/render_emopia.py` for your machine, then render audio.
4. Inspect `classifier/extract_features.py` and `classifier/train_classifier.py` to regenerate features and training results.

`classifier/data/` and `classifier/wav/` were excluded because they contain downloaded third-party data and about 8 GB of derived rendering assets. These files remain in the original local project. Obtain the dataset and soundfont separately under their applicable terms. Cached files and secrets are also excluded.

No participant response CSV was found in the supplied project snapshot. The listening interface is included, but the reported human results cannot be fully reconstructed from this archive alone. Exact Lyria segmentation provenance and service version should be documented when available. Do not treat the assigned segment labels as independently verified emotions.

## Archive status

This is the first repository snapshot of the supplied VS Code project. Source files were copied without changing the experimental implementation. The repository is initially private. No licence is assigned on behalf of third-party datasets, models or audio.
