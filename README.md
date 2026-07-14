# FightLens

FightLens is a text-to-video semantic search system for combat sports footage.

The goal of the project is to let a user describe a fight moment in natural language, such as:

> "fighter lands a right hand"  
> "clinch near the ropes"  
> "body attack"

and retrieve the most relevant video clips from a recorded fight.

## Planned pipeline

```text
Fight video
→ short clips
→ sampled frames
→ Gemini-generated descriptions
→ text embeddings
→ semantic retrieval
→ LLM reranking
→ ranked video results
```

## Current progress

- [x] Initial project structure
- [x] Gemini API integration
- [x] YAML-based configuration
- [x] Video loading with OpenCV
- [x] Time-window splitting with evenly sampled frames
- [x] Per-window folders and a processing manifest
- [x] Frame visualization with source-frame labels
- [ ] Automatic Gemini frame descriptions
- [ ] Text embedding generation
- [ ] Semantic search index
- [ ] LLM reranking
- [ ] Final video search interface

## Gemini integration

FightLens is connected to the Gemini API through the `google-genai` SDK.

The API key and model name are loaded from environment variables, keeping sensitive credentials outside the source code.

Gemini will later be used to generate semantic descriptions of fight clips and frames.

The current frame-extraction pipeline does not automatically call Gemini, so preprocessing does not consume API tokens.

## Time-window frame extraction

The preprocessing module splits a source video into short **time windows** and keeps a few representative frames from each one. One window will later map to a single multimodal request to Gemini.

Two parameters control this:

- `n_sec_per_window` — how many seconds of video one window spans. With 30 FPS and `n_sec_per_window: 2`, a full window covers ~60 source frames (window 0 ≈ frames 0–59, window 1 ≈ 60–119, ...).
- `n_img_per_window` — how many frames to keep from each window. They are sampled **evenly** across the whole window (not the first N in a row), with the first and last picks near the window boundaries.

FPS is read automatically from the video metadata. An optional `fps_override` acts as a fallback for videos with missing or broken metadata. The final, shorter-than-full slice of the video is **not** discarded — it becomes the last (partial) window.

### Output layout

```text
data/processed/<video_name>/
    manifest.json
    windows/
        window_000000/
            img_00_frame_00000000_0000000.00s.jpg
            img_01_frame_00000012_0000000.40s.jpg
            ...
        window_000001/
            ...
```

Each image name encodes its local position in the window, its source frame index, and its timestamp. The `manifest.json` records per-window metadata (frame ranges, timestamps, saved images, whether the window is full or partial) plus global video info, and is used later to tie Gemini descriptions and embeddings back to a specific moment.

All parameters are configured through YAML.

## Frame sampling preview

The extracted frames can be displayed in a vertical sequence with labels showing their positions in the original video.

![Extracted frames preview](docs/images/result.png)

![Additional extracted frames preview](docs/images/result_2.png)

## Running the project

Configure the input video, window duration, and images per window in:

```text
configs/default.yaml
```

Then run:

```bash
python -m fightlens
```

The application reads the YAML configuration and starts the video preprocessing pipeline.