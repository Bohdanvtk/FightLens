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
- [x] Automatic Gemini frame descriptions
- [x] Text embedding generation
- [ ] Semantic search index
- [ ] LLM reranking
- [ ] Final video search interface

## Gemini integration

FightLens is connected to the Gemini API through the `google-genai` SDK.

The API key and model name are loaded from environment variables, keeping sensitive credentials outside the source code.

Gemini generates a description per extracted time window (see below). Frame extraction and description generation are two **separate** commands, so preprocessing never consumes API tokens.

## Time-window frame extraction

The preprocessing module splits a source video into short **time windows** and keeps a few representative frames from each one. One window will later map to a single multimodal request to Gemini.

Two parameters control this:

- `n_sec_per_window` — how many seconds of video one window spans. With 30 FPS and `n_sec_per_window: 2`, a full window covers ~60 source frames (window 0 ≈ frames 0–59, window 1 ≈ 60–119, ...).
- `n_img_per_window` — how many frames to keep from each window. They are sampled **evenly** across the whole window (not the first N in a row), with the first and last picks near the window boundaries.

FPS is read automatically from the video metadata. An optional `fps_override` acts as a fallback for videos with missing or broken metadata. The final, shorter-than-full slice of the video is **not** discarded — it becomes the last (partial) window.

The processed scope can be limited so only part of a long fight is extracted (and later described, keeping token costs under control):

- `start_seconds` / `end_seconds` — bound the extracted time range (`end_seconds: null` = until the end of the video). Timestamps stay relative to the original video.
- `max_windows` — cap on how many windows are kept (`null` = no limit).

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

## Gemini window descriptions

A separate step sends each extracted window to Gemini: all of the window's frames go into **one** multimodal request, in chronological order, together with a boxing-analyst prompt. Gemini reads them as a short motion sequence and answers with 2–4 sentences describing the action (who attacks, punch type, target, result, defense, ring position).

The results accumulate in `descriptions.json` **inside the video's own processed folder** (`data/processed/<video>/descriptions.json`, next to `manifest.json`), so every video's artifacts stay self-contained. One entry per window:

```json
{
  "window_id": "window_000000",
  "start_sec": 0.0,
  "end_sec": 2.0,
  "frames": ["data/processed/test_video/windows/window_000000/img_00_....jpg"],
  "description": "The fighter in red shorts ..."
}
```

The step is idempotent: windows already present in the JSON are skipped, and the file is saved after every window, so an interrupted run can simply be restarted. Requests run sequentially with a configurable pause between calls (`request_delay_seconds`, for free-tier rate limits), and a failed call is retried once.

## Local window embeddings

A separate, **purely local** step turns each window's description into an embedding vector. It never calls Gemini or any paid API — it runs a [sentence-transformers](https://www.sbert.net/) model on this machine (the first run downloads and caches the model, ~80MB, from Hugging Face).

For every video it reads `data/processed/<video>/descriptions.json`, encodes each window's description into a 384-dimensional, L2-normalized vector (normalizing here makes later similarity a plain dot product), and stores all vectors together in a single `data/processed/<video>/embeddings.npz` alongside their `window_ids`. Storing normalized vectors keeps the whole video's index in one small file that the search step can load directly.

The step is idempotent and content-addressed: each vector is keyed by a hash of its exact description text, so re-running only re-embeds windows whose description changed (or everything, if the configured model changed — vectors from different models are not comparable). Running it twice with no input changes embeds nothing the second time.

It is configured under `embedding:` in the YAML:

- `model_name` — the local sentence-transformers model (default `all-MiniLM-L6-v2`, 384-dim). The output dimension is fixed by the model, so it is not a config key.
- `batch_size` — how many descriptions to encode per forward pass (default `32`).
- `device` — `auto` (let the model pick cuda/cpu), or force `cpu` / `cuda`.
- `normalize` — L2-normalize each vector (default `true`).

## Per-video manifest as the artifact index

Each video's `manifest.json` is the single index of that video's artifacts. The data files stay pure and never point at each other; instead every step registers what it produced under an `"artifacts"` section (written atomically, only after the data file is fully saved):

```json
"artifacts": {
  "descriptions": { "path": "descriptions.json", "model": "gemini-2.5-flash", "count": 13 },
  "embeddings":   { "path": "embeddings.npz", "model": "all-MiniLM-L6-v2", "dim": 384, "count": 13 }
}
```

## Frame sampling preview

The frames of a single window can be displayed in a vertical sequence with source-frame and timestamp labels parsed from the file names (see `scripts/inspect_frames.py`, `VIDEO_NAME` / `WINDOW_ID`).

![Extracted frames preview](docs/images/result.png)

![Additional extracted frames preview](docs/images/result_2.png)

## Running the project

Configure the input video, window duration, images per window, and the descriptions step in:

```text
configs/default.yaml
```

Then run the two pipeline steps separately:

```bash
# 1. Extract window frames (no API calls, spends no tokens).
python -m fightlens extract

# 2. Generate Gemini descriptions for the extracted windows.
python -m fightlens describe

# 3. Embed the window descriptions locally (no API calls, spends no tokens).
python -m fightlens embed

# Or run extract + describe in one go:
python -m fightlens full
```

`python -m fightlens` without a command still runs extraction only. Steps 2 and 3 are separate on purpose: `describe` spends Gemini tokens, while `embed` is purely local. The description prompt and retry count live in the `descriptions:` section of the YAML (`prompt`, `retry_attempts`); the embedding model and device live in `embedding:` (`model_name`, `batch_size`, `device`, `normalize`).