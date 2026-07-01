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
- [x] Extraction of every N-th video frame
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

## Frame extraction

The preprocessing module loads a source video and extracts one frame for every configurable `N` video frames.

For example, with a frame step of `30`, the extracted sequence represents approximately:

```text
Source frame 1
Source frame 30
Source frame 60
Source frame 90
...
```

The video path, output directory, and frame step are configured through YAML.

## Frame sampling preview

The extracted frames can be displayed in a vertical sequence with labels showing their positions in the original video.

![Extracted frames preview](docs/images/result.png)

![Additional extracted frames preview](docs/images/result_2.png)

## Running the project

Configure the input video and frame step in:

```text
configs/default.yaml
```

Then run:

```bash
python -m fightlens
```

The application reads the YAML configuration and starts the video preprocessing pipeline.