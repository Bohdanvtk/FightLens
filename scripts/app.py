"""Minimal Streamlit GUI — a thin UI layer over the existing pipeline/search/rerank logic.

Launch with `streamlit run scripts/app.py`. Reads the same YAML config as the
CLI and calls the exact same run_extract/run_describe/run_embed/Retriever/
rerank building blocks the CLI (`fightlens.__main__`) uses — no pipeline,
search, or rerank logic lives here. Writes nothing except what the Pipeline
tab's steps normally write (extract/describe/embed, same as the CLI); the
Config tab's overrides only apply for this session. Calls Gemini only for
"Rerank" and the "Describe" pipeline step.
"""

import copy
import io
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# Make `import fightlens...` work regardless of how/where this script is
# launched from, the same way the package's own PROJECT_ROOT is derived.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fightlens import gemini
from fightlens.__main__ import run_describe, run_embed, run_extract
from fightlens.config import (
    load_config,
    resolve_project_path,
    validate_descriptions_config,
    validate_embedding_config,
    validate_error_log_dir,
    validate_preview_config,
    validate_rerank_config,
    validate_search_config,
    validate_video_config,
    video_processed_dir,
)
from fightlens.embeddings import Embedder
from fightlens.errorlog import ErrorLog
from fightlens.rerank import rerank
from fightlens.search import Retriever, format_timecode

_FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_GIF_MAX_WIDTH = 480  # a result card is far narrower than the source frames

# "auto" preview fps: pick a rate so a window's frames play over ~this many
# seconds regardless of how many were sampled, clamped to a watchable range.
_AUTO_TARGET_SECONDS = 2.5
_AUTO_MIN_FPS = 1.0
_AUTO_MAX_FPS = 24.0


@st.cache_data(show_spinner=False)
def _count_window_frames(window_dir: str) -> int:
    """Count the sampled frame images in a window folder (0 if missing/empty)."""

    frame_dir = Path(window_dir)
    if not frame_dir.is_dir():
        return 0
    return sum(
        1
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _FRAME_SUFFIXES
    )


def _resolve_fps(window_dir: str, preview_fps: float | str) -> float:
    """Turn preview.fps (a number or "auto") into a concrete fps for this window."""

    if preview_fps != "auto":
        return float(preview_fps)

    count = _count_window_frames(window_dir)
    if count == 0:
        return _AUTO_MIN_FPS
    return min(max(count / _AUTO_TARGET_SECONDS, _AUTO_MIN_FPS), _AUTO_MAX_FPS)


@st.cache_resource(show_spinner=False)
def _ffmpeg_available() -> bool:
    """Whether the system `ffmpeg` binary is on PATH; checked once per session."""

    return shutil.which("ffmpeg") is not None


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_retriever(
    embedding_params: dict, embeddings_path: str, descriptions_path: str
) -> Retriever:
    """Build the Embedder and Retriever exactly as `_run_search` does; cached across reruns."""

    embedder = Embedder(
        model_name=embedding_params["model_name"],
        device=embedding_params["device"],
        normalize=embedding_params["normalize"],
        batch_size=embedding_params["batch_size"],
    )
    return Retriever.from_paths(
        embeddings_path=embeddings_path,
        descriptions_path=descriptions_path,
        embedder=embedder,
    )


@st.cache_data(show_spinner=False)
def _build_window_video(window_dir: str, fps: float) -> bytes | None:
    """Assemble a window folder's sampled frames into a browser-playable H.264 MP4 clip.

    Shells out to the system `ffmpeg` (OpenCV's own VideoWriter codecs, e.g.
    mp4v, are not reliably decodable by browsers) piping the frames straight
    in as image2pipe input. Returns None (never raises) on any failure — no
    ffmpeg, empty/missing folder, bad frames — so a result card without a
    playable window just skips the video, gracefully.
    """

    if not _ffmpeg_available():
        return None

    frame_dir = Path(window_dir)
    if not frame_dir.is_dir():
        return None

    frame_paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _FRAME_SUFFIXES
    )
    if not frame_paths:
        return None

    frame_bytes = b"".join(path.read_bytes() for path in frame_paths)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-movflags", "+faststart",
        str(tmp_path),
    ]
    try:
        subprocess.run(command, input=frame_bytes, capture_output=True, timeout=30, check=True)
        data = tmp_path.read_bytes()
    except (subprocess.SubprocessError, OSError):
        data = b""
    finally:
        tmp_path.unlink(missing_ok=True)

    return data or None


@st.cache_data(show_spinner=False)
def _build_window_gif(window_dir: str, fps: float) -> bytes | None:
    """Assemble a window folder's sampled frames into an animated GIF preview.

    Pure Python (Pillow), no system binaries — works on every machine that
    can run Streamlit itself, unlike the ffmpeg-based MP4 player. Returns
    None (never raises) on any failure, so a result card without a playable
    window just skips the preview, gracefully.
    """

    frame_dir = Path(window_dir)
    if not frame_dir.is_dir():
        return None

    frame_paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _FRAME_SUFFIXES
    )
    if not frame_paths:
        return None

    try:
        frames = [Image.open(path).convert("RGB") for path in frame_paths]
    except OSError:
        return None

    # Downscale to card size — the source frames are far bigger than they're
    # ever displayed at, and GIF is not an efficient codec for full-res photos.
    if frames[0].width > _GIF_MAX_WIDTH:
        ratio = _GIF_MAX_WIDTH / frames[0].width
        target_size = (_GIF_MAX_WIDTH, round(frames[0].height * ratio))
        frames = [frame.resize(target_size) for frame in frames]

    buffer = BytesIO()
    duration_ms = max(round(1000 / fps), 20)
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return buffer.getvalue()


def _build_window_preview(window_dir: str, fps: float, player: str) -> tuple[bytes | None, str]:
    """Render the requested player, falling back to the GIF (always available) on any failure."""

    if player == "mp4":
        video_bytes = _build_window_video(window_dir, fps)
        if video_bytes:
            return video_bytes, "mp4"

    return _build_window_gif(window_dir, fps), "gif"


def _effective_config(base_config: dict) -> dict:
    """Merge this session's Config-tab overrides onto the YAML config (session-only, never saved)."""

    config = copy.deepcopy(base_config)
    for key, values in st.session_state.config_overrides.items():
        if isinstance(values, dict):
            config.setdefault(key, {})
            config[key].update(values)
        else:
            config[key] = values  # top-level scalar, e.g. error_log_dir
    return config


def _render_config_tab(base_config: dict) -> None:
    """Let the user override every YAML parameter for this session only (nothing is saved to disk)."""

    st.caption(
        "Override YAML parameters for this run only — nothing is written to "
        "`configs/default.yaml`. Nullable fields use 0 to mean the null "
        "option (see each field's hint)."
    )

    current = _effective_config(base_config)
    try:
        video_c = validate_video_config(current.get("video"))
        desc_c = validate_descriptions_config(current.get("descriptions"))
        embedding_c = validate_embedding_config(current.get("embedding"))
        search_c = validate_search_config(current.get("search"))
        rerank_c = validate_rerank_config(current.get("rerank"))
        preview_c = validate_preview_config(current.get("preview"))
        error_log_c = validate_error_log_dir(current.get("error_log_dir"))
    except ValueError as error:
        st.error(f"Current overrides are invalid: {error}")
        if st.session_state.config_overrides:
            st.button(
                "Reset to configs/default.yaml",
                on_click=lambda: st.session_state.update(config_overrides={}),
            )
        return

    devices = ["auto", "cpu", "cuda"]
    players = ["gif", "mp4"]

    with st.form("config_overrides_form"):
        st.markdown("**video**")
        c1, c2 = st.columns(2)
        input_path = c1.text_input("input_path", value=video_c["input_path"])
        output_dir = c2.text_input("output_dir", value=video_c["output_dir"])
        c1, c2, c3 = st.columns(3)
        n_sec_per_window = c1.number_input(
            "n_sec_per_window", min_value=0.1, value=video_c["n_sec_per_window"], step=0.1
        )
        n_img_per_window = c2.number_input(
            "n_img_per_window", min_value=1, value=video_c["n_img_per_window"], step=1
        )
        overwrite = c3.checkbox("overwrite", value=video_c["overwrite"])
        c1, c2, c3, c4 = st.columns(4)
        fps_override = c1.number_input(
            "fps_override", min_value=0.0, value=video_c["fps_override"] or 0.0,
            step=1.0, help="0 = read FPS from video metadata.",
        )
        start_seconds = c2.number_input(
            "start_seconds", min_value=0.0, value=video_c["start_seconds"], step=1.0
        )
        end_seconds = c3.number_input(
            "end_seconds", min_value=0.0, value=video_c["end_seconds"] or 0.0,
            step=1.0, help="0 = until the end of the video.",
        )
        max_windows = c4.number_input(
            "max_windows", min_value=0, value=video_c["max_windows"] or 0,
            step=1, help="0 = no limit.",
        )

        st.markdown("**descriptions**")
        manifest_path = st.text_input("manifest_path", value=desc_c["manifest_path"])
        c1, c2, c3 = st.columns(3)
        request_delay = c1.number_input(
            "request_delay_seconds", min_value=0.0, value=desc_c["request_delay_seconds"], step=0.5
        )
        retry_attempts = c2.number_input(
            "retry_attempts", min_value=0, value=desc_c["retry_attempts"], step=1
        )
        response_timeout = c3.number_input(
            "response_timeout_seconds", min_value=0.0,
            value=desc_c["response_timeout_seconds"] or 0.0, step=5.0,
            help="0 = wait forever (no timeout).",
        )
        prompt = st.text_area("prompt", value=desc_c["prompt"], height=180)

        st.markdown("**embedding**")
        c1, c2, c3, c4 = st.columns(4)
        model_name = c1.text_input("model_name", value=embedding_c["model_name"])
        device = c2.selectbox(
            "device", devices,
            index=devices.index(embedding_c["device"]) if embedding_c["device"] in devices else 0,
        )
        batch_size = c3.number_input(
            "batch_size", min_value=1, value=embedding_c["batch_size"], step=1
        )
        normalize = c4.checkbox("normalize", value=embedding_c["normalize"])

        st.markdown("**search · rerank · preview · general**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            top_k = st.number_input("search.top_k", min_value=1, value=search_c["top_k"], step=1)
            error_log_dir = st.text_input("error_log_dir", value=error_log_c)
        with c2:
            rerank_enabled = st.checkbox("rerank.enabled", value=rerank_c["enabled"])
            top_n = st.number_input("rerank.top_n", min_value=1, value=rerank_c["top_n"], step=1)
        with c3:
            fps_is_auto = preview_c["fps"] == "auto"
            fps_auto = st.checkbox(
                "preview.fps = auto",
                value=fps_is_auto,
                help="Set fps automatically per window (frame count / ~2.5s) so every "
                "preview plays over roughly the same short, comfortable duration.",
            )
            fps_num = st.number_input(
                "preview.fps (when not auto)",
                min_value=0.1,
                value=6.0 if fps_is_auto else preview_c["fps"],
                step=0.5,
                disabled=fps_auto,
            )
            player = st.selectbox(
                "preview.player", players, index=players.index(preview_c["player"]),
                help="gif: pure Python, always works. mp4: needs ffmpeg, falls back to gif.",
            )

        applied = st.form_submit_button("Apply for this run")

    if applied:
        st.session_state.config_overrides = {
            "video": {
                "input_path": input_path,
                "output_dir": output_dir,
                "n_sec_per_window": float(n_sec_per_window),
                "n_img_per_window": int(n_img_per_window),
                "overwrite": overwrite,
                "fps_override": float(fps_override) if fps_override > 0 else None,
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds) if end_seconds > 0 else None,
                "max_windows": int(max_windows) if max_windows > 0 else None,
            },
            "descriptions": {
                "manifest_path": manifest_path,
                "request_delay_seconds": float(request_delay),
                "retry_attempts": int(retry_attempts),
                "response_timeout_seconds": float(response_timeout) if response_timeout > 0 else None,
                "prompt": prompt,
            },
            "embedding": {
                "model_name": model_name,
                "device": device,
                "batch_size": int(batch_size),
                "normalize": normalize,
            },
            "search": {"top_k": int(top_k)},
            "rerank": {"enabled": rerank_enabled, "top_n": int(top_n)},
            "preview": {"fps": "auto" if fps_auto else float(fps_num), "player": player},
            "error_log_dir": error_log_dir,
        }
        st.success("Applied — the Pipeline and Search tabs now use these values.")
        st.rerun()

    if st.session_state.config_overrides:
        st.button(
            "Reset to configs/default.yaml",
            on_click=lambda: st.session_state.update(config_overrides={}),
        )


_PIPELINE_STEPS = {
    "Extract": ("extract",),
    "Describe": ("describe",),
    "Embed": ("embed",),
    "Full (extract → describe → embed)": ("extract", "describe", "embed"),
}


def _render_pipeline_tab(config: dict) -> None:
    """Run extract/describe/embed — the exact same functions `python -m fightlens <step>` calls."""

    st.caption(
        "Runs against the config from the Config tab above (session "
        "overrides included). Same steps as the CLI, same files written."
    )

    # Render every button unconditionally, then see which was clicked — a
    # short-circuiting next() over the generator would stop drawing buttons
    # after the first clicked one, hiding the rest of the pipeline.
    columns = st.columns(len(_PIPELINE_STEPS))
    clicked = None
    for (label, column) in zip(_PIPELINE_STEPS, columns):
        if column.button(label, use_container_width=True):
            clicked = label

    if clicked is None:
        return

    output = io.StringIO()
    try:
        with st.spinner(f"Running {clicked}…"), redirect_stdout(output):
            for step in _PIPELINE_STEPS[clicked]:
                if step == "extract":
                    run_extract(config)
                elif step == "describe":
                    error_log = ErrorLog(
                        log_dir=resolve_project_path(
                            validate_error_log_dir(config.get("error_log_dir"))
                        )
                    )
                    run_describe(config, error_log)
                elif step == "embed":
                    run_embed(config)
    except Exception as error:
        st.error(f"{clicked} failed: {error}")

    if output.getvalue():
        st.code(output.getvalue(), language=None)


def _render_search_tab(config: dict) -> None:
    """The query box, controls, and result cards — reads whatever config (base or overridden) it's given."""

    try:
        search_defaults = validate_search_config(config.get("search"))
        rerank_defaults = validate_rerank_config(config.get("rerank"))
        embedding_params = validate_embedding_config(config.get("embedding"))
        preview_params = validate_preview_config(config.get("preview"))
        processed_dir = video_processed_dir(config)
    except (ValueError, FileNotFoundError) as error:
        st.error(f"Config error: {error}")
        return

    embeddings_path = processed_dir / "embeddings.npz"
    descriptions_path = processed_dir / "descriptions.json"

    if not embeddings_path.is_file():
        st.info(
            f"No embeddings found at `{embeddings_path}` — run "
            "`python -m fightlens embed` first."
        )
        return

    if preview_params["player"] == "mp4" and not _ffmpeg_available():
        st.warning(
            "`ffmpeg` was not found on this machine's PATH, so preview clips "
            "will use the GIF fallback instead of MP4 (search and rerank "
            "still work). Install ffmpeg for the MP4 player — e.g. "
            "`sudo apt install ffmpeg`, `brew install ffmpeg`, or "
            "`winget install ffmpeg` — or switch to `player: gif` (default) "
            "in the Config tab, which needs no extra install."
        )

    # Everything that would otherwise trigger a rerun (typing + losing focus,
    # dragging the number input, flipping the toggle) lives inside this form,
    # so a new search only fires on Enter in the query field or on "Search" —
    # never as a side effect of adjusting a control.
    with st.form("search_form"):
        query_input = st.text_input(
            "Query", placeholder='e.g. "left hook that slips past the guard"'
        )
        control_col1, control_col2, control_col3 = st.columns([2, 2, 1])
        with control_col1:
            top_k_input = st.number_input(
                "Results", min_value=1, value=search_defaults["top_k"], step=1
            )
        with control_col2:
            rerank_input = st.toggle("Rerank", value=rerank_defaults["enabled"])
        with control_col3:
            st.write("")  # aligns the button with the inputs above
            submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted:
        st.session_state.last_search = {
            "query": query_input,
            "top_k": top_k_input,
            "rerank_on": rerank_input,
        }

    st.divider()

    search_state = st.session_state.get("last_search")
    if not search_state or not search_state["query"]:
        st.caption("Type a query above and press Enter, or click Search.")
        return

    query = search_state["query"]
    top_k = search_state["top_k"]
    rerank_on = search_state["rerank_on"]

    retriever = _load_retriever(embedding_params, str(embeddings_path), str(descriptions_path))

    # Same two-stage flow as _run_search: rerank OFF -> retrieve top_k directly;
    # rerank ON -> retrieve the wider top_n net, rerank it, then trim to top_k.
    effective_k = rerank_defaults["top_n"] if rerank_on else top_k
    with st.spinner("Searching…"):
        results = retriever.search(query, effective_k)

    rerank_seconds = None
    if rerank_on and results:
        with st.spinner("Reranking…"):
            start = time.perf_counter()
            results = rerank(query, results, gemini.generate_text, model=gemini.GEMINI_MODEL)
            rerank_seconds = time.perf_counter() - start
        results = results[:top_k]

    if not results:
        st.info("No windows to search — run `python -m fightlens embed` first.")
        return

    status = f"Top {len(results)} of {len(retriever)} windows"
    if rerank_seconds is not None:
        status += f" · reranked in {rerank_seconds:.1f}s"
    st.caption(status)

    # Building each card's preview clip is the slow part (an ffmpeg subprocess
    # per window for the mp4 player, or GIF encoding for the default one) —
    # show visible progress so the app never looks frozen while it works.
    progress = st.progress(0.0, text=f"Rendering 0/{len(results)} previews…")
    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            col_media, col_body = st.columns([2, 5])

            with col_media:
                window_dir = processed_dir / "windows" / result.window_id
                fps = _resolve_fps(str(window_dir), preview_params["fps"])
                preview_bytes, kind = _build_window_preview(
                    str(window_dir), fps, preview_params["player"]
                )
                if preview_bytes and kind == "mp4":
                    st.video(preview_bytes, format="video/mp4")
                elif preview_bytes:
                    st.image(preview_bytes)

            with col_body:
                if result.start_sec is not None and result.end_sec is not None:
                    timecode = (
                        f"{format_timecode(result.start_sec)}"
                        f"–{format_timecode(result.end_sec)}"
                    )
                else:
                    timecode = "--:--.-–--:--.-"

                st.markdown(
                    f"**#{rank}**&nbsp;&nbsp;"
                    f"<span class='fl-badge'>{result.score:.2f}</span>&nbsp;&nbsp;"
                    f"`{timecode}`&nbsp;&nbsp;·&nbsp;&nbsp;{result.window_id}",
                    unsafe_allow_html=True,
                )
                st.write(result.description or "*(no description)*")

        progress.progress(rank / len(results), text=f"Rendering {rank}/{len(results)} previews…")

    progress.empty()


st.set_page_config(page_title="FightLens", page_icon="🥊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
    }
    .fl-badge {
        display: inline-block;
        padding: 0.05rem 0.55rem;
        border-radius: 999px;
        background-color: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🥊 FightLens")
st.caption("Video pipeline and semantic search.")

if "config_overrides" not in st.session_state:
    st.session_state.config_overrides = {}

base_config = load_config()

tab_pipeline, tab_search, tab_config = st.tabs(["Pipeline", "Search", "Config"])

with tab_pipeline:
    _render_pipeline_tab(_effective_config(base_config))

with tab_search:
    _render_search_tab(_effective_config(base_config))

with tab_config:
    _render_config_tab(base_config)
