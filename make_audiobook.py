#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.0.0",
#   "pymupdf>=1.24.0",
#   "pillow>=10.0.0",
#   "tiktoken>=0.7.0",
#   "kokoro>=0.9.2",
#   "audiotsm>=0.1.2",
#   "imageio-ffmpeg>=0.5.1",
#   "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git",
# ]
# ///

"""Build a resumable audiobook from a scanned PDF.

Run with:

    uv run make_audiobook.py book.pdf

The script uses inline uv metadata, so uv installs the Python dependencies
automatically. Set OPENAI_API_KEY in the environment before running it.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import difflib
import io
import json
import os
import re
import subprocess
import sys
import time
import warnings
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from PIL import Image


DEFAULT_OCR_MODEL = "gpt-5.4-nano"
DEFAULT_CLEAN_MODEL = "gpt-5.4-nano"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
DEFAULT_KOKORO_VOICE = "am_adam"
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_MLX_CHATTERBOX_MODEL = "mlx-community/chatterbox-fp16"
DEFAULT_IMAGE_SIZE = 1024
MAX_TTS_TOKENS = 1800
PAGE_RENDER_DPI = 300
RETRY_ATTEMPTS = 3
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"


MODEL_PRICES_USD_PER_1M = {
    "gpt-5.5": {"input_tokens": 5.00, "cached_input_tokens": 0.50, "output_tokens": 30.00},
    "gpt-5.4": {"input_tokens": 2.50, "cached_input_tokens": 0.25, "output_tokens": 15.00},
    "gpt-5.4-mini": {"input_tokens": 0.75, "cached_input_tokens": 0.075, "output_tokens": 4.50},
    "gpt-5.4-nano": {"input_tokens": 0.20, "cached_input_tokens": 0.02, "output_tokens": 1.25},
    "gpt-5": {"input_tokens": 1.25, "cached_input_tokens": 0.125, "output_tokens": 10.00},
    "gpt-5-mini": {"input_tokens": 0.25, "cached_input_tokens": 0.025, "output_tokens": 2.00},
    "gpt-5-nano": {"input_tokens": 0.05, "cached_input_tokens": 0.005, "output_tokens": 0.40},
    "gpt-4.1": {"input_tokens": 2.00, "cached_input_tokens": 0.50, "output_tokens": 8.00},
    "gpt-4.1-mini": {"input_tokens": 0.40, "cached_input_tokens": 0.10, "output_tokens": 1.60},
    "gpt-4.1-nano": {"input_tokens": 0.10, "cached_input_tokens": 0.025, "output_tokens": 0.40},
}


KOKORO_VOICES = {
    "American English": [
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
    ],
    "British English": [
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    ],
    "Other languages": [
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ],
}


OCR_PROMPT = """\
Convert this scanned book page to Markdown.

Rules:
- Transcribe the page; do not summarize or add commentary.
- Preserve paragraphs, headings, lists, emphasis, and obvious scene breaks.
- Preserve table-of-contents lines and chapter-opening lines as written,
  including bare numbered chapter headings such as "1" followed by a title.
- Omit page numbers, running headers, and running footers when they are not part
  of the book text.
- Never turn a standalone page number into a chapter heading.
- If a word is unreadable, write [unclear] rather than guessing wildly.
- Return only Markdown.
"""


CLEAN_PROMPT = """\
Clean this OCR Markdown page for audiobook narration.

Rules:
- Keep the author's words and paragraph order.
- If the page begins with a chapter heading and chapter title, keep both at the
  top. Chapter titles are part of the audiobook structure and must not be
  removed.
- Some books use bare numbered chapter openers, such as "1" followed by the
  title on the next line. Preserve those opener numbers and title lines instead
  of dropping them or turning them into ordinary section headings.
- Remove scanning artifacts, page numbers, running headers, running footers,
  footnote markers, and anything that should not be read aloud.
- Do not summarize, modernize, censor, or add commentary.
- Preserve section titles that should be narrated.
- Return only the cleaned narration text.
"""


@dataclass(frozen=True)
class Config:
    pdf: Path | None
    output_dir: Path
    audio_engine: str
    ocr_model: str
    clean_model: str
    tts_model: str
    voice: str
    chapters: tuple[int, ...]
    mlx_model: str
    mlx_ref_audio: Path | None
    mlx_ref_text: str | None
    mlx_max_tokens: int
    mlx_chunk_chars: int
    mlx_speed: float
    kokoro_language: str | None
    kokoro_speed: float
    chapter_announcement_lead_silence: float
    chapter_announcement_trail_silence: float
    m4b: bool
    intro_wavs: tuple[Path, ...]
    cover: Path | None
    no_cover: bool
    m4b_bitrate: str
    sample_text: str | None
    sample_output: Path | None
    image_size: int
    raw_ocr: bool
    text_only: bool
    overwrite: bool
    refresh: tuple[str, ...]
    list_voices: bool


@dataclass(frozen=True)
class RunPaths:
    root: Path
    images: Path
    markdown_pages: Path
    cleaned_pages: Path
    audio_chunks: Path
    chapter_audio: Path
    raw_book: Path
    cleaned_book: Path
    audiobook_text: Path
    audiobook: Path
    m4b: Path
    m4b_metadata: Path
    m4b_cover: Path
    chapters: Path
    chapter_texts: Path
    cost_report: Path
    manifest: Path


@dataclass(frozen=True)
class Chapter:
    number: int
    page: int
    title: str
    source_file: str
    raw_heading: str


@dataclass(frozen=True)
class TocEntry:
    number: int
    title: str
    source_file: str
    source_page: int
    raw_line: str
    printed_page: int | None = None


@dataclass(frozen=True)
class FrontMatterSection:
    title: str
    page: int
    source_file: str
    raw_heading: str


@dataclass(frozen=True)
class AudioTrack:
    title: str
    path: Path
    duration_ms: int
    text_path: Path | None = None


def parse_chapter_selection(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()

    chapters = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) <= 0:
            raise argparse.ArgumentTypeError("--chapters must contain positive integers.")
        chapters.append(int(part))
    return tuple(dict.fromkeys(chapters))


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(
        description="Build a resumable audiobook from a scanned PDF."
    )
    parser.add_argument("pdf", type=Path, nargs="?", help="Scanned PDF to narrate.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where converted files will be written.",
    )
    parser.add_argument(
        "--audio-engine",
        choices=("openai", "kokoro", "mlx-chatterbox"),
        default="openai",
        help="Audio backend used when synthesizing the audiobook.",
    )
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help="OpenAI model used to convert page images to Markdown.",
    )
    parser.add_argument(
        "--clean-model",
        default=DEFAULT_CLEAN_MODEL,
        help="OpenAI model used to clean Markdown for narration.",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help="OpenAI text-to-speech model used for the audiobook.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice used for audio synthesis. Defaults to alloy for OpenAI, am_adam for Kokoro.",
    )
    parser.add_argument(
        "--chapters",
        help="Comma-separated chapter numbers to synthesize, e.g. 1 or 1,2,3.",
    )
    parser.add_argument(
        "--mlx-model",
        default=DEFAULT_MLX_CHATTERBOX_MODEL,
        help="MLX-Audio model used by the mlx-chatterbox backend.",
    )
    parser.add_argument(
        "--mlx-ref-audio",
        type=Path,
        help="Reference WAV for Chatterbox voice cloning.",
    )
    parser.add_argument(
        "--mlx-ref-text",
        help="Transcript for the Chatterbox reference WAV.",
    )
    parser.add_argument(
        "--mlx-max-tokens",
        type=int,
        default=4_000,
        help="Maximum generated tokens for each MLX-Audio segment.",
    )
    parser.add_argument(
        "--mlx-chunk-chars",
        type=int,
        default=350,
        help="Approximate text characters per MLX-Audio segment before stitching.",
    )
    parser.add_argument(
        "--mlx-speed",
        type=float,
        default=1.0,
        help="Post-process MLX Chatterbox audio speed. Use values below 1.0 to slow speech.",
    )
    parser.add_argument(
        "--kokoro-language",
        choices=("a", "b", "e", "f", "h", "i", "p", "j", "z"),
        help="Kokoro language code. Defaults to the first character of the Kokoro voice.",
    )
    parser.add_argument(
        "--kokoro-speed",
        type=float,
        default=1.0,
        help="Kokoro speech speed.",
    )
    parser.add_argument(
        "--chapter-announcement-lead-silence",
        type=float,
        default=0.4,
        help="Seconds of silence before each chapter announcement.",
    )
    parser.add_argument(
        "--chapter-announcement-trail-silence",
        type=float,
        default=1.0,
        help="Seconds of silence after each chapter announcement.",
    )
    parser.add_argument(
        "--m4b",
        action="store_true",
        help="Package per-chapter WAV files into a chapterized audiobook.m4b.",
    )
    parser.add_argument(
        "--intro-wav",
        action="append",
        type=Path,
        default=[],
        help="Intro WAV to prepend to the M4B. Can be used more than once.",
    )
    parser.add_argument(
        "--cover",
        type=Path,
        help="Cover image to attach to the M4B. Defaults to the first rendered PDF page.",
    )
    parser.add_argument(
        "--no-cover",
        action="store_true",
        help="Do not attach cover art to the M4B.",
    )
    parser.add_argument(
        "--m4b-bitrate",
        default="64k",
        help="AAC audio bitrate for M4B packaging.",
    )
    parser.add_argument(
        "--sample-text",
        help="Generate a short Kokoro voice sample from this text and exit.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        help="Sample WAV path or directory. Defaults to voice_samples/<voice>.wav.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List known Kokoro voice IDs and exit.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Maximum page image width/height sent for OCR.",
    )
    parser.add_argument(
        "--raw-ocr",
        action="store_true",
        help="Narrate raw OCR text instead of cleaning it for audiobook narration.",
    )
    parser.add_argument(
        "--skip-clean",
        dest="raw_ocr",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Build the resumable text artifacts but do not synthesize audio.",
    )
    parser.add_argument(
        "--skip-audio",
        dest="text_only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing page Markdown, cleaned text, and audio chunks.",
    )
    parser.add_argument(
        "--refresh",
        action="append",
        choices=("ocr", "clean", "audio", "all"),
        default=[],
        help="Regenerate one cached stage without rebuilding the whole pipeline.",
    )

    args = parser.parse_args(argv)
    if args.mlx_speed <= 0:
        parser.error("--mlx-speed must be greater than 0.")
    if args.chapter_announcement_lead_silence < 0:
        parser.error("--chapter-announcement-lead-silence must not be negative.")
    if args.chapter_announcement_trail_silence < 0:
        parser.error("--chapter-announcement-trail-silence must not be negative.")
    if args.sample_text is not None and not args.sample_text.strip():
        parser.error("--sample-text must not be empty.")
    if not args.m4b and (args.intro_wav or args.cover):
        args.m4b = True
    voice = args.voice or (
        DEFAULT_KOKORO_VOICE
        if args.audio_engine == "kokoro" or args.sample_text is not None
        else DEFAULT_VOICE
    )
    chapters = parse_chapter_selection(args.chapters)
    return Config(
        pdf=args.pdf.expanduser() if args.pdf else None,
        output_dir=args.output_dir.expanduser(),
        audio_engine=args.audio_engine,
        ocr_model=args.ocr_model,
        clean_model=args.clean_model,
        tts_model=args.tts_model,
        voice=voice,
        chapters=chapters,
        mlx_model=args.mlx_model,
        mlx_ref_audio=args.mlx_ref_audio.expanduser() if args.mlx_ref_audio else None,
        mlx_ref_text=args.mlx_ref_text,
        mlx_max_tokens=args.mlx_max_tokens,
        mlx_chunk_chars=args.mlx_chunk_chars,
        mlx_speed=args.mlx_speed,
        kokoro_language=args.kokoro_language,
        kokoro_speed=args.kokoro_speed,
        chapter_announcement_lead_silence=args.chapter_announcement_lead_silence,
        chapter_announcement_trail_silence=args.chapter_announcement_trail_silence,
        m4b=args.m4b,
        intro_wavs=tuple(path.expanduser() for path in args.intro_wav),
        cover=args.cover.expanduser() if args.cover else None,
        no_cover=args.no_cover,
        m4b_bitrate=args.m4b_bitrate,
        sample_text=args.sample_text,
        sample_output=args.sample_output.expanduser() if args.sample_output else None,
        image_size=args.image_size,
        raw_ocr=args.raw_ocr,
        text_only=args.text_only,
        overwrite=args.overwrite,
        refresh=tuple(args.refresh),
        list_voices=args.list_voices,
    )


def make_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI()


def make_paths(config: Config) -> RunPaths:
    if config.pdf is None:
        raise ValueError("A PDF path is required.")
    root = config.output_dir / config.pdf.stem
    return RunPaths(
        root=root,
        images=root / "page_images",
        markdown_pages=root / "markdown_pages",
        cleaned_pages=root / "cleaned_pages",
        audio_chunks=root / "audio_chunks",
        chapter_audio=root / "chapter_audio",
        raw_book=root / "book.md",
        cleaned_book=root / "cleaned_book.md",
        audiobook_text=root / "audiobook_text.md",
        audiobook=root / "audiobook.wav",
        m4b=root / "audiobook.m4b",
        m4b_metadata=root / "audiobook.ffmetadata",
        m4b_cover=root / "audiobook_cover.jpg",
        chapters=root / "chapters.json",
        chapter_texts=root / "chapters",
        cost_report=root / "cost_report.json",
        manifest=root / "run_manifest.jsonl",
    )


def ensure_dirs(paths: RunPaths) -> None:
    for path in (
        paths.root,
        paths.images,
        paths.markdown_pages,
        paths.cleaned_pages,
        paths.audio_chunks,
        paths.chapter_audio,
        paths.chapter_texts,
    ):
        path.mkdir(parents=True, exist_ok=True)


def log_event(paths: RunPaths, event: str, **fields: object) -> None:
    payload = {"event": event, "time": round(time.time(), 3), **fields}
    with paths.manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def call_with_retries(label: str, func):
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - we want to retry API/network errors.
            last_error = exc
            if attempt == RETRY_ATTEMPTS:
                break
            sleep_for = 2**attempt
            print(f"{label} failed on attempt {attempt}; retrying in {sleep_for}s: {exc}")
            time.sleep(sleep_for)
    raise RuntimeError(f"{label} failed after {RETRY_ATTEMPTS} attempts") from last_error


def should_refresh(config: Config, stage: str) -> bool:
    return config.overwrite or "all" in config.refresh or stage in config.refresh


def render_pdf_pages(config: Config, paths: RunPaths) -> list[Path]:
    print("Rendering PDF pages...")
    log_event(paths, "stage_started", stage="render_pages", pdf=str(config.pdf))
    doc = fitz.open(str(config.pdf))
    try:
        image_paths: list[Path] = []
        rendered_count = 0
        reused_count = 0
        for page_index in range(len(doc)):
            image_path = paths.images / f"page_{page_index + 1:03d}.png"
            image_paths.append(image_path)
            if image_path.exists() and not should_refresh(config, "ocr"):
                reused_count += 1
                continue

            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(dpi=PAGE_RENDER_DPI)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.thumbnail(
                (config.image_size, config.image_size),
                Image.Resampling.LANCZOS,
            )
            image.save(image_path)
            log_event(
                paths,
                "rendered_page",
                page=page_index + 1,
                file=str(image_path),
                width=image.width,
                height=image.height,
            )
            rendered_count += 1
    finally:
        doc.close()

    print(f"Rendered or reused {len(image_paths)} page image(s).")
    log_event(
        paths,
        "stage_completed",
        stage="render_pages",
        pages=len(image_paths),
        rendered=rendered_count,
        reused=reused_count,
    )
    return image_paths


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def ocr_page(client: OpenAI, config: Config, image_path: Path) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=config.ocr_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        max_completion_tokens=4096,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    return content.strip(), usage


def convert_images_to_markdown(
    client: OpenAI | None, config: Config, paths: RunPaths, image_paths: Iterable[Path]
) -> list[Path]:
    print("Converting page images to Markdown...")
    log_event(paths, "stage_started", stage="ocr_pages", model=config.ocr_model)
    markdown_paths: list[Path] = []
    generated_count = 0
    reused_count = 0
    for index, image_path in enumerate(image_paths, start=1):
        markdown_path = paths.markdown_pages / f"page_{index:03d}.md"
        markdown_paths.append(markdown_path)

        if markdown_path.exists() and not should_refresh(config, "ocr"):
            reused_count += 1
            continue

        print(f"OCR page {index:03d}...")
        if client is None:
            client = make_client()
        markdown, usage = call_with_retries(
            f"OCR page {index:03d}",
            lambda image_path=image_path: ocr_page(client, config, image_path),
        )
        markdown_path.write_text(markdown + "\n", encoding="utf-8")
        log_event(
            paths,
            "ocr_page",
            page=index,
            file=str(markdown_path),
            model=config.ocr_model,
            usage=usage,
        )
        generated_count += 1

    print(f"Wrote or reused {len(markdown_paths)} Markdown page(s).")
    log_event(
        paths,
        "stage_completed",
        stage="ocr_pages",
        pages=len(markdown_paths),
        generated=generated_count,
        reused=reused_count,
    )
    return markdown_paths


def clean_markdown_page(client: OpenAI, config: Config, text: str) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=config.clean_model,
        messages=[
            {"role": "system", "content": CLEAN_PROMPT},
            {"role": "user", "content": text},
        ],
        max_completion_tokens=4096,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    return content.strip(), usage


def clean_markdown_pages(
    client: OpenAI | None, config: Config, paths: RunPaths, markdown_paths: Iterable[Path]
) -> list[Path]:
    if config.raw_ocr:
        markdown_paths = list(markdown_paths)
        log_event(
            paths,
            "stage_skipped",
            stage="clean_pages",
            reason="raw_ocr",
            pages=len(markdown_paths),
        )
        return markdown_paths

    print("Cleaning Markdown for narration...")
    log_event(paths, "stage_started", stage="clean_pages", model=config.clean_model)
    cleaned_paths: list[Path] = []
    generated_count = 0
    reused_count = 0
    for markdown_path in markdown_paths:
        cleaned_path = paths.cleaned_pages / markdown_path.name
        cleaned_paths.append(cleaned_path)

        if cleaned_path.exists() and not should_refresh(config, "clean"):
            reused_count += 1
            continue

        print(f"Clean {markdown_path.stem}...")
        if client is None:
            client = make_client()
        raw_text = markdown_path.read_text(encoding="utf-8")
        cleaned, usage = call_with_retries(
            f"Clean {markdown_path.stem}",
            lambda raw_text=raw_text: clean_markdown_page(client, config, raw_text),
        )
        cleaned_path.write_text(cleaned + "\n", encoding="utf-8")
        log_event(
            paths,
            "clean_page",
            page=markdown_path.stem,
            file=str(cleaned_path),
            model=config.clean_model,
            usage=usage,
        )
        generated_count += 1

    print(f"Wrote or reused {len(cleaned_paths)} cleaned page(s).")
    log_event(
        paths,
        "stage_completed",
        stage="clean_pages",
        pages=len(cleaned_paths),
        generated=generated_count,
        reused=reused_count,
    )
    return cleaned_paths


def merge_pages(page_paths: Iterable[Path], output_path: Path, paths: RunPaths | None = None) -> None:
    parts = []
    page_paths = list(page_paths)
    if paths:
        log_event(
            paths,
            "stage_started",
            stage="merge_pages",
            output=str(output_path),
            pages=len(page_paths),
        )
    for page_path in page_paths:
        text = page_path.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    output_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Merged text written to {output_path}")
    if paths:
        log_event(
            paths,
            "stage_completed",
            stage="merge_pages",
            output=str(output_path),
            pages=len(page_paths),
            nonempty_pages=len(parts),
        )


def page_number_from_path(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer page number from {path}")
    return int(match.group(1))


def clean_title(text: str) -> str:
    text = re.sub(r"^[#*\s`]+|[#*\s`]+$", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip(":- ")


def plain_markdown_line(text: str) -> str:
    text = re.sub(r"^\s*#{1,6}\s*", "", text.strip())
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :-")


def normalize_match_text(text: str) -> str:
    text = plain_markdown_line(text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cleaned_toc_title(raw_title: str, printed_page: int | None) -> str:
    title = raw_title
    if printed_page is not None:
        title = re.sub(rf"\s+{printed_page}\s*$", "", title)
    title = re.sub(r"\.{2,}\s*\d+\s*$", "", title)
    title = re.sub(r"\s+\d{1,4}\s*$", "", title)
    return clean_title(plain_markdown_line(title))


def looks_like_toc_page(lines: list[str], page: int) -> bool:
    if page > 40:
        return False

    plain_lines = [plain_markdown_line(line) for line in lines[:30] if line.strip()]
    has_contents_heading = any(
        line.lower() in {"contents", "table of contents"} for line in plain_lines[:8]
    )
    numbered_entries = sum(
        1
        for line in plain_lines
        if re.match(r"^\d{1,3}[.)]?\s+\S+", line)
    )
    entry_numbers = [
        int(match.group(1))
        for line in plain_lines
        if (match := re.match(r"^(\d{1,3})[.)]?\s+\S+", line))
    ]
    starts_like_toc = 1 in entry_numbers and 2 in entry_numbers
    return has_contents_heading or (numbered_entries >= 3 and starts_like_toc)


def extract_toc_entries(markdown_paths: Iterable[Path]) -> list[TocEntry]:
    entries: list[TocEntry] = []
    seen_numbers: set[int] = set()

    for markdown_path in sorted(markdown_paths):
        page = page_number_from_path(markdown_path)
        lines = markdown_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not looks_like_toc_page(lines, page):
            continue

        for raw_line in lines:
            line = plain_markdown_line(raw_line)
            match = re.match(
                r"^(?P<number>\d{1,3})[.)]?\s+"
                r"(?P<title>.+?)"
                r"(?:\s+\.{2,}\s*|\s+)?"
                r"(?P<printed_page>\d{1,4})?\s*$",
                line,
            )
            if not match:
                continue

            number = int(match.group("number"))
            printed_page = (
                int(match.group("printed_page"))
                if match.group("printed_page")
                else None
            )
            title = cleaned_toc_title(match.group("title"), printed_page)
            normalized_title = normalize_match_text(title)
            if (
                number in seen_numbers
                or number <= 0
                or number > 100
                or len(normalized_title) < 6
                or normalized_title in {"contents", "index"}
            ):
                continue

            seen_numbers.add(number)
            entries.append(
                TocEntry(
                    number=number,
                    title=title,
                    source_file=str(markdown_path),
                    source_page=page,
                    raw_line=raw_line.strip(),
                    printed_page=printed_page,
                )
            )

    return sorted(entries, key=lambda entry: entry.number)


def page_opener_lines(markdown_path: Path) -> list[str]:
    lines = markdown_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    clean_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            continue
        cleaned = plain_markdown_line(line)
        if cleaned:
            clean_lines.append(cleaned)
        if len(clean_lines) >= 12:
            break
    return clean_lines


def toc_entry_match_score(entry: TocEntry, opener_lines: list[str]) -> tuple[float, str]:
    if not opener_lines:
        return 0.0, ""

    title = normalize_match_text(entry.title)
    joined = normalize_match_text(" ".join(opener_lines[:8]))
    if not title or not joined:
        return 0.0, ""

    best_line_ratio = max(
        difflib.SequenceMatcher(None, title, normalize_match_text(line)).ratio()
        for line in opener_lines[:8]
    )
    title_words = set(title.split())
    joined_words = set(joined.split())
    word_coverage = len(title_words & joined_words) / max(len(title_words), 1)

    title_score = 0.0
    if title in joined:
        title_score = 5.0
    elif best_line_ratio >= 0.88:
        title_score = 4.0
    elif word_coverage >= 0.78:
        title_score = 3.0

    number_score = 0.0
    for index, line in enumerate(opener_lines[:4]):
        if re.fullmatch(rf"{entry.number}[.)]?", line):
            number_score = 3.0 if index <= 1 else 1.5
            break

    # Running headers often repeat only the title. Prefer true opener pages when
    # a numbered chapter marker is present, while still allowing chapter 1 in
    # books whose first opener omits the number.
    if title_score and (number_score or entry.number == 1):
        score = title_score + number_score
    elif title_score >= 4.0:
        score = title_score - 1.0
    else:
        score = 0.0

    heading_preview = " / ".join(opener_lines[:6])
    if len(heading_preview) > 360:
        heading_preview = heading_preview[:357].rstrip() + "..."
    return score, heading_preview


def detect_chapters_from_toc(
    entries: list[TocEntry],
    markdown_paths: list[Path],
) -> list[Chapter]:
    if not entries:
        return []

    path_pages = [(page_number_from_path(path), path) for path in sorted(markdown_paths)]
    toc_pages = {entry.source_page for entry in entries}
    accepted: list[Chapter] = []
    min_page = 1

    for entry in entries:
        best_score = 0.0
        best_page: int | None = None
        best_path: Path | None = None
        best_heading = ""

        for page, markdown_path in path_pages:
            if page < min_page or page in toc_pages:
                continue

            opener_lines = page_opener_lines(markdown_path)
            if looks_like_toc_page(opener_lines, page):
                continue

            score, raw_heading = toc_entry_match_score(entry, opener_lines)
            if score > best_score:
                best_score = score
                best_page = page
                best_path = markdown_path
                best_heading = raw_heading

        if best_path is None or best_page is None or best_score < 4.0:
            continue

        accepted.append(
            Chapter(
                number=entry.number,
                page=best_page,
                title=entry.title,
                source_file=str(best_path),
                raw_heading=best_heading,
            )
        )
        min_page = best_page + 1

    return validate_chapter_candidates(accepted)


def find_chapter_candidate(markdown_path: Path) -> Chapter | None:
    lines = markdown_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    nonempty = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    page = page_number_from_path(markdown_path)

    for position, (line_index, line) in enumerate(nonempty[:10]):
        if line.startswith("```"):
            continue

        match = re.match(
            r"^(?P<hashes>#{0,6})\s*chapter\s+(?P<number>\d{1,3})\b\s*:?\s*(?P<title>.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        number = int(match.group("number"))
        title = clean_title(match.group("title"))
        if not title:
            for _, later_line in nonempty[position + 1 : position + 5]:
                title_match = re.match(r"^#{1,6}\s+(?P<title>.+)$", later_line)
                if title_match:
                    title = clean_title(title_match.group("title"))
                    break
                mostly_caps = later_line == later_line.upper() and len(later_line) > 6
                if mostly_caps and not re.fullmatch(r"\d+", later_line):
                    title = clean_title(later_line.title())
                    break

        return Chapter(
            number=number,
            page=page,
            title=title,
            source_file=str(markdown_path),
            raw_heading=line,
        )

    return None


def validate_chapter_candidates(candidates: Iterable[Chapter]) -> list[Chapter]:
    accepted: list[Chapter] = []

    for candidate in sorted(candidates, key=lambda chapter: (chapter.page, chapter.number)):
        if candidate.number <= 0:
            continue

        if not accepted:
            if candidate.number <= 3 and candidate.title:
                accepted.append(candidate)
            continue

        previous = accepted[-1]
        if candidate.number == previous.number:
            # Duplicate chapter starts can happen when rerendered or duplicated pages
            # exist in the source scan. Keep the first occurrence.
            continue

        expected = previous.number + 1
        if candidate.number == expected:
            accepted.append(candidate)
            continue

        small_gap = expected < candidate.number <= previous.number + 3
        if small_gap and candidate.title:
            accepted.append(candidate)

    return accepted


def detect_chapters(markdown_paths: Iterable[Path], paths: RunPaths) -> list[Chapter]:
    log_event(paths, "stage_started", stage="detect_chapters")
    markdown_paths = sorted(markdown_paths)
    candidates = [
        candidate
        for markdown_path in markdown_paths
        if (candidate := find_chapter_candidate(markdown_path)) is not None
    ]
    chapters = validate_chapter_candidates(candidates)
    toc_entries = extract_toc_entries(markdown_paths)
    toc_chapters: list[Chapter] = []
    if not chapters and toc_entries:
        toc_chapters = detect_chapters_from_toc(toc_entries, markdown_paths)
        chapters = toc_chapters

    payload = {
        "chapters": [asdict(chapter) for chapter in chapters],
        "rejected_candidates": [
            asdict(candidate) for candidate in candidates if candidate not in chapters
        ],
        "toc_entries": [asdict(entry) for entry in toc_entries],
        "toc_detected_chapters": [asdict(chapter) for chapter in toc_chapters],
    }
    paths.chapters.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Detected {len(chapters)} chapter start(s).")
    log_event(
        paths,
        "stage_completed",
        stage="detect_chapters",
        chapters=len(chapters),
        candidates=len(candidates),
        toc_entries=len(toc_entries),
        toc_detected_chapters=len(toc_chapters),
        rejected=len(candidates) - len(chapters),
        output=str(paths.chapters),
    )
    return chapters


def chapter_heading(chapter: Chapter) -> str:
    if chapter.title:
        return f"Chapter {chapter.number}: {chapter.title}"
    return f"Chapter {chapter.number}"


FRONT_MATTER_HEADINGS = {
    "foreword": "Foreword",
    "forward": "Foreword",
    "preface": "Preface",
    "introduction": "Introduction",
    "intro": "Introduction",
    "prologue": "Prologue",
    "author s note": "Author's Note",
    "author note": "Author's Note",
    "note to the reader": "Note to the Reader",
}


def front_matter_title_from_line(line: str) -> str | None:
    cleaned = normalize_match_text(line)
    if cleaned in FRONT_MATTER_HEADINGS:
        return FRONT_MATTER_HEADINGS[cleaned]

    for key, title in FRONT_MATTER_HEADINGS.items():
        if cleaned.startswith(key + " ") and len(cleaned.split()) <= len(key.split()) + 3:
            return title
    return None


def record_front_matter_metadata(
    paths: RunPaths, front_matter: FrontMatterSection | None
) -> None:
    try:
        payload = json.loads(paths.chapters.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}

    payload["front_matter"] = asdict(front_matter) if front_matter else None
    if front_matter:
        payload["narration_start_page"] = front_matter.page
        payload["narration_start_title"] = front_matter.title
    else:
        payload.pop("narration_start_page", None)
        payload.pop("narration_start_title", None)

    paths.chapters.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_chapter_metadata(paths: RunPaths, **fields: object) -> None:
    try:
        payload = json.loads(paths.chapters.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}

    for key, value in fields.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value

    paths.chapters.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def detect_front_matter_section(
    markdown_paths: Iterable[Path], chapters: list[Chapter], paths: RunPaths
) -> FrontMatterSection | None:
    log_event(paths, "stage_started", stage="detect_front_matter")
    first_chapter_page = min((chapter.page for chapter in chapters), default=None)
    search_limit = min(first_chapter_page - 1, 60) if first_chapter_page else 60
    front_matter: FrontMatterSection | None = None

    for markdown_path in sorted(markdown_paths):
        page = page_number_from_path(markdown_path)
        if page > search_limit:
            continue
        if first_chapter_page and page >= first_chapter_page:
            continue

        lines = page_opener_lines(markdown_path)
        if looks_like_toc_page(lines, page):
            continue

        for line in lines[:4]:
            title = front_matter_title_from_line(line)
            if not title:
                continue
            front_matter = FrontMatterSection(
                title=title,
                page=page,
                source_file=str(markdown_path),
                raw_heading=line,
            )
            break

        if front_matter:
            break

    record_front_matter_metadata(paths, front_matter)
    if front_matter:
        print(
            f"Detected audiobook start at page {front_matter.page}: "
            f"{front_matter.title}."
        )
    log_event(
        paths,
        "stage_completed",
        stage="detect_front_matter",
        detected=front_matter is not None,
        page=front_matter.page if front_matter else None,
        title=front_matter.title if front_matter else None,
    )
    return front_matter


def detect_narration_stop_page(
    markdown_paths: Iterable[Path], chapters: list[Chapter], paths: RunPaths
) -> int | None:
    log_event(paths, "stage_started", stage="detect_narration_stop")
    start_after = min((chapter.page for chapter in chapters), default=1)
    stop_page = None
    stop_title = None

    for markdown_path in sorted(markdown_paths):
        page = page_number_from_path(markdown_path)
        if page <= start_after:
            continue

        lines = page_opener_lines(markdown_path)
        if not lines:
            continue
        first_line = normalize_match_text(lines[0])
        if first_line == "index":
            stop_page = page
            stop_title = "Index"
            break

    update_chapter_metadata(
        paths,
        narration_stop_page=stop_page,
        narration_stop_title=stop_title,
    )
    if stop_page:
        print(f"Detected audiobook stop before page {stop_page}: {stop_title}.")
    log_event(
        paths,
        "stage_completed",
        stage="detect_narration_stop",
        detected=stop_page is not None,
        page=stop_page,
        title=stop_title,
    )
    return stop_page


def strip_leading_chapter_heading(text: str, chapter: Chapter) -> str:
    lines = text.splitlines()

    def trim_blank_prefix() -> None:
        while lines and not lines[0].strip():
            lines.pop(0)

    def pop_chapter_marker() -> bool:
        if lines and re.match(
            r"^#*\s*chapter\s+\d+\b", lines[0].strip(), re.IGNORECASE
        ):
            lines.pop(0)
            return True
        if lines and re.fullmatch(
            rf"#*\s*{chapter.number}[.)]?\s*", lines[0].strip()
        ):
            lines.pop(0)
            return True
        return False

    def pop_title_block() -> bool:
        if not chapter.title or not lines:
            return False

        title = normalize_match_text(chapter.title)
        title_words = set(title.split())
        consumed = 0
        for index in range(min(6, len(lines))):
            candidate_lines = [plain_markdown_line(line) for line in lines[: index + 1]]
            candidate = normalize_match_text(" ".join(candidate_lines))
            if not candidate:
                consumed = index + 1
                continue
            candidate_words = set(candidate.split())
            word_coverage = len(title_words & candidate_words) / max(len(title_words), 1)
            if (
                candidate == title
                or title in candidate
                or (title.endswith(candidate) and len(candidate) >= 6)
                or (title.startswith(candidate) and len(candidate) >= 12)
                or word_coverage >= 0.85
            ):
                consumed = index + 1
                break

        if consumed:
            del lines[:consumed]
            return True
        return False

    for _ in range(2):
        trim_blank_prefix()
        changed = pop_chapter_marker()
        trim_blank_prefix()
        changed = pop_title_block() or changed
        trim_blank_prefix()
        if not changed:
            break

    return "\n".join(lines).strip()


def repair_chapter_titles_in_pages(
    page_paths: list[Path], chapters: list[Chapter], paths: RunPaths
) -> None:
    if not chapters:
        return

    pages_by_number = {page_number_from_path(path): path for path in page_paths}
    repaired = 0
    log_event(paths, "stage_started", stage="repair_chapter_titles", chapters=len(chapters))

    for chapter in chapters:
        page_path = pages_by_number.get(chapter.page)
        if not page_path:
            continue

        original = page_path.read_text(encoding="utf-8")
        body = strip_leading_chapter_heading(original, chapter)
        heading_lines = [f"Chapter {chapter.number}"]
        if chapter.title:
            heading_lines.append(chapter.title)
        repaired_text = "\n".join(heading_lines).strip()
        if body:
            repaired_text += "\n\n" + body
        repaired_text += "\n"

        if repaired_text != original:
            page_path.write_text(repaired_text, encoding="utf-8")
            repaired += 1

    log_event(
        paths,
        "stage_completed",
        stage="repair_chapter_titles",
        chapters=len(chapters),
        repaired_pages=repaired,
    )


def filter_chapter_paths(chapter_paths: Iterable[Path], chapters: tuple[int, ...]) -> list[Path]:
    chapter_paths = sorted(chapter_paths)
    if not chapters:
        return chapter_paths

    selected = set(chapters)
    filtered = []
    for chapter_path in chapter_paths:
        match = re.search(r"chapter_(\d+)", chapter_path.stem)
        if match and int(match.group(1)) in selected:
            filtered.append(chapter_path)
    missing = selected - {
        int(match.group(1))
        for chapter_path in filtered
        if (match := re.search(r"chapter_(\d+)", chapter_path.stem))
    }
    if missing:
        missing_list = ", ".join(str(chapter) for chapter in sorted(missing))
        raise RuntimeError(f"No chapter text file found for chapter(s): {missing_list}")
    return filtered


def selected_audio_text_paths(
    paths: RunPaths, config: Config, chapters: list[Chapter]
) -> list[Path]:
    if chapters:
        chapter_paths = filter_chapter_paths(
            paths.chapter_texts.glob("chapter_*.md"), config.chapters
        )
        if config.chapters:
            return chapter_paths
        front_matter_path = paths.chapter_texts / "front_matter.md"
        if front_matter_path.exists():
            return [front_matter_path, *chapter_paths]
        return chapter_paths
    if config.chapters:
        selected = ", ".join(str(chapter) for chapter in config.chapters)
        raise RuntimeError(
            f"Cannot synthesize chapter(s) {selected}; no chapters were detected."
        )
    return [paths.audiobook_text]


def strip_leading_section_heading(text: str, title: str) -> str:
    sentinel = Chapter(number=0, page=0, title=title, source_file="", raw_heading="")
    return strip_leading_chapter_heading(text, sentinel)


def write_chapter_texts(
    page_paths: list[Path],
    chapters: list[Chapter],
    front_matter: FrontMatterSection | None,
    stop_page: int | None,
    output_path: Path,
    chapter_dir: Path,
    paths: RunPaths,
) -> Path:
    log_event(
        paths,
        "stage_started",
        stage="write_chapter_texts",
        chapters=len(chapters),
        front_matter=front_matter.title if front_matter else None,
        stop_page=stop_page,
        output=str(output_path),
    )
    pages_by_number = {page_number_from_path(path): path for path in page_paths}
    sorted_pages = [
        page for page in sorted(pages_by_number) if stop_page is None or page < stop_page
    ]

    if not chapters:
        for stale_chapter in chapter_dir.glob("chapter_*.md"):
            stale_chapter.unlink()
        (chapter_dir / "front_matter.md").unlink(missing_ok=True)
        start_page = front_matter.page if front_matter else None
        page_texts = [
            pages_by_number[page].read_text(encoding="utf-8").strip()
            for page in sorted_pages
            if start_page is None or page >= start_page
        ]
        book_text = "\n\n".join(text for text in page_texts if text).strip()
        if front_matter and book_text:
            book_text = f"{front_matter.title}\n\n" + strip_leading_section_heading(
                book_text, front_matter.title
            )
        output_path.write_text(book_text + "\n", encoding="utf-8")
        print(f"Audiobook text written to {output_path}")
        print("No chapters detected; using one book-level audio file.")
        log_event(
            paths,
            "stage_completed",
            stage="write_chapter_texts",
            chapters=0,
            front_matter=front_matter.title if front_matter else None,
            start_page=start_page,
            stop_page=stop_page,
            book_level_audio=True,
            output=str(output_path),
        )
        return output_path

    chapter_parts: list[str] = []

    front_matter_file = chapter_dir / "front_matter.md"
    if front_matter:
        first_chapter_page = min(chapter.page for chapter in chapters)
        front_pages = [
            page
            for page in sorted_pages
            if front_matter.page <= page < first_chapter_page
        ]
        body_parts = []
        for page in front_pages:
            text = pages_by_number[page].read_text(encoding="utf-8").strip()
            text = strip_leading_section_heading(text, front_matter.title)
            if text:
                body_parts.append(text)
        front_text = f"{front_matter.title}\n\n" + "\n\n".join(body_parts).strip()
        front_text = front_text.strip() + "\n"
        front_matter_file.write_text(front_text, encoding="utf-8")
        chapter_parts.append(front_text.strip())
    else:
        front_matter_file.unlink(missing_ok=True)

    for index, chapter in enumerate(chapters):
        next_page = chapters[index + 1].page if index + 1 < len(chapters) else None
        section_pages = [
            page
            for page in sorted_pages
            if page >= chapter.page and (next_page is None or page < next_page)
        ]
        body_parts = []
        for page in section_pages:
            text = pages_by_number[page].read_text(encoding="utf-8").strip()
            text = strip_leading_chapter_heading(text, chapter)
            if text:
                body_parts.append(text)

        section = f"{chapter_heading(chapter)}\n\n" + "\n\n".join(body_parts).strip()
        section = section.strip() + "\n"
        chapter_file = chapter_dir / f"chapter_{chapter.number:03d}.md"
        chapter_file.write_text(section, encoding="utf-8")
        chapter_parts.append(section.strip())

    output_path.write_text("\n\n".join(chapter_parts) + "\n", encoding="utf-8")
    print(f"Chapter-aware audiobook text written to {output_path}")
    log_event(
        paths,
        "stage_completed",
        stage="write_chapter_texts",
        chapters=len(chapters),
        front_matter=front_matter.title if front_matter else None,
        stop_page=stop_page,
        chapter_files=len(chapter_parts),
        output=str(output_path),
    )
    return output_path


def normalize_for_audio(text: str) -> str:
    text = re.sub(r"(?m)^\s*-{3,}\s*$", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_text_by_tokens(text: str, max_tokens: int, encoding) -> list[str]:
    tokens = encoding.encode(text)
    chunks = []
    for start in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[start : start + max_tokens]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text_for_tts(text: str, max_tokens: int = MAX_TTS_TOKENS) -> list[str]:
    encoding = tiktoken.encoding_for_model("gpt-4")
    text = normalize_for_audio(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = len(encoding.encode(sentence))
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
            chunks.extend(split_long_text_by_tokens(sentence, max_tokens, encoding))
            continue

        candidate = " ".join([*current, sentence]).strip()
        if len(encoding.encode(candidate)) > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def speech_bytes(response) -> bytes:
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "read"):
        return response.read()
    raise TypeError("Unsupported audio response type from OpenAI client.")


def synthesize_chunk(client: OpenAI, config: Config, text: str) -> bytes:
    response = client.audio.speech.create(
        model=config.tts_model,
        voice=config.voice,
        input=text,
        response_format="wav",
    )
    return speech_bytes(response)


def synthesize_audio_chunks(
    client: OpenAI, config: Config, paths: RunPaths, text: str
) -> list[Path]:
    chunks = chunk_text_for_tts(text)
    print(f"Synthesizing {len(chunks)} audio chunk(s)...")
    log_event(
        paths,
        "stage_started",
        stage="synthesize_audio",
        model=config.tts_model,
        voice=config.voice,
        chunks=len(chunks),
    )

    chunk_paths: list[Path] = []
    generated_count = 0
    reused_count = 0
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = paths.audio_chunks / f"chunk_{index:03d}.wav"
        chunk_paths.append(chunk_path)

        if chunk_path.exists() and not should_refresh(config, "audio"):
            reused_count += 1
            continue

        print(f"TTS chunk {index:03d}/{len(chunks):03d}...")
        audio = call_with_retries(
            f"TTS chunk {index:03d}",
            lambda chunk=chunk: synthesize_chunk(client, config, chunk),
        )
        chunk_path.write_bytes(audio)
        log_event(
            paths,
            "tts_chunk",
            chunk=index,
            file=str(chunk_path),
            model=config.tts_model,
            voice=config.voice,
            characters=len(chunk),
        )
        generated_count += 1

    log_event(
        paths,
        "stage_completed",
        stage="synthesize_audio",
        chunks=len(chunk_paths),
        generated=generated_count,
        reused=reused_count,
    )
    return chunk_paths


def kokoro_language_for(config: Config) -> str:
    return config.kokoro_language or config.voice[0].lower()


@contextlib.contextmanager
def quiet_kokoro_output():
    old_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")
            warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*")
            with contextlib.redirect_stderr(io.StringIO()):
                yield
    finally:
        if old_progress is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_progress


def kokoro_result_audio(result):
    if hasattr(result, "audio"):
        return result.audio
    if isinstance(result, tuple) and len(result) >= 3:
        return result[2]
    return None


def split_chapter_announcement(text: str) -> tuple[str, str]:
    parts = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    announcement = parts[0].strip() if parts else ""
    body = parts[1].strip() if len(parts) > 1 else ""
    if not re.match(r"^chapter\s+\d+\b", announcement, re.IGNORECASE):
        return "", text.strip()
    return announcement, body


def silence_frame_count(seconds: float, sample_rate: int = 24_000) -> int:
    return max(0, round(seconds * sample_rate))


def write_silence_frames(
    wav_file: wave.Wave_write,
    seconds: float,
    sample_rate: int = 24_000,
    channels: int = 1,
    sampwidth: int = 2,
) -> None:
    frames = silence_frame_count(seconds, sample_rate)
    if frames:
        wav_file.writeframes(b"\0" * frames * channels * sampwidth)


def write_silence_wav(path: Path, seconds: float, sample_rate: int = 24_000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        write_silence_frames(wav_file, seconds, sample_rate=sample_rate)


def write_kokoro_audio_frames(wav_file: wave.Wave_write, text: str, pipeline, config: Config) -> None:
    if not text.strip():
        return

    import numpy as np

    for result in pipeline(
        text,
        voice=config.voice,
        speed=config.kokoro_speed,
        split_pattern=r"\n+",
    ):
        audio = kokoro_result_audio(result)
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio_bytes = (np.asarray(audio) * 32767).astype(np.int16).tobytes()
        wav_file.writeframes(audio_bytes)


def write_kokoro_wav(
    output_path: Path,
    text: str,
    pipeline,
    config: Config,
    pad_chapter_announcement: bool = False,
) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)

        if pad_chapter_announcement:
            announcement, body = split_chapter_announcement(text)
            if announcement:
                write_silence_frames(
                    wav_file,
                    config.chapter_announcement_lead_silence,
                )
                write_kokoro_audio_frames(wav_file, announcement, pipeline, config)
                write_silence_frames(
                    wav_file,
                    config.chapter_announcement_trail_silence,
                )
                write_kokoro_audio_frames(wav_file, body, pipeline, config)
                return

        write_kokoro_audio_frames(wav_file, text, pipeline, config)


def audio_output_path_for_text_path(
    output_dir: Path, paths: RunPaths, text_path: Path, chapter_number: int | None
) -> Path:
    if chapter_number is not None:
        return output_dir / f"chapter_{chapter_number:03d}.wav"
    if text_path == paths.audiobook_text:
        return output_dir / "book.wav"
    return output_dir / f"{text_path.stem}.wav"


def synthesize_kokoro_chapters(
    config: Config, paths: RunPaths, chapter_paths: Iterable[Path]
) -> list[Path]:
    chapter_paths = list(chapter_paths)
    output_dir = paths.chapter_audio / "kokoro"
    output_dir.mkdir(parents=True, exist_ok=True)
    language = kokoro_language_for(config)
    print(f"Synthesizing {len(chapter_paths)} Kokoro audio WAV(s)...")
    log_event(
        paths,
        "stage_started",
        stage="synthesize_kokoro_chapters",
        voice=config.voice,
        language=language,
        speed=config.kokoro_speed,
        announcement_lead_silence=config.chapter_announcement_lead_silence,
        announcement_trail_silence=config.chapter_announcement_trail_silence,
        chapters=len(chapter_paths),
    )

    with quiet_kokoro_output():
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=language, repo_id=KOKORO_REPO_ID)
    output_paths: list[Path] = []
    generated_count = 0
    reused_count = 0

    for chapter_path in chapter_paths:
        match = re.search(r"chapter_(\d+)", chapter_path.stem)
        chapter_number = int(match.group(1)) if match else None
        output_path = audio_output_path_for_text_path(
            output_dir, paths, chapter_path, chapter_number
        )
        output_paths.append(output_path)

        if output_path.exists() and not should_refresh(config, "audio"):
            reused_count += 1
            continue

        label = (
            f"chapter {chapter_number:03d}"
            if chapter_number is not None
            else pretty_title_from_path(chapter_path)
        )
        print(f"Kokoro {label} ({len(output_paths):03d}/{len(chapter_paths):03d})...")
        text = chapter_path.read_text(encoding="utf-8").strip()
        with quiet_kokoro_output():
            write_kokoro_wav(
                output_path,
                text,
                pipeline,
                config,
                pad_chapter_announcement=chapter_number is not None,
            )
        generated_count += 1
        log_event(
            paths,
            "kokoro_chapter",
            chapter=chapter_number,
            file=str(output_path),
            voice=config.voice,
            language=language,
            speed=config.kokoro_speed,
            announcement_lead_silence=config.chapter_announcement_lead_silence,
            announcement_trail_silence=config.chapter_announcement_trail_silence,
            characters=len(text),
            bytes=output_path.stat().st_size,
        )

    log_event(
        paths,
        "stage_completed",
        stage="synthesize_kokoro_chapters",
        chapters=len(output_paths),
        generated=generated_count,
        reused=reused_count,
        output=str(paths.chapter_audio),
    )
    return output_paths


def find_generated_wav(output_dir: Path, before: set[Path]) -> Path:
    wavs = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)
    new_wavs = [path for path in wavs if path not in before]
    candidates = new_wavs or wavs
    if not candidates:
        raise RuntimeError(f"No WAV file was generated in {output_dir}")
    return candidates[-1]


def split_text_by_chars(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text.strip()] if text.strip() else []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n\n".join([*current, paragraph]).strip()
        if len(candidate) <= max_chars:
            current.append(paragraph)
            continue

        if current:
            chunks.append("\n\n".join(current).strip())
            current = []

        if len(paragraph) <= max_chars:
            current = [paragraph]
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentence_group: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_candidate = " ".join([*sentence_group, sentence]).strip()
            if len(sentence_candidate) <= max_chars:
                sentence_group.append(sentence)
            else:
                if sentence_group:
                    chunks.append(" ".join(sentence_group).strip())
                sentence_group = [sentence]
        if sentence_group:
            current = [" ".join(sentence_group).strip()]

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def patch_transformers_string_tokenizer_registration() -> None:
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    except Exception:
        return

    original_register = AutoTokenizer.register
    if getattr(original_register, "_audiobook_patched", False):
        return

    def safe_register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
        if isinstance(config_class, str):
            return None
        return original_register(
            config_class,
            slow_tokenizer_class=slow_tokenizer_class,
            fast_tokenizer_class=fast_tokenizer_class,
            exist_ok=exist_ok,
        )

    safe_register._audiobook_patched = True
    AutoTokenizer.register = safe_register


def time_stretch_wav(input_path: Path, speed: float) -> None:
    if abs(speed - 1.0) < 0.001:
        return

    from audiotsm import wsola
    from audiotsm.io.wav import WavReader, WavWriter

    temp_path = input_path.with_suffix(f".speed-{speed:g}.tmp.wav")
    try:
        with wave.open(str(input_path), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise RuntimeError(
                    f"Cannot time-stretch {input_path}; expected 16-bit WAV."
                )
        with WavReader(str(input_path)) as reader:
            with WavWriter(str(temp_path), reader.channels, reader.samplerate) as writer:
                wsola(reader.channels, speed=speed).run(reader, writer)
        temp_path.replace(input_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def synthesize_mlx_chatterbox_chapters(
    config: Config, paths: RunPaths, chapter_paths: Iterable[Path]
) -> list[Path]:
    patch_transformers_string_tokenizer_registration()
    from mlx_audio.tts.generate import generate_audio

    chapter_paths = list(chapter_paths)
    output_dir = paths.chapter_audio / "mlx-chatterbox"
    scratch_dir = output_dir / "_scratch"
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing {len(chapter_paths)} MLX Chatterbox audio WAV(s)...")
    log_event(
        paths,
        "stage_started",
        stage="synthesize_mlx_chatterbox_chapters",
        model=config.mlx_model,
        chapters=len(chapter_paths),
        ref_audio=str(config.mlx_ref_audio) if config.mlx_ref_audio else None,
        max_tokens=config.mlx_max_tokens,
        speed=config.mlx_speed,
        announcement_lead_silence=config.chapter_announcement_lead_silence,
        announcement_trail_silence=config.chapter_announcement_trail_silence,
    )

    output_paths: list[Path] = []
    generated_count = 0
    reused_count = 0

    for chapter_path in chapter_paths:
        match = re.search(r"chapter_(\d+)", chapter_path.stem)
        chapter_number = int(match.group(1)) if match else None
        output_path = audio_output_path_for_text_path(
            output_dir, paths, chapter_path, chapter_number
        )
        output_paths.append(output_path)
        track_label = (
            f"chapter {chapter_number:03d}"
            if chapter_number is not None
            else pretty_title_from_path(chapter_path).lower()
        )
        track_prefix = (
            f"chapter_{chapter_number:03d}"
            if chapter_number is not None
            else re.sub(r"[^a-z0-9]+", "_", chapter_path.stem.lower()).strip("_")
            or "section"
        )

        if output_path.exists() and not should_refresh(config, "audio"):
            reused_count += 1
            continue

        print(f"MLX Chatterbox {track_label} ({len(output_paths):03d}/{len(chapter_paths):03d})...")
        text = chapter_path.read_text(encoding="utf-8").strip()
        announcement, body = split_chapter_announcement(text) if chapter_number is not None else ("", text)
        body_text = body if announcement else text
        segments = split_text_by_chars(body_text, config.mlx_chunk_chars)
        segment_wavs: list[Path] = []

        if announcement:
            if config.chapter_announcement_lead_silence:
                lead_silence = scratch_dir / f"{track_prefix}_lead_silence.wav"
                write_silence_wav(lead_silence, config.chapter_announcement_lead_silence)
                segment_wavs.append(lead_silence)

            print(f"MLX Chatterbox {track_label} announcement...")
            before = set(scratch_dir.glob("*.wav"))
            kwargs = {
                "text": announcement,
                "model": config.mlx_model,
                "file_prefix": f"{track_prefix}_announcement",
                "output_path": str(scratch_dir),
                "max_tokens": config.mlx_max_tokens,
                "join_audio": True,
            }
            if config.mlx_ref_audio:
                kwargs["ref_audio"] = str(config.mlx_ref_audio)
            if config.mlx_ref_text:
                kwargs["ref_text"] = config.mlx_ref_text
            generate_audio(**kwargs)
            segment_wavs.append(find_generated_wav(scratch_dir, before))

            if config.chapter_announcement_trail_silence:
                trail_silence = scratch_dir / f"{track_prefix}_trail_silence.wav"
                write_silence_wav(trail_silence, config.chapter_announcement_trail_silence)
                segment_wavs.append(trail_silence)

        for segment_index, segment in enumerate(segments, start=1):
            print(
                f"MLX Chatterbox {track_label} "
                f"segment {segment_index:02d}/{len(segments):02d}..."
            )
            before = set(scratch_dir.glob("*.wav"))
            kwargs = {
                "text": segment,
                "model": config.mlx_model,
                "file_prefix": f"{track_prefix}_part_{segment_index:03d}",
                "output_path": str(scratch_dir),
                "max_tokens": config.mlx_max_tokens,
                "join_audio": True,
            }
            if config.mlx_ref_audio:
                kwargs["ref_audio"] = str(config.mlx_ref_audio)
            if config.mlx_ref_text:
                kwargs["ref_text"] = config.mlx_ref_text

            generate_audio(**kwargs)
            segment_wavs.append(find_generated_wav(scratch_dir, before))

        concatenate_wavs(segment_wavs, output_path)
        if abs(config.mlx_speed - 1.0) >= 0.001:
            print(f"Adjusting MLX Chatterbox chapter speed to {config.mlx_speed:g}x...")
            time_stretch_wav(output_path, config.mlx_speed)
        generated_count += 1
        log_event(
            paths,
            "mlx_chatterbox_chapter",
            chapter=chapter_number,
            file=str(output_path),
            model=config.mlx_model,
            ref_audio=str(config.mlx_ref_audio) if config.mlx_ref_audio else None,
            max_tokens=config.mlx_max_tokens,
            chunk_chars=config.mlx_chunk_chars,
            speed=config.mlx_speed,
            announcement_lead_silence=config.chapter_announcement_lead_silence,
            announcement_trail_silence=config.chapter_announcement_trail_silence,
            segments=len(segments),
            characters=len(text),
            bytes=output_path.stat().st_size,
        )

    log_event(
        paths,
        "stage_completed",
        stage="synthesize_mlx_chatterbox_chapters",
        chapters=len(output_paths),
        generated=generated_count,
        reused=reused_count,
        output=str(output_dir),
    )
    return output_paths


def concatenate_wavs(chunk_paths: Iterable[Path], output_path: Path, paths: RunPaths | None = None) -> None:
    chunk_paths = list(chunk_paths)
    if not chunk_paths:
        raise RuntimeError("No audio chunks were produced.")

    if paths:
        log_event(
            paths,
            "stage_started",
            stage="concatenate_audio",
            chunks=len(chunk_paths),
            output=str(output_path),
        )

    with wave.open(str(output_path), "wb") as output:
        expected_params = None
        for chunk_path in chunk_paths:
            with wave.open(str(chunk_path), "rb") as chunk:
                params = chunk.getparams()
                if expected_params is None:
                    expected_params = params
                    output.setparams(params)
                elif params[:3] != expected_params[:3]:
                    raise RuntimeError(
                        f"Cannot concatenate {chunk_path}; WAV parameters differ."
                    )
                output.writeframes(chunk.readframes(chunk.getnframes()))

    print(f"Audiobook written to {output_path}")
    if paths:
        log_event(
            paths,
            "stage_completed",
            stage="concatenate_audio",
            chunks=len(chunk_paths),
            output=str(output_path),
            bytes=output_path.stat().st_size,
        )


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wav_file:
        return round(wav_file.getnframes() * 1000 / wav_file.getframerate())


def pretty_title_from_path(path: Path) -> str:
    title = re.sub(r"[_-]+", " ", path.stem).strip()
    return title.title() if title else path.stem


def chapter_number_from_audio_path(path: Path) -> int | None:
    match = re.search(r"chapter_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def text_path_for_audio_path(paths: RunPaths, audio_path: Path) -> Path | None:
    chapter_number = chapter_number_from_audio_path(audio_path)
    if chapter_number is not None:
        return paths.chapter_texts / f"chapter_{chapter_number:03d}.md"
    if audio_path.stem == "front_matter":
        return paths.chapter_texts / "front_matter.md"
    if audio_path.stem == "book":
        return paths.audiobook_text
    return None


def m4b_chapter_title(chapter_number: int, chapters: list[Chapter], config: Config) -> str:
    for chapter in chapters:
        if chapter.number == chapter_number:
            title = chapter.title.strip()
            return f"Chapter {chapter.number}: {title}" if title else f"Chapter {chapter.number}"
    if chapter_number == 1 and not chapters and config.pdf:
        return config.pdf.stem.replace("_", " ")
    return f"Chapter {chapter_number}"


def markdown_heading_title(line: str) -> str | None:
    match = re.match(r"^\s*#{1,6}\s+(?P<title>.+?)\s*#*\s*$", line)
    if not match:
        return None
    title = clean_title(plain_markdown_line(match.group("title")))
    return title or None


def include_m4b_section_heading(title: str, track_title: str, config: Config) -> bool:
    normalized = normalize_match_text(title)
    if not normalized:
        return False
    if re.match(r"^\d{1,4}[.)]?\s+\S+", title.strip()):
        return False
    if normalized == normalize_match_text(track_title):
        return False
    if config.pdf:
        pdf_title = normalize_match_text(config.pdf.stem.replace("_", " "))
        if normalized == pdf_title or normalized in pdf_title:
            return False
    if normalized in {"behavior", "approach", "behavior approach", "index"}:
        return False
    if "|" in title or "/" in title:
        return False
    if title.endswith(".") or len(title) > 90:
        return False
    return len(normalized) >= 4


def m4b_heading_key(title: str) -> str:
    normalized = normalize_match_text(title)
    return re.sub(r"^\d+\s+", "", normalized).strip()


def section_headings_for_text(
    text_path: Path, track_title: str, config: Config
) -> list[dict[str, object]]:
    if not text_path.exists():
        return []

    text = text_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    total_chars = max(len(normalize_for_audio(text)), 1)
    seen_titles = {normalize_match_text(track_title)}
    heading_counts: dict[str, int] = {}
    for line in lines:
        heading = markdown_heading_title(line)
        if heading:
            key = m4b_heading_key(heading)
            heading_counts[key] = heading_counts.get(key, 0) + 1

    sections: list[dict[str, object]] = []
    chars_before = 0

    for line_number, line in enumerate(lines, start=1):
        heading = markdown_heading_title(line)
        if heading and include_m4b_section_heading(heading, track_title, config):
            normalized_heading = normalize_match_text(heading)
            repeated_page_heading = heading_counts.get(m4b_heading_key(heading), 0) > 1
            if normalized_heading not in seen_titles and not repeated_page_heading:
                relative_position = chars_before / total_chars
                sections.append(
                    {
                        "title": heading,
                        "line": line_number,
                        "char_offset": chars_before,
                        "relative_position": round(relative_position, 6),
                    }
                )
                seen_titles.add(normalized_heading)

        line_text = plain_markdown_line(line)
        chars_before += len(line_text) + 1

    return sections


def m4b_track_markers(track: AudioTrack, config: Config) -> list[tuple[int, str]]:
    markers: list[tuple[int, str]] = [(0, track.title)]
    if not track.text_path or not track.text_path.exists() or track.duration_ms <= 0:
        return markers

    for section in section_headings_for_text(track.text_path, track.title, config):
        relative_position = float(section["relative_position"])
        offset = round(relative_position * track.duration_ms)
        if 1_000 <= offset <= track.duration_ms - 1_000:
            markers.append((offset, str(section["title"])))

    markers.sort(key=lambda marker: marker[0])
    deduped: list[tuple[int, str]] = []
    for offset, title in markers:
        if deduped and offset - deduped[-1][0] < 1_000:
            continue
        deduped.append((offset, title))
    return deduped


def detect_section_headings(
    paths: RunPaths,
    config: Config,
    chapters: list[Chapter],
    front_matter: FrontMatterSection | None,
) -> list[dict[str, object]]:
    log_event(paths, "stage_started", stage="detect_section_headings")
    detected: list[dict[str, object]] = []

    front_matter_path = paths.chapter_texts / "front_matter.md"
    if front_matter and front_matter_path.exists():
        sections = section_headings_for_text(front_matter_path, front_matter.title, config)
        detected.append(
            {
                "type": "front_matter",
                "title": front_matter.title,
                "file": str(front_matter_path),
                "sections": sections,
            }
        )

    for chapter in chapters:
        chapter_file = paths.chapter_texts / f"chapter_{chapter.number:03d}.md"
        if not chapter_file.exists():
            continue
        sections = section_headings_for_text(
            chapter_file, chapter_heading(chapter), config
        )
        detected.append(
            {
                "type": "chapter",
                "number": chapter.number,
                "title": chapter.title,
                "file": str(chapter_file),
                "sections": sections,
            }
        )

    if not chapters and paths.audiobook_text.exists():
        title = config.pdf.stem.replace("_", " ") if config.pdf else "Audiobook"
        sections = section_headings_for_text(paths.audiobook_text, title, config)
        detected.append(
            {
                "type": "book",
                "title": title,
                "file": str(paths.audiobook_text),
                "sections": sections,
            }
        )

    total_sections = sum(len(group["sections"]) for group in detected)
    update_chapter_metadata(paths, section_headings=detected)
    print(f"Detected {total_sections} section heading(s).")
    log_event(
        paths,
        "stage_completed",
        stage="detect_section_headings",
        groups=len(detected),
        sections=total_sections,
    )
    return detected


def m4b_escape(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    escaped = []
    for char in value:
        if char in "\\=;#":
            escaped.append("\\" + char)
        else:
            escaped.append(char)
    return "".join(escaped)


def write_m4b_metadata(
    paths: RunPaths,
    config: Config,
    tracks: list[AudioTrack],
    include_chapters: bool,
) -> None:
    lines = [
        ";FFMETADATA1",
        f"title={m4b_escape(config.pdf.stem.replace('_', ' ') if config.pdf else 'Audiobook')}",
        "genre=Audiobook",
    ]
    if include_chapters:
        markers: list[tuple[int, str]] = []
        track_start_ms = 0
        for track in tracks:
            for offset_ms, title in m4b_track_markers(track, config):
                markers.append((track_start_ms + offset_ms, title))
            track_start_ms += track.duration_ms

        total_duration_ms = max(track_start_ms, 1)
        for index, (start_ms, title) in enumerate(markers):
            next_start = (
                markers[index + 1][0] if index + 1 < len(markers) else total_duration_ms
            )
            end_ms = max(next_start, start_ms + 1)
            lines.extend(
                [
                    "",
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start_ms}",
                    f"END={end_ms}",
                    f"title={m4b_escape(title)}",
                ]
            )
    paths.m4b_metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_cover_for_m4b(
    config: Config, paths: RunPaths, default_cover: Path | None
) -> Path | None:
    if config.no_cover:
        return None
    cover_source = config.cover or default_cover
    if not cover_source:
        return None
    if not cover_source.exists():
        raise FileNotFoundError(f"Cover image does not exist: {cover_source}")

    with Image.open(cover_source) as image:
        image = image.convert("RGB")
        image.thumbnail((1600, 1600))
        image.save(paths.m4b_cover, "JPEG", quality=90, optimize=True)
    return paths.m4b_cover


def chapter_audio_dir_for_engine(paths: RunPaths, config: Config) -> Path:
    if config.audio_engine not in {"kokoro", "mlx-chatterbox"}:
        raise RuntimeError("--m4b requires --audio-engine kokoro or mlx-chatterbox.")
    return paths.chapter_audio / config.audio_engine


def discover_chapter_audio_paths(paths: RunPaths, config: Config) -> list[Path]:
    audio_dir = chapter_audio_dir_for_engine(paths, config)
    chapter_paths = filter_chapter_paths(audio_dir.glob("chapter_*.wav"), config.chapters)
    if chapter_paths and not config.chapters:
        front_matter_path = audio_dir / "front_matter.wav"
        if front_matter_path.exists():
            return [front_matter_path, *chapter_paths]
        return chapter_paths
    if not chapter_paths and not config.chapters:
        book_path = audio_dir / "book.wav"
        if book_path.exists():
            return [book_path]
    if not chapter_paths:
        raise RuntimeError(
            f"No chapter or book WAV files found in {audio_dir}. Generate audio first."
        )
    return chapter_paths


def build_m4b(
    config: Config,
    paths: RunPaths,
    chapters: list[Chapter],
    chapter_audio_paths: Iterable[Path],
    default_cover: Path | None,
) -> Path:
    chapter_audio_paths = list(chapter_audio_paths)
    if not chapter_audio_paths:
        chapter_audio_paths = discover_chapter_audio_paths(paths, config)

    for intro_wav in config.intro_wavs:
        if not intro_wav.exists():
            raise FileNotFoundError(f"Intro WAV does not exist: {intro_wav}")

    tracks: list[AudioTrack] = []
    for index, intro_wav in enumerate(config.intro_wavs, start=1):
        title = pretty_title_from_path(intro_wav)
        if len(config.intro_wavs) > 1:
            title = f"Intro {index}: {title}"
        tracks.append(
            AudioTrack(title=title, path=intro_wav, duration_ms=wav_duration_ms(intro_wav))
        )

    for chapter_wav in chapter_audio_paths:
        chapter_number = chapter_number_from_audio_path(chapter_wav)
        title = (
            m4b_chapter_title(chapter_number, chapters, config)
            if chapter_number is not None
            else pretty_title_from_path(chapter_wav)
        )
        tracks.append(
            AudioTrack(
                title=title,
                path=chapter_wav,
                duration_ms=wav_duration_ms(chapter_wav),
                text_path=text_path_for_audio_path(paths, chapter_wav),
            )
        )

    if not tracks:
        raise RuntimeError("No audio tracks were provided for M4B packaging.")

    write_m4b_metadata(paths, config, tracks, include_chapters=bool(chapters))
    cover_path = normalize_cover_for_m4b(config, paths, default_cover)

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    temp_output = paths.m4b.with_suffix(".tmp.m4b")
    temp_output.unlink(missing_ok=True)

    cmd = [ffmpeg, "-hide_banner", "-y"]
    for track in tracks:
        cmd.extend(["-i", str(track.path)])
    metadata_input = len(tracks)
    cmd.extend(["-f", "ffmetadata", "-i", str(paths.m4b_metadata)])

    cover_input = None
    if cover_path:
        cover_input = metadata_input + 1
        cmd.extend(["-i", str(cover_path)])

    filter_parts = []
    for index in range(len(tracks)):
        filter_parts.append(
            f"[{index}:a]aresample=24000,"
            f"aformat=sample_fmts=fltp:channel_layouts=mono[a{index}]"
        )
    concat_inputs = "".join(f"[a{index}]" for index in range(len(tracks)))
    filter_parts.append(f"{concat_inputs}concat=n={len(tracks)}:v=0:a=1[a]")
    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[a]",
        ]
    )
    if cover_input is not None:
        cmd.extend(
            [
                "-map",
                f"{cover_input}:v:0",
                "-c:v",
                "copy",
                "-disposition:v:0",
                "attached_pic",
                "-metadata:s:v",
                "title=Cover",
                "-metadata:s:v",
                "comment=Cover (front)",
            ]
        )
    cmd.extend(
        [
            "-map_metadata",
            str(metadata_input),
            "-map_chapters",
            str(metadata_input),
            "-c:a",
            "aac",
            "-b:a",
            config.m4b_bitrate,
            "-movflags",
            "+faststart",
            str(temp_output),
        ]
    )

    print(f"Writing M4B to {paths.m4b}...")
    log_event(
        paths,
        "stage_started",
        stage="build_m4b",
        tracks=len(tracks),
        chapters=len(chapter_audio_paths),
        intros=len(config.intro_wavs),
        cover=str(cover_path) if cover_path else None,
        bitrate=config.m4b_bitrate,
        output=str(paths.m4b),
    )
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        temp_output.unlink(missing_ok=True)
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffmpeg failed while writing M4B:\n{stderr[-4000:]}")

    temp_output.replace(paths.m4b)
    log_event(
        paths,
        "stage_completed",
        stage="build_m4b",
        tracks=len(tracks),
        chapters=len(chapter_audio_paths),
        intros=len(config.intro_wavs),
        output=str(paths.m4b),
        bytes=paths.m4b.stat().st_size,
    )
    print(f"M4B written to {paths.m4b}")
    return paths.m4b


def token_usage_cost(model: str, usage: dict) -> float | None:
    prices = MODEL_PRICES_USD_PER_1M.get(model)
    if not prices:
        return None

    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_tokens = details.get("cached_tokens") or usage.get("cached_tokens") or 0
    fresh_input_tokens = max(input_tokens - cached_tokens, 0)

    return (
        fresh_input_tokens * prices["input_tokens"]
        + cached_tokens * prices.get("cached_input_tokens", prices["input_tokens"])
        + output_tokens * prices["output_tokens"]
    ) / 1_000_000


def iter_manifest_events(paths: RunPaths) -> Iterable[dict]:
    if not paths.manifest.exists():
        return
    for line in paths.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def cost_report_scope(events: list[dict]) -> dict:
    totals_by_stage: dict[str, dict[str, float]] = {}
    totals_by_model: dict[str, dict[str, float]] = {}
    unknown_models: set[str] = set()
    tts_characters = 0
    kokoro_characters = 0
    mlx_chatterbox_characters = 0

    for event in events:
        event_name = event.get("event")
        if event_name in {"ocr_page", "clean_page"}:
            stage = "ocr" if event_name == "ocr_page" else "clean"
            model = event.get("model")
            usage = event.get("usage") or {}
            if not isinstance(model, str) or not isinstance(usage, dict):
                continue
            cost = token_usage_cost(model, usage)
            if cost is None:
                unknown_models.add(model)
                continue

            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            cached_tokens = details.get("cached_tokens") or usage.get("cached_tokens") or 0

            for bucket, key in ((totals_by_stage, stage), (totals_by_model, model)):
                totals = bucket.setdefault(
                    key,
                    {
                        "estimated_usd": 0.0,
                        "input_tokens": 0.0,
                        "cached_input_tokens": 0.0,
                        "output_tokens": 0.0,
                        "requests": 0.0,
                    },
                )
                totals["estimated_usd"] += cost
                totals["input_tokens"] += input_tokens
                totals["cached_input_tokens"] += cached_tokens
                totals["output_tokens"] += output_tokens
                totals["requests"] += 1

        elif event_name == "tts_chunk":
            tts_characters += int(event.get("characters") or 0)
        elif event_name == "kokoro_chapter":
            kokoro_characters += int(event.get("characters") or 0)
        elif event_name == "mlx_chatterbox_chapter":
            mlx_chatterbox_characters += int(event.get("characters") or 0)

    def clean_totals(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                field: round(value, 6) if field == "estimated_usd" else int(value)
                for field, value in totals.items()
            }
            for key, totals in sorted(values.items())
        }

    total_known_cost = sum(stage["estimated_usd"] for stage in totals_by_stage.values())
    return {
        "estimated_text_and_vision_usd": round(total_known_cost, 6),
        "by_stage": clean_totals(totals_by_stage),
        "by_model": clean_totals(totals_by_model),
        "tts_characters": tts_characters,
        "kokoro_characters": kokoro_characters,
        "mlx_chatterbox_characters": mlx_chatterbox_characters,
        "unknown_models": sorted(unknown_models),
    }


def events_for_current_run(events: list[dict]) -> list[dict]:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("event") == "run_started":
            return events[index:]
    return events


def write_cost_report(paths: RunPaths) -> None:
    log_event(paths, "stage_started", stage="write_cost_report", output=str(paths.cost_report))
    all_events = list(iter_manifest_events(paths))
    current_scope = cost_report_scope(events_for_current_run(all_events))
    all_logged_scope = cost_report_scope(all_events)

    report = {
        "pricing_source": PRICING_SOURCE,
        "note": (
            "This is a local estimate from API usage returned by text/vision calls. "
            "Exact posted spend is available from OpenAI's organization Costs endpoint "
            "or dashboard, and may include audio output tokens, regional uplift, service "
            "tier, retries, credits, taxes, or account-specific adjustments."
        ),
        "current_run": current_scope,
        "all_logged": all_logged_scope,
        "tts_note": (
            "Speech API responses do not currently give this script per-call audio "
            "output token usage, so TTS spend is best reconciled through the OpenAI "
            "organization Costs endpoint for exact billing."
        ),
    }
    paths.cost_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Cost report written to {paths.cost_report}")
    log_event(
        paths,
        "stage_completed",
        stage="write_cost_report",
        output=str(paths.cost_report),
        estimated_text_and_vision_usd=current_scope["estimated_text_and_vision_usd"],
        tts_characters=current_scope["tts_characters"],
        kokoro_characters=current_scope["kokoro_characters"],
        mlx_chatterbox_characters=current_scope["mlx_chatterbox_characters"],
        unknown_models=current_scope["unknown_models"],
    )


def write_config(paths: RunPaths, config: Config) -> None:
    def json_value(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [json_value(item) for item in value]
        if isinstance(value, list):
            return [json_value(item) for item in value]
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        return value

    serializable = {
        key: json_value(value)
        for key, value in asdict(config).items()
    }
    (paths.root / "config.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_kokoro_voices() -> None:
    print("Known Kokoro voices:")
    for group, voices in KOKORO_VOICES.items():
        print(f"\n{group}")
        for voice in voices:
            print(f"  {voice}")
    print("\nKokoro also supports comma-separated voice blends, such as af_bella,af_nicole.")


def sample_output_path(config: Config) -> Path:
    voice_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.voice).strip("_") or "voice"
    default_name = f"kokoro_{voice_slug}.wav"
    if config.sample_output:
        if config.sample_output.suffix.lower() == ".wav":
            return config.sample_output
        return config.sample_output / default_name
    return Path("voice_samples") / default_name


def generate_kokoro_sample(config: Config) -> Path:
    if config.sample_text is None:
        raise ValueError("--sample-text is required.")

    output_path = sample_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    language = kokoro_language_for(config)
    with quiet_kokoro_output():
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=language, repo_id=KOKORO_REPO_ID)
        write_kokoro_wav(output_path, config.sample_text.strip(), pipeline, config)
    print(f"Kokoro sample written to {output_path}")
    return output_path


def run(config: Config) -> RunPaths:
    if config.pdf is None:
        raise ValueError("A PDF path is required. Use --list-voices without a PDF to list Kokoro voices.")
    if not config.pdf.exists():
        raise FileNotFoundError(f"PDF does not exist: {config.pdf}")
    if config.image_size < 256:
        raise ValueError("--image-size must be at least 256.")
    if config.m4b and config.audio_engine not in {"kokoro", "mlx-chatterbox"}:
        raise ValueError("--m4b requires --audio-engine kokoro or mlx-chatterbox.")

    paths = make_paths(config)
    ensure_dirs(paths)
    write_config(paths, config)
    log_event(
        paths,
        "run_started",
        pdf=str(config.pdf),
        output=str(paths.root),
        audio_engine=config.audio_engine,
        ocr_model=config.ocr_model,
        clean_model=config.clean_model,
        tts_model=config.tts_model,
        voice=config.voice,
        chapters=list(config.chapters),
        mlx_model=config.mlx_model,
        mlx_ref_audio=str(config.mlx_ref_audio) if config.mlx_ref_audio else None,
        mlx_max_tokens=config.mlx_max_tokens,
        mlx_speed=config.mlx_speed,
        kokoro_language=config.kokoro_language,
        kokoro_speed=config.kokoro_speed,
        chapter_announcement_lead_silence=config.chapter_announcement_lead_silence,
        chapter_announcement_trail_silence=config.chapter_announcement_trail_silence,
        m4b=config.m4b,
        intro_wavs=[str(path) for path in config.intro_wavs],
        cover=str(config.cover) if config.cover else None,
        no_cover=config.no_cover,
        m4b_bitrate=config.m4b_bitrate,
        text_only=config.text_only,
        raw_ocr=config.raw_ocr,
        overwrite=config.overwrite,
        refresh=list(config.refresh),
    )

    image_paths = render_pdf_pages(config, paths)
    markdown_paths = convert_images_to_markdown(None, config, paths, image_paths)
    merge_pages(markdown_paths, paths.raw_book, paths)
    chapters = detect_chapters(markdown_paths, paths)
    front_matter = detect_front_matter_section(markdown_paths, chapters, paths)
    stop_page = detect_narration_stop_page(markdown_paths, chapters, paths)

    final_page_paths = clean_markdown_pages(None, config, paths, markdown_paths)
    if not config.raw_ocr:
        repair_chapter_titles_in_pages(final_page_paths, chapters, paths)
    final_book = paths.raw_book if config.raw_ocr else paths.cleaned_book
    merge_pages(final_page_paths, final_book, paths)
    audiobook_text = write_chapter_texts(
        final_page_paths,
        chapters,
        front_matter,
        stop_page,
        paths.audiobook_text,
        paths.chapter_texts,
        paths,
    )
    detect_section_headings(paths, config, chapters, front_matter)

    chapter_audio_paths: list[Path] = []
    if not config.text_only:
        if config.audio_engine == "kokoro":
            audio_text_paths = selected_audio_text_paths(paths, config, chapters)
            chapter_audio_paths = synthesize_kokoro_chapters(config, paths, audio_text_paths)
        elif config.audio_engine == "mlx-chatterbox":
            audio_text_paths = selected_audio_text_paths(paths, config, chapters)
            chapter_audio_paths = synthesize_mlx_chatterbox_chapters(config, paths, audio_text_paths)
        else:
            client = make_client()
            final_text = audiobook_text.read_text(encoding="utf-8")
            chunk_paths = synthesize_audio_chunks(client, config, paths, final_text)
            concatenate_wavs(chunk_paths, paths.audiobook, paths)
    else:
        log_event(paths, "stage_skipped", stage="synthesize_audio", reason="text_only")
        log_event(paths, "stage_skipped", stage="concatenate_audio", reason="text_only")

    if config.m4b:
        if not chapter_audio_paths:
            chapter_audio_paths = discover_chapter_audio_paths(paths, config)
        default_cover = image_paths[0] if image_paths else None
        build_m4b(config, paths, chapters, chapter_audio_paths, default_cover)
    else:
        log_event(paths, "stage_skipped", stage="build_m4b", reason="m4b_disabled")

    write_cost_report(paths)
    log_event(paths, "run_completed", output=str(paths.root))
    return paths


def main(argv: list[str] | None = None) -> int:
    config = parse_args(sys.argv[1:] if argv is None else argv)
    if config.list_voices:
        print_kokoro_voices()
        return 0
    if config.sample_text is not None:
        try:
            generate_kokoro_sample(config)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as exc:  # noqa: BLE001 - CLI should show concise failures.
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        paths = run(config)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should show concise failures.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nDone.")
    print(f"Output: {paths.root}")
    print(f"Raw Markdown: {paths.raw_book}")
    if not config.raw_ocr:
        print(f"Cleaned Markdown: {paths.cleaned_book}")
    print(f"Audiobook text: {paths.audiobook_text}")
    print(f"Chapters: {paths.chapters}")
    print(f"Cost report: {paths.cost_report}")
    if not config.text_only and config.audio_engine in {"kokoro", "mlx-chatterbox"}:
        print(f"Chapter audio: {paths.chapter_audio / config.audio_engine}")
    elif not config.text_only:
        print(f"Audiobook: {paths.audiobook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
