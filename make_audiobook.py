#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.0.0",
#   "pymupdf>=1.24.0",
#   "pillow>=10.0.0",
#   "tiktoken>=0.7.0",
#   "kokoro>=0.9.2",
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
import io
import json
import os
import re
import sys
import time
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
    mlx_lang_code: str
    mlx_max_tokens: int
    mlx_chunk_chars: int
    kokoro_language: str | None
    kokoro_speed: float
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
        "--mlx-lang-code",
        default="en",
        help="Language code for Chatterbox generation.",
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
    voice = args.voice or (DEFAULT_KOKORO_VOICE if args.audio_engine == "kokoro" else DEFAULT_VOICE)
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
        mlx_lang_code=args.mlx_lang_code,
        mlx_max_tokens=args.mlx_max_tokens,
        mlx_chunk_chars=args.mlx_chunk_chars,
        kokoro_language=args.kokoro_language,
        kokoro_speed=args.kokoro_speed,
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
    candidates = [
        candidate
        for markdown_path in markdown_paths
        if (candidate := find_chapter_candidate(markdown_path)) is not None
    ]
    chapters = validate_chapter_candidates(candidates)
    payload = {
        "chapters": [asdict(chapter) for chapter in chapters],
        "rejected_candidates": [
            asdict(candidate) for candidate in candidates if candidate not in chapters
        ],
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
        rejected=len(candidates) - len(chapters),
        output=str(paths.chapters),
    )
    return chapters


def chapter_heading(chapter: Chapter) -> str:
    if chapter.title:
        return f"Chapter {chapter.number}: {chapter.title}"
    return f"Chapter {chapter.number}"


def strip_leading_chapter_heading(text: str, chapter: Chapter) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"^#*\s*chapter\s+\d+\b", lines[0].strip(), re.IGNORECASE):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if chapter.title and lines:
        possible_title = clean_title(lines[0])
        if possible_title.lower() == chapter.title.lower():
            lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
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


def write_chapter_texts(
    page_paths: list[Path],
    chapters: list[Chapter],
    output_path: Path,
    chapter_dir: Path,
    paths: RunPaths,
) -> Path:
    log_event(
        paths,
        "stage_started",
        stage="write_chapter_texts",
        chapters=len(chapters),
        output=str(output_path),
    )
    if not chapters:
        merge_pages(page_paths, output_path, paths)
        log_event(
            paths,
            "stage_completed",
            stage="write_chapter_texts",
            chapters=0,
            output=str(output_path),
        )
        return output_path

    pages_by_number = {page_number_from_path(path): path for path in page_paths}
    sorted_pages = sorted(pages_by_number)
    chapter_parts: list[str] = []

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
            if page == chapter.page:
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


def kokoro_result_audio(result):
    if hasattr(result, "audio"):
        return result.audio
    if isinstance(result, tuple) and len(result) >= 3:
        return result[2]
    return None


def write_kokoro_wav(output_path: Path, text: str, pipeline, config: Config) -> None:
    import wave

    import numpy as np

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

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


def synthesize_kokoro_chapters(
    config: Config, paths: RunPaths, chapter_paths: Iterable[Path]
) -> list[Path]:
    from kokoro import KPipeline

    chapter_paths = list(chapter_paths)
    output_dir = paths.chapter_audio / "kokoro"
    output_dir.mkdir(parents=True, exist_ok=True)
    language = kokoro_language_for(config)
    print(f"Synthesizing {len(chapter_paths)} Kokoro chapter WAV(s)...")
    log_event(
        paths,
        "stage_started",
        stage="synthesize_kokoro_chapters",
        voice=config.voice,
        language=language,
        speed=config.kokoro_speed,
        chapters=len(chapter_paths),
    )

    pipeline = KPipeline(lang_code=language)
    output_paths: list[Path] = []
    generated_count = 0
    reused_count = 0

    for chapter_path in chapter_paths:
        match = re.search(r"chapter_(\d+)", chapter_path.stem)
        chapter_number = int(match.group(1)) if match else len(output_paths) + 1
        output_path = output_dir / f"chapter_{chapter_number:03d}.wav"
        output_paths.append(output_path)

        if output_path.exists() and not should_refresh(config, "audio"):
            reused_count += 1
            continue

        print(f"Kokoro chapter {chapter_number:03d}/{len(chapter_paths):03d}...")
        text = chapter_path.read_text(encoding="utf-8").strip()
        write_kokoro_wav(output_path, text, pipeline, config)
        generated_count += 1
        log_event(
            paths,
            "kokoro_chapter",
            chapter=chapter_number,
            file=str(output_path),
            voice=config.voice,
            language=language,
            speed=config.kokoro_speed,
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

    print(f"Synthesizing {len(chapter_paths)} MLX Chatterbox chapter WAV(s)...")
    log_event(
        paths,
        "stage_started",
        stage="synthesize_mlx_chatterbox_chapters",
        model=config.mlx_model,
        chapters=len(chapter_paths),
        ref_audio=str(config.mlx_ref_audio) if config.mlx_ref_audio else None,
        lang_code=config.mlx_lang_code,
        max_tokens=config.mlx_max_tokens,
        voice=config.voice,
    )

    output_paths: list[Path] = []
    generated_count = 0
    reused_count = 0

    for chapter_path in chapter_paths:
        match = re.search(r"chapter_(\d+)", chapter_path.stem)
        chapter_number = int(match.group(1)) if match else len(output_paths) + 1
        output_path = output_dir / f"chapter_{chapter_number:03d}.wav"
        output_paths.append(output_path)

        if output_path.exists() and not should_refresh(config, "audio"):
            reused_count += 1
            continue

        print(f"MLX Chatterbox chapter {chapter_number:03d}/{len(chapter_paths):03d}...")
        text = chapter_path.read_text(encoding="utf-8").strip()
        segments = split_text_by_chars(text, config.mlx_chunk_chars)
        segment_wavs: list[Path] = []
        for segment_index, segment in enumerate(segments, start=1):
            print(
                f"MLX Chatterbox chapter {chapter_number:03d} "
                f"segment {segment_index:02d}/{len(segments):02d}..."
            )
            before = set(scratch_dir.glob("*.wav"))
            kwargs = {
                "text": segment,
                "model": config.mlx_model,
                "file_prefix": f"chapter_{chapter_number:03d}_part_{segment_index:03d}",
                "output_path": str(scratch_dir),
                "lang_code": config.mlx_lang_code,
                "voice": config.voice,
                "speed": config.kokoro_speed,
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
        generated_count += 1
        log_event(
            paths,
            "mlx_chatterbox_chapter",
            chapter=chapter_number,
            file=str(output_path),
            model=config.mlx_model,
            voice=config.voice,
            ref_audio=str(config.mlx_ref_audio) if config.mlx_ref_audio else None,
            lang_code=config.mlx_lang_code,
            max_tokens=config.mlx_max_tokens,
            chunk_chars=config.mlx_chunk_chars,
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
    serializable = {
        key: str(value) if isinstance(value, Path) else value
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


def run(config: Config) -> RunPaths:
    if config.pdf is None:
        raise ValueError("A PDF path is required. Use --list-voices without a PDF to list Kokoro voices.")
    if not config.pdf.exists():
        raise FileNotFoundError(f"PDF does not exist: {config.pdf}")
    if config.image_size < 256:
        raise ValueError("--image-size must be at least 256.")

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
        mlx_lang_code=config.mlx_lang_code,
        mlx_max_tokens=config.mlx_max_tokens,
        kokoro_language=config.kokoro_language,
        kokoro_speed=config.kokoro_speed,
        text_only=config.text_only,
        raw_ocr=config.raw_ocr,
        overwrite=config.overwrite,
        refresh=list(config.refresh),
    )

    image_paths = render_pdf_pages(config, paths)
    markdown_paths = convert_images_to_markdown(None, config, paths, image_paths)
    merge_pages(markdown_paths, paths.raw_book, paths)
    chapters = detect_chapters(markdown_paths, paths)

    final_page_paths = clean_markdown_pages(None, config, paths, markdown_paths)
    if not config.raw_ocr:
        repair_chapter_titles_in_pages(final_page_paths, chapters, paths)
    final_book = paths.raw_book if config.raw_ocr else paths.cleaned_book
    merge_pages(final_page_paths, final_book, paths)
    audiobook_text = write_chapter_texts(
        final_page_paths,
        chapters,
        paths.audiobook_text,
        paths.chapter_texts,
        paths,
    )

    if not config.text_only:
        if config.audio_engine == "kokoro":
            chapter_paths = filter_chapter_paths(paths.chapter_texts.glob("chapter_*.md"), config.chapters)
            synthesize_kokoro_chapters(config, paths, chapter_paths)
        elif config.audio_engine == "mlx-chatterbox":
            chapter_paths = filter_chapter_paths(paths.chapter_texts.glob("chapter_*.md"), config.chapters)
            synthesize_mlx_chatterbox_chapters(config, paths, chapter_paths)
        else:
            client = make_client()
            final_text = audiobook_text.read_text(encoding="utf-8")
            chunk_paths = synthesize_audio_chunks(client, config, paths, final_text)
            concatenate_wavs(chunk_paths, paths.audiobook, paths)
    else:
        log_event(paths, "stage_skipped", stage="synthesize_audio", reason="text_only")
        log_event(paths, "stage_skipped", stage="concatenate_audio", reason="text_only")

    write_cost_report(paths)
    log_event(paths, "run_completed", output=str(paths.root))
    return paths


def main(argv: list[str] | None = None) -> int:
    config = parse_args(sys.argv[1:] if argv is None else argv)
    if config.list_voices:
        print_kokoro_voices()
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
