# Resumable Audiobook Pipeline

Build an audiobook from a scanned PDF with resumable OCR, text cleanup, and
audio synthesis stages.

The preferred entry point is the single-file `uv` script:

```bash
OPENAI_API_KEY=... uv run make_audiobook.py book.pdf
```

`uv` reads the dependency metadata embedded in `make_audiobook.py`, downloads the
needed Python packages, and runs the pipeline.

Markdown is still produced, but it is an intermediate artifact: the main output
is a resumable audiobook build.

## Output

For `book.pdf`, files are written under `output/book/`:

```text
page_images/       rendered PDF pages
markdown_pages/    raw OCR Markdown, one file per page
cleaned_pages/     narration-cleaned text, one file per page
chapters/          chapter-aware narration text, one file per chapter
chapter_audio/     per-engine chapter WAV output
audio_chunks/      generated WAV chunks
book.md            merged raw OCR Markdown
cleaned_book.md    merged narration-cleaned Markdown
audiobook_text.md  chapter-aware text used for speech synthesis
chapters.json      detected chapter starts and rejected candidates
audiobook.wav      final concatenated audiobook
config.json        effective run configuration
cost_report.json   estimated text/vision API cost from logged usage
run_manifest.jsonl page/chunk processing log
```

Existing files are reused by default, so reruns resume where they left off.
Use `--overwrite` to regenerate existing page text and audio chunks.

Chapter detection is based on the raw OCR pages, then validated against page
order and chapter sequence before headings are inserted into the final
`audiobook_text.md`. This keeps narration chapter markers even when the cleanup
stage removes or normalizes them. The cleaned page files are also repaired from
the validated chapter list so chapter titles remain visible during audits.

The cost report uses actual token usage returned by text/vision API calls and a
small rate table sourced from OpenAI's published pricing page. It separates the
current run from all logged work, and breaks text/vision cost down by OCR and
cleanup stages. Exact billed spend is still reconciled through the OpenAI
dashboard or organization Costs endpoint, especially for audio generation.

`run_manifest.jsonl` is append-only and records major pipeline events as they
happen: run start/completion, stage start/completion/skips, rendered pages, OCR
pages, cleaned pages, chapter detection, chapter text creation, TTS chunks, WAV
concatenation, and cost report creation.

## Meaningful Options

```bash
uv run make_audiobook.py book.pdf --text-only
uv run make_audiobook.py book.pdf --voice nova
uv run make_audiobook.py book.pdf --raw-ocr
uv run make_audiobook.py book.pdf --image-size 1400
uv run make_audiobook.py book.pdf --ocr-model gpt-5.4-mini
uv run make_audiobook.py book.pdf --text-only --refresh clean
uv run make_audiobook.py book.pdf --audio-engine kokoro --chapters 1
```

The CLI intentionally exposes only options that change produced files, model
choices, voice, image quality/cost, output location, or resume behavior.

## Kokoro Chapter Audio

List known Kokoro voice IDs:

```bash
uv run make_audiobook.py --list-voices
```

Generate a separate WAV file for each detected chapter:

```bash
uv run make_audiobook.py book.pdf --audio-engine kokoro --voice am_adam
```

Files are written to `output/book/chapter_audio/`:

```text
chapter_001.wav
chapter_002.wav
chapter_003.wav
```

Kokoro uses the first character of the voice ID as the language by default:
`a` for American English, `b` for British English, `p` for Brazilian Portuguese,
and so on. You can override that when needed:

```bash
uv run make_audiobook.py book.pdf --audio-engine kokoro --voice bf_emma
uv run make_audiobook.py book.pdf --audio-engine kokoro --voice af_nicole --kokoro-speed 0.95
uv run make_audiobook.py book.pdf --audio-engine kokoro --voice pm_alex --kokoro-language p
uv run make_audiobook.py book.pdf --audio-engine kokoro --voice af_heart --refresh audio
```

Generate only chapter 1:

```bash
uv run make_audiobook.py book.pdf --audio-engine kokoro --voice am_adam --chapters 1
```

## MLX-Audio Chatterbox

`mlx-chatterbox` uses `mlx-audio` with the `mlx-community/chatterbox-fp16`
model. Chatterbox supports voice cloning through a reference WAV, so a practical
comparison is to generate a short male Kokoro reference and then use that as the
Chatterbox reference. Keep the reference clip short and pass its exact transcript;
using a full chapter as the reference can produce unusable audio.

```bash
uv run --with kokoro --with numpy python - <<'PY'
from pathlib import Path
import wave
import numpy as np
from kokoro import KPipeline

out = Path("output/book/voice_refs/kokoro_am_adam_reference.wav")
out.parent.mkdir(parents=True, exist_ok=True)
text = "This is a clean male reference voice for chapter narration. The pacing should be calm, clear, and steady."
pipeline = KPipeline(lang_code="a")
audio = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in pipeline(text, voice="am_adam")])
pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
with wave.open(str(out), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    wav.writeframes(pcm.tobytes())
PY

uv run make_audiobook.py book.pdf \
  --audio-engine mlx-chatterbox \
  --chapters 1 \
  --voice am_adam \
  --mlx-chunk-chars 350 \
  --mlx-max-tokens 4000 \
  --mlx-ref-audio output/book/voice_refs/kokoro_am_adam_reference.wav \
  --mlx-ref-text "This is a clean male reference voice for chapter narration. The pacing should be calm, clear, and steady."
```

Chatterbox output is written to `output/book/chapter_audio/mlx-chatterbox/`.
Long chapters are split into smaller MLX-Audio segments and stitched into one
chapter WAV to avoid single-call truncation. This backend is much slower than
Kokoro in local testing, so start with a short sample before regenerating a full
chapter.
