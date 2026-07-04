# pdf2mb4

Build a chapterized M4B audiobook from a scanned PDF with resumable OCR, text
cleanup, chapter detection, and audio synthesis stages.

The preferred entry point is the single-file `uv` script:

```bash
OPENAI_API_KEY=... uv run make_audiobook.py book.pdf
```

`uv` reads the dependency metadata embedded in `make_audiobook.py`, downloads the
needed Python packages, and runs the pipeline.

Markdown is still produced, but it is an intermediate artifact: the main output
is a resumable audiobook build ending in chapter WAV files and, optionally, an
M4B package.

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
audiobook.m4b      optional chapterized audiobook package
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
uv run make_audiobook.py book.pdf --audio-engine kokoro --m4b
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

## M4B Packaging

Generate all chapter WAV files and package them into a chapterized M4B:

```bash
uv run make_audiobook.py book.pdf \
  --audio-engine kokoro \
  --voice am_adam \
  --m4b
```

The M4B is written to `output/book/audiobook.m4b`. By default, the first
rendered PDF page is attached as the cover. Override it with `--cover cover.jpg`,
or disable cover art with `--no-cover`.

Intro WAV files can be prepended before chapter 1:

```bash
uv run make_audiobook.py book.pdf \
  --audio-engine kokoro \
  --m4b \
  --intro-wav intro.wav \
  --intro-wav dedication.wav \
  --cover cover.jpg
```

If chapter WAV files already exist, reruns reuse them. Use `--refresh m4b` to
rebuild only the M4B package after changing intro files, cover art, or bitrate.

## MLX-Audio Chatterbox

`mlx-chatterbox` uses `mlx-audio` with the `mlx-community/chatterbox-fp16`
model. It can synthesize speech with the model's shipped conditionals, so a
reference clip is not required.

```bash
uv run make_audiobook.py book.pdf \
  --audio-engine mlx-chatterbox \
  --chapters 1 \
  --mlx-speed 0.88 \
  --mlx-chunk-chars 350 \
  --mlx-max-tokens 4000 \
  --refresh audio
```

For voice cloning, pass a short reference WAV and its transcript with
`--mlx-ref-audio` and `--mlx-ref-text`. Keep the reference clip short and
transcript-matched; using a full chapter as the reference can produce unusable
audio.

Chatterbox output is written to `output/book/chapter_audio/mlx-chatterbox/`.
Long chapters are split into smaller MLX-Audio segments and stitched into one
chapter WAV to avoid single-call truncation. `--mlx-speed` post-processes the
chapter WAV after generation; values below `1.0` slow speech while preserving
pitch. Use `--refresh audio` when changing speed for an already-generated
chapter. This backend is much slower than Kokoro in local testing, so start with
a short sample before regenerating a full chapter.
