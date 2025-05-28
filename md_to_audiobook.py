import os
import re
import sys
import time
from openai import OpenAI
import tiktoken
import subprocess

# ───────── CONFIG ─────────
TTS_MODEL = "gpt-4o-mini-tts"
VOICE = "alloy"
MAX_TOKENS = 2000
TMP_DIR = "tts_chunks"
OUTPUT_FILE = "audiobook.mp3"
MODEL_FOR_TOKENIZATION = "gpt-4"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

# ───────── HELPERS ─────────
def chunk_text(text, max_tokens):
    enc = tiktoken.encoding_for_model(MODEL_FOR_TOKENIZATION)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []

    for sentence in sentences:
        current.append(sentence)
        tokens = len(enc.encode(" ".join(current)))
        if tokens > max_tokens:
            # Remove last sentence to stay within budget
            current.pop()
            chunks.append(" ".join(current))
            current = [sentence]

    if current:
        chunks.append(" ".join(current))
    return chunks

def synthesize_chunks(chunks, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        print(f"[TTS] Synthesizing chunk {i+1}/{len(chunks)}...")
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=chunk
        )
        out_path = os.path.join(tmp_dir, f"chunk_{i+1:03d}.mp3")
        with open(out_path, "wb") as f:
            f.write(response.content)
        time.sleep(1.0)

def merge_audio_chunks(tmp_dir, output_file):
    files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".mp3"))
    concat_file = os.path.join(tmp_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for file in files:
            f.write(f"file '{os.path.join(tmp_dir, file)}'\n")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-c", "copy", output_file
    ])
    print(f"\n✅ Audiobook saved as {output_file}")

# ───────── MAIN ─────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_audiobook.py <cleaned_markdown.md>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, MAX_TOKENS)
    synthesize_chunks(chunks, TMP_DIR)
    merge_audio_chunks(TMP_DIR, OUTPUT_FILE)
