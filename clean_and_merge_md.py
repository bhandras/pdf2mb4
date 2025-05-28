import os
import time
from openai import OpenAI

MODEL = "gpt-4.1-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_text_with_gpt(text, file_name=""):
    print(f"Cleaning {file_name}...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are cleaning a book page for audio narration. Remove scanning artifacts like 'page 42', headings, footnotes, and anything not meant to be read aloud. Keep the language natural."},
            {"role": "user", "content": text}
        ],
        max_tokens=2048
    )
    return response.choices[0].message.content.strip()

def clean_markdown_pages(input_dir, output_dir):
    ensure_dir(output_dir)
    md_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".md"))

    for fname in md_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path):
            print(f"Skipping already cleaned: {fname}")
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        try:
            cleaned = clean_text_with_gpt(raw_text, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            time.sleep(1.2)
        except Exception as e:
            print(f"Failed on {fname}: {e}")

def merge_cleaned_pages(cleaned_dir, output_file):
    files = sorted(f for f in os.listdir(cleaned_dir) if f.endswith(".md"))
    with open(output_file, "w", encoding="utf-8") as out:
        for fname in files:
            with open(os.path.join(cleaned_dir, fname), "r", encoding="utf-8") as f:
                out.write(f.read())
                out.write("\n\n---\n\n")
    print(f"✅ Merged cleaned text saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clean_and_merge_md.py <markdown_pages_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    cleaned_dir = os.path.join(input_dir, "..", "cleaned_pages")
    merged_output = os.path.join(input_dir, "..", "cleaned_book.md")

    clean_markdown_pages(input_dir, cleaned_dir)
    merge_cleaned_pages(cleaned_dir, merged_output)

