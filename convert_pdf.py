import os
import sys
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from openai import OpenAI
import time

# ───────────────────────────────
# GLOBAL CONFIGURATION
# ───────────────────────────────
MODEL_NAME = "gpt-4.1-nano"
TOKEN_RATE_PER_TOKEN_INPUT = 0.10 / 1_000_000  # $0.10 per 1M
TOKEN_RATE_PER_TOKEN_OUTPUT = 0.40 / 1_000_000  # $0.40 per 1M
IMAGE_TOKEN_MULTIPLIER = 2.46  # For gpt-4.1-nano
IMAGE_PATCH_SIZE = 32
IMAGE_TOKEN_CAP = 1536

MAX_WIDTH, MAX_HEIGHT = 1024, 1024

total_spent = 0.0  # Global variable to track total cost

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

# ───────────────────────────────
# UTILS
# ───────────────────────────────
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_image_tokens_and_cost(width, height):
    patches_w = (width + IMAGE_PATCH_SIZE - 1) // IMAGE_PATCH_SIZE
    patches_h = (height + IMAGE_PATCH_SIZE - 1) // IMAGE_PATCH_SIZE
    patch_count = patches_w * patches_h
    capped_patches = min(patch_count, IMAGE_TOKEN_CAP)

    billed_tokens = capped_patches * IMAGE_TOKEN_MULTIPLIER
    cost = billed_tokens * TOKEN_RATE_PER_TOKEN_INPUT

    return int(capped_patches), int(billed_tokens), round(cost, 4)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ───────────────────────────────
# MAIN PROCESSING
# ───────────────────────────────
def extract_pdf_pages_to_images(pdf_path, output_dir):
    ensure_dir(output_dir)
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        image_path = os.path.join(output_dir, f"page_{page_number+1:03d}.png")
        if not os.path.exists(image_path):
            page = doc.load_page(page_number)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            print(f"[Debug] original image size = {img.width} x {img.height}")
            img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
            print(f"[Debug] resized image size = {img.width} x {img.height}")
            img.save(image_path)
            print(f"Saved {image_path}")
        else:
            print(f"Skipping existing image {image_path}")

def extract_markdown_from_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    patches, image_tokens, image_cost = calculate_image_tokens_and_cost(width, height)
    print(f"[Image] {patches} patches, billed as {image_tokens} tokens, cost: ${image_cost:.4f}")

    b64_img = image_to_base64(image_path)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert this scanned page to clean Markdown format."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]
            }
        ],
        max_completion_tokens=2048
    )

    content = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens + image_tokens  # Add image token contribution
    completion_tokens = usage.completion_tokens

    prompt_cost = prompt_tokens * TOKEN_RATE_PER_TOKEN_INPUT
    completion_cost = completion_tokens * TOKEN_RATE_PER_TOKEN_OUTPUT
    total_cost = round(prompt_cost + completion_cost, 4)

    print(f"[Tokens] prompt: {prompt_tokens} (text + image), completion: {completion_tokens}")
    print(f"[Cost] prompt: ${prompt_cost:.4f}, completion: ${completion_cost:.4f}, total: ${total_cost:.4f}")

    return content, total_cost

def convert_images_to_markdown(image_dir, output_dir):
    global total_spent
    ensure_dir(output_dir)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        page_number = os.path.splitext(image_file)[0]
        md_output_path = os.path.join(output_dir, f"{page_number}.md")

        if os.path.exists(md_output_path):
            print(f"Skipping existing markdown for {image_file}")
            continue

        print(f"Processing {image_file}...")
        try:
            markdown, cost = extract_markdown_from_image(image_path)
            total_spent += cost
            with open(md_output_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"Saved {md_output_path} (cost: ${cost:.4f})\n")
        except Exception as e:
            print(f"Failed on {image_file}: {e}")

def graceful_exit():
    print(f"\n✅ Estimated total cost up to this point: ${total_spent:.4f}")

# ───────────────────────────────
# ENTRY POINT
# ───────────────────────────────
if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python convert_pdf.py <your_document.pdf>")
            sys.exit(1)

        pdf_path = sys.argv[1]
        if not os.path.isfile(pdf_path):
            print(f"File does not exist: {pdf_path}")
            sys.exit(1)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        working_dir = os.path.join("output", base_name)
        cache_img_dir = os.path.join(working_dir, "pdf_pages")
        markdown_output_dir = os.path.join(working_dir, "markdown_pages")

        extract_pdf_pages_to_images(pdf_path, cache_img_dir)
        convert_images_to_markdown(cache_img_dir, markdown_output_dir)

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user (Ctrl+C).")
        graceful_exit()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        graceful_exit()
        sys.exit(1)

    graceful_exit()
