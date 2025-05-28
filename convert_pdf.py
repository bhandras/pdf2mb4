import os
import sys
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from openai import OpenAI
import time

# Get API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_pdf_pages_to_images(pdf_path, output_dir):
    ensure_dir(output_dir)
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        image_path = os.path.join(output_dir, f"page_{page_number+1:03d}.png")
        if not os.path.exists(image_path):
            page = doc.load_page(page_number)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img.save(image_path)
            print(f"Saved {image_path}")
        else:
            print(f"Skipping existing image {image_path}")

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def extract_markdown_from_image(image_path):
    b64_img = image_to_base64(image_path)
    response = client.chat.completions.create(
        model="o4-mini",
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
    return response.choices[0].message.content

def convert_images_to_markdown(image_dir, output_dir):
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
            markdown = extract_markdown_from_image(image_path)
            with open(md_output_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"Saved {md_output_path}")
        except Exception as e:
            print(f"Failed on {image_file}: {e}")
        time.sleep(1.5)

if __name__ == "__main__":
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

