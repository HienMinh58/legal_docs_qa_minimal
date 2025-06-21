import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import logging
import fitz
import requests
from io import BytesIO

from app.pre_processing.chunking import Chunking

logger = logging.getLogger(__name__)

def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Không tải được file PDF.")

    pdf_bytes = BytesIO(response.content)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def pdf_to_images(pdf_path, output_dir):
    logger.debug(f"Processing PDF at {pdf_path}")
    if not os.path.isfile(pdf_path):
        raise Exception(f"{pdf_path} is not a valid file")
    os.makedirs(output_dir, exist_ok=True)
    try:
        images = convert_from_path(pdf_path, dpi=300)
        logger.debug(f"Successfully converted PDF to {len(images)} images")
    except Exception as e:
        logger.error(f"Error in convert_from_path: {e}")
        raise
    image_files = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        image.save(image_path, 'JPEG')
        image_files.append(image_path)
    return image_files

def images_to_text(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    texts = {}
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='vie', config='--psm 6')
        text_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        texts[image_file] = text
    return texts

def chunk_by_sentences(text, max_words=100, overlap_sentences=1):
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        word_count = len(words)
        
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_word_count = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_word_count += word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_and_chunk(image_dir, output_text_dir, output_chunk_dir, max_words=100, overlap_sentences=1):
    os.makedirs(output_chunk_dir, exist_ok=True)
    texts = images_to_text(image_dir, output_text_dir)
    chunks = {}
    for image_file, text in texts.items():
        chunked = chunk_by_sentences(text, max_words, overlap_sentences)
        chunks[image_file] = chunked
        for i, chunk in enumerate(chunked):
            chunk_path = os.path.join(output_chunk_dir, f"{os.path.splitext(image_file)[0]}_chunk_{i+1}.txt")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
    return chunks

def chunk_pdf_text(url):
    text = extract_text_from_pdf_url(url)
    chunker = Chunking(max_characters=1000, overlap_size=50)
    chunks = chunker.split_document_with_order_overlap(text)
    
    for c in chunks:
        print(f"[Chunk {c['chunk_id']}] {c['content'][:100]}...")
    
    return chunks
