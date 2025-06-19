import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

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