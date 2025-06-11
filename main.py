import os
from src import data_processing, embedding, rag
from pymilvus import connections, Collection

def main():
    # Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "data/input_pdfs")
    image_dir = os.path.join(base_dir, "data/pdf_images")
    text_dir = os.path.join(base_dir, "data/extracted_text")
    chunk_dir = os.path.join(base_dir, "data/chunked_text")

    # Kiểm tra và tạo thư mục nếu cần
    for dir_path in [pdf_dir, image_dir, text_dir, chunk_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Kiểm tra file PDF đầu vào
    if not os.listdir(pdf_dir):
        print(f"Không tìm thấy file PDF trong {pdf_dir}. Vui lòng thêm file.")
        return

    # Xử lý PDF thành ảnh
    data_processing.pdf_to_images(pdf_dir, image_dir)

    # Trích xuất văn bản và chunking
    chunked_texts = data_processing.process_and_chunk(image_dir, text_dir, chunk_dir, max_words=100, overlap_sentences=1)
    if not chunked_texts:
        print(f"Không trích xuất được văn bản hoặc chunk từ {image_dir}.")
        return

    # Tạo danh sách chunks với metadata
    chunks = []
    for image_file, chunk_list in chunked_texts.items():
        text_path = os.path.join(text_dir, f"{os.path.splitext(image_file)[0]}.txt")
        with open(text_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        for i, chunk in enumerate(chunk_list):
            chunk_data = {"chunk_id": i + 1, "data": chunk, "metadata": {}}
            chunks.append(chunk_data)

    # Tạo embeddings
    embeddings = embedding.generate_embeddings(chunks)

    # Lưu vào Milvus (sử dụng read_chunk_files để điền metadata)
    processed_chunks = embedding.read_chunk_files(chunk_dir)  # Thêm bước này
    embedding.save_to_milvus(processed_chunks, embeddings)

    # Tải collection để dùng RAG
    try:
        collection = Collection("chunked_legal_vectors")
        collection.load()
    except Exception as e:
        print(f"Lỗi tải collection: {e}")
        return

    # Chạy query từ terminal
    while True:
        query = input("Nhập câu hỏi (hoặc 'quit' để thoát): ")
        if query.lower() == 'quit':
            break
        try:
            answer = rag.rag_query(query, collection)
            print(f"Trả lời: {answer}")
        except Exception as e:
            print(f"Lỗi khi xử lý query: {e}")

if __name__ == "__main__":
    main()