from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import os
import re

def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [model.encode(chunk["data"]) for chunk in chunks]
    return embeddings
def read_chunk_files(chunk_dir):
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.txt')]
    chunks = []
    for file_name in chunk_files:
        with open(os.path.join(chunk_dir, file_name), 'r', encoding='utf-8') as f:
            content = f.read()
            chunk_id = int(file_name.split('_')[-1].split('.')[0])
            ngay_ban_hanh = re.search(r'ngày \d{1,2} tháng \d{1,2} năm \d{4}', content)
            ngay_ban_hanh = ngay_ban_hanh.group(0) if ngay_ban_hanh else "Không rõ"
            dieu_match = re.search(r'Điều \d+\.', content)
            dieu = dieu_match.group(0) if dieu_match else "Không rõ"
            khoan_match = re.search(r'Khoản \d+\.\d+\.', content)
            khoan = khoan_match.group(0) if khoan_match else "Không rõ"
            
            words = content.split()
            word_count = len(words)
            start_word = 1
            end_word = word_count
            avg_word_per_page = word_count / content.count('--------------------------------------------------') if content.count('--------------------------------------------------') else 1
            start_page = int((chunk_id - 1) * avg_word_per_page) + 1
            end_page = int(chunk_id * avg_word_per_page)
            chunks.append({
                "chunk_id": chunk_id,
                "data": content,
                "metadata": {
                    "ngay_ban_hanh": ngay_ban_hanh,
                    "dieu": dieu,
                    "khoan": khoan,
                    "phan_loai_theo_luat": "Luật Phòng cháy chữa cháy",
                    "word_range": f"Từ {start_word} đến {end_word}",
                    "estimated_page_range": f"Trang {start_page} đến {end_page}"
                }
            })
    return sorted(chunks, key=lambda x: x["chunk_id"])
def save_to_milvus(chunks, embeddings):
    connections.connect(host='localhost', port='19530')
    if utility.has_collection("chunked_legal_vectors"):
        utility.drop_collection("chunked_legal_vectors")
    
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="data_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="ngay_ban_hanh", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="dieu", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="khoan", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="phan_loai_theo_luat", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="word_range", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="estimated_page_range", dtype=DataType.VARCHAR, max_length=256)
    ]
    schema = CollectionSchema(fields=fields, description="Chunked legal document vectors with metadata")
    collection = Collection("chunked_legal_vectors", schema)
    
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
    collection.create_index("embedding", index_params)
    
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    data_ids = chunk_ids
    datas = [chunk["data"] for chunk in chunks]
    embeddings_list = embeddings
    metadata = [chunk["metadata"] for chunk in chunks]
    ngay_ban_hanhs = [m["ngay_ban_hanh"] for m in metadata]
    dieus = [m["dieu"] for m in metadata]
    khoans = [m["khoan"] for m in metadata]
    phan_loai_theo_luats = [m["phan_loai_theo_luat"] for m in metadata]
    word_ranges = [m["word_range"] for m in metadata]
    estimated_page_ranges = [m["estimated_page_range"] for m in metadata]
    
    collection.insert([
        chunk_ids, data_ids, embeddings_list, datas, ngay_ban_hanhs, dieus, khoans,
        phan_loai_theo_luats, word_ranges, estimated_page_ranges
    ])
    collection.flush()
    collection.load()
    print(f"Đã lưu {len(chunks)} chunks vào Milvus.")