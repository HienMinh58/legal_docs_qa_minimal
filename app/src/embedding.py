from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging
import pprint

from app.src.data_processing import chunk_pdf_text


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

model = SentenceTransformer('all-MiniLM-L6-v2')
collection = None

def _init_milvus_collection(drop_existing: bool = False):
    try:
        connections.connect("default", host="localhost", port="19530")
        logger.debug("Connected to Milvus at localhost:19530")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise

    if drop_existing and utility.has_collection("legal_docs"):
        utility.drop_collection("legal_docs")
        logger.debug("Dropped existing collection 'legal_docs'")

    meta_fields = [
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="issue_date", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=32)
    ]

    if not utility.has_collection("legal_docs"):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            *meta_fields
        ]
        schema = CollectionSchema(fields, description="Text embeddings with metadata")
        col = Collection(name="legal_docs", schema=schema)
        logger.debug("Created new collection 'legal_docs'")
    else:
        col = Collection(name="legal_docs")
        logger.debug("Loaded existing collection 'legal_docs'")

    if not col.has_index():
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            col.create_index(field_name="embedding", index_params=index_params)
            logger.debug("Created index for 'embedding' field")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    try:
        col.load()
        logger.debug("Collection loaded")
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        raise

    return col

def init_collection():
    global collection
    if collection is None:
        collection = _init_milvus_collection(drop_existing=True)

def embed_text(text: str) -> list:
    embedding = model.encode(text).tolist()
    logger.debug(f"Generated embedding for text (length: {len(embedding)})")
    return embedding

def insert_embedding(
    url: str,
    doc_type: str = None,
    code: str = None,
    issue_date: str = None,
    effective_date: str = None
) -> list:
    global collection
    if collection is None:
        raise ValueError("Chưa tạo collection.")

    chunks = chunk_pdf_text(url)
    
    # logger.debug("Nội dung đầy đủ của chunks:\n%s", pprint.pformat(chunks, indent=2, width=120))
    


    

    insert_data = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        logger.debug(f"Chunk {i+1} length: {len(content)}")
        logger.debug(f"Nội dung text (length={len(content)}):\n{content[:500]}...")
        vector = embed_text(content)
        insert_data.append({
            "embedding": vector,
            "text": content,
            "doc_type": doc_type or "",
            "code": code or "",
            "issue_date": issue_date or "",
            "effective_date": effective_date or ""
        })

    try:
        result = collection.insert(insert_data)
        collection.flush()
        logger.debug(f"Nội dung text (length={len(content)}):\n{content[:500]}...") 
        logger.debug(f"Đã lưu {len(insert_data[0])} chunk vào Milvus, primary keys: {result.primary_keys}")
        return result.primary_keys
    except Exception as e:
        logger.exception("Lỗi khi insert vào Milvus")
        raise


def get_collection():
    global collection
    return collection
