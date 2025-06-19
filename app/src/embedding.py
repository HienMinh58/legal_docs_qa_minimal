from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Global collection variable
collection = None

def _init_milvus_collection(drop_existing: bool = False):
    
    
    # Connect to Milvus with error handling
    try:
        connections.connect("default", host="localhost", port="19530")
        logger.debug("Connected to Milvus at localhost:19530")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise
    
    # Drop collection if drop_existing is True
    if drop_existing and utility.has_collection("legal_docs"):
        utility.drop_collection("legal_docs")
        logger.debug("Dropped existing collection 'legal_docs'")
    
    # Define metadata fields
    meta_fields = [
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="issue_date", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=32)
    ]
    
    # Create collection if it doesn't exist
    if not utility.has_collection("legal_docs"):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            *meta_fields
        ]
        schema = CollectionSchema(fields, description="Text embeddings with metadata")
        collection = Collection(name="legal_docs", schema=schema)
        logger.debug("Created new collection 'legal_docs'")
    else:
        collection = Collection(name="legal_docs")
        logger.debug("Loaded existing collection 'legal_docs'")
        # Add any missing metadata fields
        existing = {f.name for f in collection.schema.fields}
        to_add = [f for f in meta_fields if f.name not in existing]
        if to_add:
            collection.add_fields(to_add)
            logger.debug(f"Added new fields: {[f.name for f in to_add]}")
    
    # Create index if not exists
    if not collection.has_index():
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.debug("Created index for 'embedding' field")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    # Load collection
    try:
        collection.load()
        logger.debug("Collection loaded")
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        raise
    
    return collection

def init_collection():
    global collection
    if collection is None:
        collection = _init_milvus_collection(drop_existing=False)
        
def embed_text(text: str) -> list:
    """Generate embedding vector for given text."""
    embedding = model.encode(text).tolist()
    logger.debug(f"Generated embedding for text (length: {len(embedding)})")
    return embedding

def insert_embedding(
    text: str,
    id: int = None,
    doc_type: str = None,
    code: str = None,
    issue_date: str = None,
    effective_date: str = None
) -> int:
    global collection
    if collection is None:
        raise ValueError("Chưa tạo collection.")
    
    vector = embed_text(text)
    record_id = id or collection.num_entities + 1
    meta_values = [
        doc_type or "", code or "", issue_date or "", effective_date or ""
    ]
    collection.insert([
        [record_id],
        [vector],
        [meta_values[0]],
        [meta_values[1]],
        [meta_values[2]],
        [meta_values[3]]
    ])
    collection.flush()
    logger.debug(f"Đã lưu dữ liệu vào Milvus: {record_id}")
    return record_id

def get_collection():
    global collection
    return collection

