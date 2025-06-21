from sentence_transformers import SentenceTransformer
from pymilvus import Collection
import logging

logger = logging.getLogger(__name__)

# Load model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar_metadata(query: str, collection: Collection, doc_type=None, code=None, top_k=3):
    logger.debug(f"[RAG] Encoding query: {query}")
    query_embedding = embedding_model.encode(query).tolist()
    # Tạo filter biểu thức nếu có điều kiện lọc
    filters = []
    if doc_type:
        filters.append(f"doc_type == '{doc_type}'")
    if code:
        filters.append(f"code == '{code}'")
    filter_expr = " and ".join(filters) if filters else None

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    logger.debug(f"[RAG] Searching with filter: {filter_expr}")
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["doc_type", "code", "issue_date", "effective_date"],
        filter=filter_expr
    )

    logger.debug(f"[RAG] Found {len(results[0])} results")

    # Trả về metadata của các bản ghi gần nhất
    output = []
    for hit in results[0]:
        item = {
            "score": hit.distance,
            "doc_type": hit.entity.get("doc_type"),
            "code": hit.entity.get("code"),
            "issue_date": hit.entity.get("issue_date"),
            "effective_date": hit.entity.get("effective_date"),
        }
        output.append(item)

    return output

def rag_query(query: str, collection: Collection, doc_type=None, code=None):
    results = retrieve_similar_metadata(query, collection, doc_type, code)
    if not results:
        return "Không tìm thấy văn bản phù hợp."
    return results

def retrieve_metadata_by_query(query, collection, top_k=3):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode([query])
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["doc_type", "code", "issue_date", "effective_date", "text"]
    )
    
    docs = []
    for hit in results[0]:
        entity = hit.entity
        docs.append({
            "score": hit.score,
            "doc_type": entity.get("doc_type"),
            "code": entity.get("code"),
            "issue_date": entity.get("issue_date"),
            "effective_date": entity.get("effective_date"),
            "text": entity.get("text")
        })

    return docs
