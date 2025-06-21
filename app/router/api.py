import os
import logging
import tempfile
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

from app.src.data_processing import pdf_to_images, process_and_chunk
from app.src.rag import rag_query
from app.src.embedding import insert_embedding, get_collection
from app.src.chatbot import ask_chatbot

from pymilvus import connections, Collection
import shutil

logger = logging.getLogger(__name__)
router = APIRouter()


class UploadRequest(BaseModel):
    url: str
    doc_type: str = None  # e.g., luật, nghị định, thông tư, ...
    code: str = None      # mã ký hiệu
    issue_date: str = None  # ngày ban hành (YYYY-MM-DD)
    effective_date: str = None  # ngày hiệu lực (YYYY-MM-DD)



@router.post("/upload/")
def upload_and_store(request: UploadRequest):
    """Endpoint to receive a URL and metadata, embed the URL text, and store with metadata."""
    col = get_collection()
    if not col:
        raise HTTPException(status_code=500, detail="Collection not initialized")
    try:
        record_id = insert_embedding(
            url=request.url,
            doc_type=request.doc_type,
            code=request.code,
            issue_date=request.issue_date,
            effective_date=request.effective_date
        )
        return {
            "status": "inserted",
            "metadata": {
                "url": request.url,
                "doc_type": request.doc_type,
                "code": request.code,
                "issue_date": request.issue_date,
                "effective_date": request.effective_date
            }
        }
    except Exception as e:
        logger.error(f"Error in upload_and_store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query/")
async def query_data(keyword: str):
    col = get_collection()
    if not col:
        return JSONResponse(content={"error": "Collection chưa được khởi tạo."}, status_code=500)
    
    try:
        logger.debug(f"[QUERY] keyword: {keyword}")
        logger.debug(f"[QUERY] Collection entities: {col.num_entities}")
        if not col.has_index():
            logger.warning("[QUERY] Collection chưa có index!")
        answer = rag_query(keyword, col)
        logger.debug(f"[QUERY] Kết quả: {answer}")
        return JSONResponse(content={"query": keyword, "answer": answer})
    except Exception as e:
        logger.error(f"Error in query_data: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


class ChatRequest(BaseModel):
    message: str

@router.post("/chat/")
def chat(request: ChatRequest):
    try:
        response = ask_chatbot(request.message)
        return {"response": response}
    except Exception as e:
        logger.error("[CHAT ERROR] %s", e)
        raise HTTPException(status_code=500, detail="Chatbot gặp lỗi.")
