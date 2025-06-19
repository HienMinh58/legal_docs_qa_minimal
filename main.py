from fastapi import FastAPI
import logging
from app.router import api
from app.src import embedding

embedding.init_collection()


# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(docs_url="/docs", title="Legal Doc QA API", description="API for uploading PDF and querying legal documents with BERT")
app.include_router(api.router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)