from openai import OpenAI
from app.src.embedding import get_collection
from app.src.rag import retrieve_metadata_by_query

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def ask_chatbot(prompt: str) -> str:
    col = get_collection()
    if not col:
        return "Không có dữ liệu để truy vấn"

    docs = retrieve_metadata_by_query(prompt, col)

    context = "\n\n".join([
        f"Văn bản: {doc['doc_type']}\nMã số: {doc['code']}\nNgày ban hành: {doc['issue_date']}\nNgày hiệu lực: {doc['effective_date']}\nNội dung:\n{doc['text']}"
        for doc in docs
    ])

    final_prompt = f"""Dưới đây là các thông tin văn bản pháp luật. Trả lời câu hỏi người dùng dựa trên dữ liệu:

{context}

Câu hỏi: {prompt}
"""

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý hiểu luật pháp Việt Nam."},
            {"role": "user", "content": final_prompt}
        ],
        extra_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "LegalDoc Chatbot"
        }
    )

    return completion.choices[0].message.content
