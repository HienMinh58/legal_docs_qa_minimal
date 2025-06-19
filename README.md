# 🧾 Legal Document Q&A API

Hệ thống API sử dụng FastAPI + Milvus để lưu trữ, tìm kiếm và truy vấn văn bản pháp luật (luật, nghị định, thông tư...) kết hợp chatbot AI để trả lời câu hỏi từ người dùng.

---

## Mục tiêu

- Upload file url (file pdf) và metadata đi kèm.
- Lưu vector embedding vào Milvus để truy vấn nhanh.
- Cho phép tìm kiếm theo `doc_type`, `code`.
- Tích hợp chatbot để trả lời câu hỏi người dùng về các văn bản đã lưu.

---

## Cấu trúc dự án

```bash
legal_doc_qa/
├── app/
│   ├── main.py              # Entry point FastAPI
│   ├── router/
│   │   └── api.py           # Router định nghĩa các endpoint
│   └── src/
│       ├── embedding.py     # Xử lý và lưu vector vào Milvus
│       ├── rag.py           # Truy vấn vector từ Milvus
│       └── chatbot.py       # Giao tiếp với OpenRouter AI
├── requirements.txt         # Thư viện cần thiết
├── docker-compose.yml       # Khởi tạo Milvus & MinIO
└── README.md
