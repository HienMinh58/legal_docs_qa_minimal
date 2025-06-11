from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from transformers import pipeline, AutoTokenizer

def retrieve_relevant_chunks(query, collection, top_k=3, max_tokens=700):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode([query])
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["data"]
    )
    chunks = [hit.entity.get("data") for hit in results[0]]
    
    # Sử dụng tokenizer để tính độ dài token
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    context = ""
    current_tokens = 0
    for chunk in chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        if current_tokens + len(tokens) <= max_tokens:
            context += chunk + "\n"
            current_tokens += len(tokens)
        else:
            break
    
    return context.strip()

def generate_answer(query, context):
    generator = pipeline('text-generation', model='distilgpt2', clean_up_tokenization_spaces=True)
    prompt = f"Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng tiếng Việt một cách rõ ràng và chính xác. Nếu không tìm thấy thông tin phù hợp, trả lời 'Không tìm thấy thông tin phù hợp.'\n\nNgữ cảnh: {context}\nCâu hỏi: {query}\nTrả lời:"
    try:
        # Kiểm tra độ dài prompt
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if len(prompt_tokens) > 1024:
            return "Ngữ cảnh quá dài, không thể xử lý. Vui lòng thử câu hỏi ngắn hơn hoặc giảm số chunk."
        
        response = generator(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)
        answer = response[0]['generated_text']
        answer_part = answer.split("Trả lời:")[-1].strip()
        return answer_part if answer_part else "Không tìm thấy câu trả lời cụ thể."
    except Exception as e:
        return f"Lỗi khi sinh câu trả lời: {e}"

def rag_query(query, collection):
    relevant_context = retrieve_relevant_chunks(query, collection)
    answer = generate_answer(query, relevant_context)
    return answer