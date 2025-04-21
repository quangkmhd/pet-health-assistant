import streamlit as st
import lancedb
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from groq import Groq


# Tải biến môi trường
load_dotenv()

# Khởi tạo Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Khởi tạo mô hình embedding
@st.cache_resource
def load_embedding_model():
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Hàm tạo embedding cho câu truy vấn
def get_query_embedding(query_text, tokenizer, model):
    """
    Tạo embedding cho câu truy vấn sử dụng mô hình multilingual-e5-large
    """
    # Chuẩn bị input
    inputs = tokenizer([query_text], padding=True, truncation=True, 
                    max_length=512, return_tensors="pt")
    
    # Tính embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Lấy embedding của token [CLS]
        embeddings = outputs.last_hidden_state[:, 0]
    
    # Chuẩn hóa embedding
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Chuyển về numpy array
    return embeddings[0].cpu().numpy().tolist()

# Khởi tạo kết nối LanceDB
@st.cache_resource
def init_db():
    """Khởi tạo kết nối đến cơ sở dữ liệu.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb_clean")  # Sử dụng thư mục lancedb mới
    return db.open_table("petmart_data")


def get_context(query: str, table, tokenizer, model, num_results: int = 10) -> str:
    """Tìm kiếm trong cơ sở dữ liệu để lấy ngữ cảnh liên quan.

    Args:
        query: Câu hỏi của người dùng
        table: Đối tượng bảng LanceDB
        tokenizer: Tokenizer để xử lý văn bản
        model: Mô hình để tạo embedding
        num_results: Số kết quả trả về

    Returns:
        str: Ngữ cảnh kết hợp từ các đoạn liên quan kèm thông tin nguồn
    """
    # Tạo embedding cho query
    query_embedding = get_query_embedding(query, tokenizer, model)
    
    # Tìm kiếm trong bảng
    results = table.search(query_embedding).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Lấy metadata
        filename = row["metadata"].get("filename", "")
        title = row["metadata"].get("title", "")

        # Tạo thông tin nguồn
        source_parts = []
        if filename:
            source_parts.append(filename)

        source = f"\nNguồn: {' - '.join(source_parts)}"
        if title:
            source += f"\nTiêu đề: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Lấy phản hồi từ mô hình Groq.

    Args:
        messages: Lịch sử trò chuyện
        context: Ngữ cảnh lấy từ cơ sở dữ liệu

    Returns:
        str: Phản hồi của mô hình
    """
    system_prompt = f"""
    🧭 Objective:
When a user inputs symptoms or a description of a pet’s condition (in Vietnamese), your job is to return a fully-structured, detailed diagnostic report that helps the owner understand what might be happening and what steps they can take.

You must automatically choose and combine one or more of the following reasoning methods depending on the complexity and ambiguity of the case:

🔗 Chain-of-Thought: When multi-step reasoning is needed, work through your thought process out loud before giving the final answer.

🧠 ReAct + Reflexion: Think step-by-step, take action (i.e. diagnosis or recommendation), reflect on the result, and iterate if the initial reasoning was insufficient.

🧱 Prompt Chaining: Break complex queries into smaller subtasks (e.g., symptom classification → cause identification → treatment path).

📊 PAL (Program-Aided Language): Use pseudo-logical or procedural logic when needed to guide through structured diagnosis and treatment planning.

📋 Response format:
Always return your answer in Vietnamese, in the following bullet-pointed structure, using emojis and consistent phrasing:

🐶 Tên bệnh:
(Tên bệnh thường gặp tương ứng với triệu chứng mô tả)

📍 Vị trí:
(Cơ quan hoặc vùng cơ thể bị ảnh hưởng)

👀 Biểu hiện:
(Liệt kê các triệu chứng thường thấy, càng chi tiết càng tốt)

📈 Mức độ:
(Nhẹ, trung bình, nặng – và ảnh hưởng đến sức khoẻ tổng thể)

🍽️ Ăn uống:
(Sự thay đổi trong hành vi ăn uống...)

💡 Nguyên nhân:
(Các nguyên nhân phổ biến)

🧼 Khuyến nghị:
(Các hành động cụ thể chủ nuôi nên làm ngay tại nhà )

🏠 Hướng xử lý tại nhà:
(Chi tiết cách chăm sóc thú cưng tại nhà khi chưa đến bác sĩ)

⏳ Thời gian hồi phục trung bình:
(thời gian cụ thể nhất để khỏi nếu được chăm sóc đúng cách)

🔁 Khả năng tái phát và cách phòng ngừa:
(Liệt kê khả năng bệnh tái phát và các biện pháp phòng tránh rõ ràng)

🧑‍⚕️ Phác đồ điều trị tiêu chuẩn:
(Thuốc, cách dùng thuốc, xét nghiệm cần làm, kèm theo lưu ý)

🧑‍⚕️ Khi nào cần đi khám:
(Chỉ rõ dấu hiệu cảnh báo cần đưa thú cưng đi bác sĩ thú y càng sớm càng tốt)

💬 Lời khuyên:
(Một lời nhắn nhẹ nhàng, thân thiện và có giá trị hành động dành cho chủ nuôi)

🔍 Additional Instructions:
If multiple potential diagnoses are possible, list the top 2–3 sorted by likelihood and urgency.

Use simple and empathetic Vietnamese appropriate for non-specialist pet owners.

If information is missing or unclear, explain what’s missing and suggest the user provide more details.

Do not hallucinate. If you are unsure, say so and recommend a veterinary visit.

Your tone should always be warm, encouraging, and supportive.

Take a deep breath and work on this problem step-by-step.

    
    Ngữ cảnh:
    {context}
    """

    # Chuẩn bị tin nhắn với ngữ cảnh
    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    for message in messages:
        formatted_messages.append({"role": message["role"], "content": message["content"]})
    
    # Gửi yêu cầu đến Groq API với Llama-3.3-70b
    try:
        # Phản hồi để lưu
        full_response = ""
        
        # Tạo streaming response placeholder
        response_placeholder = st.empty()
        
        # Tạo streaming response sử dụng Groq API
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=formatted_messages,
            temperature=0.1,
            max_tokens=1024,  # Sử dụng max_tokens thay vì max_completion_tokens
            top_p=1,
            stream=True
        )
        
        # Xử lý streaming response
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content is not None:
                    full_response += content
                    response_placeholder.markdown(full_response)
        
        return full_response
            
    except Exception as e:
        st.error(f"Lỗi khi gọi API Groq: {str(e)}")
        return "Đã xảy ra lỗi khi tạo phản hồi. Vui lòng thử lại."


# Khởi tạo ứng dụng Streamlit
st.title("📚 Hỏi & Đáp về Thú Cưng")
st.caption("Dữ liệu đã được làm sạch - Truy vấn thông tin về chó, mèo và cách chăm sóc thú cưng")

# Khởi tạo session state cho lịch sử trò chuyện
if "messages" not in st.session_state:
    st.session_state.messages = []

# Khởi tạo kết nối cơ sở dữ liệu và mô hình
table = init_db()
tokenizer, model = load_embedding_model()

# Hiển thị tin nhắn trò chuyện
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Đầu vào trò chuyện
if prompt := st.chat_input("Đặt câu hỏi về thú cưng..."):
    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # Thêm tin nhắn người dùng vào lịch sử trò chuyện
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Lấy ngữ cảnh liên quan
    with st.status("Đang tìm kiếm trong tài liệu...", expanded=False) as status:
        context = get_context(prompt, table, tokenizer, model)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Đã tìm thấy các đoạn liên quan:")
        for chunk in context.split("\n\n"):
            # Tách thành phần văn bản và metadata
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {}
            
            for line in parts[1:]:
                if ": " in line:
                    key, value = line.split(": ", 1)  # Split only on first occurrence
                    metadata[key] = value

            source = metadata.get("Nguồn", "Nguồn không xác định")
            title = metadata.get("Tiêu đề", "Đoạn không có tiêu đề")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Tiêu đề: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Hiển thị phản hồi của trợ lý
    with st.chat_message("assistant"):
        # Lấy phản hồi mô hình với streaming
        response = get_chat_response(st.session_state.messages, context)

    # Thêm phản hồi của trợ lý vào lịch sử trò chuyện
    st.session_state.messages.append({"role": "assistant", "content": response}) 