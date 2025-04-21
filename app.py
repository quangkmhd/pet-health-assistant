import sys
from flask import Flask, render_template, request, jsonify, session
import lancedb
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from groq import Groq
import requests
import json
from werkzeug.serving import run_simple

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key_for_development")

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize OpenRouter.ai API key
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize embedding model
def load_embedding_model():
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Create embedding for query
def get_query_embedding(query_text, tokenizer, model):
    """
    Create embedding for query using multilingual-e5-large model
    """
    # Prepare input
    inputs = tokenizer([query_text], padding=True, truncation=True, 
                    max_length=512, return_tensors="pt")
    
    # Calculate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Get embedding of [CLS] token
        embeddings = outputs.last_hidden_state[:, 0]
    
    # Normalize embedding
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Convert to numpy array
    return embeddings[0].cpu().numpy().tolist()

# Initialize LanceDB connection
def init_db():
    """Initialize connection to database.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb_clean")
    return db.open_table("petmart_data")

def get_context(query, table, tokenizer, model, num_results=10):
    """Search database to get relevant context.

    Args:
        query: User question
        table: LanceDB table object
        tokenizer: Tokenizer for text processing
        model: Model for embedding generation
        num_results: Number of results to return

    Returns:
        str: Combined context from relevant passages with source information
    """
    # Create embedding for query
    query_embedding = get_query_embedding(query, tokenizer, model)
    
    # Search in table
    results = table.search(query_embedding).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Get metadata
        filename = row["metadata"].get("filename", "")
        title = row["metadata"].get("title", "")

        # Create source information
        source_parts = []
        if filename:
            source_parts.append(filename)

        source = f"\nNguồn: {' - '.join(source_parts)}"
        if title:
            source += f"\nTiêu đề: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)

def get_chat_response(messages, context, model_choice="groq"):
    """Get response from selected model.

    Args:
        messages: Chat history
        context: Context from database
        model_choice: Which model to use ("groq" or "openrouter")

    Returns:
        str: Model response
    """
    system_prompt = f"""
    Act like a professional veterinary assistant and pet health advisor.
You have supported thousands of pet owners over the past 15 years. You specialize in interpreting symptoms, triaging based on urgency, and guiding pet owners with accurate, step-by-step advice using evidence-based veterinary knowledge. You are fluent in Vietnamese and communicate with warmth and empathy.

🎯 Objective:
When a user describes symptoms of their pet in Vietnamese, your job is to return a complete, structured, easy-to-understand diagnostic report — even when information is incomplete. Your response should help the pet owner:

Understand what might be wrong with their pet.

Know what to do at home immediately.

Know when to visit a veterinarian.

Receive follow-up questions if more info is needed.

🧠 Reasoning Strategy:
Based on the complexity of the symptoms, automatically combine one or more of the following cognitive tools:

🔗 Chain-of-Thought: Think step-by-step before concluding.

🧠 ReAct + Reflexion: Observe → Diagnose → Reflect → Refine.

🧱 Prompt Chaining: Break into subtasks:

Step 1: Symptom classification

Step 2: Potential diseases

Step 3: Risk and urgency level

Step 4: Home vs Clinic actions

📊 PAL (Program-Aided Language): Use structured pseudo-code logic to determine outcomes, especially for multi-symptom cases.

📋 Response Format in Vietnamese (bullet-pointed with emojis):

Hãy luôn giữ văn phong gần gũi, dễ hiểu, không dùng từ chuyên môn phức tạp.

🐶 Tên bệnh:
(Tên các bệnh phổ biến nhất dựa trên mô tả triệu chứng)

📍 Vị trí:
(Bộ phận hoặc cơ quan bị ảnh hưởng)

👀 Biểu hiện:
(Các triệu chứng rõ ràng và đặc trưng, càng chi tiết càng tốt)

📈 Mức độ:
(Nhẹ / Trung bình / Nặng — ảnh hưởng tổng thể ra sao)

🍽️ Ăn uống:
(Có thay đổi gì về khẩu vị, lượng nước, tần suất ăn uống không?)

💡 Nguyên nhân:
(Những nguyên nhân thường gặp và yếu tố nguy cơ)

🧼 Khuyến nghị:
(Chủ nuôi cần làm gì ngay bây giờ tại nhà)

🏠 Hướng xử lý tại nhà:
(Chi tiết từng bước chăm sóc tại nhà)

⏳ Thời gian hồi phục trung bình:
(Xác định khoảng thời gian nếu được chăm sóc đúng cách)

🔁 Khả năng tái phát & phòng tránh:
(Các yếu tố dẫn đến tái phát và cách ngăn ngừa)

🧑‍⚕️ Phác đồ điều trị tiêu chuẩn:
(Tên thuốc, liều dùng, xét nghiệm, và những lưu ý đặc biệt)

🧑‍⚕️ Khi nào cần đi khám:
(Dấu hiệu cảnh báo cần đưa thú cưng đến bác sĩ càng sớm càng tốt)

💬 Lời khuyên:
(Một lời nhắn nhẹ nhàng, thực tế và trấn an tinh thần chủ nuôi)

💬 Follow-up Questions (nếu thông tin chưa đủ):
Nếu triệu chứng mô tả quá mơ hồ hoặc không đầy đủ để đưa ra chẩn đoán, hãy đưa ra một danh sách các câu hỏi ngắn gọn, dễ hiểu, tập trung vào:

Thời gian phát bệnh

Các triệu chứng cụ thể hơn (ví dụ: sốt? ho? tiêu chảy?...)

Hành vi ăn uống / ngủ nghỉ

Các bệnh nền, tiền sử tiêm phòng

Thú cưng có ra ngoài gần đây không?

Loài, tuổi, cân nặng, giống thú cưng

📌 Sau mỗi câu hỏi, kèm một ví dụ cụ thể để người dùng dễ hình dung.

🔍 Extra Notes:

Nếu có nhiều khả năng chẩn đoán, liệt kê top 2–3 bệnh phổ biến, kèm mức độ khẩn cấp.

Không được đoán bừa. Nếu không chắc chắn, hãy nói thẳng và đề nghị đưa thú cưng đến phòng khám.

Giữ tone thân thiện, nhẹ nhàng như người hướng dẫn tận tâm.

Take a deep breath and work on this problem step-by-step.

    
    Ngữ cảnh:
    {context}
    """

    # Prepare messages with context
    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    for message in messages:
        formatted_messages.append({"role": message["role"], "content": message["content"]})
    
    try:
        if model_choice == "groq":
            # Use Groq's Llama-3.3-70b model (faster but paid)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=formatted_messages,
                temperature=0.1,
                max_tokens=1024,
                top_p=1
            )
            return response.choices[0].message.content
            
        else:  # model_choice == "openrouter"
            # Use OpenRouter.ai's DeepSeek Chat model (slower but free)
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": formatted_messages,
                "temperature": 0.1,
                "max_tokens": 1024,
                "top_p": 1.0
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error when calling OpenRouter API: {response.status_code} - {response.text}"
                
    except Exception as e:
        return f"Error when calling API: {str(e)}"

# Load models and database
tokenizer, model = load_embedding_model()
table = init_db()

@app.route('/')
def index():
    # Initialize model choice in session if not present
    if 'model_choice' not in session:
        session['model_choice'] = 'groq'  # Default to Groq
    return render_template('index.html', model_choice=session['model_choice'])

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api')
def api_info():
    return render_template('api.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/get_chat_history')
def get_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return jsonify(session['chat_history'])

@app.route('/save_chat_history', methods=['POST'])
def save_chat_history():
    data = request.json
    session['chat_history'] = data.get('history', [])
    return jsonify({"status": "success"})

@app.route('/set_model', methods=['POST'])
def set_model():
    data = request.json
    model_choice = data.get('model_choice')
    if model_choice in ['groq', 'openrouter']:
        session['model_choice'] = model_choice
        return jsonify({"status": "success", "model": model_choice})
    return jsonify({"status": "error", "message": "Invalid model choice"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    chat_history = data.get('history', [])
    model_choice = session.get('model_choice', 'groq')
    
    # Get context from database
    context = get_context(query, table, tokenizer, model)
    
    # Get response from selected model
    response = get_chat_response(chat_history + [{"role": "user", "content": query}], context, model_choice)
    
    # Update session history
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'] = chat_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response}
    ]
    
    return jsonify({
        'response': response,
        'context': context
    })

if __name__ == '__main__':
    # Configure Flask to ignore site-packages when watching for changes
    extra_files = None
    if app.debug:
        # Get only project files, not packages from site-packages
        import os
        extra_dirs = ['templates/', 'static/']
        extra_files = []
        for extra_dir in extra_dirs:
            for dirname, dirs, files in os.walk(extra_dir):
                for filename in files:
                    filename = os.path.join(dirname, filename)
                    if os.path.isfile(filename):
                        extra_files.append(filename)
    
    # Run with custom reloader configuration
    run_simple(
        '127.0.0.1', 
        5000, 
        app,
        use_reloader=app.debug,
        use_debugger=app.debug,
        extra_files=extra_files
    )
