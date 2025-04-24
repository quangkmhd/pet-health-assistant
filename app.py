from flask import Flask, render_template, request, jsonify, session
import lancedb
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq
import requests
import json
from livereload import Server

load_dotenv()

app = Flask(__name__)
app.secret_key = "123"
app.debug = True  


groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

model = SentenceTransformer("intfloat/multilingual-e5-large")

def get_query_embedding(query):
    # Chuẩn hóa L2 embedding cho query và trả về list float
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    return query_embedding.tolist()

def init_db():
    db = lancedb.connect("data/lancedb_clean")
    return db.open_table("petmart_data")

def get_context(query, table, num_results=10):
    query_embedding = get_query_embedding(query)
    
    # Tìm kiếm trong bảng lancedb
    results = table.search(query_embedding).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
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

def get_chat_response(messages, context, model_choice="groq"):

    system_prompt = f"""
    Bạn là một trợ lý thú y chuyên nghiệp và cố vấn sức khỏe vật nuôi giàu kinh nghiệm, đã hỗ trợ hàng ngàn chủ nuôi suốt 15 năm qua. 
    Bạn giỏi lắng nghe, giải thích dễ hiểu, đưa ra các bước hành động cụ thể dựa trên kiến thức thú y cập nhật, có khả năng giải thích thân thiện như bác sĩ, 
    đồng thời trình bày chuyên nghiệp như ChatGPT.

🚩 Nhiệm vụ chính:
Khi người dùng mô tả triệu chứng của thú cưng bằng tiếng Việt, hãy phản hồi bằng một bản chẩn đoán đầy đủ, trình bày đẹp mắt, chia phần rõ ràng bằng biểu tượng cảm xúc, giúp chủ nuôi:

Hiểu rõ bệnh có thể là gì

Biết nên xử lý thế nào tại nhà

Nhận diện lúc nào cần đi khám gấp

Nhận thêm các câu hỏi chuyên sâu nếu cần bổ sung thông tin

🧠 Cách xử lý thông minh (áp dụng tự động):
🔗 Chain-of-Thought reasoning để giải thích từng bước

🧠 ReAct + Reflexion: Quan sát → Phân tích → Suy xét → Hiệu chỉnh

📊 PAL-style logic để xử lý nhiều triệu chứng

🧱 Prompt-chaining chia nhỏ tác vụ:

Phân loại triệu chứng

Liệt kê bệnh có thể

Đánh giá rủi ro

Khuyến nghị hành động rõ ràng

📝 Cấu trúc phản hồi tiêu chuẩn (bắt buộc tuân thủ):
Sử dụng gạch đầu dòng, biểu tượng cảm xúc, trình bày giống ChatGPT. Văn phong nhẹ nhàng như một người bác sĩ thú y tận tâm nói chuyện trực tiếp với chủ nuôi, 
không sử dụng **, ## các kí tự đặc biệt khác ở đầu câu.

🐶 Các bệnh có thể gặp:  
1. [Tên bệnh 1] – 📈 Mức độ: Trung bình/Cao  
   👉 Dấu hiệu: …  
   👉 Vì sao có thể mắc: …  

2. [Tên bệnh 2] – 📈 Mức độ: …  
   👉 Dấu hiệu: …  
   👉 Vì sao có thể mắc: …  

3. [Tên bệnh 3] (nếu cần) – 📈 Mức độ: …  
   👉 Dấu hiệu: …  
   👉 Vì sao có thể mắc: …  

📍 Vị trí có thể ảnh hưởng:  
👀 Các biểu hiện đã ghi nhận:  
🍽️ Tình trạng ăn uống:  
💡 Nguyên nhân phổ biến:  
🧼 Khuyến nghị chăm sóc:  
🏠 Hướng dẫn chăm sóc tại nhà theo từng bước:  
⏳ Thời gian hồi phục (ước tính):  
🔁 Nguy cơ tái phát & cách phòng tránh:  
🧑‍⚕️ Phác đồ điều trị phổ biến (theo từng khả năng bệnh):  
🧑‍⚕️ Khi nào cần đến bác sĩ thú y:  
💬 Lời khuyên:


❓ Hảy trả lời câu hỏi sau đây để PET HEALTH biết thêm thông tin về bệnh để có thể đưa ra bệnh chính sác nhất:
Dựa vào thông tin ban đầu, bạn cần đưa ra 10 câu hỏi dạng Có/Không giúp người dùng xác định xem PET có thể đang mắc một bệnh cụ thể nào đó. có ví dụ minh họa,:


⚠️ Lưu ý quan trọng:
Luôn đưa ra câu trả lời chi tiết đầy đủ, rõ ràng cho từng phần, không ngắn gọn, không bao giờ đoán bừa. Nếu nghi ngờ, hãy khuyên đi khám.
Luôn viết bằng ngôn ngữ gần gũi, giải thích dễ hiểu.
Phản hồi phải trông "xịn" như một bác sĩ, nhưng dễ tiếp cận như người bạn đáng tin cậy.
🔍 Luôn nhấn mạnh bạn không thay thế bác sĩ thú y thực thụ. Điều chỉnh phản hồi theo từng loài (chó, mèo, v.v.).
    
    Ngữ cảnh:
    {context}
    """

    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    for message in messages:
        formatted_messages.append({"role": message["role"], "content": message["content"]})
    
    try:
        if model_choice == "groq":
            # Sử dụng mô hình Llama-3.3-70b của Groq (nhanh hơn nhưng tốn phí)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=formatted_messages,
                temperature=0.1,
                max_tokens= 2048,
                top_p=1
            )
            return response.choices[0].message.content
            
        else:  # model_choice == "openrouter"
            # Sử dụng mô hình DeepSeek Chat của OpenRouter.ai (chậm hơn nhưng miễn phí)
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": formatted_messages,
                "temperature": 0.1,
                "max_tokens": 2048,
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

table = init_db()

# Khởi tạo dữ liệu đánh giá sao
RATINGS_FILE = 'data/ratings.json'
if os.path.exists(RATINGS_FILE):
    with open(RATINGS_FILE, 'r') as f:
        try:
            ratings_data = json.load(f)
        except json.JSONDecodeError:
            ratings_data = {'count': 0, 'total': 0}
else:
    ratings_data = {'count': 0, 'total': 0}

@app.route('/')
def index():
    # Khởi tạo tùy chọn mô hình trong session nếu chưa có
    if 'model_choice' not in session:
        session['model_choice'] = 'groq'  # Mặc định là Groq
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
    # Lấy dữ liệu JSON từ client
    data = request.json
    query = data.get('message', '')
    chat_history = data.get('history', [])
    # Lấy model_choice từ client gửi lên, nếu hợp lệ thì override session
    client_model = data.get('model_choice')
    if client_model in ['groq', 'openrouter']:
        model_choice = client_model
        session['model_choice'] = client_model
    else:
        model_choice = session.get('model_choice', 'groq')
    
    # Lấy ngữ cảnh từ cơ sở dữ liệu
    context = get_context(query, table)
    
    # Lấy phản hồi từ mô hình đã chọn
    response = get_chat_response(chat_history + [{"role": "user", "content": query}], context, model_choice)
    
    # Cập nhật lịch sử phiên chat
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'] = chat_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response}
    ]
    
    return jsonify({
        'response': response,
        'model_used': model_choice,
        'context': context
    })

@app.route('/rating', methods=['GET'])
def get_rating():
    """Trả về đánh giá trung bình và số lượng đánh giá."""
    avg = ratings_data['total'] / ratings_data['count'] if ratings_data['count'] > 0 else 0
    return jsonify({'average': avg, 'count': ratings_data['count']})

@app.route('/rate', methods=['POST'])
def rate():
    """Nhận đánh giá của người dùng và cập nhật trung bình."""
    data = request.json
    rating = int(data.get('rating', 0))
    if rating < 1 or rating > 5:
        return jsonify({'status': 'error', 'message': 'Rating không hợp lệ'}), 400
    ratings_data['total'] += rating
    ratings_data['count'] += 1
    avg = ratings_data['total'] / ratings_data['count']
    # Lưu vào file
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings_data, f)
    return jsonify({'average': avg, 'count': ratings_data['count']})

if __name__ == '__main__':
    
    # Khởi tạo livereload server
    server = Server(app.wsgi_app)
    
    # Theo dõi các file và thư mục
    server.watch('*.css')  # Theo dõi tất cả file trong thư mục templates/
    server.watch('*.html')    # Theo dõi tất cả file trong thư mục static/
    server.watch('*.py')        # Theo dõi tất cả file Python trong thư mục hiện tại
    
    # Chạy server với livereload
    server.serve(
        host='127.0.0.1',
        port=5000,
        debug=True  # Kích hoạt chế độ debug
    )
