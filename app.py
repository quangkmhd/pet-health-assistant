from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
import lancedb
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import secrets
from groq import Groq
from openai import OpenAI
import requests
import json
from livereload import Server
import datetime
from system_prompt import get_system_prompt

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))
app.debug = True  


groq_api_key = os.getenv("GROQ_API_KEY")
Groq_client = Groq(api_key=groq_api_key)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

model = SentenceTransformer("intfloat/multilingual-e5-large")

def get_query_embedding(query):
    # Chuẩn hóa L2 embedding cho query và trả về list float
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    return query_embedding.tolist()

def init_db():
    db = lancedb.connect("data/lancedb_clean")
    return db.open_table("petmart_data")

def get_context(query, table, num_results=5, similarity_threshold=0.7):
    query_embedding = get_query_embedding(query)
    
    # Tìm kiếm trong bảng lancedb với nhiều kết quả hơn để lọc
    results = table.search(query_embedding).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Kiểm tra ngưỡng điểm tương đồng
        # Lưu ý: Trong LanceDB, điểm L2 càng thấp càng giống nhau, nên chuyển đổi thành điểm cosine
        # distance_score là L2 distance, chuyển sang cosine similarity: 1 - (distance_score/2)
        similarity_score = 1 - (row['_distance']/2)
        
        # Chỉ sử dụng những kết quả có độ tương đồng trên ngưỡng
        if similarity_score >= similarity_threshold:
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

    # Nếu không có kết quả nào vượt ngưỡng, trả về chuỗi rỗng
    if not contexts:
        return ""
        
    return "\n\n".join(contexts)

def get_chat_response(messages, context, model_choice="groq", pet_info=None):
    # Lấy system prompt từ file system_prompt.py
    system_prompt = get_system_prompt(context)

    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    # Thêm thông tin thú cưng vào formatted_messages nếu có
    if pet_info:
        pet_info_summary = "THÔNG TIN THÚ CƯNG:\n"
        if pet_info.get('pet_name'):
            pet_info_summary += f"- Tên: {pet_info['pet_name']}\n"
        if pet_info.get('pet_type'):
            pet_info_summary += f"- Loài: {pet_info['pet_type']}\n"
        if pet_info.get('pet_breed'):
            pet_info_summary += f"- Giống: {pet_info['pet_breed']}\n"
        if pet_info.get('pet_age'):
            pet_info_summary += f"- Tuổi: {pet_info['pet_age']}\n"
        if pet_info.get('pet_weight'):
            pet_info_summary += f"- Cân nặng: {pet_info['pet_weight']} kg\n"
        if pet_info.get('pet_gender'):
            pet_info_summary += f"- Giới tính: {pet_info['pet_gender']}\n"
        if pet_info.get('pet_health_history'):
            pet_info_summary += f"- Tiền sử bệnh: {pet_info['pet_health_history']}\n"
        if pet_info.get('pet_diet'):
            pet_info_summary += f"- Chế độ ăn: {pet_info['pet_diet']}\n"
        
        # Thêm thông tin thú cưng vào formatted_messages
        formatted_messages.append({"role": "system", "content": pet_info_summary})
    
    # Thêm lịch sử chat với role tương ứng (user hoặc assistant)
    for message in messages[:-1]:  # Bỏ qua tin nhắn cuối cùng vì sẽ được xử lý riêng
        formatted_messages.append({"role": message["role"], "content": message["content"]})
    
    # Thêm tin nhắn hiện tại của người dùng vào role user
    if messages and messages[-1]["role"] == "user":
        formatted_messages.append({"role": "user", "content": messages[-1]["content"]})
    
    try:
        if model_choice == "groq":
            # Sử dụng mô hình Llama-3.3-70b của Groq (nhanh hơn nhưng tốn phí)
            response = Groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=formatted_messages,
                temperature=0.1,
                max_tokens= 2048,
                top_p=1
            )
            return response.choices[0].message.content
            
        elif model_choice == "openrouter":
            # Sử dụng mô hình DeepSeek Chat của OpenRouter.ai
            response = openrouter_client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=formatted_messages,
                temperature=0.1,
                max_tokens=2048,
                top_p=1.0,
            )
            return response.choices[0].message.content
                
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
    
    # Kiểm tra xem có thông tin thú cưng trong cookie không
    pet_info_cookie = request.cookies.get('pet_info')
    if pet_info_cookie and 'pet_info' not in session:
        try:
            # Giải mã JSON từ cookie và lưu vào session
            pet_info = json.loads(pet_info_cookie)
            session['pet_info'] = pet_info
        except:
            # Nếu có lỗi giải mã, bỏ qua
            pass
    
    return render_template('index.html', model_choice=session['model_choice'])

@app.route('/pet-info', methods=['GET', 'POST'])
def pet_info():
    """Hiển thị trang thông tin thú cưng"""
    # Nếu đã có thông tin thú cưng trong session, hiển thị lại
    pet_info = session.get('pet_info', None)
    return render_template('pet_info.html', pet_info=pet_info)

@app.route('/save-pet-info', methods=['POST'])
def save_pet_info():
    """Lưu thông tin thú cưng vào session và cookie"""
    # Lấy dữ liệu từ form
    pet_info = {
        'owner_name': request.form.get('owner_name', ''),
        'pet_name': request.form.get('pet_name', ''),
        'pet_type': request.form.get('pet_type', ''),
        'pet_breed': request.form.get('pet_breed', ''),
        'pet_age': request.form.get('pet_age', ''),
        'pet_weight': request.form.get('pet_weight', ''),
        'pet_gender': request.form.get('pet_gender', ''),
        'pet_health_history': request.form.get('pet_health_history', ''),
        'pet_diet': request.form.get('pet_diet', '')
    }
    
    # Lưu vào session
    session['pet_info'] = pet_info
    
    # Tạo response và thiết lập cookie
    response = make_response(redirect(url_for('index')))
    
    # Thiết lập thời gian hết hạn cho cookie (1 năm)
    expire_date = datetime.datetime.now() + datetime.timedelta(days=365)
    
    # Lưu thông tin thú cưng vào cookie
    response.set_cookie('pet_info', json.dumps(pet_info), expires=expire_date)
    
    return response

@app.route('/clear-pet-info', methods=['POST'])
def clear_pet_info():
    """Xóa thông tin thú cưng khỏi session và cookie"""
    # Xóa khỏi session
    if 'pet_info' in session:
        session.pop('pet_info')
    
    # Tạo response và xóa cookie
    response = make_response(redirect(url_for('index')))
    response.set_cookie('pet_info', '', expires=0)  # Xóa cookie bằng cách đặt thời gian hết hạn là 0
    
    return response

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
    
    # Lấy thông tin thú cưng từ session nếu có
    pet_info = session.get('pet_info')
    
    # Giới hạn lịch sử tin nhắn để tiết kiệm token (chỉ lấy 5 tin nhắn gần nhất)
    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    
    # Thêm tin nhắn hiện tại của người dùng vào lịch sử
    messages = recent_history + [{"role": "user", "content": query}]
    
    # Lấy phản hồi từ mô hình đã chọn, truyền thêm thông tin thú cưng
    response = get_chat_response(messages, context, model_choice, pet_info)
    
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
