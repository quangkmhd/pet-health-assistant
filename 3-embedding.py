from typing import List
import os
import json
import glob
from tqdm import tqdm

import lancedb
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

load_dotenv()

# --------------------------------------------------------------
# Cài đặt mô hình Hugging Face và tokenizer
# --------------------------------------------------------------

# Tải mô hình và tokenizer từ huggingface
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Số tokens tối đa cho mỗi chunk
MAX_TOKENS = 512

# --------------------------------------------------------------
# Hàm embedding sử dụng mô hình multilingual-e5-large
# --------------------------------------------------------------

def get_embeddings(texts, batch_size=8):
    """
    Hàm tạo embedding cho văn bản sử dụng mô hình multilingual-e5-large
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i+batch_size]
        
        # Chuẩn bị input cho mô hình
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                         max_length=512, return_tensors="pt")
        
        # Chuyển đến GPU nếu có
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model.to("cuda")
        
        # Tính embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy embedding của token [CLS] (đầu tiên)
            embeddings = outputs.last_hidden_state[:, 0]
            
        # Chuẩn hóa embedding
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Chuyển về CPU và numpy
        embeddings_np = embeddings.cpu().numpy()
        all_embeddings.extend(embeddings_np.tolist())
    
    return all_embeddings

# --------------------------------------------------------------
# Triển khai chunking đơn giản
# --------------------------------------------------------------

def chunk_text(text, max_tokens=512, overlap=50):
    """
    Chia nhỏ văn bản thành các đoạn với số lượng token tối đa và độ chồng lấp
    """
    # Tokenize văn bản
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Chia văn bản thành các đoạn
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        # Lấy tokens cho chunk
        chunk_tokens = tokens[i:i + max_tokens]
        
        # Chuyển tokens thành text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Bỏ qua chunk quá ngắn
        if len(chunk_tokens) > 50:
            chunks.append(chunk_text)
    
    return chunks

# --------------------------------------------------------------
# Tạo database và định nghĩa schema cho LanceDB
# --------------------------------------------------------------

# Tạo database LanceDB trong thư mục mới
LANCEDB_PATH = "data/lancedb_clean"
os.makedirs(LANCEDB_PATH, exist_ok=True)
db = lancedb.connect(LANCEDB_PATH)

# Kích thước vector của mô hình multilingual-e5-large
VECTOR_SIZE = 1024

# Định nghĩa schema metadata
class ChunkMetadata(LanceModel):
    """
    Các trường phải được sắp xếp theo thứ tự bảng chữ cái.
    Đây là yêu cầu của triển khai Pydantic.
    """
    filename: str | None
    title: str | None

# Định nghĩa schema chính
class Chunks(LanceModel):
    text: str
    vector: Vector(VECTOR_SIZE)
    metadata: ChunkMetadata

# Tạo bảng trong LanceDB
table = db.create_table("petmart_data", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Tìm và xử lý tất cả file JSON trong thư mục petmart_data
# --------------------------------------------------------------

# Lấy danh sách tất cả file JSON trong thư mục petmart_data
json_files = glob.glob("petmart_data/*.json")
print(f"Tìm thấy {len(json_files)} file JSON để xử lý")

# Danh sách lưu tất cả các chunk đã xử lý
all_processed_chunks = []

# Tổng số chunk được tạo
total_chunks = 0

# Xử lý từng file
for json_file in tqdm(json_files, desc="Đang xử lý file"):
    try:
        # Đọc file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy filename và title
        filename = os.path.basename(json_file)
        title = data.get("title", "")
        
        # Kết hợp nội dung văn bản từ dữ liệu JSON
        content = "\n".join(data.get("content", []))
        
        # Bỏ qua file có nội dung quá ngắn
        if len(content) < 50:
            print(f"Bỏ qua file {filename}: nội dung quá ngắn")
            continue
        
        # Chia văn bản thành các đoạn
        chunks = chunk_text(content, max_tokens=MAX_TOKENS)
        
        print(f"File {filename}: {len(chunks)} chunks được tạo")
        total_chunks += len(chunks)
        
        # Chuẩn bị chunks cho bảng
        file_processed_chunks = [
            {
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "title": title,
                },
            }
            for chunk in chunks
        ]
        
        all_processed_chunks.extend(file_processed_chunks)
        
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_file}: {str(e)}")

# --------------------------------------------------------------
# Tạo embedding và thêm vào LanceDB
# --------------------------------------------------------------

print(f"Tổng số chunks cần embedding: {len(all_processed_chunks)}")
print(f"Tổng số chunks được tạo từ {len(json_files)} file: {total_chunks}")

# Lấy văn bản từ các chunk
texts = [chunk["text"] for chunk in all_processed_chunks]

# Tạo embedding
embeddings = get_embeddings(texts)

# Thêm embedding vào chunks
for i, embedding in enumerate(embeddings):
    all_processed_chunks[i]["vector"] = embedding

# Thêm chunks vào bảng LanceDB
print("Đang thêm chunks vào LanceDB...")
table.add(all_processed_chunks)

# --------------------------------------------------------------
# Hiển thị thông tin về bảng
# --------------------------------------------------------------

print("Hoàn thành việc xử lý và lưu trữ dữ liệu")
print(f"Tổng số chunks đã lưu: {table.count_rows()}")

# Hiển thị một số mẫu từ bảng
try:
    df = table.to_pandas()
    print("Năm hàng đầu tiên từ bảng LanceDB:")
    print(df.head())
except Exception as e:
    print(f"Không thể hiển thị dữ liệu từ bảng: {str(e)}")

print(f"Dữ liệu đã được lưu trong thư mục: {LANCEDB_PATH}")