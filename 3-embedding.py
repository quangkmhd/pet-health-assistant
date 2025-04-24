from typing import List
import os
import json
import glob
from tqdm import tqdm

import lancedb
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import numpy as np

load_dotenv()

# Tải mô hình và tokenizer từ huggingface
# Tải mô hình
model_name = "intfloat/multilingual-e5-large"
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Số tokens tối đa cho mỗi chunk
MAX_TOKENS = 512

def get_embeddings(texts, batch_size=8, show_progress=True, save_path=None):
    """
    Hàm tạo embedding cho văn bản sử dụng mô hình multilingual-e5-large với SentenceTransformer
    """
    
    # Điều chỉnh batch_size dựa trên bộ nhớ GPU, dùng google colab chạy để đc sử dụng gpu
    # if torch.cuda.is_available():
    #     total_memory = torch.cuda.get_device_properties(0).total_memory
    #     batch_size = min(batch_size, int(total_memory // (512 * 1024 * 4)))
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress, # Hiển thị thanh tiến trình.
        normalize_embeddings=True, #Chuẩn hóa L2 embedding.
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return embeddings.tolist()


# Triển khai chunking đơn giản
def chunk_text(text, max_tokens=512, overlap=50):
    """
    Chia nhỏ văn bản thành các đoạn sử dụng RecursiveCharacterTextSplitter từ langchain
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens, 
        chunk_overlap=overlap,
        length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False)),
        separators=["\n\n", "\n", ". ", " ", ""],  # Ưu tiên chia theo đoạn, câu
    )
    
    # Chia văn bản thành các đoạn
    chunks = splitter.split_text(text)
    
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

# Định nghĩa schema cho bảng LanceDB
schema = {
    "text": str,
    "vector": Vector(VECTOR_SIZE),
    "metadata": {
        "filename": str | None,
        "title": str | None
    }
}

# Tạo bảng trong LanceDB
table = db.create_table("petmart_data", schema=schema, mode="overwrite")

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
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filename = os.path.basename(json_file)
        title = data.get("title", "")
        
        
        content = "\n".join(data.get("content", []))
        
        
        chunks = chunk_text(content, max_tokens=MAX_TOKENS)
        
        print(f"File {filename}: {len(chunks)} chunks được tạo")
        total_chunks += len(chunks)
        
        # Chuẩn bị chunks cho bảng
        file_chunks = []
        for chunk in chunks:
            chunk_data = {
                "text": chunk,                  
                "vector": None,                   
                "metadata": {                    
                    "filename": filename,        
                    "title": title                
                }
            }
            file_chunks.append(chunk_data)
        
        all_processed_chunks.extend(file_chunks)
        
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_file}: {str(e)}")

# --------------------------------------------------------------
# Tạo embedding và thêm vào LanceDB
# --------------------------------------------------------------

print(f"Tổng số chunks cần embedding: {len(all_processed_chunks)}")
print(f"Tổng số chunks được tạo từ {len(json_files)} file: {total_chunks}")

# Lấy văn bản và tạo embedding
texts = []
for chunk in all_processed_chunks:
    texts.append(chunk["text"])

embeddings = get_embeddings(texts)

# Gán embedding vào chunks
for i in range(len(embeddings)):
    all_processed_chunks[i]["vector"] = embeddings[i]

# Thêm chunks vào bảng LanceDB
print("Đang thêm chunks vào LanceDB...")
table.add(all_processed_chunks)


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