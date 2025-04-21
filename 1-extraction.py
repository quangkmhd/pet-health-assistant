import requests
from bs4 import BeautifulSoup
import time
import os
import json
from tqdm import tqdm

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------



# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------



# --------------------------------------------------------------
# Chỉ lấy các liên kết có class="plain" từ trang web petmart.vn
# --------------------------------------------------------------

def get_plain_links(base_url, num_pages=50, existing_links=None):
    """
    # Hàm lấy tất cả các liên kết có class="plain" từ trang petmart.vn
    # Tham số:
    #   base_url: URL cơ sở của trang web
    #   num_pages: Số trang cần cào dữ liệu
    #   existing_links: Danh sách các liên kết đã tồn tại (nếu có)
    # Trả về:
    #   Danh sách các URL có class="plain" mới (chưa có trong existing_links)
    """
    # Khởi tạo danh sách liên kết đã tồn tại nếu không được cung cấp
    if existing_links is None:
        existing_links = []
    
    all_plain_links = []
    new_links_count = 0
    
    # Tạo thanh tiến độ cho việc thu thập liên kết
    for page in tqdm(range(1, num_pages + 1), desc="Thu thập liên kết từ các trang"):
        # Tạo URL cho từng trang
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}/page/{page}"
            
        try:
            # Gửi yêu cầu HTTP đến trang web
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Không thể truy cập trang {url}. Mã trạng thái: {response.status_code}")
                break
                
            # Phân tích cú pháp HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm tất cả thẻ a có class="plain"
            plain_links = soup.find_all('a', class_='plain')
            
            # Lấy thuộc tính href từ các thẻ a tìm được
            page_new_links = 0
            for link in plain_links:
                if 'href' in link.attrs:
                    url = link['href']
                    # Kiểm tra nếu liên kết chưa tồn tại thì thêm vào
                    if url not in existing_links and url not in all_plain_links:
                        all_plain_links.append(url)
                        new_links_count += 1
                        page_new_links += 1
            
            print(f"Trang {page}: Tìm thấy {len(plain_links)} liên kết, {page_new_links} liên kết mới")
            
            # Tạm dừng để tránh gửi quá nhiều yêu cầu trong thời gian ngắn
            time.sleep(1)
            
        except Exception as e:
            print(f"Lỗi khi xử lý trang {url}: {str(e)}")
            break
    
    print(f"Tổng số liên kết mới: {new_links_count}")
    return all_plain_links

def load_existing_data(json_file_path):
    """
    # Đọc dữ liệu từ file JSON
    # Tham số:
    #   json_file_path: Đường dẫn đến file JSON
    # Trả về:
    #   Dữ liệu từ file JSON hoặc một dict rỗng nếu file không tồn tại
    """
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            return data
        except Exception as e:
            print(f"Lỗi khi đọc file {json_file_path}: {str(e)}")
    return {}

def main(base_url="https://www.petmart.vn/cho-canh", num_pages=50):
    """
    # Hàm chính để lấy và lưu trữ các liên kết
    # Tham số:
    #   base_url: URL cơ sở để bắt đầu thu thập
    #   num_pages: Số trang tối đa cần duyệt qua
    """
    print("=" * 50)
    print("CÔNG CỤ CÀO LIÊN KẾT")
    print("=" * 50)
    print("Chú ý: Tool này chỉ thu thập và lưu trữ các liên kết.")
    print("Để cào dữ liệu từ các liên kết, hãy sử dụng extraction_helper.py sau khi chạy xong.")
    print("=" * 50)
    
    # Đường dẫn đến file JSON lưu trữ liên kết
    links_file_path = "plain_links.json"
    
    # Đọc danh sách liên kết đã tồn tại
    links_data = load_existing_data(links_file_path)
    existing_links = links_data.get("links", [])
    print(f"Đọc {len(existing_links)} liên kết đã tồn tại từ file {links_file_path}")
    
    # Lấy danh sách các liên kết mới
    plain_urls = get_plain_links(base_url, num_pages, existing_links)
    print(f"Tổng số liên kết mới thu thập được: {len(plain_urls)}")
    
    # Kết hợp liên kết mới với liên kết đã tồn tại
    all_links = existing_links + plain_urls
    all_links = list(set(all_links))  # Loại bỏ trùng lặp
    
    # Lưu tất cả liên kết vào file JSON
    if all_links:
        links_data = {
            "source": base_url,
            "total_links": len(all_links),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "links": all_links
        }
        
        with open(links_file_path, "w", encoding="utf-8") as json_file:
            json.dump(links_data, json_file, ensure_ascii=False, indent=4)
        
        print(f"Đã lưu {len(all_links)} liên kết vào file {links_file_path}")
        print("=" * 50)
        print(f"CÁC BƯỚC TIẾP THEO:")
        print(f"1. Các liên kết đã được lưu vào file {links_file_path}")
        print(f"2. Để cào dữ liệu từ các liên kết này, chạy: python extraction_helper.py")
        print("=" * 50)
    else:
        print("Không tìm thấy liên kết nào có class='plain'")

if __name__ == "__main__":
    # Chạy hàm main với các tham số mặc định
    main()

