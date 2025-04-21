import json
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging
import sys
from datetime import datetime

class FirecrawlApp:
    def __init__(self, storage_dir="crawled_data"):
        """Khởi tạo FirecrawlApp với thư mục lưu trữ dữ liệu."""
        self.storage_dir = storage_dir
        self.crawled_links_file = os.path.join(storage_dir, "crawled_links.json")
        self.crawled_links = set()
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        # Thiết lập logging với xử lý encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(storage_dir, "crawl.log"), encoding='utf-8'),
                # Sử dụng StreamHandler với encoding đúng hoặc chỉ ghi log vào file
                # logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FirecrawlApp")
        
        # Thêm console handler với xử lý lỗi encoding đặc biệt
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Tải danh sách các liên kết đã cào
        self._load_crawled_links()
    
    def safe_log(self, level, message):
        """Ghi log an toàn, tránh lỗi encoding."""
        try:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
        except UnicodeEncodeError:
            # Nếu có lỗi encoding, ghi log với ASCII
            ascii_message = message.encode('ascii', 'replace').decode('ascii')
            if level == "info":
                self.logger.info(ascii_message)
            elif level == "error":
                self.logger.error(ascii_message)
            elif level == "warning":
                self.logger.warning(ascii_message)
    
    def _load_crawled_links(self):
        """Tải danh sách các liên kết đã cào từ file."""
        if os.path.exists(self.crawled_links_file):
            try:
                with open(self.crawled_links_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.crawled_links = set(data.get("crawled_links", []))
                self.safe_log("info", f"Da tai {len(self.crawled_links)} lien ket da cao truoc do.")
            except Exception as e:
                self.safe_log("error", f"Loi khi tai danh sach lien ket da cao: {e}")
    
    def _save_crawled_links(self):
        """Lưu danh sách các liên kết đã cào vào file."""
        try:
            with open(self.crawled_links_file, 'w', encoding='utf-8') as f:
                json.dump({"crawled_links": list(self.crawled_links)}, f, ensure_ascii=False, indent=2)
            self.safe_log("info", f"Da luu {len(self.crawled_links)} lien ket da cao.")
        except Exception as e:
            self.safe_log("error", f"Loi khi luu danh sach lien ket da cao: {e}")
    
    def _get_domain_from_url(self, url):
        """Trích xuất tên miền từ URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def _generate_filename(self, url):
        """Tạo tên file từ URL."""
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        if not path:
            path = "index"
        else:
            path = path.replace('/', '_')
        return f"{path}.json"
    
    def crawl_url(self, url, delay=1):
        """Cào dữ liệu từ một URL cụ thể."""
        if url in self.crawled_links:
            self.safe_log("info", f"Da cao truoc do: {url}")
            return False
        
        try:
            self.safe_log("info", f"Bat dau cao: {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                self.safe_log("warning", f"Khong the truy cap URL {url} - Ma trang thai: {response.status_code}")
                return False
            
            # Phân tích HTML với BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Chỉ lấy nội dung từ thẻ p
            paragraphs = soup.find_all('p')
            text_content = [p.get_text(strip=True) for p in paragraphs]
            
            # Lọc bỏ các đoạn văn bản trống
            text_content = [text for text in text_content if text.strip()]
            
            # Lấy tiêu đề nếu có
            title = soup.title.get_text() if soup.title else "Khong co tieu de"
            
            # Chuẩn bị dữ liệu để lưu
            data = {
                "url": url,
                "title": title,
                "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": text_content
            }
            
            # Lưu dữ liệu vào file
            filename = self._generate_filename(url)
            filepath = os.path.join(self.storage_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Đánh dấu URL đã được cào
            self.crawled_links.add(url)
            self._save_crawled_links()
            
            self.safe_log("info", f"Da cao thanh cong: {url}")
            
            # Nghỉ giữa các yêu cầu để tránh tải quá mức cho server
            time.sleep(delay)
            return True
            
        except Exception as e:
            self.safe_log("error", f"Loi khi cao {url}: {e}")
            return False
    
    def crawl_from_json(self, json_file, delay=1):
        """Cào dữ liệu từ danh sách URL trong file JSON."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            links = data.get("links", [])
            total_links = len(links)
            
            self.safe_log("info", f"Bat dau cao {total_links} lien ket tu file {json_file}")
            
            successful_crawls = 0
            skipped_crawls = 0
            
            for i, link in enumerate(links, 1):
                self.safe_log("info", f"Dang xu ly {i}/{total_links}: {link}")
                
                if link in self.crawled_links:
                    self.safe_log("info", f"Da bo qua (da cao truoc do): {link}")
                    skipped_crawls += 1
                    continue
                
                success = self.crawl_url(link, delay)
                if success:
                    successful_crawls += 1
                
                # Lưu tiến độ sau mỗi 10 URL
                if i % 10 == 0:
                    self._save_crawled_links()
            
            self._save_crawled_links()
            
            result = {
                "total_links": total_links,
                "successful_crawls": successful_crawls,
                "skipped_crawls": skipped_crawls,
                "failed_crawls": total_links - successful_crawls - skipped_crawls
            }
            
            self.safe_log("info", f"Da hoan thanh qua trinh cao: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            self.safe_log("error", f"Loi khi cao tu file {json_file}: {e}")
            return {"error": str(e)}

# Sử dụng FirecrawlApp
if __name__ == "__main__":
    # Khởi tạo ứng dụng
    crawler = FirecrawlApp(storage_dir="petmart_data")
    
    # Cào dữ liệu từ file JSON
    results = crawler.crawl_from_json("plain_links.json", delay=2)
    
    print("Ket qua cao du lieu:")
    print(f"Tong so lien ket: {results.get('total_links', 0)}")
    print(f"Cao thanh cong: {results.get('successful_crawls', 0)}")
    print(f"Bo qua (da cao): {results.get('skipped_crawls', 0)}")
    print(f"That bai: {results.get('failed_crawls', 0)}")