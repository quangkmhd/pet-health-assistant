# Pet Health Assistant Project

## Overview
This project creates a veterinary assistant chatbot that helps pet owners with health concerns. It uses web crawling to collect pet health data from petmart.vn (a trusted Vietnamese pet care resource), vector embeddings for semantic search, and large language model integration for intelligent responses.

<div align="center">
  <img src="static\images\web.png" alt="Pet Health Assistant" width="800"/>
</div>

## Features
- Data collection through web crawling from petmart.vn
- Vietnamese-focused pet health knowledge base
- Vector database for semantic search
- AI-powered conversation with veterinary knowledge
- Bilingual support (Vietnamese and English)
- Multiple LLM backend options (Groq and OpenRouter)

## System Architecture

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐    ┌────────────┐
│ Web Crawler │───►│ Content Parser │───►│ Text Embedding │───►│  LanceDB   │
│ (petmart.vn)│    └───────────────┘    └────────────────┘    └────────┬───┘
└─────────────┘                                                        │
                                                                       │
┌──────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────┐    ┌──────────────┐    ┌───────────────┐
│  User Question │───►│ Flask Server │◄───┤ LLM (Groq or  │
│                │    │              │    │  OpenRouter)  │
└────────────────┘    └──────────────┘    └───────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   Response   │
                      └──────────────┘
```

## Project Components

### 1. Data Collection (`1-extraction.py`)
- Web crawler that extracts links with class="plain" from petmart.vn
- Focuses on the pet health and veterinary sections
- Stores links in JSON format for further processing
- Handles pagination and implements rate limiting

### 2. Content Extraction (`2-crawler.py`)
- `FirecrawlApp` class for structured content extraction
- Processes links collected in the previous step
- Saves crawled data with metadata
- Implements logging and error handling

### 3. Vector Database Creation (`3-embedding.py`)
- Text chunking for better search results
- Embedding generation using `multilingual-e5-large` model (optimized for Vietnamese language)
- LanceDB integration for efficient vector storage
- Progress tracking using tqdm

### 4. Web Application (`app.py`)
- Flask web application with responsive UI
- Integration with Groq and OpenRouter APIs
- Context-aware responses using vector similarity search
- Session management for conversation history

## Data Flow

1. **Data Collection Phase**:
   - Web crawler extracts pet health information from petmart.vn
   - Content is parsed and cleaned
   - Text is split into manageable chunks

2. **Embedding Phase**:
   - Text chunks are converted to vector embeddings
   - Embeddings are stored in LanceDB

3. **Query Phase**:
   - User asks a question about pet health
   - Question is converted to vector embedding
   - Similar vectors are retrieved from database
   - Context is sent to LLM with user query
   - LLM generates a veterinary-knowledge based response

## Setup Instructions

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pet-health-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
GROQ_API_KEY= your_groq_api_key
HUGGINGFACE_API_KEY= your_huggingface_api_key
FIRECRAWL_API_KEY= your_firecrawl_api_key
OPENROUTER_API_KEY = your_openrouter_api_key
```

### Running the Data Pipeline

1. Collect links:
```bash
python 1-extraction.py
```

2. Extract content:
```bash
python 2-crawler.py
```

3. Generate embeddings and create vector database:
```bash
python 3-embedding.py
```

### Starting the Web Application

```bash
python app.py
```

The application will be available at http://127.0.0.1:5000/

## Usage
1. Visit the web interface in your browser
2. Type questions about pet health 
3. Optionally switch between LLM backends in the settings

## Example Questions (Vietnamese)
- "Chó của tôi bị nôn và tiêu chảy, tôi nên làm gì?"
- "Làm thế nào để điều trị ghẻ cho mèo?"
- "Những triệu chứng của bệnh viêm da ở thú cưng?"
- "Cách chăm sóc thú cưng sau phẫu thuật?"
- "Thức ăn nào tốt cho chó bị bệnh dạ dày?"

## Acknowledgments
- Special thanks to petmart.vn for being the knowledge source for this project
- This project is for educational purposes only