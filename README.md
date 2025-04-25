# 🐾 Pet Health - Pet Healthcare Assistant

<div align="center">
  <img src="static\images\web.png" alt="Pet Health Logo" width="300px">
  <p><i>Your intelligent companion for pet care</i></p>
</div>

## 📋 Introduction

**Pet Health** is an intelligent veterinary assistant built to help pet owners better understand their pets' health. The project uses artificial intelligence, natural language processing, and a specialized veterinary knowledge database to provide useful advice and information.

### ✨ Key Features

- 💬 **Intelligent Conversation**: Natural communication with AI assistant about pet health issues
- 🔍 **Contextual Search**: Using embedding vectors to find relevant information from the database
- 📊 **Pet Information Storage**: Save and use personalized information about your pet
- 🧠 **Advanced Language Models**: Integration with cutting-edge AI models (Llama-3.3, DeepSeek)
- 🇻🇳 **Vietnamese Support**: Specially designed to support Vietnamese language
- 🔄 **Easy Customization**: Can be transformed into any type of chatbot by changing the data source

<div align="center">
  <img src="static\images\web1.png" alt="Pet Health" width="700px">
  <p><i>Pet Health chat interface</i></p>
</div>

## 🔍 System Architecture

Pet Health is built with a modern architecture to ensure high performance and scalability:

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   1. Crawling   │────►│  2. Embedding   │────►│  3. Vector Store │
│   & Data Prep   │     │    Generation   │     │    (LanceDB)     │
└─────────────────┘     └─────────────────┘     └────────┬─────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  6. User Input  │◄────│    5. Flask     │◄────│  4. Context      │
│     & Output    │────►│     Server      │────►│     Retrieval    │
└─────────────────┘     └────────┬────────┘     └──────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  7. LLM Models   │
                        │ (llama/deepseek) │
                        └──────────────────┘
```

## 📊 Data Preparation

The Pet Health project uses a 3-step process to prepare data:

1. **Data Collection**
```bash
python 1-extraction.py
```
This tool collects links.

2. **Web Crawling**
```bash
python 2-crawler.py
```
Collects detailed content from the stored links.

3. **Creating Embeddings**
```bash
python 3-embedding.py
```
Processes text and creates vector embeddings stored in LanceDB.

## 🧩 Project Structure

```
Pet Health/
├── app.py                # Main Flask server
├── system_prompt.py      # System prompt for LLM
├── requirements.txt      # Library requirements
├── 1-extraction.py       # Link collection tool
├── 2-crawler.py          # Data crawling tool from links
├── 3-embedding.py        # Vector embeddings creation tool
├── static/               # Static resources (CSS, JS, images)
├── templates/            # HTML templates
├── petmart_data/         # Raw data from crawling
└── data/                 # Processed data and vector store
```

## 🚀 Usage

### Main Features

1. **Chat with the Assistant**
   - Ask questions about pet health issues
   - Receive advice and guidance from the veterinary knowledge base

2. **Manage Pet Information**
   - Store information about your pet
   - Receive personalized responses based on saved information

3. **Choose AI Model**
   - Switch between different LLM models (llama and deepseek)
   - Optimize performance and response quality

<div align="center">
  <img src="static\images\model.png" alt="Pet Information" width="650px">
  <p><i>Pet information management</i></p>
</div>

## 🔧 Advanced Customization

### Changing the Embedding Model
Pet Health uses the `intfloat/multilingual-e5-large` model as default. To change it:

1. Open the `3-embedding.py` file
2. Change the `model_name` variable to another model compatible with SentenceTransformer
3. Run the embedding process again

### Changing the Number of Search Results
By default, Pet Health returns the 5 most relevant results. To adjust:

1. Open the `app.py` file
2. In the `get_context()` function, change the `num_results=5` parameter
3. Restart the application

### 🔄 Transform into a Different Chatbot
You can easily convert Pet Health into any other type of chatbot by:

1. **Changing the Data Source**
   - Modify the `1-extraction.py` file to collect links from websites specialized in your desired field (e.g., cooking, travel, human health)
   - Adjust the search expressions according to the target website structure

2. **Adjusting Data Crawling**
   - Modify the `2-crawler.py` file to extract information suitable for the new website structure
   - Change selectors or patterns to capture the necessary content

3. **Customizing the Prompt**
   - Adjust the `system_prompt.py` file to guide the LLM to respond according to the new field
   - Change the response format to suit the topic

4. **Changing the Interface**
   - Update HTML templates to reflect the new chatbot theme
   - Adjust images and CSS to create an appropriate interface

With this approach, you can create specialized chatbots in various fields: legal assistance, financial advice, travel, education, etc.

## 📝 Author

**Nguyen Huu Quang**
- GitHub: [quangkmhd](https://github.com/quangkmhd)

---

<div align="center">
  <p>Thank you for reading! Note that this code is just a demo, you can transform it into any chatbot by changing the data.</p>
</div>