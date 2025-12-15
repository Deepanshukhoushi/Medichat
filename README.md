# MediChat - Medical Assistant

A deployment-ready medical chatbot that uses RAG (Retrieval-Augmented Generation) to answer medical questions based on indexed medical documents.

## Features

- **Vector Database First**: Always checks Pinecone vector database for relevant medical information
- **Clear Data Sources**: Explicitly indicates whether answers come from indexed medical data or general knowledge
- **Fast Performance**: Optimized for quick response times
- **Dual Interface**: Available as both Flask web app and Streamlit app
- **Guest Mode**: No authentication required for immediate use

## Data Source

Currently indexed with "Anatomy and Physiology 2e" textbook for comprehensive medical knowledge.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Required environment variables:
- `PINECONE_API_KEY`: Your Pinecone API key
- `COHERE_API_KEY`: Your Cohere API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anon key

### 3. Index Your Data
```bash
python store_index.py
```

### 4. Run the Application

**Quick Deployment (Recommended):**
```bash
python deploy.py
```

**Manual Setup:**

**Flask Web App:**
```bash
python app.py
```
Access at: http://localhost:8000

**Streamlit App:**
```bash
streamlit run app_streamlit.py
```
Access at: http://localhost:8501

## How It Works

1. **Question Processing**: Every medical question is first searched against the vector database
2. **Data Source Priority**:
   - If relevant medical data is found: Uses RAG with indexed content + "📚 Based on indexed medical data"
   - If no relevant data found: Uses general medical knowledge + "🧠 Based on general medical knowledge"
3. **Performance**: Optimized retrieval with MMR search and conversation memory

## Deployment

### Railway/Render/Vercel
1. Set environment variables in your deployment platform
2. Use `gunicorn` for Flask: `gunicorn app:app`
3. For Streamlit: Use their built-in deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
```

## API Usage

The core function `get_answer(user_input, user_id, conversation_id)` returns answers with clear source indicators.

## Project Structure

```
├── app.py                 # Main Flask application
├── app_streamlit.py       # Streamlit interface
├── store_index.py         # Data indexing script
├── src/                   # Helper modules
├── Data/                  # Source documents
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## Performance Notes

- Vector database queries are optimized for speed
- Conversation history is maintained per session
- Responses include timing information in logs
- Error handling ensures graceful fallbacks
