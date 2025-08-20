from flask import Flask, render_template, request
from src.helper import download_embeddings  # updated helper to support Cohere
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, template_folder="templates")

# ---------------------------
# Load embeddings
# ---------------------------
embeddings = download_embeddings()

index_name = "medichat"

# Load Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ---------------------------
# Cohere LLM
# ---------------------------
llm = ChatCohere(
    model="command",  # "command-light" for cheaper usage
    temperature=0.9,
    max_tokens=500,
    cohere_api_key=COHERE_API_KEY
)

# ---------------------------
# Prompt Template (must include {context} for LangChain v0.2+)
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}\n\nContext:\n{context}")
])

# ---------------------------
# RAG Chain
# ---------------------------
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------
# Frontend route
# ---------------------------
@app.route("/")
def home():
    return render_template("chat.html")

# ---------------------------
# API route
# ---------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import time
from functools import lru_cache

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, template_folder="templates")

# ---------------------------
# Load embeddings
# ---------------------------
embeddings = download_embeddings()

index_name = "medichat"

# ---------------------------
# Cohere LLM - Optimized for speed
# ---------------------------
llm = ChatCohere(
    model="command-light",  # Faster than 'command'
    temperature=0.7,        # Less creative but faster
    max_tokens=300,         # Shorter responses
    cohere_api_key=COHERE_API_KEY
)

# ---------------------------
# Prompt Template
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Question: {input}\nContext: {context}")
])

# ---------------------------
# Cache setup
# ---------------------------
CACHE_SIZE = 200  # Adjust based on memory availability
query_cache = {}

# ---------------------------
# Initialize vector store with pre-warming
# ---------------------------
print("Initializing Pinecone vector store...")
start_time = time.time()

# Initialize vector store
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever with optimized parameters
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 2}  # Reduced from 3 to 2
)

# Pre-warm with a simple query
_ = retriever.get_relevant_documents("health")
print(f"Vector store initialized in {time.time() - start_time:.2f} seconds")

# ---------------------------
# RAG Chain
# ---------------------------
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------
# Optimized processing function with caching
# ---------------------------
@lru_cache(maxsize=CACHE_SIZE)
def process_query(query: str) -> str:
    """Process a query with caching and optimized parameters"""
    try:
        start_time = time.time()
        
        # Run retrieval and generation with additional constraints
        response = rag_chain.invoke({
            "input": query,
            "max_tokens": 250  # Enforce shorter response
        })
        
        answer = str(response["answer"])
        print(f"Processed query in {time.time() - start_time:.2f} seconds")
        return answer
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return "I'm having trouble answering that right now. Please try again with a different question."

# ---------------------------
# Frontend route
# ---------------------------
@app.route("/")
def home():
    return render_template("chat.html")

# ---------------------------
# API route - Optimized with caching
# ---------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip()
    
    # Return cached response if available
    if msg in query_cache:
        print(f"Cache hit for: {msg}")
        return query_cache[msg]
    
    print("Processing new query:", msg)
    
    # Process the query
    response = process_query(msg)
    
    # Update cache
    if len(query_cache) >= CACHE_SIZE:
        # Remove oldest item if cache is full
        oldest_key = next(iter(query_cache))
        del query_cache[oldest_key]
    
    query_cache[msg] = response
    
    return response

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)