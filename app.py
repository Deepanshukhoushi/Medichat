from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os
import time

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("Missing required environment variables: PINECONE_API_KEY or COHERE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

print("Loading embeddings...")
start_time = time.time()
embeddings = download_embeddings()
print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

index_name = "medichat"

print("Initializing Pinecone vector store...")
start_time = time.time()
try:
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print(f"✅ Vector store initialized in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"❌ Error initializing vector store: {str(e)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="mmr",  # Max Marginal Relevance to reduce noise
    search_kwargs={"k": 5, "lambda_mult": 0.7}
)

# ---------------------------
# Cohere LLM
# ---------------------------
print("Initializing Cohere LLM...")
start_time = time.time()

llm = ChatCohere(
    model="command-light",   # faster model
    temperature=0.3,         # less hallucination
    max_tokens=200,          # concise answers
    cohere_api_key=COHERE_API_KEY
)

print(f"✅ LLM initialized in {time.time() - start_time:.2f} seconds")

# ---------------------------
# Prompt Template (with history)
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are MediChat, a helpful medical assistant. 

Instructions:
- Use the conversation history to resolve vague references (e.g., "this", "that").
- Stick to the medical topic unless the user explicitly changes it.
- If uncertain, ask the user to clarify instead of guessing.
- Use retrieved context if it is relevant to the user's current question.
- Keep responses accurate, short, and medically relevant.
"""),

    ("human", """Conversation so far:
{history}

User Question: {input}

Relevant Context from knowledge base:
{context}""")
])

# ---------------------------
# RAG Chain
# ---------------------------
print("Creating RAG chain...")
start_time = time.time()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print(f"✅ RAG chain created in {time.time() - start_time:.2f} seconds")

#---------------------------
# Chat History with Memory
#---------------------------
session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

chat_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)


query_cache = {}
CACHE_SIZE = 100

def get_cached_response(query):
    return query_cache.get(query)

def cache_response(query, response):
    if len(query_cache) >= CACHE_SIZE:
        oldest_key = next(iter(query_cache))
        del query_cache[oldest_key]
    query_cache[query] = response

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip()
    session_id = request.remote_addr  # could also use flask session or JWT

    if not msg:
        return "Please enter a valid question."

    # Cache check
    cached_response = get_cached_response(msg)
    if cached_response:
        print(f"⚡ Cache hit for: {msg}")
        return cached_response

    print(f"💬 New query: {msg}")
    try:
        start_time = time.time()
        response = chat_with_memory.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}},
        )
        answer = str(response["answer"])
        print(f"✅ Answer in {time.time() - start_time:.2f}s")

        cache_response(msg, answer)
        return answer

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        error_message = "I'm having trouble answering right now. Please try again."
        cache_response(msg, error_message)
        return error_message

@app.route("/health")
def health():
    return {"status": "healthy", "service": "MediChat API"}

if __name__ == "__main__":
    print("🚀 Starting MediChat server...")
    app.run(host="0.0.0.0", port=8000, debug=True)

