# from flask import Flask, render_template, request, make_response, jsonify
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import MessagesPlaceholder
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from supabase import create_client, Client
# from dotenv import load_dotenv
# from uuid import uuid4
# import os
# import time

# # ---------------------------
# # Load environment variables
# # ---------------------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# if not PINECONE_API_KEY or not GEMINI_API_KEY:
#     raise ValueError("❌ Missing required environment variables: PINECONE_API_KEY or GEMINI_API_KEY")
# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise ValueError("❌ Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# # ---------------------------
# # Supabase client
# # ---------------------------
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # ---------------------------
# # Flask app (only used if you run this file directly)
# # ---------------------------
# app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# # ---------------------------
# # Embeddings + Pinecone Vector Store
# # ---------------------------
# print("📥 Loading embeddings...")
# t0 = time.time()
# embeddings = download_embeddings()
# print(f"✅ Embeddings loaded in {time.time() - t0:.2f}s")

# index_name = "medichat"
# print("🗂️ Initializing Pinecone vector store...")
# try:
#     vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
#     print(f"✅ Vector store ready in {time.time() - t0:.2f}s")
# except Exception as e:
#     raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# # Retriever (looser search is nicer for Q/A)
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 5, "fetch_k": 20},
# )

# # ---------------------------
# # Gemini LLM
# # ---------------------------
# print("🤖 Initializing Gemini LLM...")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_output_tokens=1000,  # Increased output tokens
#     google_api_key=GEMINI_API_KEY,
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Improved Prompt Template
# # ---------------------------
# # ... (previous imports remain the same)

# # ---------------------------
# # Improved Prompt Template with stronger context instructions
# # ---------------------------
# system_prompt = """
# You are MediChat, an intelligent and reliable medical assistant chatbot. 
# Your primary role is to answer medical questions based on the retrieved context provided.

# IMPORTANT INSTRUCTIONS:
# 1. ALWAYS prioritize and use the retrieved context to answer questions
# 2. If the context contains relevant information, base your answer SOLELY on it
# 3. If the context doesn't contain enough information, you can supplement with your general medical knowledge
#    but clearly indicate what's from the context vs. general knowledge
# 4. NEVER say "the provided text does not contain information about..." - instead, use your knowledge if needed
# 5. Structure your answers clearly when appropriate
# 6. Keep the tone supportive, professional, and empathetic

# Retrieved Context:
# {context}

# Current conversation:
# {history}

# User Question: {input}
# """

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "{input}"),
# ])

# # ... (rest of the app.py remains the same until the get_answer function)

# def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
#     """
#     Generate an answer with RAG + past chat memory.
#     Also persists both user and assistant chat_history to Supabase.
#     """
#     try:
#         conv_id = conversation_id or ensure_conversation(user_id)
        
#         # First, retrieve relevant documents
#         retrieved_docs = retriever.invoke(user_input)
        
#         # Debug: print what was retrieved
#         print(f"Retrieved {len(retrieved_docs)} documents for query: {user_input}")
#         for i, doc in enumerate(retrieved_docs):
#             print(f"Document {i+1}: {doc.page_content[:200]}...")  # First 200 chars
        
#         # If no relevant documents found, use a different approach
#         if not retrieved_docs or all(len(doc.page_content.strip()) < 10 for doc in retrieved_docs):
#             print("No relevant documents found, using LLM without context")
#             # Use LLM directly without context
#             direct_response = llm.invoke(f"Answer this medical question: {user_input}")
#             answer = str(direct_response.content).strip()
#         else:
#             # Run RAG with memory (memory reads from DB via _get_session_history)
#             response = chat_with_memory.invoke(
#                 {"input": user_input},
#                 config={"configurable": {"session_id": conv_id}},  # key for RunnableWithMessageHistory
#             )
#             answer = str(response["answer"]).strip()

#         # Persist turns
#         save_message(conv_id, user_id, "user", user_input)
#         save_message(conv_id, user_id, "assistant", answer)

#         return answer
#     except Exception as e:
#         print(f"❌ Error during get_answer: {e}")
#         return "⚠️ I'm having trouble answering right now. Please try again."

# # ... (rest of the file remains the same)

# # RAG chain
# print("🔗 Building RAG chain...")
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Supabase persistence helpers (CONVERSATIONS + CHAT_HISTORY)
# # ---------------------------

# # ... (previous imports remain the same)

# # ---------------------------
# # Supabase persistence helpers (CONVERSATIONS + CHAT_HISTORY)
# # ---------------------------

# def ensure_conversation(user_id: str) -> str:
#     """
#     Ensure there is at least one conversation for this user.
#     Returns the most recent conversation_id (creates one if not present).
#     """
#     # Check if this is a guest user
#     if user_id.startswith("guest_"):
#         # For guest users, don't use the database, just return a UUID
#         return str(uuid4())
    
#     try:
#         res = (
#             supabase.table("conversations")
#             .select("id")
#             .eq("user_id", user_id)
#             .order("created_at", desc=True)
#             .limit(1)
#             .execute()
#         )
#         if res.data:
#             return res.data[0]["id"]
#         # Create a new conversation
#         created = (
#             supabase.table("conversations")
#             .insert({"user_id": user_id, "title": "New Conversation"})
#             .execute()
#         )
#         return created.data[0]["id"]
#     except Exception as e:
#         print(f"⚠️ ensure_conversation error: {e}")
#         # As a fallback, create a local UUID
#         return str(uuid4())


# def save_message(conversation_id: str, user_id: str, role: str, message: str) -> None:
#     """
#     Save a single message to Supabase (chat_history).
#     Skip saving for guest users.
#     """
#     # Don't save messages for guest users
#     if user_id.startswith("guest_"):
#         return
        
#     try:
#         supabase.table("chat_history").insert({
#             "conversation_id": conversation_id,
#             "user_id": user_id,
#             "role": role,
#             "message": message,
#         }).execute()
#     except Exception as e:
#         print(f"⚠️ Failed to save message: {e}")


# def load_history_as_langchain(conversation_id: str) -> ChatMessageHistory:
#     """
#     Load conversation chat_history and return a LangChain ChatMessageHistory
#     suitable for RunnableWithMessageHistory.
#     For guest users, return empty history.
#     """
#     history = ChatMessageHistory()
    
#     # Don't load history for guest users (conversation_id is not in database)
#     if not conversation_id or "guest_" in conversation_id:
#         return history
        
#     try:
#         res = (
#             supabase.table("chat_history")
#             .select("role, message, created_at")
#             .eq("conversation_id", conversation_id)
#             .order("created_at")
#             .execute()
#         )
#         for row in (res.data or []):
#             if row["role"] == "user":
#                 history.add_user_message(row["message"])
#             elif row["role"] == "assistant":
#                 history.add_ai_message(row["message"])
#             # ignore other roles for rendering (e.g., "system")
#     except Exception as e:
#         print(f"⚠️ Failed to load history: {e}")
#     return history


# def load_history_for_ui(conversation_id: str):
#     """
#     Load chat_history for UI (list of (role, message) tuples).
#     For guest users, return empty list.
#     """
#     # Don't load history for guest users
#     if not conversation_id or "guest_" in conversation_id:
#         return []
        
#     msgs = []
#     try:
#         res = (
#             supabase.table("chat_history")
#             .select("role, message, created_at")
#             .eq("conversation_id", conversation_id)
#             .order("created_at")
#             .execute()
#         )
#         msgs = [(row["role"], row["message"]) for row in (res.data or [])]
#     except Exception as e:
#         print(f"⚠️ Failed to load history_for_ui: {e}")
#     return msgs


# def get_user_conversations(user_id: str):
#     """
#     Get all conversations for a user.
#     Skip for guest users.
#     """
#     # Don't get conversations for guest users
#     if user_id.startswith("guest_"):
#         return []
        
#     try:
#         res = (
#             supabase.table("conversations")
#             .select("id, title, created_at")
#             .eq("user_id", user_id)
#             .order("created_at", desc=True)
#             .execute()
#         )
#         return res.data if res.data else []
#     except Exception as e:
#         print(f"⚠️ Failed to get user conversations: {e}")
#         return []


# # ---------------------------
# # Message-history wrapper for RAG
# # ---------------------------
# _session_histories = {}

# def _get_session_history(session_key: str) -> ChatMessageHistory:
#     # session_key will be conversation_id (so history is per-conversation)
#     if session_key not in _session_histories:
#         _session_histories[session_key] = load_history_as_langchain(session_key)
#     return _session_histories[session_key]

# # ... (rest of the file remains the same)

# # ---------------------------
# # Public function used by Streamlit
# # ---------------------------
# def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
#     """
#     Generate an answer with RAG + past chat memory.
#     Also persists both user and assistant chat_history to Supabase.
#     """
#     try:
#         conv_id = conversation_id or ensure_conversation(user_id)
        
#         # Debug: print the query
#         print(f"Processing query: {user_input}")
        
#         # First, test the retriever to see if it's working
#         try:
#             retrieved_docs = retriever.invoke(user_input)
#             print(f"Retrieved {len(retrieved_docs)} documents")
            
#             # If no relevant documents found, use a different approach
#             if not retrieved_docs or all(len(doc.page_content.strip()) < 10 for doc in retrieved_docs):
#                 print("No relevant documents found, using LLM without context")
#                 # Use LLM directly without context
#                 direct_response = llm.invoke(f"Answer this medical question: {user_input}")
#                 answer = str(direct_response.content).strip()
#             else:
#                 # Display some content from the retrieved documents for debugging
#                 for i, doc in enumerate(retrieved_docs[:2]):  # Show first 2 docs
#                     print(f"Document {i+1}: {doc.page_content[:100]}...")
                
#                 # Run RAG with memory
#                 response = chat_with_memory.invoke(
#                     {"input": user_input},
#                     config={"configurable": {"session_id": conv_id}},
#                 )
#                 answer = str(response["answer"]).strip()
                
#         except Exception as e:
#             print(f"❌ Error in retriever or RAG chain: {e}")
#             # Fallback to direct LLM response
#             direct_response = llm.invoke(f"Answer this medical question: {user_input}")
#             answer = str(direct_response.content).strip()

#         # Persist turns (skipped for guest users as per previous fix)
#         save_message(conv_id, user_id, "user", user_input)
#         save_message(conv_id, user_id, "assistant", answer)

#         return answer
#     except Exception as e:
#         print(f"❌ Error during get_answer: {e}")
#         # Try a simple direct response as a last resort
#         try:
#             direct_response = llm.invoke(f"Answer this medical question: {user_input}")
#             return str(direct_response.content).strip()
#         except:
#             return "⚠️ I'm having trouble answering right now. Please try again."

# # ---------------------------
# # Minimal Flask routes (optional; only when running app.py directly)
# # ---------------------------
# def _get_user_id_from_cookie():
#     sid = request.cookies.get("session_id")
#     return sid or f"guest_{uuid4()}"

# @app.route("/")
# def home():
#     resp = make_response(render_template("chat.html"))
#     if not request.cookies.get("session_id"):
#         resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
#     return resp

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form.get("msg", "").strip()
#     user_id = _get_user_id_from_cookie()
#     if not msg:
#         return "⚠️ Please enter a valid question."
#     start_time_local = time.time()
#     answer = get_answer(msg, user_id)
#     print(f"✅ Answer generated in {time.time() - start_time_local:.2f}s")
#     return answer

# # Auth endpoints (optional)
# @app.route("/signup", methods=["POST"])
# def signup():
#     email = request.form.get("email") or (request.json or {}).get("email")
#     password = request.form.get("password") or (request.json or {}).get("password")
#     if not email or not password:
#         return jsonify({"error": "email and password are required"}), 400
#     try:
#         res = supabase.auth.sign_up({"email": email, "password": password})
#         return jsonify({"message": "Signup successful. Check your email for confirmation.", "user": getattr(res, "user", None)}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# @app.route("/login", methods=["POST"])
# def login():
#     email = request.form.get("email") or (request.json or {}).get("email")
#     password = request.form.get("password") or (request.json or {}).get("password")
#     if not email or not password:
#         return jsonify({"error": "email and password are required"}), 400
#     try:
#         res = supabase.auth.sign_in_with_password({"email": email, "password": password})
#         if not getattr(res, "session", None):
#             return jsonify({"error": "Invalid credentials"}), 401
#         user = getattr(res, "user", None)
#         if not user:
#             return jsonify({"error": "Login failed"}), 401
#         resp = make_response(jsonify({"message": "Login successful"}))
#         resp.set_cookie("session_id", user.id, max_age=60*60*24*30, httponly=True, samesite="Lax")
#         return resp
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# @app.route("/logout", methods=["POST"])
# def logout():
#     try:
#         supabase.auth.sign_out()
#     except Exception:
#         pass
#     resp = make_response(jsonify({"message": "Logged out"}))
#     resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
#     return resp

# @app.route("/health")
# def health():
#     return {"status": "healthy", "service": "MediChat API"}

# if __name__ == "__main__":
#     print("🚀 Starting MediChat with Gemini (RAG + Supabase Auth & History)…")
#     app.run(host="0.0.0.0", port=8000, debug=True)


























from flask import Flask, render_template, request, make_response, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from supabase import create_client, Client
from dotenv import load_dotenv
from uuid import uuid4
import os
import time

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not PINECONE_API_KEY or not COHERE_API_KEY:
    raise ValueError("❌ Missing required environment variables: PINECONE_API_KEY or COHERE_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# ---------------------------
# Supabase client
# ---------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# ---------------------------
# Embeddings + Pinecone Vector Store
# ---------------------------
print("📥 Loading embeddings...")
t0 = time.time()
# You can use your helper if it wraps CohereEmbeddings; else directly:
# embeddings = download_embeddings()
embeddings = CohereEmbeddings(model="embed-english-v3.0")
print(f"✅ Embeddings loaded in {time.time() - t0:.2f}s")

index_name = "medichat"
print("🗂️ Initializing Pinecone vector store...")
try:
    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    print(f"✅ Vector store ready in {time.time() - t0:.2f}s")
except Exception as e:
    raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

# ---------------------------
# Cohere LLM
# ---------------------------
print("🤖 Initializing Cohere Command-A model...")
llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0.3,
    cohere_api_key=COHERE_API_KEY,
)
print("✅ Cohere LLM ready")

# ---------------------------
# Prompt Template
# ---------------------------
system_prompt = """
You are MediChat, an intelligent and reliable medical assistant chatbot. 
Your primary role is to answer medical questions based on the retrieved context provided.

IMPORTANT INSTRUCTIONS:
1. ALWAYS prioritize and use the retrieved context to answer questions.
2. If the context contains relevant information, base your answer SOLELY on it.
3. If the context doesn't contain enough information, you can supplement with your general medical knowledge
   but clearly indicate what's from the context vs. general knowledge.
4. NEVER say "the provided text does not contain information about..." — instead, use your knowledge if needed.
5. Structure your answers clearly when appropriate.
6. Keep the tone supportive, professional, and empathetic.

Retrieved Context:
{context}

Current conversation:
{history}

User Question: {input}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# ---------------------------
# Build RAG Chain
# ---------------------------
print("🔗 Building RAG chain...")
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)
print("✅ RAG chain ready")

# ---------------------------
# Supabase persistence helpers
# ---------------------------
def ensure_conversation(user_id: str) -> str:
    if user_id.startswith("guest_"):
        return str(uuid4())
    try:
        res = (
            supabase.table("conversations")
            .select("id")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["id"]
        created = (
            supabase.table("conversations")
            .insert({"user_id": user_id, "title": "New Conversation"})
            .execute()
        )
        return created.data[0]["id"]
    except Exception as e:
        print(f"⚠️ ensure_conversation error: {e}")
        return str(uuid4())

def save_message(conversation_id: str, user_id: str, role: str, message: str) -> None:
    if user_id.startswith("guest_"):
        return
    try:
        supabase.table("chat_history").insert({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "message": message,
        }).execute()
    except Exception as e:
        print(f"⚠️ Failed to save message: {e}")

def load_history_as_langchain(conversation_id: str) -> ChatMessageHistory:
    history = ChatMessageHistory()
    if not conversation_id or "guest_" in conversation_id:
        return history
    try:
        res = (
            supabase.table("chat_history")
            .select("role, message, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        for row in (res.data or []):
            if row["role"] == "user":
                history.add_user_message(row["message"])
            elif row["role"] == "assistant":
                history.add_ai_message(row["message"])
    except Exception as e:
        print(f"⚠️ Failed to load history: {e}")
    return history

def get_user_conversations(user_id: str):
    if user_id.startswith("guest_"):
        return []
    try:
        res = (
            supabase.table("conversations")
            .select("id, title, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return res.data if res.data else []
    except Exception as e:
        print(f"⚠️ Failed to get user conversations: {e}")
        return []

# ---------------------------
# Message-history wrapper
# ---------------------------
_session_histories = {}
def _get_session_history(session_key: str) -> ChatMessageHistory:
    if session_key not in _session_histories:
        _session_histories[session_key] = load_history_as_langchain(session_key)
    return _session_histories[session_key]

chat_with_memory = RunnableWithMessageHistory(
    rag_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------------------------
# Public function for Streamlit
# ---------------------------
def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
    try:
        conv_id = conversation_id or ensure_conversation(user_id)
        print(f"Processing query: {user_input}")
        retrieved_docs = retriever.invoke(user_input)
        print(f"Retrieved {len(retrieved_docs)} documents")

        if not retrieved_docs or all(len(doc.page_content.strip()) < 10 for doc in retrieved_docs):
            print("No relevant documents found, using LLM without context")
            direct_response = llm.invoke(f"Answer this medical question: {user_input}")
            answer = str(direct_response.content).strip()
        else:
            for i, doc in enumerate(retrieved_docs[:2]):
                print(f"Document {i+1}: {doc.page_content[:100]}...")
            response = chat_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": conv_id}},
            )
            answer = str(response["answer"]).strip()

        save_message(conv_id, user_id, "user", user_input)
        save_message(conv_id, user_id, "assistant", answer)
        return answer

    except Exception as e:
        print(f"❌ Error during get_answer: {e}")
        try:
            direct_response = llm.invoke(f"Answer this medical question: {user_input}")
            return str(direct_response.content).strip()
        except:
            return "⚠️ I'm having trouble answering right now. Please try again."

# ---------------------------
# Flask routes
# ---------------------------
def _get_user_id_from_cookie():
    sid = request.cookies.get("session_id")
    return sid or f"guest_{uuid4()}"

@app.route("/")
def home():
    resp = make_response(render_template("chat.html"))
    if not request.cookies.get("session_id"):
        resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
    return resp

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    user_id = _get_user_id_from_cookie()
    if not msg:
        return "⚠️ Please enter a valid question."
    start_time_local = time.time()
    answer = get_answer(msg, user_id)
    print(f"✅ Answer generated in {time.time() - start_time_local:.2f}s")
    return answer

@app.route("/signup", methods=["POST"])
def signup():
    email = request.form.get("email") or (request.json or {}).get("email")
    password = request.form.get("password") or (request.json or {}).get("password")
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return jsonify({"message": "Signup successful. Check your email for confirmation."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email") or (request.json or {}).get("email")
    password = request.form.get("password") or (request.json or {}).get("password")
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if not getattr(res, "session", None):
            return jsonify({"error": "Invalid credentials"}), 401
        user = getattr(res, "user", None)
        if not user:
            return jsonify({"error": "Login failed"}), 401
        resp = make_response(jsonify({"message": "Login successful"}))
        resp.set_cookie("session_id", user.id, max_age=60*60*24*30, httponly=True, samesite="Lax")
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/logout", methods=["POST"])
def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    resp = make_response(jsonify({"message": "Logged out"}))
    resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
    return resp

@app.route("/health")
def health():
    return {"status": "healthy", "service": "MediChat API (Cohere)"}

if __name__ == "__main__":
    print("🚀 Starting MediChat with Cohere Command-A (RAG + Supabase Auth & History)…")
    app.run(host="0.0.0.0", port=8000, debug=True)
