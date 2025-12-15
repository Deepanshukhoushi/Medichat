import os
# Disable LangChain tracing for performance
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_HANDLER'] = 'false'

# Disable all LangChain telemetry
os.environ['LANGCHAIN_TELEMETRY'] = 'false'

import langchain
langchain.debug = False
langchain.verbose = False

# Try to disable RootListenersTracer specifically
try:
    from langchain_core.tracers import RootListenersTracer
    # Override the problematic callback method
    original_on_chain_end = RootListenersTracer.on_chain_end

    def patched_on_chain_end(self, outputs, **kwargs):
        try:
            return original_on_chain_end(self, outputs, **kwargs)
        except KeyError:
            # Silently ignore KeyError for 'output'
            return

    RootListenersTracer.on_chain_end = patched_on_chain_end
except ImportError:
    pass
except Exception:
    pass

from flask import Flask, render_template, request, make_response, jsonify
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from supabase import create_client, Client
from dotenv import load_dotenv
from uuid import uuid4
import time

# Load environment variables
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

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# Embeddings + Pinecone Vector Store
embeddings = CohereEmbeddings(model="embed-english-v3.0")
index_name = "medichat"
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Cohere LLM
llm = ChatCohere(model="command-a-03-2025", temperature=0.3, cohere_api_key=COHERE_API_KEY)

# Prompt Template
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

# Build RAG Chain
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Supabase persistence helpers
def ensure_conversation(user_id: str) -> str:
    if user_id.startswith("guest_"):
        return str(uuid4())
    try:
        res = supabase.table("conversations").select("id").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data:
            return res.data[0]["id"]
        created = supabase.table("conversations").insert({"user_id": user_id, "title": "New Conversation"}).execute()
        return created.data[0]["id"]
    except Exception as e:
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
        pass

def load_history_as_langchain(conversation_id: str) -> ChatMessageHistory:
    history = ChatMessageHistory()
    if not conversation_id or "guest_" in conversation_id:
        return history
    try:
        res = supabase.table("chat_history").select("role, message, created_at").eq("conversation_id", conversation_id).order("created_at").execute()
        for row in (res.data or []):
            if row["role"] == "user":
                history.add_user_message(row["message"])
            elif row["role"] == "assistant":
                history.add_ai_message(row["message"])
    except Exception as e:
        pass
    return history

# Message-history wrapper
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

# Answer Generation
def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
    """Optimized answer generation that prioritizes vector database retrieval."""
    try:
        conv_id = conversation_id or ensure_conversation(user_id)

        # Check vector database first
        retrieved_docs = retriever.invoke(user_input)
        has_relevant_docs = retrieved_docs and any(len(doc.page_content.strip()) > 20 for doc in retrieved_docs)

        if has_relevant_docs:
            # Use RAG with indexed data
            try:
                response = chat_with_memory.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": conv_id}},
                )
                answer = str(response["answer"]).strip()
                if not answer.endswith("📚 Based on indexed medical data"):
                    answer += "\n\n📚 *Based on indexed medical data*"
            except Exception as e:
                # Fallback with context
                context_text = "\n".join([doc.page_content[:500] for doc in retrieved_docs[:3]])
                enhanced_prompt = f"""Answer this medical question using the provided context:

Context: {context_text}

Question: {user_input}

If the context doesn't contain enough information, use your general medical knowledge but clearly indicate this."""
                direct_response = llm.invoke(enhanced_prompt)
                answer = str(direct_response.content).strip() + "\n\n📚 *Based on indexed medical data*"
        else:
            # Use general medical knowledge
            direct_prompt = f"""Answer this medical question using your general medical knowledge:

Question: {user_input}

Note: This answer is based on general medical knowledge as the specific information was not found in our indexed medical database."""
            direct_response = llm.invoke(direct_prompt)
            answer = str(direct_response.content).strip() + "\n\n🧠 *Based on general medical knowledge (not from indexed data)*"

        # Save conversation
        try:
            save_message(conv_id, user_id, "user", user_input)
            save_message(conv_id, user_id, "assistant", answer)
        except Exception as e:
            pass

        return answer

    except Exception as e:
        return "⚠️ I'm having trouble answering right now. Please try again."

# Flask routes
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
    answer = get_answer(msg, user_id)
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
    return {"status": "healthy", "service": "MediChat API"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
