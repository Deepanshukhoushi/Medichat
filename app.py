# from flask import Flask, render_template, request
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_cohere import ChatCohere
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from dotenv import load_dotenv
# import os
# import time

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# if not PINECONE_API_KEY or not COHERE_API_KEY:
#     raise ValueError("Missing required environment variables: PINECONE_API_KEY or COHERE_API_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# # ---------------------------
# # Flask app
# # ---------------------------
# app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# print("Loading embeddings...")
# start_time = time.time()
# embeddings = download_embeddings()
# print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

# index_name = "medichat"

# print("Initializing Pinecone vector store...")
# start_time = time.time()
# try:
#     vectorstore = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )
#     print(f"✅ Vector store initialized in {time.time() - start_time:.2f} seconds")
# except Exception as e:
#     print(f"❌ Error initializing vector store: {str(e)}")
#     raise

# # ---------------------------
# # Retriever with relevance filtering
# # ---------------------------
# retriever = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.65}  # ignore irrelevant matches
# )

# # ---------------------------
# # Cohere LLM
# # ---------------------------
# print("Initializing Cohere LLM...")
# start_time = time.time()

# llm = ChatCohere(
#     model="command-r",           # more reliable than command-light
#     temperature=0.2,             # less hallucination
#     max_tokens=400,              # allow more detailed medical answers
#     cohere_api_key=COHERE_API_KEY
# )

# print(f"✅ LLM initialized in {time.time() - start_time:.2f} seconds")

# # ---------------------------
# # Prompt Template (with safety)
# # ---------------------------
# prompt = ChatPromptTemplate.from_chat_history([
#     ("system", """You are MediChat, a safe and reliable medical assistant. 

# Guidelines:
# - Only use medically accurate and relevant information.
# - Use the retrieved context as your primary source. If the context is weak, admit it.
# - If you are unsure, say: "I’m not certain about that. Please consult a qualified doctor."
# - Keep responses clear, short, and focused on the medical question.
# - Do NOT mix unrelated diseases or conditions in the same answer.
# - Do NOT invent symptoms or treatments.
# """),

#     ("human", """Conversation so far:
# {history}

# User Question: {input}

# Relevant Context from knowledge base:
# {context}""")
# ])

# # ---------------------------
# # RAG Chain
# # ---------------------------
# print("Creating RAG chain...")
# start_time = time.time()
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print(f"✅ RAG chain created in {time.time() - start_time:.2f} seconds")

# #---------------------------
# # Chat History with Memory
# #---------------------------
# session_histories = {}

# def get_session_history(session_id: str) -> ChatMessageHistory:
#     if session_id not in session_histories:
#         session_histories[session_id] = ChatMessageHistory()
#     return session_histories[session_id]

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_chat_history_key="input",
#     history_chat_history_key="history",
#     output_chat_history_key="answer",
# )

# # ---------------------------
# # Cache
# # ---------------------------
# query_cache = {}
# CACHE_SIZE = 100

# def get_cached_response(query):
#     return query_cache.get(query)

# def cache_response(query, response):
#     if len(query_cache) >= CACHE_SIZE:
#         oldest_key = next(iter(query_cache))
#         del query_cache[oldest_key]
#     query_cache[query] = response

# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/")
# def home():
#     return render_template("chat.html")

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form["msg"].strip()
#     session_id = request.remote_addr

#     if not msg:
#         return "Please enter a valid question."

#     # Cache check
#     cached_response = get_cached_response(msg)
#     if cached_response:
#         print(f"⚡ Cache hit for: {msg}")
#         return cached_response

#     print(f"💬 New query: {msg}")
#     try:
#         start_time = time.time()
#         response = chat_with_memory.invoke(
#             {"input": msg},
#             config={"configurable": {"session_id": session_id}},
#         )
#         answer = str(response["answer"]).strip()

#         # Extra safeguard: if answer looks empty or irrelevant
#         if not answer or "synarthrosis" in answer.lower() or "toys" in answer.lower():
#             answer = "I’m not certain about that. Please consult a healthcare professional."

#         print(f"✅ Answer in {time.time() - start_time:.2f}s")
#         cache_response(msg, answer)
#         return answer

#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         error_message = "I'm having trouble answering right now. Please try again."
#         cache_response(msg, error_message)
#         return error_message

# @app.route("/health")
# def health():
#     return {"status": "healthy", "service": "MediChat API"}

# if __name__ == "__main__":
#     print("🚀 Starting MediChat server...")
#     app.run(host="0.0.0.0", port=8000, debug=True)












# from flask import Flask, render_template, request
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv
# import os
# import time

# # ---------------------------
# # Load environment variables
# # ---------------------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not PINECONE_API_KEY or not GEMINI_API_KEY:
#     raise ValueError("❌ Missing required environment variables: PINECONE_API_KEY or GEMINI_API_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# # ---------------------------
# # Flask app
# # ---------------------------
# app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# # ---------------------------
# # Load Embeddings
# # ---------------------------
# print("📥 Loading embeddings...")
# start_time = time.time()
# embeddings = download_embeddings()
# print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

# # ---------------------------
# # Pinecone Vector Store
# # ---------------------------
# index_name = "medichat"
# print("🗂️ Initializing Pinecone vector store...")
# try:
#     vectorstore = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )
#     print(f"✅ Vector store ready in {time.time() - start_time:.2f} seconds")
# except Exception as e:
#     raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# # ---------------------------
# # Retriever (looser search)
# # ---------------------------
# retriever = vectorstore.as_retriever(
#     search_type="mmr",   # better diversity of results
#     search_kwargs={"k": 5, "fetch_k": 20}
# )

# # ---------------------------
# # Gemini LLM
# # ---------------------------
# print("🤖 Initializing Gemini LLM...")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_output_tokens=700,
#     google_api_key=GEMINI_API_KEY
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Prompt Template (relaxed)
# # ---------------------------
# prompt = ChatPromptTemplate.from_chat_history([
#     ("system", 
#      "You are MediChat, an intelligent and reliable medical assistant chatbot. "
#      "Your job is to explain medical concepts clearly, accurately, and in detail. "
#      "Always be factual, structured, and student-friendly. "
#      "If a question is too broad, guide the user to focus. "
#      "Never make up medical facts. If unsure, say you are unsure and suggest referring to reliable resources. "
#      "When explaining, try to include: "
#      "1. Definition/Overview "
#      "2. Location (if anatomy-related) "
#      "3. Function/Mechanism "
#      "4. Clinical Relevance or Disorders (if applicable) "
#      "5. Summary (easy-to-revise points). "
#      "Keep the tone supportive and educational."
#     ),

#     ("human", """Conversation so far:
# {history}

# User Question: {input}

# Relevant Context (may be empty):
# {context}""")
# ])

# # ---------------------------
# # RAG Chain
# # ---------------------------
# print("🔗 Building RAG chain...")
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Chat History (Memory per session)
# # ---------------------------
# session_histories = {}

# def get_session_history(session_id: str) -> ChatMessageHistory:
#     if session_id not in session_histories:
#         session_histories[session_id] = ChatMessageHistory()
#     return session_histories[session_id]

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_chat_history_key="input",
#     history_chat_history_key="history",
#     output_chat_history_key="answer",
# )

# # ---------------------------
# # Simple Response Cache
# # ---------------------------
# query_cache = {}
# CACHE_SIZE = 100

# def get_cached_response(query):
#     return query_cache.get(query)

# def cache_response(query, response):
#     if len(query_cache) >= CACHE_SIZE:
#         query_cache.pop(next(iter(query_cache)))
#     query_cache[query] = response

# # ---------------------------
# # Hybrid Answer Function
# # ---------------------------
# def get_answer(msg, session_id):
#     try:
#         # Step 1: Try retrieval
#         retrieved_docs = retriever.get_relevant_documents(msg)
#         if not retrieved_docs:  # No useful context → fallback
#             print("⚠️ No context found → direct Gemini answer")
#             return llm.invoke(msg).content

#         # Step 2: Use RAG with memory
#         response = chat_with_memory.invoke(
#             {"input": msg},
#             config={"configurable": {"session_id": session_id}},
#         )
#         return str(response["answer"]).strip()

#     except Exception as e:
#         print(f"❌ Error during get_answer: {str(e)}")
#         return "⚠️ I’m having trouble answering right now. Please try again."

# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/")
# def home():
#     return render_template("chat.html")

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form.get("msg", "").strip()
#     session_id = request.remote_addr

#     if not msg:
#         return "⚠️ Please enter a valid question."

#     # Cache check
#     cached_response = get_cached_response(msg)
#     if cached_response:
#         print(f"⚡ Cache hit: {msg}")
#         return cached_response

#     print(f"💬 Query: {msg}")
#     start_time = time.time()
#     answer = get_answer(msg, session_id)
#     print(f"✅ Answer generated in {time.time() - start_time:.2f}s")

#     cache_response(msg, answer)
#     return answer

# @app.route("/health")
# def health():
#     return {"status": "healthy", "service": "MediChat API"}

# # ---------------------------
# # Start App
# # ---------------------------
# if __name__ == "__main__":
#     print("🚀 Starting MediChat with Gemini (Hybrid Mode)...")
#     app.run(host="0.0.0.0", port=8000, debug=True)







# from flask import Flask, render_template, request
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from dotenv import load_dotenv
# import os
# import time

# # ---------------------------
# # Load environment variables
# # ---------------------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not PINECONE_API_KEY or not GEMINI_API_KEY:
#     raise ValueError("❌ Missing required environment variables: PINECONE_API_KEY or GEMINI_API_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# # ---------------------------
# # Flask app
# # ---------------------------
# app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# # ---------------------------
# # Load Embeddings
# # ---------------------------
# print("📥 Loading embeddings...")
# start_time = time.time()
# embeddings = download_embeddings()
# print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

# # ---------------------------
# # Pinecone Vector Store
# # ---------------------------
# index_name = "medichat"
# print("🗂️ Initializing Pinecone vector store...")
# try:
#     vectorstore = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )
#     print(f"✅ Vector store ready in {time.time() - start_time:.2f} seconds")
# except Exception as e:
#     raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# # ---------------------------
# # Retriever (looser search)
# # ---------------------------
# retriever = vectorstore.as_retriever(
#     search_type="mmr",   # better diversity of results
#     search_kwargs={"k": 5, "fetch_k": 20}
# )

# # ---------------------------
# # Gemini LLM
# # ---------------------------
# print("🤖 Initializing Gemini LLM...")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_output_tokens=700,
#     google_api_key=GEMINI_API_KEY
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Prompt Template (relaxed)
# # ---------------------------
# prompt = ChatPromptTemplate.from_chat_history([
#     ("system", 
#      "You are MediChat, an intelligent and reliable medical assistant chatbot. "
#      "Your job is to explain medical concepts clearly, accurately, and in detail. "
#      "Always be factual, structured, and student-friendly. "
#      "If a question is too broad, guide the user to focus. "
#      "Never make up medical facts. If unsure, say you are unsure and suggest referring to reliable resources. "
#      "When explaining, try to include: "
#      "1. Definition/Overview "
#      "2. Location (if anatomy-related) "
#      "3. Function/Mechanism "
#      "4. Clinical Relevance or Disorders (if applicable) "
#      "5. Summary (easy-to-revise points). "
#      "Keep the tone supportive and educational."
#     ),

#     ("human", """Conversation so far:
# {history}

# User Question: {input}

# Relevant Context (may be empty):
# {context}""")
# ])

# # ---------------------------
# # RAG Chain
# # ---------------------------
# print("🔗 Building RAG chain...")
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Chat History (Memory per session)
# # ---------------------------
# session_histories = {}

# def get_session_history(session_id: str) -> ChatMessageHistory:
#     if session_id not in session_histories:
#         session_histories[session_id] = ChatMessageHistory()
#     return session_histories[session_id]

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_chat_history_key="input",
#     history_chat_history_key="history",
#     output_chat_history_key="answer",
# )

# # ---------------------------
# # Simple Response Cache
# # ---------------------------
# query_cache = {}
# CACHE_SIZE = 100

# def get_cached_response(query):
#     return query_cache.get(query)

# def cache_response(query, response):
#     if len(query_cache) >= CACHE_SIZE:
#         query_cache.pop(next(iter(query_cache)))
#     query_cache[query] = response

# # ---------------------------
# # Hybrid Answer Function (with memory)
# # ---------------------------
# def get_answer(msg, session_id):
#     try:
#         response = chat_with_memory.invoke(
#             {"input": msg},
#             config={"configurable": {"session_id": session_id}},
#         )
#         return str(response["answer"]).strip()

#     except Exception as e:
#         print(f"❌ Error during get_answer: {str(e)}")
#         return "⚠️ I’m having trouble answering right now. Please try again."

# # def get_answer(msg, session_id):
# #     try:
# #         # Try retrieval
# #         retrieved_docs = retriever.get_relevant_documents(msg)

# #         if not retrieved_docs:  
# #             print("⚠️ No context found → direct Gemini answer (but storing in history)")
# #             response = llm.invoke(msg)
# #             # store in memory manually
# #             history = get_session_history(session_id)
# #             history.add_user_message(msg)
# #             history.add_ai_message(response.content)
# #             return response.content

# #         # Use RAG with memory
# #         response = chat_with_memory.invoke(
# #             {"input": msg},
# #             config={"configurable": {"session_id": session_id}},
# #         )
# #         return str(response["answer"]).strip()

# #     except Exception as e:
# #         print(f"❌ Error during get_answer: {str(e)}")
# #         return "⚠️ I’m having trouble answering right now. Please try again."

# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/")
# def home():
#     return render_template("chat.html")

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form.get("msg", "").strip()
#     # session_id = request.remote_addr  # user session tracking
#     session_id = request.cookies.get("session_id", "default_user")

#     if not msg:
#         return "⚠️ Please enter a valid question."

#     # Cache check
#     cached_response = get_cached_response(msg)
#     if cached_response:
#         print(f"⚡ Cache hit: {msg}")
#         return cached_response

#     print(f"💬 Query: {msg}")
#     start_time = time.time()
#     answer = get_answer(msg, session_id)
#     print(f"✅ Answer generated in {time.time() - start_time:.2f}s")

#     cache_response(msg, answer)
#     return answer

# @app.route("/health")
# def health():
#     return {"status": "healthy", "service": "MediChat API"}

# # ---------------------------
# # Start App
# # ---------------------------
# if __name__ == "__main__":
#     print("🚀 Starting MediChat with Gemini (Hybrid Mode + Memory)...")
#     app.run(host="0.0.0.0", port=8000, debug=True)









# from flask import Flask, render_template, request, make_response
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from supabase import create_client
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

# if not PINECONE_API_KEY or not GEMINI_API_KEY:
#     raise ValueError("❌ Missing required environment variables: PINECONE_API_KEY or GEMINI_API_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# # ---------------------------
# # Flask app
# # ---------------------------
# # (Keeping your folders the same; change static_folder if your assets live elsewhere)
# app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# # ---------------------------
# # Load Embeddings
# # ---------------------------
# print("📥 Loading embeddings...")
# start_time = time.time()
# embeddings = download_embeddings()
# print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

# # ---------------------------
# # Pinecone Vector Store
# # ---------------------------
# index_name = "medichat"
# print("🗂️ Initializing Pinecone vector store...")
# try:
#     vectorstore = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )
#     print(f"✅ Vector store ready in {time.time() - start_time:.2f} seconds")
# except Exception as e:
#     raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# # ---------------------------
# # Retriever (looser search)
# # ---------------------------
# retriever = vectorstore.as_retriever(
#     search_type="mmr",   # better diversity of results
#     search_kwargs={"k": 5, "fetch_k": 20}
# )

# # ---------------------------
# # Gemini LLM
# # ---------------------------
# print("🤖 Initializing Gemini LLM...")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_output_tokens=700,
#     google_api_key=GEMINI_API_KEY
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Prompt Template (relaxed)
# # ---------------------------
# prompt = ChatPromptTemplate.from_chat_history([
#     ("system", 
#      "You are MediChat, an intelligent and reliable medical assistant chatbot. "
#      "don't rely on your own data which gemini trained you on."
#      "Your job is to explain medical concepts clearly, accurately, and in detail. "
#      "Always be factual, structured, and student-friendly. "
#      "If a question is too broad, guide the user to focus. "
#      "Never make up medical facts. If unsure, say you are unsure and suggest referring to reliable resources. "
#      "When explaining, try to include: "
#      "1. Definition/Overview "
#      "2. Location (if anatomy-related) "
#      "3. Function/Mechanism "
#      "4. Clinical Relevance or Disorders (if applicable) "
#      "5. Summary (easy-to-revise points). "
#      "Keep the tone supportive and educational."
#     ),
#     ("human", """Conversation so far:
# {history}

# User Question: {input}

# Relevant Context (may be empty):
# {context}""")
# ])

# # ---------------------------
# # RAG Chain
# # ---------------------------
# print("🔗 Building RAG chain...")
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Chat History (Memory per session)
# # ---------------------------
# session_histories = {}

# def get_session_history(session_id: str) -> ChatMessageHistory:
#     """Return/create the per-session in-memory history."""
#     if session_id not in session_histories:
#         session_histories[session_id] = ChatMessageHistory()
#     return session_histories[session_id]

# def format_history(history_obj: ChatMessageHistory) -> str:
#     """Flatten LangChain's structured history into readable text the prompt expects."""
#     out_lines = []
#     for m in history_obj.chat_history:
#         # msg.type is typically "human" or "ai"
#         role = "User" if getattr(m, "type", "") == "human" else "Assistant"
#         content = getattr(m, "content", "")
#         out_lines.append(f"{role}: {content}")
#     return "\n".join(out_lines).strip()

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_chat_history_key="input",
#     history_chat_history_key="history",
#     output_chat_history_key="answer",
# )

# # ---------------------------
# # Simple Response Cache (optional)
# # ---------------------------
# query_cache = {}
# CACHE_SIZE = 100

# def get_cached_response(query):
#     return query_cache.get(query)

# def cache_response(query, response):
#     if len(query_cache) >= CACHE_SIZE:
#         query_cache.pop(next(iter(query_cache)))
#     query_cache[query] = response

# # ---------------------------
# # Answer Function (ALWAYS uses memory)
# # ---------------------------
# def get_answer(msg, session_id):
#     try:
#         # Pull current history and stringify it for the prompt
#         history_obj = get_session_history(session_id)
#         response = chat_with_memory.invoke(
#             {"input": msg, "history": format_history(history_obj)},
#             config={"configurable": {"session_id": session_id}},
#         )
#         return str(response["answer"]).strip()

#     except Exception as e:
#         print(f"❌ Error during get_answer: {str(e)}")
#         return "⚠️ I’m having trouble answering right now. Please try again."

# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/")
# def home():
#     """
#     Serve the chat UI and ensure the user has a stable session cookie.
#     """
#     resp = make_response(render_template("chat.html"))
#     if not request.cookies.get("session_id"):
#         sid = str(uuid4())
#         # Cookie for ~30 days, adjust as you like
#         resp.set_cookie("session_id", sid, max_age=60*60*24*30, httponly=True, samesite="Lax")
#     return resp

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form.get("msg", "").strip()
#     # Use cookie-based stable session id (fallback to a default)
#     session_id = request.cookies.get("session_id", "default_user")

#     if not msg:
#         return "⚠️ Please enter a valid question."

#     # Optional: skip cache for conversational flow (or keep it if you like)
#     cached_response = get_cached_response(msg)
#     if cached_response:
#         print(f"⚡ Cache hit: {msg}")
#         return cached_response

#     print(f"💬 Query: {msg} | session: {session_id}")
#     start_time = time.time()
#     answer = get_answer(msg, session_id)
#     print(f"✅ Answer generated in {time.time() - start_time:.2f}s")

#     cache_response(msg, answer)
#     return answer

# @app.route("/health")
# def health():
#     return {"status": "healthy", "service": "MediChat API"}

# # ---------------------------
# # Start App
# # ---------------------------
# if __name__ == "__main__":
#     print("🚀 Starting MediChat with Gemini (Memory fixed, cookie sessions)…")
#     app.run(host="0.0.0.0", port=8000, debug=True)








































# from flask import Flask, render_template, request, make_response, jsonify
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import chat_historyPlaceholder
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
# # Flask app
# # ---------------------------
# app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# # ---------------------------
# # Load Embeddings
# # ---------------------------
# print("📥 Loading embeddings...")
# start_time = time.time()
# embeddings = download_embeddings()
# print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

# # ---------------------------
# # Pinecone Vector Store
# # ---------------------------
# index_name = "medichat"
# print("🗂️ Initializing Pinecone vector store...")
# try:
#     vectorstore = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings,
#     )
#     print(f"✅ Vector store ready in {time.time() - start_time:.2f} seconds")
# except Exception as e:
#     raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# # ---------------------------
# # Retriever (looser search)
# # ---------------------------
# retriever = vectorstore.as_retriever(
#     search_type="mmr",  # better diversity of results
#     search_kwargs={"k": 5, "fetch_k": 20},
# )

# # ---------------------------
# # Gemini LLM
# # ---------------------------
# print("🤖 Initializing Gemini LLM...")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0,
#     max_output_tokens=700,
#     google_api_key=GEMINI_API_KEY,
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Prompt Template (uses chat_historyPlaceholder for proper history)
# # ---------------------------
# prompt = ChatPromptTemplate.from_chat_history([
#     (
#         "system",
#         "You are MediChat, an intelligent and reliable medical assistant chatbot. "
#         "Do not rely on proprietary pretraining facts when unsure; prefer retrieved context. "
#         "Explain medical concepts clearly, accurately, and in detail. "
#         "Be factual, structured, student-friendly. If a question is too broad, help narrow it. "
#         "Never invent medical facts. If unsure, say so and suggest reputable sources. "
#         "When explaining, try to include: 1) Definition/Overview 2) Location 3) Function/Mechanism "
#         "4) Clinical Relevance/Disorders 5) Summary points. Keep the tone supportive.",
#     ),
#     # Prior turns will be injected here by RunnableWithMessageHistory
#     chat_historyPlaceholder(variable_name="history"),
#     (
#         "human",
#         (
#             "User Question: {input}\n\n"
#             "Relevant Context (may be empty):\n{context}"
#         ),
#     ),
# ])

# # ---------------------------
# # RAG Chain
# # ---------------------------
# print("🔗 Building RAG chain...")
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Persistence helpers (Supabase)
# # ---------------------------

# def save_message(user_id: str, role: str, content: str):
#     try:
#         supabase.table("chat_history").insert({
#             "user_id": user_id,
#             "role": role,
#             "content": content,
#         }).execute()
#     except Exception as e:
#         print(f"⚠️ Failed to save message: {e}")


# def load_history(user_id: str) -> ChatMessageHistory:
#     history = ChatMessageHistory()
#     try:
#         res = (
#             supabase.table("chat_history")
#             .select("role, content, created_at")
#             .eq("user_id", user_id)
#             .order("created_at")
#             .execute()
#         )
#         for row in (res.data or []):
#             if row["role"] == "user":
#                 history.add_user_message(row["content"])
#             else:
#                 history.add_ai_message(row["content"])
#     except Exception as e:
#         print(f"⚠️ Failed to load history: {e}")
#     return history


# # ---------------------------
# # Chat History (Memory per user via Supabase)
# # ---------------------------
# session_histories: dict[str, ChatMessageHistory] = {}


# def get_session_history(user_id: str) -> ChatMessageHistory:
#     # Always rebuild from Supabase so memory persists across restarts
#     session_histories[user_id] = load_history(user_id)
#     return session_histories[user_id]


# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_chat_history_key="input",      # maps to {input}
#     history_chat_history_key="history",  # fills chat_historyPlaceholder("history")
#     output_chat_history_key="answer",    # captured for history object (not auto-persisted)
# )

# # ---------------------------
# # Utility: ensure session cookie / user id
# # ---------------------------

# def get_user_id_from_cookie():
#     sid = request.cookies.get("session_id")
#     if sid:
#         return sid
#     # create guest session id
#     return f"guest_{uuid4()}"


# # ---------------------------
# # Answer Function (ALWAYS uses memory wrapper and persists to Supabase)
# # ---------------------------

# def get_answer(msg: str, user_id: str) -> str:
#     try:
#         # 1) Generate answer with memory-aware RAG
#         response = chat_with_memory.invoke(
#             {"input": msg},
#             config={"configurable": {"session_id": user_id}},
#         )
#         answer = str(response["answer"]).strip()

#         # 2) Persist both turns to Supabase
#         save_message(user_id, "user", msg)
#         save_message(user_id, "assistant", answer)

#         return answer
#     except Exception as e:
#         print(f"❌ Error during get_answer: {str(e)}")
#         return "⚠️ I’m having trouble answering right now. Please try again."


# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/")
# def home():
#     resp = make_response(render_template("chat.html"))
#     current_sid = request.cookies.get("session_id")
#     if not current_sid:
#         # assign a guest session cookie (switches to Supabase user id after login)
#         resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
#     return resp


# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form.get("msg", "").strip()
#     user_id = get_user_id_from_cookie()

#     if not msg:
#         return "⚠️ Please enter a valid question."

#     print(f"💬 Query: {msg} | user: {user_id}")
#     start_time_local = time.time()
#     answer = get_answer(msg, user_id)
#     print(f"✅ Answer generated in {time.time() - start_time_local:.2f}s")

#     return answer


# # ---------------------------
# # Auth endpoints (Supabase Auth)
# # ---------------------------
# @app.route("/signup", methods=["POST"])
# def signup():
#     email = request.form.get("email") or (request.json or {}).get("email")
#     password = request.form.get("password") or (request.json or {}).get("password")
#     if not email or not password:
#         return jsonify({"error": "email and password are required"}), 400
#     try:
#         res = supabase.auth.sign_up({"email": email, "password": password})
#         # Depending on project settings, user may need to confirm email.
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
#         # Store Supabase user id in cookie so chats bind to real account
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
#     # Reset to a new guest session
#     resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
#     return resp


# @app.route("/health")
# def health():
#     return {"status": "healthy", "service": "MediChat API"}


# # ---------------------------
# # Start App
# # ---------------------------
# if __name__ == "__main__":
#     print("🚀 Starting MediChat with Gemini (RAG + Supabase Auth & History)…")
#     app.run(host="0.0.0.0", port=8000, debug=True)


# app.py
# from flask import Flask, render_template, request, make_response, jsonify
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import chat_historyPlaceholder
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
#     max_output_tokens=700,
#     google_api_key=GEMINI_API_KEY,
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Prompt Template (history-aware)
# # ---------------------------
# prompt = ChatPromptTemplate.from_chat_history([
#     (
#         "system",
#         "You are MediChat, an intelligent and reliable medical assistant chatbot. "
#         "Prefer retrieved context over guesswork. Explain clearly and factually. "
#         "If unsure, say so and suggest reputable sources. "
#         "When explaining, try to include: 1) Definition/Overview 2) Location 3) Function/Mechanism "
#         "4) Clinical Relevance/Disorders 5) Summary points. Keep the tone supportive."
#     ),
#     chat_historyPlaceholder(variable_name="history"),
#     (
#         "human",
#         "User Question: {input}\n\nRelevant Context (may be empty):\n{context}"
#     ),
# ])

# # RAG chain
# print("🔗 Building RAG chain...")
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Supabase persistence helpers (CONVERSATIONS + CHAT_HISTORY)
# # ---------------------------

# def ensure_conversation(user_id: str) -> str:
#     """
#     Ensure there is at least one conversation for this user.
#     Returns the most recent conversation_id (creates one if not present).
#     """
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
#         # As a fallback (should not happen if DB reachable), create a local UUID
#         return str(uuid4())


# def save_message(conversation_id: str, user_id: str, role: str, message: str) -> None:
#     """
#     Save a single message to Supabase (chat_history).
#     Schema columns expected: conversation_id, user_id, role, message
#     """
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
#     """
#     history = ChatMessageHistory()
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
#     """
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


# # ---------------------------
# # Message-history wrapper for RAG
# # ---------------------------
# _session_histories = {}

# def _get_session_history(session_key: str) -> ChatMessageHistory:
#     # session_key will be conversation_id (so history is per-conversation)
#     _session_histories[session_key] = load_history_as_langchain(session_key)
#     return _session_histories[session_key]

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     _get_session_history,
#     input_chat_history_key="input",
#     history_chat_history_key="history",
#     output_chat_history_key="answer",
# )

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

#         # Run RAG with memory (memory reads from DB via _get_session_history)
#         response = chat_with_memory.invoke(
#             {"input": user_input},
#             config={"configurable": {"session_id": conv_id}},  # key for RunnableWithMessageHistory
#         )
#         answer = str(response["answer"]).strip()

#         # Persist turns
#         save_message(conv_id, user_id, "user", user_input)
#         save_message(conv_id, user_id, "assistant", answer)

#         return answer
#     except Exception as e:
#         print(f"❌ Error during get_answer: {e}")
#         return "⚠️ I’m having trouble answering right now. Please try again."


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









# from flask import Flask, render_template, request, make_response, jsonify
# from src.helper import download_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import MessagesPlaceholder  # Fixed import
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
#     max_output_tokens=700,
#     google_api_key=GEMINI_API_KEY,
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Prompt Template (history-aware)
# # ---------------------------
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are MediChat, an intelligent and reliable medical assistant chatbot. "
#         "Prefer retrieved context over guesswork. Explain clearly and factually. "
#         "If unsure, say so and suggest reputable sources. "
#         "When explaining, try to include: 1) Definition/Overview 2) Location 3) Function/Mechanism "
#         "4) Clinical Relevance/Disorders 5) Summary points. Keep the tone supportive."
#     ),
#     MessagesPlaceholder(variable_name="history"),  # Fixed this line
#     (
#         "human",
#         "User Question: {input}\n\nRelevant Context (may be empty):\n{context}"
#     ),
# ])

# # RAG chain
# print("🔗 Building RAG chain...")
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Supabase persistence helpers (CONVERSATIONS + CHAT_HISTORY)
# # ---------------------------

# def ensure_conversation(user_id: str) -> str:
#     """
#     Ensure there is at least one conversation for this user.
#     Returns the most recent conversation_id (creates one if not present).
#     """
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
#         # As a fallback (should not happen if DB reachable), create a local UUID
#         return str(uuid4())


# def save_message(conversation_id: str, user_id: str, role: str, message: str) -> None:
#     """
#     Save a single message to Supabase (chat_history).
#     Schema columns expected: conversation_id, user_id, role, message
#     """
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
#     """
#     history = ChatMessageHistory()
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
#     """
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
#     """
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
#     _session_histories[session_key] = load_history_as_langchain(session_key)
#     return _session_histories[session_key]

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     _get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
#     output_messages_key="answer",
# )

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

#         # Run RAG with memory (memory reads from DB via _get_session_history)
#         response = chat_with_memory.invoke(
#             {"input": user_input},
#             config={"configurable": {"session_id": conv_id}},  # key for RunnableWithMessageHistory
#         )
#         answer = str(response["answer"]).strip()

#         # Persist turns
#         save_message(conv_id, user_id, "user", user_input)
#         save_message(conv_id, user_id, "assistant", answer)

#         return answer
#     except Exception as e:
#         print(f"❌ Error during get_answer: {e}")
#         return "⚠️ I'm having trouble answering right now. Please try again."


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
from langchain_google_genai import ChatGoogleGenerativeAI
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("❌ Missing required environment variables: PINECONE_API_KEY or GEMINI_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ---------------------------
# Supabase client
# ---------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Flask app (only used if you run this file directly)
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# ---------------------------
# Embeddings + Pinecone Vector Store
# ---------------------------
print("📥 Loading embeddings...")
t0 = time.time()
embeddings = download_embeddings()
print(f"✅ Embeddings loaded in {time.time() - t0:.2f}s")

index_name = "medichat"
print("🗂️ Initializing Pinecone vector store...")
try:
    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    print(f"✅ Vector store ready in {time.time() - t0:.2f}s")
except Exception as e:
    raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# Retriever (looser search is nicer for Q/A)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20},
)

# ---------------------------
# Gemini LLM
# ---------------------------
print("🤖 Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=1000,  # Increased output tokens
    google_api_key=GEMINI_API_KEY,
)
print("✅ Gemini LLM ready")

# ---------------------------
# Improved Prompt Template
# ---------------------------
# ... (previous imports remain the same)

# ---------------------------
# Improved Prompt Template with stronger context instructions
# ---------------------------
system_prompt = """
You are MediChat, an intelligent and reliable medical assistant chatbot. 
Your primary role is to answer medical questions based on the retrieved context provided.

IMPORTANT INSTRUCTIONS:
1. ALWAYS prioritize and use the retrieved context to answer questions
2. If the context contains relevant information, base your answer SOLELY on it
3. If the context doesn't contain enough information, you can supplement with your general medical knowledge
   but clearly indicate what's from the context vs. general knowledge
4. NEVER say "the provided text does not contain information about..." - instead, use your knowledge if needed
5. Structure your answers clearly when appropriate
6. Keep the tone supportive, professional, and empathetic

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

# ... (rest of the app.py remains the same until the get_answer function)

def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
    """
    Generate an answer with RAG + past chat memory.
    Also persists both user and assistant chat_history to Supabase.
    """
    try:
        conv_id = conversation_id or ensure_conversation(user_id)
        
        # First, retrieve relevant documents
        retrieved_docs = retriever.invoke(user_input)
        
        # Debug: print what was retrieved
        print(f"Retrieved {len(retrieved_docs)} documents for query: {user_input}")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1}: {doc.page_content[:200]}...")  # First 200 chars
        
        # If no relevant documents found, use a different approach
        if not retrieved_docs or all(len(doc.page_content.strip()) < 10 for doc in retrieved_docs):
            print("No relevant documents found, using LLM without context")
            # Use LLM directly without context
            direct_response = llm.invoke(f"Answer this medical question: {user_input}")
            answer = str(direct_response.content).strip()
        else:
            # Run RAG with memory (memory reads from DB via _get_session_history)
            response = chat_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": conv_id}},  # key for RunnableWithMessageHistory
            )
            answer = str(response["answer"]).strip()

        # Persist turns
        save_message(conv_id, user_id, "user", user_input)
        save_message(conv_id, user_id, "assistant", answer)

        return answer
    except Exception as e:
        print(f"❌ Error during get_answer: {e}")
        return "⚠️ I'm having trouble answering right now. Please try again."

# ... (rest of the file remains the same)

# RAG chain
print("🔗 Building RAG chain...")
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)
print("✅ RAG chain ready")

# ---------------------------
# Supabase persistence helpers (CONVERSATIONS + CHAT_HISTORY)
# ---------------------------

# ... (previous imports remain the same)

# ---------------------------
# Supabase persistence helpers (CONVERSATIONS + CHAT_HISTORY)
# ---------------------------

def ensure_conversation(user_id: str) -> str:
    """
    Ensure there is at least one conversation for this user.
    Returns the most recent conversation_id (creates one if not present).
    """
    # Check if this is a guest user
    if user_id.startswith("guest_"):
        # For guest users, don't use the database, just return a UUID
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
        # Create a new conversation
        created = (
            supabase.table("conversations")
            .insert({"user_id": user_id, "title": "New Conversation"})
            .execute()
        )
        return created.data[0]["id"]
    except Exception as e:
        print(f"⚠️ ensure_conversation error: {e}")
        # As a fallback, create a local UUID
        return str(uuid4())


def save_message(conversation_id: str, user_id: str, role: str, message: str) -> None:
    """
    Save a single message to Supabase (chat_history).
    Skip saving for guest users.
    """
    # Don't save messages for guest users
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
    """
    Load conversation chat_history and return a LangChain ChatMessageHistory
    suitable for RunnableWithMessageHistory.
    For guest users, return empty history.
    """
    history = ChatMessageHistory()
    
    # Don't load history for guest users (conversation_id is not in database)
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
            # ignore other roles for rendering (e.g., "system")
    except Exception as e:
        print(f"⚠️ Failed to load history: {e}")
    return history


def load_history_for_ui(conversation_id: str):
    """
    Load chat_history for UI (list of (role, message) tuples).
    For guest users, return empty list.
    """
    # Don't load history for guest users
    if not conversation_id or "guest_" in conversation_id:
        return []
        
    msgs = []
    try:
        res = (
            supabase.table("chat_history")
            .select("role, message, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        msgs = [(row["role"], row["message"]) for row in (res.data or [])]
    except Exception as e:
        print(f"⚠️ Failed to load history_for_ui: {e}")
    return msgs


def get_user_conversations(user_id: str):
    """
    Get all conversations for a user.
    Skip for guest users.
    """
    # Don't get conversations for guest users
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
# Message-history wrapper for RAG
# ---------------------------
_session_histories = {}

def _get_session_history(session_key: str) -> ChatMessageHistory:
    # session_key will be conversation_id (so history is per-conversation)
    if session_key not in _session_histories:
        _session_histories[session_key] = load_history_as_langchain(session_key)
    return _session_histories[session_key]

# ... (rest of the file remains the same)

# ---------------------------
# Public function used by Streamlit
# ---------------------------
def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
    """
    Generate an answer with RAG + past chat memory.
    Also persists both user and assistant chat_history to Supabase.
    """
    try:
        conv_id = conversation_id or ensure_conversation(user_id)
        
        # Debug: print the query
        print(f"Processing query: {user_input}")
        
        # First, test the retriever to see if it's working
        try:
            retrieved_docs = retriever.invoke(user_input)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # If no relevant documents found, use a different approach
            if not retrieved_docs or all(len(doc.page_content.strip()) < 10 for doc in retrieved_docs):
                print("No relevant documents found, using LLM without context")
                # Use LLM directly without context
                direct_response = llm.invoke(f"Answer this medical question: {user_input}")
                answer = str(direct_response.content).strip()
            else:
                # Display some content from the retrieved documents for debugging
                for i, doc in enumerate(retrieved_docs[:2]):  # Show first 2 docs
                    print(f"Document {i+1}: {doc.page_content[:100]}...")
                
                # Run RAG with memory
                response = chat_with_memory.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": conv_id}},
                )
                answer = str(response["answer"]).strip()
                
        except Exception as e:
            print(f"❌ Error in retriever or RAG chain: {e}")
            # Fallback to direct LLM response
            direct_response = llm.invoke(f"Answer this medical question: {user_input}")
            answer = str(direct_response.content).strip()

        # Persist turns (skipped for guest users as per previous fix)
        save_message(conv_id, user_id, "user", user_input)
        save_message(conv_id, user_id, "assistant", answer)

        return answer
    except Exception as e:
        print(f"❌ Error during get_answer: {e}")
        # Try a simple direct response as a last resort
        try:
            direct_response = llm.invoke(f"Answer this medical question: {user_input}")
            return str(direct_response.content).strip()
        except:
            return "⚠️ I'm having trouble answering right now. Please try again."

# ---------------------------
# Minimal Flask routes (optional; only when running app.py directly)
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

# Auth endpoints (optional)
@app.route("/signup", methods=["POST"])
def signup():
    email = request.form.get("email") or (request.json or {}).get("email")
    password = request.form.get("password") or (request.json or {}).get("password")
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        return jsonify({"message": "Signup successful. Check your email for confirmation.", "user": getattr(res, "user", None)}), 200
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
    print("🚀 Starting MediChat with Gemini (RAG + Supabase Auth & History)…")
    app.run(host="0.0.0.0", port=8000, debug=True)



























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
#     max_output_tokens=1000,
#     google_api_key=GEMINI_API_KEY,
# )
# print("✅ Gemini LLM ready")

# # ---------------------------
# # Improved Prompt Template
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

# # RAG chain
# print("🔗 Building RAG chain...")
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)
# print("✅ RAG chain ready")

# # ---------------------------
# # Message-history wrapper for RAG
# # ---------------------------
# _session_histories = {}

# def _get_session_history(session_key: str) -> ChatMessageHistory:
#     # session_key will be conversation_id (so history is per-conversation)
#     if session_key not in _session_histories:
#         _session_histories[session_key] = ChatMessageHistory()  # Start with empty history
#     return _session_histories[session_key]

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     _get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
#     output_messages_key="answer",
# )

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

#         # Persist turns (skipped for guest users)
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