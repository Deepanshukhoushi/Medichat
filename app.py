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
# prompt = ChatPromptTemplate.from_messages([
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
#     input_messages_key="input",
#     history_messages_key="history",
#     output_messages_key="answer",
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
# prompt = ChatPromptTemplate.from_messages([
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
#     input_messages_key="input",
#     history_messages_key="history",
#     output_messages_key="answer",
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
# prompt = ChatPromptTemplate.from_messages([
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
#     input_messages_key="input",
#     history_messages_key="history",
#     output_messages_key="answer",
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
# prompt = ChatPromptTemplate.from_messages([
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
#     for m in history_obj.messages:
#         # msg.type is typically "human" or "ai"
#         role = "User" if getattr(m, "type", "") == "human" else "Assistant"
#         content = getattr(m, "content", "")
#         out_lines.append(f"{role}: {content}")
#     return "\n".join(out_lines).strip()

# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
#     output_messages_key="answer",
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
# Flask app
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/")

# ---------------------------
# Load Embeddings
# ---------------------------
print("📥 Loading embeddings...")
start_time = time.time()
embeddings = download_embeddings()
print(f"✅ Embeddings loaded in {time.time() - start_time:.2f} seconds")

# ---------------------------
# Pinecone Vector Store
# ---------------------------
index_name = "medichat"
print("🗂️ Initializing Pinecone vector store...")
try:
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    print(f"✅ Vector store ready in {time.time() - start_time:.2f} seconds")
except Exception as e:
    raise RuntimeError(f"❌ Error initializing vector store: {str(e)}")

# ---------------------------
# Retriever (looser search)
# ---------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",  # better diversity of results
    search_kwargs={"k": 5, "fetch_k": 20},
)

# ---------------------------
# Gemini LLM
# ---------------------------
print("🤖 Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=700,
    google_api_key=GEMINI_API_KEY,
)
print("✅ Gemini LLM ready")

# ---------------------------
# Prompt Template (uses MessagesPlaceholder for proper history)
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are MediChat, an intelligent and reliable medical assistant chatbot. "
        "Do not rely on proprietary pretraining facts when unsure; prefer retrieved context. "
        "Explain medical concepts clearly, accurately, and in detail. "
        "Be factual, structured, student-friendly. If a question is too broad, help narrow it. "
        "Never invent medical facts. If unsure, say so and suggest reputable sources. "
        "When explaining, try to include: 1) Definition/Overview 2) Location 3) Function/Mechanism "
        "4) Clinical Relevance/Disorders 5) Summary points. Keep the tone supportive.",
    ),
    # Prior turns will be injected here by RunnableWithMessageHistory
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        (
            "User Question: {input}\n\n"
            "Relevant Context (may be empty):\n{context}"
        ),
    ),
])

# ---------------------------
# RAG Chain
# ---------------------------
print("🔗 Building RAG chain...")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("✅ RAG chain ready")

# ---------------------------
# Persistence helpers (Supabase)
# ---------------------------

def save_message(user_id: str, role: str, content: str):
    try:
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "role": role,
            "content": content,
        }).execute()
    except Exception as e:
        print(f"⚠️ Failed to save message: {e}")


def load_history(user_id: str) -> ChatMessageHistory:
    history = ChatMessageHistory()
    try:
        res = (
            supabase.table("chat_history")
            .select("role, content, created_at")
            .eq("user_id", user_id)
            .order("created_at")
            .execute()
        )
        for row in (res.data or []):
            if row["role"] == "user":
                history.add_user_message(row["content"])
            else:
                history.add_ai_message(row["content"])
    except Exception as e:
        print(f"⚠️ Failed to load history: {e}")
    return history


# ---------------------------
# Chat History (Memory per user via Supabase)
# ---------------------------
session_histories: dict[str, ChatMessageHistory] = {}


def get_session_history(user_id: str) -> ChatMessageHistory:
    # Always rebuild from Supabase so memory persists across restarts
    session_histories[user_id] = load_history(user_id)
    return session_histories[user_id]


chat_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",      # maps to {input}
    history_messages_key="history",  # fills MessagesPlaceholder("history")
    output_messages_key="answer",    # captured for history object (not auto-persisted)
)

# ---------------------------
# Utility: ensure session cookie / user id
# ---------------------------

def get_user_id_from_cookie():
    sid = request.cookies.get("session_id")
    if sid:
        return sid
    # create guest session id
    return f"guest_{uuid4()}"


# ---------------------------
# Answer Function (ALWAYS uses memory wrapper and persists to Supabase)
# ---------------------------

def get_answer(msg: str, user_id: str) -> str:
    try:
        # 1) Generate answer with memory-aware RAG
        response = chat_with_memory.invoke(
            {"input": msg},
            config={"configurable": {"session_id": user_id}},
        )
        answer = str(response["answer"]).strip()

        # 2) Persist both turns to Supabase
        save_message(user_id, "user", msg)
        save_message(user_id, "assistant", answer)

        return answer
    except Exception as e:
        print(f"❌ Error during get_answer: {str(e)}")
        return "⚠️ I’m having trouble answering right now. Please try again."


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    resp = make_response(render_template("chat.html"))
    current_sid = request.cookies.get("session_id")
    if not current_sid:
        # assign a guest session cookie (switches to Supabase user id after login)
        resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
    return resp


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    user_id = get_user_id_from_cookie()

    if not msg:
        return "⚠️ Please enter a valid question."

    print(f"💬 Query: {msg} | user: {user_id}")
    start_time_local = time.time()
    answer = get_answer(msg, user_id)
    print(f"✅ Answer generated in {time.time() - start_time_local:.2f}s")

    return answer


# ---------------------------
# Auth endpoints (Supabase Auth)
# ---------------------------
@app.route("/signup", methods=["POST"])
def signup():
    email = request.form.get("email") or (request.json or {}).get("email")
    password = request.form.get("password") or (request.json or {}).get("password")
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        # Depending on project settings, user may need to confirm email.
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
        # Store Supabase user id in cookie so chats bind to real account
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
    # Reset to a new guest session
    resp.set_cookie("session_id", f"guest_{uuid4()}", max_age=60*60*24*30, httponly=True, samesite="Lax")
    return resp


@app.route("/health")
def health():
    return {"status": "healthy", "service": "MediChat API"}


# ---------------------------
# Start App
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting MediChat with Gemini (RAG + Supabase Auth & History)…")
    app.run(host="0.0.0.0", port=8000, debug=True)
