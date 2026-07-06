# MediChat Architecture

## Layer Map

- `app/api`: Flask controllers and route registration.
- `app/services`: business logic for chat generation, conversation management, and auth orchestration.
- `app/repositories`: Supabase data access for conversations, chat history, and users.
- `app/rag`: embeddings, vector store access, retrieval, prompt construction, and LLM chain assembly.
- `app/core`: configuration, logging, and custom error handling.
- `frontend`: Streamlit app and Flask template assets.
- `scripts`: deployment and indexing entrypoints.

## Runtime Flow

1. A request enters the Flask app or Streamlit UI.
2. The controller resolves the user and delegates to the chat service.
3. The chat service loads retrieval and generation dependencies lazily.
4. Retrieval checks Pinecone for relevant context.
5. The answer is generated through RAG when possible, or general medical knowledge otherwise.
6. Conversation history is written back through the repository layer.

