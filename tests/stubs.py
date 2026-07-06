import os
import sys
import types

def _stub(name: str, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m

def setup_stubs():
    os.environ.setdefault("PINECONE_API_KEY", "test-key")
    os.environ.setdefault("COHERE_API_KEY", "test-key")
    os.environ.setdefault("SUPABASE_URL", "http://localhost")
    os.environ.setdefault("SUPABASE_KEY", "test-key")
    os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")

    class _FakeChatMessageHistory:
        def __init__(self):
            self.messages = []
        def add_user_message(self, msg):
            self.messages.append(types.SimpleNamespace(type="human", content=msg))
        def add_ai_message(self, msg):
            self.messages.append(types.SimpleNamespace(type="ai", content=msg))

    _stub("dotenv", load_dotenv=lambda *a, **k: None)
    _stub("langchain_community.chat_message_histories", ChatMessageHistory=_FakeChatMessageHistory)
    _stub("langchain_cohere", CohereEmbeddings=object, ChatCohere=object)
    _stub("langchain_core.prompts", ChatPromptTemplate=object, MessagesPlaceholder=object)
    _stub("langchain_core.runnables.history", RunnableWithMessageHistory=object)
    _stub("langchain.chains.combine_documents", create_stuff_documents_chain=lambda *a, **k: None)
    _stub("langchain.chains.retrieval", create_retrieval_chain=lambda *a, **k: None)
    _stub("langchain.schema", Document=object)
    _stub("langchain.text_splitter", RecursiveCharacterTextSplitter=object)
    _stub("langchain_pinecone", PineconeVectorStore=object)
    _stub("pinecone", ServerlessSpec=object, Pinecone=object)
    _stub("pinecone.grpc", PineconeGRPC=object)
    _stub("supabase", create_client=lambda *a, **k: None)
    _stub("werkzeug", exceptions=types.SimpleNamespace())
    _stub("werkzeug.exceptions", HTTPException=Exception)
    _stub("flask", Flask=object, jsonify=dict, make_response=lambda v=None: v,
          send_from_directory=lambda *a, **k: None,
          render_template=lambda *a, **k: "", request=types.SimpleNamespace(
              form={}, cookies={}, get_json=lambda silent=True: None,
              headers={}, method="GET", path="/", remote_addr="127.0.0.1",
              host_url="http://localhost:8000/", endpoint="chat"),
          g=types.SimpleNamespace())
    from unittest.mock import MagicMock
    def stub_task(f):
        f.delay = MagicMock()
        return f

    _stub("celery", 
          Celery=lambda *a, **k: type("DummyCelery", (), {
              "task": lambda self, *a, **k: stub_task,
              "conf": type("Conf", (), {"update": lambda *a, **k: None})()
          })(),
          shared_task=lambda *a, **k: stub_task)

    _app_pkg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
    _app_stub = types.ModuleType("app")
    _app_stub.__path__ = [_app_pkg_path]
    _app_stub.__package__ = "app"
    _app_stub.__spec__ = None
    sys.modules.setdefault("app", _app_stub)
