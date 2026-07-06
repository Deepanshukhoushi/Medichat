from __future__ import annotations

import importlib
import os
import sys
import types
import unittest


class ImportSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
        os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
        os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
        os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
        os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-for-unit-tests-only")
        self._install_stubs()

    def _install_stubs(self) -> None:
        def stub_module(name: str, **attributes):
            module = types.ModuleType(name)
            for attribute_name, attribute_value in attributes.items():
                setattr(module, attribute_name, attribute_value)
            sys.modules[name] = module
            return module

        class StubFlask:
            def __init__(self, *args, **kwargs):
                self.name = args[0] if args else kwargs.get("import_name")
                self.template_folder = kwargs.get("template_folder")
                self.static_folder = kwargs.get("static_folder")
                self.static_url_path = kwargs.get("static_url_path")
                self.config = {}
                self.extensions = {}
                self.url_rules = []

            def before_request(self, func):
                return func

            def after_request(self, func):
                return func

            def errorhandler(self, *_args, **_kwargs):
                def decorator(func):
                    return func

                return decorator

            def add_url_rule(self, *args, **kwargs):
                self.url_rules.append((args, kwargs))
                return None

        class StubResponse:
            def __init__(self, *args, **kwargs):
                self.status_code = 200

            def set_cookie(self, *args, **kwargs):
                return None

        class StubSupabaseClient:
            def __init__(self):
                self.auth = types.SimpleNamespace(
                    sign_up=lambda *_args, **_kwargs: types.SimpleNamespace(),
                    sign_in_with_password=lambda *_args, **_kwargs: types.SimpleNamespace(
                        session=types.SimpleNamespace(),
                        user=types.SimpleNamespace(id="stub-user"),
                    ),
                    sign_out=lambda *_args, **_kwargs: None,
                )

            def table(self, *_args, **_kwargs):
                return types.SimpleNamespace(
                    select=lambda *a, **k: types.SimpleNamespace(
                        eq=lambda *a, **k: types.SimpleNamespace(
                            order=lambda *a, **k: types.SimpleNamespace(
                                limit=lambda *a, **k: types.SimpleNamespace(
                                    execute=lambda: types.SimpleNamespace(data=[])
                                ),
                            ),
                        ),
                        execute=lambda: types.SimpleNamespace(data=[]),
                    ),
                    insert=lambda *a, **k: types.SimpleNamespace(
                        execute=lambda: types.SimpleNamespace(data=[{"id": "stub-conversation"}])
                    ),
                )

        class StubVectorStore:
            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def from_existing_index(cls, *args, **kwargs):
                return cls()

            @classmethod
            def from_documents(cls, *args, **kwargs):
                return cls()

            def as_retriever(self, *args, **kwargs):
                return types.SimpleNamespace(invoke=lambda query: [])

        class StubChatModel:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, prompt):
                return types.SimpleNamespace(content="stub answer")

        class StubPrompt:
            @staticmethod
            def from_messages(*args, **kwargs):
                return StubPrompt()

        class StubRunnable:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return {"answer": "stub answer"}

        class StubPdfReader:
            def __init__(self, *args, **kwargs):
                self.pages = []

        class StubTextSplitter:
            def __init__(self, *args, **kwargs):
                pass

            def split_documents(self, documents):
                return documents

        class StubDocument:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        stub_module("flask", Flask=StubFlask, jsonify=lambda payload=None, **kwargs: payload, make_response=lambda value=None: StubResponse(), send_from_directory=lambda *args, **kwargs: None, render_template=lambda *args, **kwargs: "<html />", request=types.SimpleNamespace(form={}, cookies={}, get_json=lambda silent=True: None, headers={}, method="GET", path="/", remote_addr="127.0.0.1"), g=types.SimpleNamespace())
        stub_module("supabase", create_client=lambda *args, **kwargs: StubSupabaseClient())
        stub_module("dotenv", load_dotenv=lambda *args, **kwargs: None)
        stub_module("streamlit", set_page_config=lambda *args, **kwargs: None, session_state={}, sidebar=types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False))
        stub_module("werkzeug", exceptions=types.SimpleNamespace())
        stub_module("werkzeug.exceptions", HTTPException=Exception)
        stub_module("langchain_community.chat_message_histories", ChatMessageHistory=type("ChatMessageHistory", (), {"add_user_message": lambda self, message: None, "add_ai_message": lambda self, message: None}))
        stub_module("langchain_cohere", CohereEmbeddings=object, ChatCohere=StubChatModel)
        stub_module("langchain_core.prompts", ChatPromptTemplate=StubPrompt, MessagesPlaceholder=object)
        stub_module("langchain_core.runnables.history", RunnableWithMessageHistory=StubRunnable)
        stub_module("langchain.chains.combine_documents", create_stuff_documents_chain=lambda *args, **kwargs: object())
        stub_module("langchain.chains.retrieval", create_retrieval_chain=lambda *args, **kwargs: object())
        stub_module("langchain.schema", Document=StubDocument)
        stub_module("langchain.text_splitter", RecursiveCharacterTextSplitter=StubTextSplitter)
        stub_module("langchain_pinecone", PineconeVectorStore=StubVectorStore)
        stub_module("pinecone", ServerlessSpec=object, Pinecone=type("Pinecone", (), {"__init__": lambda self, *args, **kwargs: None, "list_indexes": lambda self: types.SimpleNamespace(names=lambda: []), "delete_index": lambda self, *args, **kwargs: None, "create_index": lambda self, *args, **kwargs: None}))
        stub_module("pinecone.grpc", PineconeGRPC=type("PineconeGRPC", (), {"__init__": lambda self, *args, **kwargs: None, "list_indexes": lambda self: types.SimpleNamespace(names=lambda: []), "delete_index": lambda self, *args, **kwargs: None, "create_index": lambda self, *args, **kwargs: None}))
        stub_module("pypdf", PdfReader=StubPdfReader)

    def test_app_package_imports(self) -> None:
        # Remove any lightweight stub registered by other test files before
        # importing the real app package (which has Flask, blueprints, etc.)
        sys.modules.pop("app", None)
        module = importlib.import_module("app")
        self.assertNotIn("app", module.__dict__)
        self.assertTrue(hasattr(module, "create_app"))
        self.assertTrue(hasattr(module, "get_app"))
        self.assertTrue(hasattr(module, "app"))
        self.assertTrue(callable(module.get_answer))
        self.assertEqual(module.app.config["MAX_CONTENT_LENGTH"], 10489856)
        registered_routes = {rule[0][0] for rule in module.app.url_rules}
        self.assertIn("/", registered_routes)
        self.assertIn("/get", registered_routes)
        self.assertIn("/health", registered_routes)

    def test_wrapper_scripts_import(self) -> None:
        self.assertTrue(callable(importlib.import_module("deploy").main))

    def test_create_app_is_callable_multiple_times(self) -> None:
        app_module = importlib.import_module("app")
        from app.core.config.settings import AppSettings

        first = app_module.create_app(
            AppSettings(pinecone_api_key="k1", cohere_api_key="k1", flask_secret_key="s1")
        )
        second = app_module.create_app(
            AppSettings(pinecone_api_key="k2", cohere_api_key="k2", flask_secret_key="s2")
        )

        self.assertIsNot(first, second)
        self.assertEqual(first.config["SECRET_KEY"], "s1")
        self.assertEqual(second.config["SECRET_KEY"], "s2")

    def test_auth_schema_validation(self) -> None:
        auth_module = importlib.import_module("app.api.schemas.auth")
        valid = auth_module.AuthRequest.from_request_payload({"email": "User@Example.com", "password": "secret123"})
        self.assertEqual(valid.email, "user@example.com")
        with self.assertRaises(auth_module.ValidationError):
            auth_module.AuthRequest.from_request_payload({"email": "bad-email", "password": "secret"})

    def test_signed_session_context(self) -> None:
        session_module = importlib.import_module("app.core.security.session_manager")
        manager = session_module.SessionManager("test-secret", "session_id", "guest_", 3600)
        auth_context = manager.create_auth_context("user-123", "access-token-abc")
        self.assertNotEqual(auth_context.cookie_value, "user-123")
        resolved = manager.resolve(auth_context.cookie_value)
        self.assertTrue(resolved.is_authenticated)
        self.assertEqual(resolved.user_id, "user-123")
        self.assertEqual(resolved.access_token, "access-token-abc")
