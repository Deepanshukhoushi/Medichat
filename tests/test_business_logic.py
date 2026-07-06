"""
Business logic unit tests.

These tests exercise pure-Python components with no network calls.
Coverage:
  - RAG retrieval relevance filtering (zero results, below-threshold, above-threshold)
  - CSRF validation edge cases
  - Rate-limiter boundary conditions (exact limit, over-limit, window reset)
  - SessionManager: signing, tamper detection, expiry, guest/auth round-trips
  - SessionHistoryStore: TTL expiry, set/get round-trip, unknown-key miss
  - ChatService.get_answer routing (relevant docs path, zero docs path, low-score path)
"""
from __future__ import annotations

import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from tests.stubs import setup_stubs
setup_stubs()




# ---------------------------------------------------------------------------
# 1. Retrieval relevance filtering
# ---------------------------------------------------------------------------

class TestRelevanceFiltering(unittest.TestCase):
    """is_relevant_score controls which retrieved docs are sent to the LLM."""

    def setUp(self):
        from app.rag.retrieval import is_relevant_score
        self.is_relevant = is_relevant_score
        self.threshold = 0.55

    def test_score_above_threshold_is_relevant(self):
        self.assertTrue(self.is_relevant(0.9, self.threshold))
        self.assertTrue(self.is_relevant(0.55, self.threshold))   # boundary: exactly at threshold

    def test_score_below_threshold_is_not_relevant(self):
        self.assertFalse(self.is_relevant(0.54, self.threshold))
        self.assertFalse(self.is_relevant(0.0, self.threshold))

    def test_none_score_is_not_relevant(self):
        self.assertFalse(self.is_relevant(None, self.threshold))

    def test_zero_docs_means_empty_relevant_list(self):
        """If retrieve_documents_with_scores returns nothing, relevant_documents is empty."""
        docs_with_scores = []
        relevant = [doc for doc, score in docs_with_scores
                    if self.is_relevant(score, self.threshold)]
        self.assertEqual(relevant, [])

    def test_all_low_score_docs_produces_empty_relevant_list(self):
        fake_doc = types.SimpleNamespace(page_content="some content")
        docs_with_scores = [(fake_doc, 0.1), (fake_doc, 0.3), (fake_doc, 0.0)]
        relevant = [doc for doc, score in docs_with_scores
                    if self.is_relevant(score, self.threshold)]
        self.assertEqual(relevant, [])

    def test_mixed_scores_filters_correctly(self):
        hi = types.SimpleNamespace(page_content="relevant")
        lo = types.SimpleNamespace(page_content="irrelevant")
        docs_with_scores = [(hi, 0.8), (lo, 0.3), (hi, 0.6)]
        relevant = [doc for doc, score in docs_with_scores
                    if self.is_relevant(score, self.threshold)]
        self.assertEqual(len(relevant), 2)
        self.assertNotIn(lo, relevant)


# ---------------------------------------------------------------------------
# 2. CSRF manager
# ---------------------------------------------------------------------------

class TestCsrfManager(unittest.TestCase):
    def setUp(self):
        from app.core.security.csrf import CsrfManager
        self.csrf = CsrfManager("csrf_token", "X-CSRF-Token")

    def test_generated_token_is_nonempty_string(self):
        token = self.csrf.generate()
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 16)

    def test_matching_tokens_are_valid(self):
        token = self.csrf.generate()
        self.assertTrue(self.csrf.is_valid(token, token))

    def test_mismatched_tokens_are_invalid(self):
        t1 = self.csrf.generate()
        t2 = self.csrf.generate()
        self.assertNotEqual(t1, t2)
        self.assertFalse(self.csrf.is_valid(t1, t2))

    def test_none_cookie_token_is_invalid(self):
        token = self.csrf.generate()
        self.assertFalse(self.csrf.is_valid(None, token))

    def test_none_header_token_is_invalid(self):
        token = self.csrf.generate()
        self.assertFalse(self.csrf.is_valid(token, None))

    def test_both_none_is_invalid(self):
        self.assertFalse(self.csrf.is_valid(None, None))

    def test_empty_string_tokens_are_invalid(self):
        self.assertFalse(self.csrf.is_valid("", ""))

    def test_partial_match_is_invalid(self):
        token = self.csrf.generate()
        self.assertFalse(self.csrf.is_valid(token, token[:-1]))


# ---------------------------------------------------------------------------
# 3. Rate limiter (in-process backend)
# ---------------------------------------------------------------------------

class TestRateLimiter(unittest.TestCase):
    """Tests the in-process fallback path (no Redis)."""

    def _make_limiter(self):
        from app.core.security.rate_limiter import RateLimiter
        return RateLimiter(redis_url=None)  # always use in-process store

    def test_requests_within_limit_are_allowed(self):
        rl = self._make_limiter()
        for _ in range(3):
            self.assertTrue(rl.check("key:within", limit=3, window_seconds=60))

    def test_request_exactly_at_limit_is_allowed(self):
        rl = self._make_limiter()
        # first (limit - 1) calls are definitely allowed; the limit-th should also be allowed
        for _ in range(4):
            rl.check("key:boundary", limit=5, window_seconds=60)
        # exactly the 5th call
        result = rl.check("key:boundary", limit=5, window_seconds=60)
        self.assertTrue(result)

    def test_request_over_limit_is_rejected(self):
        rl = self._make_limiter()
        limit = 3
        key = "key:overlimit"
        for _ in range(limit):
            rl.check(key, limit=limit, window_seconds=60)
        # one beyond the limit must be rejected
        self.assertFalse(rl.check(key, limit=limit, window_seconds=60))

    def test_different_keys_are_independent(self):
        rl = self._make_limiter()
        for _ in range(5):
            rl.check("key:a", limit=2, window_seconds=60)
        # key:a is exhausted; key:b should be unaffected
        self.assertTrue(rl.check("key:b", limit=2, window_seconds=60))

    def test_window_reset_allows_new_requests(self):
        rl = self._make_limiter()
        key = "key:window"
        limit = 2
        for _ in range(limit + 1):
            rl.check(key, limit=limit, window_seconds=0)  # window of 0 s expires instantly

        # After expiry the counter resets — next call inside a new window should pass
        time.sleep(0.01)
        self.assertTrue(rl.check(key, limit=limit, window_seconds=60))


# ---------------------------------------------------------------------------
# 4. SessionManager: signing, tampering, expiry, round-trips
# ---------------------------------------------------------------------------

class TestSessionManager(unittest.TestCase):
    def _make_manager(self, max_age=3600):
        from app.core.security.session_manager import SessionManager
        return SessionManager("test-secret", "session_id", "guest_", max_age)

    # --- Guest sessions ---

    def test_guest_cookie_round_trip(self):
        mgr = self._make_manager()
        ctx = mgr.create_guest_cookie()
        resolved = mgr.resolve(ctx.cookie_value)
        self.assertFalse(resolved.is_authenticated)
        self.assertEqual(resolved.user_id, ctx.user_id)

    def test_missing_cookie_produces_guest_context(self):
        mgr = self._make_manager()
        ctx = mgr.resolve(None)
        self.assertFalse(ctx.is_authenticated)
        self.assertTrue(ctx.user_id.startswith("guest_"))

    def test_empty_cookie_produces_guest_context(self):
        mgr = self._make_manager()
        ctx = mgr.resolve("")
        self.assertFalse(ctx.is_authenticated)

    # --- Auth sessions ---

    def test_auth_cookie_round_trip(self):
        mgr = self._make_manager()
        ctx = mgr.create_auth_context("user-42", "tok-abc")
        resolved = mgr.resolve(ctx.cookie_value)
        self.assertTrue(resolved.is_authenticated)
        self.assertEqual(resolved.user_id, "user-42")
        self.assertEqual(resolved.access_token, "tok-abc")

    def test_cookie_value_is_not_plaintext(self):
        mgr = self._make_manager()
        ctx = mgr.create_auth_context("user-42", "tok-abc")
        self.assertNotIn("user-42", ctx.cookie_value)
        self.assertNotIn("tok-abc", ctx.cookie_value)

    # --- Tamper resistance ---

    def test_tampered_signature_falls_back_to_guest(self):
        mgr = self._make_manager()
        ctx = mgr.create_auth_context("user-42", "tok-abc")
        parts = ctx.cookie_value.split(".")
        # tamper with the first character of the signature (the third part)
        tampered_sig = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
        tampered = f"{parts[0]}.{parts[1]}.{tampered_sig}"
        resolved = mgr.resolve(tampered)
        self.assertFalse(resolved.is_authenticated)

    def test_wrong_key_falls_back_to_guest(self):
        mgr1 = self._make_manager()
        mgr2_different_key = type(mgr1).__new__(type(mgr1))
        from app.core.security.session_manager import SessionManager
        mgr2 = SessionManager("different-secret", "session_id", "guest_", 3600)
        ctx = mgr1.create_auth_context("user-42", "tok-abc")
        resolved = mgr2.resolve(ctx.cookie_value)
        self.assertFalse(resolved.is_authenticated)

    def test_garbage_cookie_falls_back_to_guest(self):
        mgr = self._make_manager()
        resolved = mgr.resolve("this.is.garbage")
        self.assertFalse(resolved.is_authenticated)

    # --- Expiry ---

    def test_expired_session_falls_back_to_guest(self):
        mgr = self._make_manager(max_age=-1)   # any token will be older than -1 s
        ctx = mgr.create_auth_context("user-42", "tok-abc")
        resolved = mgr.resolve(ctx.cookie_value)
        self.assertFalse(resolved.is_authenticated)

    def test_valid_session_within_max_age_resolves_correctly(self):
        mgr = self._make_manager(max_age=60)
        ctx = mgr.create_auth_context("user-99", "tok-xyz")
        resolved = mgr.resolve(ctx.cookie_value)
        self.assertTrue(resolved.is_authenticated)
        self.assertEqual(resolved.user_id, "user-99")


# ---------------------------------------------------------------------------
# 5. SessionHistoryStore (in-process backend)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 6. ChatService.get_answer routing (mocked dependencies)
# ---------------------------------------------------------------------------

class TestChatServiceRouting(unittest.TestCase):
    """Verifies that get_answer routes to the correct LLM path depending on
    whether relevant documents are found."""

    def _make_settings(self):
        from app.core.config.settings import AppSettings
        return AppSettings(
            pinecone_api_key="k",
            cohere_api_key="k",
            flask_secret_key="s",
            persistence_enabled=False,
            relevance_score_threshold=0.55,
            indexed_answer_suffix="[indexed]",
            general_answer_suffix="[general]",
        )

    def _make_service(self, settings=None):
        import importlib
        # Import the module directly to avoid triggering app/__init__ (Flask construction)
        chat_svc_mod = importlib.import_module("app.services.chat_service")
        return chat_svc_mod.ChatService(
            settings=settings or self._make_settings(),
            conversation_repository=None,
            chat_history_repository=None,
        )

    def _patch_retrieve(self, return_value):
        import importlib
        chat_svc_mod = importlib.import_module("app.services.chat_service")
        return patch.object(chat_svc_mod, "retrieve_documents_with_scores", return_value=return_value)

    def test_zero_retrieved_docs_uses_general_llm_path(self):
        svc = self._make_service()
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = types.SimpleNamespace(content="general answer")

        with self._patch_retrieve([]), \
             patch.object(type(svc), "llm", new_callable=lambda: property(lambda self: fake_llm)):
            answer = svc.get_answer("What is aspirin?", "guest_abc")

        fake_llm.invoke.assert_called_once()
        self.assertIn("[general]", answer)
        self.assertNotIn("[indexed]", answer)

    def test_all_low_score_docs_uses_general_llm_path(self):
        svc = self._make_service()
        fake_doc = types.SimpleNamespace(page_content="irrelevant content")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = types.SimpleNamespace(content="general answer")

        low_score_results = [(fake_doc, 0.1), (fake_doc, 0.2)]
        with self._patch_retrieve(low_score_results), \
             patch.object(type(svc), "llm", new_callable=lambda: property(lambda self: fake_llm)):
            answer = svc.get_answer("What is aspirin?", "guest_abc")

        self.assertIn("[general]", answer)

    def test_relevant_docs_uses_qa_chain_path(self):
        svc = self._make_service()
        fake_doc = types.SimpleNamespace(page_content="aspirin reduces fever")
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = {"answer": "aspirin helps with fever"}

        high_score_results = [(fake_doc, 0.9)]
        with self._patch_retrieve(high_score_results), \
             patch.object(type(svc), "qa_chain", new_callable=lambda: property(lambda self: fake_chain)):
            answer = svc.get_answer("What does aspirin do?", "guest_abc")

        fake_chain.invoke.assert_called_once()
        self.assertIn("[indexed]", answer)
        self.assertNotIn("[general]", answer)

    def test_chain_dict_response_with_answer_key(self):
        svc = self._make_service()
        fake_doc = types.SimpleNamespace(page_content="content")
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = {"answer": "  my answer  "}

        with self._patch_retrieve([(fake_doc, 0.9)]), \
             patch.object(type(svc), "qa_chain", new_callable=lambda: property(lambda self: fake_chain)):
            answer = svc.get_answer("q", "user_1")

        self.assertIn("my answer", answer)

    def test_chain_object_with_content_attribute(self):
        svc = self._make_service()
        fake_doc = types.SimpleNamespace(page_content="content")
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = types.SimpleNamespace(content="object answer")

        with self._patch_retrieve([(fake_doc, 0.9)]), \
             patch.object(type(svc), "qa_chain", new_callable=lambda: property(lambda self: fake_chain)):
            answer = svc.get_answer("q", "user_1")

        self.assertIn("object answer", answer)

    def test_chain_failure_falls_back_to_direct_llm(self):
        svc = self._make_service()
        fake_doc = types.SimpleNamespace(page_content="ctx content")
        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = RuntimeError("chain exploded")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = types.SimpleNamespace(content="fallback answer")

        with self._patch_retrieve([(fake_doc, 0.9)]), \
             patch.object(type(svc), "qa_chain", new_callable=lambda: property(lambda self: fake_chain)), \
             patch.object(type(svc), "llm", new_callable=lambda: property(lambda self: fake_llm)):
            answer = svc.get_answer("q", "user_1")

        # Chain failed → LLM direct call → indexed suffix still applied
        fake_llm.invoke.assert_called_once()
        self.assertIn("[indexed]", answer)


class TestChatServiceStreamPersistence(unittest.TestCase):
    def _make_settings(self):
        from app.core.config.settings import AppSettings
        return AppSettings(
            pinecone_api_key="k",
            cohere_api_key="k",
            flask_secret_key="s",
            persistence_enabled=False,
            relevance_score_threshold=0.55,
            indexed_answer_suffix="[indexed]",
            general_answer_suffix="[general]",
        )

    def test_partial_stream_is_persisted_on_disconnect(self):
        import importlib

        chat_svc_mod = importlib.import_module("app.services.chat_service")
        memory_service = MagicMock()
        memory_service.get_recent_messages.return_value = []
        memory_service.get_message_count.return_value = 2
        svc = chat_svc_mod.ChatService(
            settings=self._make_settings(),
            conversation_repository=MagicMock(),
            chat_history_repository=MagicMock(),
            memory_service=memory_service,
        )
        svc.conversation_service.get_or_create_conversation = MagicMock(return_value="conv-1")
        svc.topic_service = None
        svc.summary_service = None

        with patch.object(chat_svc_mod, "retrieve_documents_with_scores", return_value=[]), \
             patch.object(svc, "_generate_answer_stream", return_value=iter(["part one ", "part two"])), \
             patch.object(svc, "_run_background_memory_tasks") as mock_bg:
            stream = svc.get_answer_stream("What is pneumonia?", "student-1")

            first_event = next(stream)
            second_event = next(stream)

            self.assertIn("conversation_id", first_event)
            self.assertIn("token", second_event)

            stream.close()

        memory_service.save_message.assert_any_call("conv-1", "student-1", "user", "What is pneumonia?")
        memory_service.save_message.assert_any_call("conv-1", "student-1", "assistant", "part one ")
        mock_bg.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Password Reset
# ---------------------------------------------------------------------------

class TestPasswordReset(unittest.TestCase):
    def setUp(self):
        from app.repositories.user_repository import UserRepository
        from app.services.auth_service import AuthService
        from app.core.config.settings import AppSettings

        self.mock_supabase = MagicMock()
        self.settings = AppSettings(
            pinecone_api_key="k",
            cohere_api_key="k",
            flask_secret_key="s",
            supabase_url="http://localhost",
            supabase_key="test-key",
            persistence_enabled=True,
        )
        self.user_repo = UserRepository(self.mock_supabase, self.settings)
        self.mock_session_mgr = MagicMock()
        self.auth_service = AuthService(self.user_repo, self.mock_session_mgr)

    def test_repository_reset_password_success(self):
        self.user_repo.reset_password("test@example.com", "http://redirect")
        self.mock_supabase.auth.reset_password_for_email.assert_called_once_with(
            "test@example.com",
            options={"redirect_to": "http://redirect"},
        )

    def test_repository_reset_password_failure(self):
        from app.core.security.exceptions import RepositoryError
        self.mock_supabase.auth.reset_password_for_email.side_effect = RuntimeError("network error")
        with self.assertRaises(RepositoryError):
            self.user_repo.reset_password("test@example.com", "http://redirect")

    @patch("supabase.create_client")
    def test_repository_update_password_success(self, mock_create_client):
        mock_temp_client = MagicMock()
        mock_create_client.return_value = mock_temp_client

        self.user_repo.update_password("access-token-123", "new-password-abc")

        mock_create_client.assert_called_once_with("http://localhost", "test-key")
        mock_temp_client.auth.set_session.assert_called_once_with(
            access_token="access-token-123", refresh_token=""
        )
        mock_temp_client.auth.update_user.assert_called_once_with(
            {"password": "new-password-abc"}
        )

    @patch("supabase.create_client")
    def test_repository_update_password_failure(self, mock_create_client):
        from app.core.security.exceptions import RepositoryError
        mock_temp_client = MagicMock()
        mock_create_client.return_value = mock_temp_client
        mock_temp_client.auth.update_user.side_effect = RuntimeError("update error")

        with self.assertRaises(RepositoryError):
            self.user_repo.update_password("access-token-123", "new-password-abc")

    def test_auth_service_reset_password(self):
        with patch.object(self.user_repo, "reset_password") as mock_reset:
            self.auth_service.reset_password("test@example.com", "http://redirect")
            mock_reset.assert_called_once_with("test@example.com", "http://redirect")

    def test_auth_service_update_password(self):
        with patch.object(self.user_repo, "update_password") as mock_update:
            self.auth_service.update_password("access-token-123", "new-password-abc")
            mock_update.assert_called_once_with("access-token-123", "new-password-abc")

    def test_repository_reset_password_rate_limit(self):
        from app.core.security.exceptions import AppError
        self.mock_supabase.auth.reset_password_for_email.side_effect = RuntimeError("429 Too Many Requests")
        with self.assertRaises(AppError) as ctx:
            self.user_repo.reset_password("test@example.com", "http://redirect")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("Too many password reset requests", ctx.exception.message)

    @patch("supabase.create_client")
    def test_repository_update_password_rate_limit(self, mock_create_client):
        from app.core.security.exceptions import AppError
        mock_temp_client = MagicMock()
        mock_create_client.return_value = mock_temp_client
        mock_temp_client.auth.update_user.side_effect = RuntimeError("email rate limit exceeded")

        with self.assertRaises(AppError) as ctx:
            self.user_repo.update_password("access-token-123", "new-password-abc")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("Too many requests to update password", ctx.exception.message)

    @patch("supabase.create_client")
    def test_repository_update_password_expired(self, mock_create_client):
        from app.core.security.exceptions import AuthenticationError
        mock_temp_client = MagicMock()
        mock_create_client.return_value = mock_temp_client
        mock_temp_client.auth.update_user.side_effect = RuntimeError("session expired or invalid token")

        with self.assertRaises(AuthenticationError) as ctx:
            self.user_repo.update_password("access-token-123", "new-password-abc")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid or expired password reset link", ctx.exception.message)

    def test_repository_signup_rate_limit(self):
        from app.core.security.exceptions import AppError
        self.mock_supabase.auth.sign_up.side_effect = RuntimeError("429 Too Many Requests")
        with self.assertRaises(AppError) as ctx:
            self.user_repo.sign_up("test@example.com", "password123")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("Too many signup requests", ctx.exception.message)

    def test_repository_signin_rate_limit(self):
        from app.core.security.exceptions import AppError
        self.mock_supabase.auth.sign_in_with_password.side_effect = RuntimeError("email rate limit exceeded")
        with self.assertRaises(AppError) as ctx:
            self.user_repo.sign_in("test@example.com", "password123")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("Too many login requests", ctx.exception.message)


