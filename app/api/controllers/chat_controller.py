from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from uuid import uuid4

from flask import g, jsonify, make_response, render_template, request, send_from_directory

from app.api.schemas.auth import AuthRequest
from app.core.config.settings import AppSettings
from app.core.security.exceptions import AppError, RepositoryError, ValidationError
from app.services.audit_service import AuditService
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.services.topic_service import TopicService
from app.services.summary_service import SummaryService
from app.services.flashcard_service import FlashcardService
from app.services.quiz_service import QuizService
from app.services.study_tools_service import StudyToolsService
from app.services.analytics_service import AnalyticsService
from app.services.document_service import DocumentService
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.profile_repository import ProfileRepository
from app.rag.vector_store import get_pinecone_client


logger = logging.getLogger(__name__)
_PINECONE_HEALTH_CACHE: dict[str, tuple[float, bool]] = {}
_PINECONE_HEALTH_CACHE_TTL_SECONDS = 30.0

# JWT structural regex — three base64url segments separated by dots.
_JWT_SEGMENT_RE = re.compile(r'^[A-Za-z0-9_\-]+=*$')
_JWT_MAX_LENGTH = 2048


def _validate_jwt_structure(token: str) -> bool:
    """Return True only if *token* looks like a well-formed JWT.

    Checks:
    - Length ≤ 2048 characters
    - Exactly three dot-separated segments
    - Each segment is non-empty and contains only URL-safe Base64 chars

    This is a structural check only — it does NOT verify the signature.
    """
    if not token or len(token) > _JWT_MAX_LENGTH:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    return all(part and _JWT_SEGMENT_RE.match(part) for part in parts)


def _probe_pinecone_health(api_key: str, cache_ttl_seconds: float = _PINECONE_HEALTH_CACHE_TTL_SECONDS) -> bool:
    cached = _PINECONE_HEALTH_CACHE.get(api_key)
    now = time.monotonic()
    if cached and (now - cached[0]) < cache_ttl_seconds:
        return cached[1]

    try:
        import httpx
        httpx.get(
            "https://api.pinecone.io/indexes",
            headers={"Api-Key": api_key},
            timeout=3.0,
        )
        healthy = True
    except Exception:
        healthy = False

    _PINECONE_HEALTH_CACHE[api_key] = (now, healthy)
    return healthy


class ChatController:
    def __init__(
        self,
        chat_service: ChatService,
        auth_service: AuthService,
        settings: AppSettings,
        memory_service: MemoryService | None = None,
        topic_service: TopicService | None = None,
        summary_service: SummaryService | None = None,
        audit_service: AuditService | None = None,
        flashcard_service: FlashcardService | None = None,
        quiz_service: QuizService | None = None,
        study_tools_service: StudyToolsService | None = None,
        analytics_service: AnalyticsService | None = None,
        document_service: DocumentService | None = None,
        conversation_repository: ConversationRepository | None = None,
        profile_repository: ProfileRepository | None = None,
    ) -> None:
        self.chat_service = chat_service
        self.auth_service = auth_service
        self.settings = settings
        self.memory_service = memory_service
        self.topic_service = topic_service
        self.summary_service = summary_service
        self.audit_service = audit_service
        self.flashcard_service = flashcard_service
        self.quiz_service = quiz_service
        self.study_tools_service = study_tools_service
        self.analytics_service = analytics_service
        self.document_service = document_service
        self.conversation_repository = conversation_repository
        self.profile_repository = profile_repository

    def _get_user_id_from_cookie(self) -> str:
        session_context = getattr(g, "session_context", None)
        if session_context:
            return session_context.user_id
        session_id = request.cookies.get(self.settings.session_cookie_name)
        return session_id or f"{self.settings.guest_session_prefix}{uuid4()}"

    def _require_auth(self) -> None:
        session_context = getattr(g, "session_context", None)
        if not session_context or not session_context.is_authenticated:
            raise AppError("Unauthorized", status_code=401, error_type="unauthorized")

    def home(self):
        session_context = getattr(g, "session_context", None)
        csrf_token = getattr(g, "csrf_token", "")
        spa_root = Path(__file__).resolve().parents[3] / "frontend" / "dist" / "medichat-frontend" / "browser"
        if spa_root.exists() and (spa_root / "index.html").exists():
            response = make_response(send_from_directory(spa_root, "index.html"))
        else:
            response = make_response("Angular SPA not built. Run 'npm run build' in the frontend directory.", 404)
        return response

    def chat(self):
        message = request.form.get("msg", "").strip()
        conversation_id = request.form.get("conversation_id", "").strip() or None
        if len(message) > self.settings.max_chat_message_length:
            raise ValidationError("Message is too long", details={"msg": "max length exceeded"})
        if not message:
            raise ValidationError("Please enter a valid question.")

        user_id = self._get_user_id_from_cookie()

        # IDOR guard — verify the caller owns the conversation before reading
        # its history into the prompt or writing new messages to it.
        if conversation_id and self.conversation_repository:
            if not self.conversation_repository.user_owns_conversation(conversation_id, user_id):
                raise AppError("Not found", status_code=404, error_type="not_found")

        answer = self.chat_service.get_answer(message, user_id, conversation_id)
        response = make_response(answer)
        return response

    def chat_stream(self):
        from flask import Response, stream_with_context
        
        # We can accept form encoded data or JSON
        payload = request.get_json(silent=True) or {}
        if not payload and request.form:
            payload = request.form.to_dict()

        message = payload.get("msg", "").strip()
        conversation_id = payload.get("conversation_id", "").strip() or None
        is_regenerate = payload.get("is_regenerate", "false").lower() == "true"

        if len(message) > self.settings.max_chat_message_length:
            raise ValidationError("Message is too long", details={"msg": "max length exceeded"})
        if not message:
            raise ValidationError("Please enter a valid question.")

        user_id = self._get_user_id_from_cookie()

        # IDOR guard — verify the caller owns the conversation before reading
        # its history into the prompt or writing new messages to it.
        if conversation_id and self.conversation_repository:
            if not self.conversation_repository.user_owns_conversation(conversation_id, user_id):
                logger.info("User %s attempted to use unowned conversation %s; starting a new one.", user_id, conversation_id)
                conversation_id = None

        def generate():
            yield from self.chat_service.get_answer_stream(message, user_id, conversation_id, is_regenerate)
            
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    def signup(self):
        if not self.settings.persistence_enabled:
            logger.warning("Signup requested while persistence is disabled")
            return jsonify({"error": "persistence is disabled"}), 503

        payload = request.get_json(silent=True)
        if payload is None and request.form:
            payload = request.form.to_dict()

        try:
            auth_request = AuthRequest.from_request_payload(payload)
            self.auth_service.sign_up(auth_request.email, auth_request.password, auth_request.display_name)
            if self.audit_service:
                self.audit_service.log(
                    "signup",
                    user_id=None,
                    remote_addr=request.remote_addr,
                    details={"email": auth_request.email},
                )
            return jsonify({"message": "Signup successful. Check your email for confirmation."}), 200
        except ValidationError as exc:
            logger.warning("Signup validation failed: %s, details: %s", exc, exc.details)
            return jsonify(exc.to_dict()), exc.status_code
        except AppError as exc:
            logger.warning("Signup failed: %s", exc)
            return jsonify(exc.to_dict()), exc.status_code
        except Exception as exc:
            logger.exception("Signup failed")
            return jsonify({"error": str(exc)}), 400

    def login(self):
        if not self.settings.persistence_enabled:
            logger.warning("Login requested while persistence is disabled")
            return jsonify({"error": "persistence is disabled"}), 503

        payload = request.get_json(silent=True)
        if payload is None and request.form:
            payload = request.form.to_dict()

        try:
            auth_request = AuthRequest.from_request_payload(payload)
            auth_session = self.auth_service.sign_in(auth_request.email, auth_request.password)
            g.session_context = self.auth_service.session_manager.create_auth_context(auth_session.user_id, auth_session.access_token)
            g.remember_me = payload.get("remember_me", False) if isinstance(payload, dict) else False
            if self.audit_service:
                self.audit_service.log(
                    "login",
                    user_id=auth_session.user_id,
                    remote_addr=request.remote_addr,
                    details={"email": auth_request.email},
                )
            response = make_response(jsonify({"message": "Login successful"}))
            return response
        except ValidationError as exc:
            logger.warning("Login validation failed: %s, details: %s", exc, exc.details)
            if self.audit_service:
                self.audit_service.log(
                    "login_failed",
                    user_id=None,
                    remote_addr=request.remote_addr,
                    details={"reason": "validation_error"},
                )
            return jsonify(exc.to_dict()), exc.status_code
        except AppError as exc:
            logger.warning("Login failed: %s", exc)
            if self.audit_service:
                self.audit_service.log(
                    "login_failed",
                    user_id=None,
                    remote_addr=request.remote_addr,
                    details={"reason": exc.error_type},
                )
            return jsonify(exc.to_dict()), exc.status_code
        except Exception as exc:
            logger.exception("Login failed")
            return jsonify({"error": str(exc)}), 400

    def google_login(self):
        from flask import redirect
        import os
        render_url = os.environ.get("RENDER_EXTERNAL_URL")
        if render_url:
            redirect_url = f"{render_url}/api/auth/callback"
        else:
            host = request.headers.get("X-Forwarded-Host", request.headers.get("Host", request.host))
            scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
            redirect_url = f"{scheme}://{host}/api/auth/callback"
            
        logger.warning(f"GENERATED REDIRECT URL: {redirect_url}")
        try:
            url = self.auth_service.get_google_oauth_url(redirect_url)
            return redirect(url)
        except Exception as exc:
            logger.exception("Failed to get Google OAuth URL")
            return jsonify({"error": str(exc)}), 500

    def google_callback(self):
        from flask import redirect
        code = request.args.get("code")
        if not code:
            return jsonify({"error": "No code provided"}), 400
        try:
            auth_session = self.auth_service.exchange_oauth_code(code)
            g.session_context = self.auth_service.session_manager.create_auth_context(auth_session.user_id, auth_session.access_token)
            g.remember_me = True  # OAuth sessions usually default to remembered
            if self.audit_service:
                self.audit_service.log(
                    "login_google",
                    user_id=auth_session.user_id,
                    remote_addr=request.remote_addr,
                )
            frontend_url = self.settings.frontend_origins[0] if self.settings.frontend_origins else request.host_url.rstrip('/')
            response = make_response(redirect(f"{frontend_url}/app/chat"))
            return response
        except Exception as exc:
            logger.exception("Failed to exchange Google OAuth code")
            return redirect("/auth/login?error=oauth_failed")


    def logout(self):
        if not self.settings.persistence_enabled:
            guest_context = self.auth_service.session_manager.create_guest_cookie()
            response = make_response(jsonify({"message": "Logged out"}))
            return response

        try:
            self.auth_service.sign_out()
        except RepositoryError as exc:
            logger.warning("Logout fallback: %s", exc)
        session_context = getattr(g, "session_context", None)
        if self.audit_service:
            user_id = session_context.user_id if session_context else None
            self.audit_service.log(
                "logout",
                user_id=user_id,
                remote_addr=request.remote_addr,
            )
        guest_context = self.auth_service.session_manager.create_guest_cookie()
        g.session_context = guest_context
        response = make_response(jsonify({"message": "Logged out"}))
        return response

    def reset_password_request(self):
        if not self.settings.persistence_enabled:
            logger.warning("Password reset requested while persistence is disabled")
            return jsonify({"error": "persistence is disabled"}), 503

        payload = request.get_json(silent=True)
        if payload is None and request.form:
            payload = request.form.to_dict()

        email = (payload or {}).get("email", "").strip()
        if not email:
            return jsonify({"error": "Email is required"}), 400

        try:
            origin = request.headers.get("Origin")
            if not origin and self.settings.frontend_origins:
                origin = self.settings.frontend_origins[0]
            if not origin:
                origin = request.host_url.rstrip("/")

            redirect_url = f"{origin}/auth/reset-password"
            self.auth_service.reset_password(email, redirect_url)
            if self.audit_service:
                self.audit_service.log(
                    "password_reset_request",
                    user_id=None,
                    remote_addr=request.remote_addr,
                    details={"email": email},
                )
            return jsonify({"message": "Password reset link sent to your email"}), 200
        except AppError as exc:
            logger.warning("Password reset request failed: %s", exc)
            return jsonify(exc.to_dict()), exc.status_code
        except Exception as exc:
            logger.exception("Password reset request failed")
            return jsonify({"error": str(exc)}), 400

    def reset_password(self):
        if not self.settings.persistence_enabled:
            logger.warning("Password reset requested while persistence is disabled")
            return jsonify({"error": "persistence is disabled"}), 503

        payload = request.get_json(silent=True)
        if payload is None and request.form:
            payload = request.form.to_dict()

        access_token = (payload or {}).get("access_token", "").strip()
        new_password = (payload or {}).get("password", "").strip()

        if not access_token:
            return jsonify({"error": "Access token is required"}), 400
        # Issue #2 — structural JWT validation before hitting the database
        if not _validate_jwt_structure(access_token):
            logger.warning(
                "reset_password: malformed access_token from remote_addr=%s",
                request.remote_addr,
            )
            return jsonify({"error": "Invalid access token format"}), 400
        if not new_password or len(new_password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400

        try:
            self.auth_service.update_password(access_token, new_password)
            # Issue #4 — invalidate the reset token and force re-login
            self.auth_service.sign_out_with_token(access_token)
            if self.audit_service:
                session_context = getattr(g, "session_context", None)
                self.audit_service.log(
                    "password_changed",
                    user_id=session_context.user_id if session_context else None,
                    remote_addr=request.remote_addr,
                )
            response = make_response(
                jsonify({"message": "Password updated successfully. Please log in again."})
            )
            return response
        except AppError as exc:
            logger.warning("Password update failed: %s", exc)
            return jsonify(exc.to_dict()), exc.status_code
        except Exception as exc:
            logger.exception("Password update failed")
            return jsonify({"error": str(exc)}), 400


    def health(self):
        checks: dict[str, str] = {}
        all_healthy = True
        timeout = 3.0

        # --- Pinecone ---
        try:
            if _probe_pinecone_health(self.settings.pinecone_api_key):
                checks["pinecone"] = "ok"
            else:
                raise RuntimeError("pinecone health check failed")
        except Exception as exc:
            logger.warning("Health check: Pinecone unreachable: %s", exc)
            checks["pinecone"] = "error"
            all_healthy = False

        # --- Cohere ---
        try:
            import httpx
            resp = httpx.get("https://api.cohere.com", timeout=timeout)
            checks["cohere"] = "ok" if resp.status_code < 500 else "error"
            if checks["cohere"] == "error":
                all_healthy = False
        except Exception as exc:
            logger.warning("Health check: Cohere unreachable: %s", exc)
            checks["cohere"] = "error"
            all_healthy = False

        # --- Supabase (only when persistence is enabled) ---
        if self.settings.persistence_enabled and self.settings.supabase_url:
            try:
                import httpx
                resp = httpx.get(
                    f"{self.settings.supabase_url}/rest/v1/",
                    headers={"apikey": self.settings.supabase_key or ""},
                    timeout=timeout,
                )
                checks["supabase"] = "ok" if resp.status_code < 500 else "error"
                if checks["supabase"] == "error":
                    all_healthy = False
            except Exception as exc:
                logger.warning("Health check: Supabase unreachable: %s", exc)
                checks["supabase"] = "error"
                all_healthy = False
        else:
            checks["supabase"] = "disabled"

        status = "healthy" if all_healthy else "degraded"
        http_status = 200 if all_healthy else 503
        return jsonify({"status": status, "service": "MediChat API", "checks": checks}), http_status

    def spa(self, path: str):
        spa_root = Path(__file__).resolve().parents[3] / "frontend" / "dist" / "medichat-frontend" / "browser"
        if not spa_root.exists():
            return jsonify({"error": "frontend app not built"}), 404

        candidate = spa_root / path
        if path and candidate.exists() and candidate.is_file():
            return send_from_directory(spa_root, path)
        if not (spa_root / "index.html").exists():
            return jsonify({"error": "frontend app not built"}), 404
        return send_from_directory(spa_root, "index.html")

    # ------------------------------------------------------------------
    # Memory endpoints
    # ------------------------------------------------------------------

    def get_topic_memory(self):
        """GET /api/memory/topic — returns current_topic and related_topics."""
        session_id = self._get_user_id_from_cookie()
        if not session_id or not self.topic_service:
            return jsonify({"current_topic": None, "related_topics": []}), 200

        try:
            topic = self.topic_service.get_topic(session_id)
            if topic:
                return jsonify(topic), 200
            return jsonify({"current_topic": None, "related_topics": []}), 200
        except Exception as exc:
            logger.exception("Failed to fetch topic memory")
            return jsonify({"error": str(exc)}), 500

    def get_session_summary(self):
        """GET /api/memory/summary — returns the rolling study summary."""
        session_id = self._get_user_id_from_cookie()
        if not session_id or not self.summary_service:
            return jsonify({"summary": None}), 200

        try:
            summary = self.summary_service.get_summary(session_id)
            return jsonify({"summary": summary}), 200
        except Exception as exc:
            logger.exception("Failed to fetch session summary")
            return jsonify({"error": str(exc)}), 500


    # ------------------------------------------------------------------
    # Conversations API
    # ------------------------------------------------------------------

    def list_conversations(self):
        user_id = self._get_user_id_from_cookie()
        if not self.conversation_repository:
            return jsonify({"error": "Persistence disabled"}), 503
        limit = min(int(request.args.get("limit", 30)), 100)
        offset = int(request.args.get("offset", 0))
        data = self.conversation_repository.list_conversations(user_id, limit=limit, offset=offset)
        return jsonify(data)

    def delete_conversation(self, conversation_id: str):
        user_id = self._get_user_id_from_cookie()
        if not self.conversation_repository:
            return jsonify({"error": "Persistence disabled"}), 503
        deleted_count = self.conversation_repository.delete_conversation(conversation_id, user_id)
        if not deleted_count:
            raise AppError("Not found", status_code=404, error_type="not_found")
        return jsonify({"message": "Conversation deleted"})

    def get_conversation_messages(self, conversation_id: str):
        user_id = self._get_user_id_from_cookie()
        if not self.memory_service or not self.conversation_repository:
            return jsonify({"error": "Persistence disabled"}), 503
        if not self.conversation_repository.user_owns_conversation(conversation_id, user_id):
            raise AppError("Not found", status_code=404, error_type="not_found")
        # Use limit=50 or similar for the history
        messages = self.memory_service.get_recent_messages(conversation_id, limit=50)
        return jsonify(messages)

    def delete_message(self, conversation_id: str, message_id: str):
        user_id = self._get_user_id_from_cookie()
        if not self.conversation_repository or not self.memory_service:
            return jsonify({"error": "Persistence disabled"}), 503
        if not self.conversation_repository.user_owns_conversation(conversation_id, user_id):
            raise AppError("Not found", status_code=404, error_type="not_found")
        
        try:
            self.memory_service.chat_history_repository.delete_message(conversation_id, user_id, message_id)
            if self.audit_service:
                self.audit_service.log(
                    "message_deleted", user_id=user_id, remote_addr=request.remote_addr,
                    details={"conversation_id": conversation_id, "message_id": message_id}
                )
            return jsonify({"status": "success"})
        except Exception:
            logger.exception("Failed to delete message %s", message_id)
            return jsonify({"error": "Failed to delete message"}), 500

    def rate_message(self, conversation_id: str, message_id: str):
        user_id = self._get_user_id_from_cookie()
        if not self.conversation_repository or not self.memory_service:
            return jsonify({"error": "Persistence disabled"}), 503
        if not self.conversation_repository.user_owns_conversation(conversation_id, user_id):
            raise AppError("Not found", status_code=404, error_type="not_found")
            
        payload = request.get_json(silent=True) or {}
        liked = payload.get("liked")
        if liked is None:
            return jsonify({"error": "Missing 'liked' boolean"}), 400
            
        try:
            self.memory_service.chat_history_repository.rate_message(conversation_id, user_id, message_id, bool(liked))
            if self.audit_service:
                self.audit_service.log(
                    "message_rated", user_id=user_id, remote_addr=request.remote_addr,
                    details={"conversation_id": conversation_id, "message_id": message_id, "liked": liked}
                )
            return jsonify({"status": "success"})
        except Exception:
            logger.exception("Failed to rate message %s", message_id)
            return jsonify({"error": "Failed to rate message"}), 500

    # ------------------------------------------------------------------
    # Flashcards API
    # ------------------------------------------------------------------

    def list_flashcard_decks(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.flashcard_service:
            return jsonify({"error": "Service unavailable"}), 503
        limit = min(int(request.args.get("limit", 30)), 100)
        return jsonify(self.flashcard_service.list_decks(user_id, limit=limit))

    def generate_flashcard_deck(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.flashcard_service:
            return jsonify({"error": "Service unavailable"}), 503
            
        payload = request.get_json() or {}
        topic = payload.get("topic")
        count = payload.get("count", 5)
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
            
        deck_id = self.flashcard_service.generate_deck(user_id, topic, count, self.chat_service.llm)
        return jsonify({"message": "Deck generated", "deck_id": deck_id})

    def get_flashcard_deck(self, deck_id: str):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.flashcard_service:
            return jsonify({"error": "Service unavailable"}), 503
        
        try:
            deck = self.flashcard_service.get_deck(deck_id, user_id)
            if deck:
                return jsonify(deck)
            return jsonify({"error": "Deck not found"}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    def rate_flashcard(self, deck_id: str, card_id: str):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()

        if not self.flashcard_service or not self.flashcard_service.repository:
            return jsonify({"error": "Persistence disabled"}), 503

        data = request.get_json() or {}
        rating = data.get("rating")
        if rating not in ["known", "unknown"]:
            return jsonify({"error": "Invalid rating. Must be 'known' or 'unknown'"}), 400

        try:
            # Verify deck ownership via the service repository
            self.flashcard_service.repository.get_deck(deck_id, user_id)
            self.flashcard_service.repository.rate_card(deck_id, card_id, rating)
            return jsonify({"success": True})
        except Exception as exc:
            logger.exception("Failed to rate flashcard card_id=%s deck_id=%s", card_id, deck_id)
            return jsonify({"error": "Failed to save rating"}), 500

    # ------------------------------------------------------------------
    # Quizzes API
    # ------------------------------------------------------------------

    def list_quiz_sessions(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.quiz_service:
            return jsonify({"error": "Service unavailable"}), 503
        limit = min(int(request.args.get("limit", 30)), 100)
        return jsonify(self.quiz_service.list_sessions(user_id, limit=limit))

    def generate_quiz(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.quiz_service:
            return jsonify({"error": "Service unavailable"}), 503
            
        payload = request.get_json() or {}
        topic = payload.get("topic")
        count = payload.get("count", 5)
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
            
        session_id = self.quiz_service.generate_quiz(user_id, topic, count, self.chat_service.llm)
        return jsonify({"message": "Quiz generated", "session_id": session_id})

    def get_quiz_session(self, session_id: str):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.quiz_service:
            return jsonify({"error": "Service unavailable"}), 503
            
        try:
            session = self.quiz_service.get_session(session_id, user_id)
            if session:
                return jsonify(session)
            return jsonify({"error": "Quiz not found"}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    def submit_quiz_score(self, session_id: str):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.quiz_service:
            return jsonify({"error": "Service unavailable"}), 503
            
        payload = request.get_json() or {}
        answers = payload.get("answers")
        if not isinstance(answers, list):
            return jsonify({"error": "Answers are required"}), 400

        score = self.quiz_service.grade_answers(session_id, user_id, answers)
        return jsonify({"message": "Score submitted", "score": score})

    # ------------------------------------------------------------------
    # Study Tools API
    # ------------------------------------------------------------------

    def explain_topic(self):
        if not self.study_tools_service:
            return jsonify({"error": "Service unavailable"}), 503
        topic = (request.get_json() or {}).get("topic")
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        answer = self.study_tools_service.explain(topic, self.chat_service.llm)
        return jsonify({"result": answer})

    def summarize_text(self):
        if not self.study_tools_service:
            return jsonify({"error": "Service unavailable"}), 503
        text = (request.get_json() or {}).get("text")
        if not text:
            return jsonify({"error": "Text is required"}), 400
        answer = self.study_tools_service.summarize(text, self.chat_service.llm)
        return jsonify({"result": answer})

    def generate_mnemonics(self):
        if not self.study_tools_service:
            return jsonify({"error": "Service unavailable"}), 503
        topic = (request.get_json() or {}).get("topic")
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        answer = self.study_tools_service.generate_mnemonics(topic, self.chat_service.llm)
        return jsonify({"result": answer})

    # ------------------------------------------------------------------
    # Profile & Analytics API
    # ------------------------------------------------------------------

    def get_profile(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.profile_repository:
            return jsonify({"error": "Service unavailable"}), 503
        return jsonify(self.profile_repository.get_profile(user_id))

    def update_profile(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.profile_repository:
            return jsonify({"error": "Service unavailable"}), 503
        data = request.get_json() or {}
        profile = self.profile_repository.upsert_profile(user_id, data)
        return jsonify({"message": "Profile updated", "profile": profile})

    def get_study_stats(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.analytics_service:
            return jsonify({"error": "Service unavailable"}), 503
        return jsonify(self.analytics_service.get_study_stats(user_id))

    def get_dashboard_stats(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.analytics_service or not self.conversation_repository:
            return jsonify({"error": "Service unavailable"}), 503
            
        stats = self.analytics_service.get_study_stats(user_id)
        recent_conversations = self.conversation_repository.list_conversations(user_id, limit=3)
        return jsonify({
            "stats": stats,
            "recent_conversations": recent_conversations
        })

    # ------------------------------------------------------------------
    # Document Upload API
    # ------------------------------------------------------------------

    def upload_document(self):
        self._require_auth()
        user_id = self._get_user_id_from_cookie()
        if not self.document_service:
            return jsonify({"error": "Service unavailable"}), 503
            
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected for uploading"}), 400
            
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400
            
        file_bytes = file.read()
        result = self.document_service.process_upload(file_bytes, file.filename, self.settings, user_id)
        return jsonify(result)
