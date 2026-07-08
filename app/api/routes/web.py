from __future__ import annotations

from app.api.controllers.chat_controller import ChatController


def register_web_routes(application, controller: ChatController) -> None:
    application.add_url_rule("/", view_func=controller.home, methods=["GET"])
    application.add_url_rule("/get", view_func=controller.chat, methods=["POST"])
    application.add_url_rule("/signup", view_func=controller.signup, methods=["POST"])
    application.add_url_rule("/login", view_func=controller.login, methods=["POST"])
    application.add_url_rule("/logout", view_func=controller.logout, methods=["POST"])
    application.add_url_rule("/reset-password-request", view_func=controller.reset_password_request, methods=["POST"])
    application.add_url_rule("/reset-password", view_func=controller.reset_password, methods=["POST"])
    application.add_url_rule("/api/auth/google/login", view_func=controller.google_login, methods=["GET"])
    application.add_url_rule("/api/auth/callback", view_func=controller.google_callback, methods=["GET"])
    application.add_url_rule("/api/chat/stream", view_func=controller.chat_stream, methods=["POST"])
    application.add_url_rule("/health", view_func=controller.health, methods=["GET"])
    # Memory API
    application.add_url_rule("/api/memory/topic", view_func=controller.get_topic_memory, methods=["GET"])
    application.add_url_rule("/api/memory/summary", view_func=controller.get_session_summary, methods=["GET"])
    # Conversations
    application.add_url_rule("/api/conversations", view_func=controller.list_conversations, methods=["GET"])
    application.add_url_rule("/api/conversations/<conversation_id>", view_func=controller.delete_conversation, methods=["DELETE"])
    application.add_url_rule("/api/conversations/<conversation_id>/messages", view_func=controller.get_conversation_messages, methods=["GET"])
    application.add_url_rule("/api/conversations/<conversation_id>/messages/<message_id>", view_func=controller.delete_message, methods=["DELETE"])
    application.add_url_rule("/api/conversations/<conversation_id>/messages/<message_id>/feedback", view_func=controller.rate_message, methods=["POST"])
    # Flashcards
    application.add_url_rule("/api/flashcards", view_func=controller.list_flashcard_decks, methods=["GET"])
    application.add_url_rule("/api/flashcards/<deck_id>", view_func=controller.get_flashcard_deck, methods=["GET"])
    application.add_url_rule("/api/flashcards/generate", view_func=controller.generate_flashcard_deck, methods=["POST"])
    application.add_url_rule("/api/flashcards/<deck_id>/cards/<card_id>/rating", view_func=controller.rate_flashcard, methods=["POST"])
    # Quizzes
    application.add_url_rule("/api/quiz/sessions", view_func=controller.list_quiz_sessions, methods=["GET"])
    application.add_url_rule("/api/quiz/<session_id>", view_func=controller.get_quiz_session, methods=["GET"])
    application.add_url_rule("/api/quiz/generate", view_func=controller.generate_quiz, methods=["POST"])
    application.add_url_rule("/api/quiz/<session_id>/score", view_func=controller.submit_quiz_score, methods=["POST"])
    # Study Tools
    application.add_url_rule("/api/tools/explain", view_func=controller.explain_topic, methods=["POST"])
    application.add_url_rule("/api/tools/summarize", view_func=controller.summarize_text, methods=["POST"])
    application.add_url_rule("/api/tools/mnemonics", view_func=controller.generate_mnemonics, methods=["POST"])
    # Profile & Analytics
    application.add_url_rule("/api/profile/me", view_func=controller.get_profile, methods=["GET"])
    application.add_url_rule("/api/profile/me", view_func=controller.update_profile, methods=["PUT"])
    application.add_url_rule("/api/analytics/study-stats", view_func=controller.get_study_stats, methods=["GET"])
    application.add_url_rule("/api/dashboard/stats", view_func=controller.get_dashboard_stats, methods=["GET"])
    # Document Upload
    application.add_url_rule("/api/documents/upload", view_func=controller.upload_document, methods=["POST"])
    
    # SPA fallback via 404 error handler
    @application.errorhandler(404)
    def not_found(e):
        from flask import request, jsonify
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        # For non-API routes, let the SPA handle it (or serve static files if requested)
        return controller.spa(request.path.lstrip("/"))
