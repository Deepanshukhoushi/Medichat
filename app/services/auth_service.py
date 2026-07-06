from __future__ import annotations

from dataclasses import dataclass

from app.core.security.exceptions import AuthenticationError
from app.core.security.session_manager import SessionManager
from app.repositories.user_repository import UserRepository


@dataclass(frozen=True)
class AuthSession:
    user_id: str
    access_token: str
    cookie_value: str


class AuthService:
    def __init__(self, user_repository: UserRepository | None, session_manager: SessionManager) -> None:
        self.user_repository = user_repository
        self.session_manager = session_manager

    def sign_up(self, email: str, password: str, display_name: str | None = None):
        if self.user_repository is None:
            raise AuthenticationError("Authentication is disabled")
        return self.user_repository.sign_up(email, password, display_name)

    def sign_in(self, email: str, password: str) -> AuthSession:
        if self.user_repository is None:
            raise AuthenticationError("Authentication is disabled")
        response = self.user_repository.sign_in(email, password)
        if not getattr(response, "session", None) or not getattr(response, "user", None):
            raise AuthenticationError("Invalid credentials")
        access_token = getattr(response.session, "access_token", None)
        user_id = getattr(response.user, "id", None)
        if not access_token or not user_id:
            raise AuthenticationError("Invalid credentials")
        auth_context = self.session_manager.create_auth_context(user_id, access_token)
        return AuthSession(user_id=user_id, access_token=access_token, cookie_value=auth_context.cookie_value)

    def get_google_oauth_url(self, redirect_url: str) -> str:
        if self.user_repository is None:
            raise AuthenticationError("Authentication is disabled")
        return self.user_repository.get_google_oauth_url(redirect_url)

    def exchange_oauth_code(self, code: str) -> AuthSession:
        if self.user_repository is None:
            raise AuthenticationError("Authentication is disabled")
        response = self.user_repository.exchange_oauth_code(code)
        if not getattr(response, "session", None) or not getattr(response, "user", None):
            raise AuthenticationError("Invalid oauth session")
        access_token = getattr(response.session, "access_token", None)
        user_id = getattr(response.user, "id", None)
        if not access_token or not user_id:
            raise AuthenticationError("Invalid oauth credentials")
        auth_context = self.session_manager.create_auth_context(user_id, access_token)
        return AuthSession(user_id=user_id, access_token=access_token, cookie_value=auth_context.cookie_value)

    def sign_out(self) -> None:
        if self.user_repository is None:
            return
        self.user_repository.sign_out()

    def verify_session(self, access_token: str | None) -> bool:
        if not access_token or self.user_repository is None:
            return False
        try:
            self.user_repository.get_user(access_token)
            return True
        except AuthenticationError:
            return False

    def reset_password(self, email: str, redirect_url: str) -> None:
        if self.user_repository is None:
            raise AuthenticationError("Authentication is disabled")
        self.user_repository.reset_password(email, redirect_url)

    def update_password(self, access_token: str, new_password: str) -> None:
        if self.user_repository is None:
            raise AuthenticationError("Authentication is disabled")
        self.user_repository.update_password(access_token, new_password)

    def sign_out_with_token(self, access_token: str | None) -> None:
        """Best-effort sign-out of the session tied to *access_token*.

        Used after a password change to invalidate the reset-link token.
        Failures are silently swallowed — the caller issues a new guest
        cookie regardless.
        """
        if self.user_repository is None or not access_token:
            return
        try:
            self.user_repository.sign_out_with_token(access_token)
        except Exception:
            pass  # best-effort; do not surface to caller
