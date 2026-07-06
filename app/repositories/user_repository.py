from __future__ import annotations

import logging

from app.core.security.exceptions import AppError, AuthenticationError, RepositoryError


logger = logging.getLogger(__name__)


class UserRepository:
    def __init__(self, supabase_client, settings) -> None:
        self.supabase = supabase_client
        self.settings = settings

    def sign_up(self, email: str, password: str, display_name: str | None = None):
        try:
            response = self.supabase.auth.sign_up({"email": email, "password": password})
            user_id = getattr(getattr(response, "user", None), "id", None)
            if user_id and display_name:
                # Use the new user's session token so the insert satisfies RLS.
                # If email confirmation is required, session may be None — skip silently.
                session = getattr(response, "session", None)
                access_token = getattr(session, "access_token", None) if session else None
                if access_token:
                    from supabase import create_client
                    authed_client = create_client(self.settings.supabase_url, self.settings.supabase_key)
                    authed_client.auth.set_session(
                        access_token=access_token,
                        refresh_token=getattr(session, "refresh_token", ""),
                    )
                    authed_client.table("user_profiles").upsert({
                        "user_id": user_id,
                        "display_name": display_name.strip(),
                        "medical_year": None,
                        "specialty": "",
                        "university": ""
                    }).execute()
                else:
                    logger.info("Signup: no session returned (email confirmation required?); skipping profile insert for user %s", user_id)
            return response
        except Exception as exc:
            logger.exception("Failed to sign up user")
            exc_msg = str(exc).lower()
            if "rate limit" in exc_msg or "too many requests" in exc_msg or "429" in exc_msg:
                raise AppError("Too many signup requests. Please try again later.", status_code=429, error_type="rate_limited") from exc
            raise RepositoryError("Failed to sign up user") from exc

    def sign_in(self, email: str, password: str):
        try:
            return self.supabase.auth.sign_in_with_password({"email": email, "password": password})
        except Exception as exc:
            logger.exception("Failed to sign in user")
            exc_msg = str(exc).lower()
            if "rate limit" in exc_msg or "too many requests" in exc_msg or "429" in exc_msg:
                raise AppError("Too many login requests. Please try again later.", status_code=429, error_type="rate_limited") from exc
            if "email not confirmed" in exc_msg:
                raise AppError(
                    "Please confirm your email address before logging in. Check your inbox for a confirmation link.",
                    status_code=403,
                    error_type="email_not_confirmed",
                ) from exc
            raise AuthenticationError("Invalid credentials") from exc

    def get_google_oauth_url(self, redirect_url: str):
        try:
            res = self.supabase.auth.sign_in_with_oauth(
                {
                    "provider": "google",
                    "options": {
                        "redirect_to": redirect_url,
                        "flow_type": "pkce",
                        "skip_browser_redirect": True,
                    },
                }
            )
            return res.url
        except Exception as exc:
            logger.exception("Failed to get Google OAuth URL")
            raise RepositoryError("Failed to get Google OAuth URL") from exc

    def exchange_oauth_code(self, code: str):
        try:
            return self.supabase.auth.exchange_code_for_session({"auth_code": code})
        except Exception as exc:
            logger.exception("Failed to exchange OAuth code")
            raise AuthenticationError("Failed to authenticate with Google") from exc


    def get_user(self, access_token: str):
        try:
            return self.supabase.auth.get_user(access_token)
        except Exception as exc:
            logger.exception("Failed to validate user session")
            raise AuthenticationError("Invalid or expired session") from exc

    def sign_out(self) -> None:
        try:
            self.supabase.auth.sign_out()
        except Exception as exc:
            logger.exception("Failed to sign out user")
            raise RepositoryError("Failed to sign out user") from exc

    def reset_password(self, email: str, redirect_url: str) -> None:
        try:
            self.supabase.auth.reset_password_for_email(email, options={"redirect_to": redirect_url})
        except Exception as exc:
            logger.exception("Failed to send password reset email")
            exc_msg = str(exc).lower()
            if "rate limit" in exc_msg or "too many requests" in exc_msg or "429" in exc_msg:
                raise AppError("Too many password reset requests. Please try again later.", status_code=429, error_type="rate_limited") from exc
            raise RepositoryError("Failed to send password reset email") from exc

    def update_password(self, access_token: str, new_password: str) -> None:
        try:
            from supabase import create_client
            temp_client = create_client(self.settings.supabase_url, self.settings.supabase_key)
            temp_client.auth.set_session(access_token=access_token, refresh_token="")
            temp_client.auth.update_user({"password": new_password})
        except Exception as exc:
            logger.exception("Failed to update user password")
            exc_msg = str(exc).lower()
            if "rate limit" in exc_msg or "too many requests" in exc_msg or "429" in exc_msg:
                raise AppError("Too many requests to update password. Please try again later.", status_code=429, error_type="rate_limited") from exc
            if "invalid" in exc_msg or "expired" in exc_msg:
                raise AuthenticationError("Invalid or expired password reset link. Please request a new one.") from exc
            raise RepositoryError("Failed to update user password") from exc

    def sign_out_with_token(self, access_token: str) -> None:
        """Best-effort sign-out using the provided access token.

        Creates a temporary authenticated client and calls sign_out so the
        Supabase session is revoked server-side, invalidating the JWT.
        """
        try:
            from supabase import create_client
            temp_client = create_client(self.settings.supabase_url, self.settings.supabase_key)
            temp_client.auth.set_session(access_token=access_token, refresh_token="")
            temp_client.auth.sign_out()
        except Exception as exc:
            logger.warning("sign_out_with_token: best-effort sign-out failed: %s", exc)
            raise RepositoryError("sign_out_with_token failed") from exc
