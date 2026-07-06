from __future__ import annotations

import base64
import hmac
import json
import hashlib
import time
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class SessionContext:
    user_id: str
    conversation_id: str
    kind: str
    cookie_value: str
    access_token: str | None = None
    is_authenticated: bool = False


class SessionManager:
    def __init__(self, secret_key: str, cookie_name: str, guest_prefix: str, max_age_seconds: int) -> None:
        self.cookie_name = cookie_name
        self.guest_prefix = guest_prefix
        self.max_age_seconds = max_age_seconds
        self.secret_key = secret_key.encode("utf-8")

    def create_guest_context(self) -> SessionContext:
        guest_id = f"{self.guest_prefix}{uuid4()}"
        return self._build_guest_context(guest_id)

    def create_guest_cookie(self, guest_id: str | None = None) -> SessionContext:
        guest_value = guest_id or f"{self.guest_prefix}{uuid4()}"
        return self._build_guest_context(guest_value)

    def create_auth_context(self, user_id: str, access_token: str) -> SessionContext:
        cookie_value = self.dumps({"kind": "auth", "user_id": user_id, "access_token": access_token})
        return SessionContext(
            user_id=user_id,
            conversation_id=user_id,
            kind="auth",
            cookie_value=cookie_value,
            access_token=access_token,
            is_authenticated=True,
        )

    def dumps(self, payload: dict[str, str]) -> str:
        timestamp = str(int(time.time()))
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
        body = f"{timestamp}.{payload_b64}"
        signature = hmac.new(self.secret_key, body.encode("utf-8"), hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
        return f"{body}.{signature_b64}"

    def loads(self, token: str, max_age: int | None = None) -> dict[str, str]:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("invalid session token")

        timestamp_text, payload_b64, signature_b64 = parts
        body = f"{timestamp_text}.{payload_b64}"
        expected_signature = hmac.new(self.secret_key, body.encode("utf-8"), hashlib.sha256).digest()
        provided_signature = base64.urlsafe_b64decode(self._pad(signature_b64))
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise ValueError("invalid session signature")

        issued_at = int(timestamp_text)
        age = int(time.time()) - issued_at
        if max_age is not None and age > max_age:
            raise ValueError("session token expired")

        payload_json = base64.urlsafe_b64decode(self._pad(payload_b64))
        payload = json.loads(payload_json.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid session payload")
        return payload

    def resolve(self, cookie_value: str | None) -> SessionContext:
        if not cookie_value:
            return self.create_guest_context()

        try:
            payload = self.loads(cookie_value, max_age=self.max_age_seconds)
        except ValueError:
            return self.create_guest_context()

        kind = payload.get("kind")
        if kind == "auth":
            user_id = payload.get("user_id")
            access_token = payload.get("access_token")
            if isinstance(user_id, str) and isinstance(access_token, str) and user_id:
                return SessionContext(
                    user_id=user_id,
                    conversation_id=user_id,
                    kind="auth",
                    cookie_value=cookie_value,
                    access_token=access_token,
                    is_authenticated=True,
                )
            return self.create_guest_context()

        guest_id = payload.get("user_id")
        if isinstance(guest_id, str) and guest_id.startswith(self.guest_prefix):
            return self._build_guest_context(guest_id, cookie_value=cookie_value)

        return self.create_guest_context()

    def _build_guest_context(self, guest_id: str, cookie_value: str | None = None) -> SessionContext:
        signed_cookie = cookie_value or self.dumps({"kind": "guest", "user_id": guest_id})
        return SessionContext(
            user_id=guest_id,
            conversation_id=guest_id,
            kind="guest",
            cookie_value=signed_cookie,
            access_token=None,
            is_authenticated=False,
        )

    def _pad(self, text: str) -> bytes:
        padding = "=" * (-len(text) % 4)
        return f"{text}{padding}".encode("utf-8")

