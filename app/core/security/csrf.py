from __future__ import annotations

import secrets


class CsrfManager:
    def __init__(self, cookie_name: str, header_name: str) -> None:
        self.cookie_name = cookie_name
        self.header_name = header_name

    def generate(self) -> str:
        return secrets.token_urlsafe(32)

    def is_valid(self, cookie_token: str | None, header_token: str | None) -> bool:
        return bool(cookie_token) and bool(header_token) and secrets.compare_digest(cookie_token, header_token)

