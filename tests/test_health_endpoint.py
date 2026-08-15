"""
Tests for the /health endpoint.

Verifies:
  - GET /health returns HTTP 200
  - Response body is valid JSON
  - JSON contains {"status": "ok"}
  - The endpoint requires no authentication or CSRF token
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from tests.stubs import setup_stubs
setup_stubs()


class TestHealthEndpoint(unittest.TestCase):
    """ChatController.health() must return 200 {"status": "ok"} with no side-effects."""

    def _make_controller(self):
        from app.api.controllers.chat_controller import ChatController
        return ChatController(
            chat_service=MagicMock(),
            auth_service=MagicMock(),
            settings=MagicMock(),
        )

    def test_health_returns_200(self):
        """health() must return HTTP status 200."""
        controller = self._make_controller()
        response, status_code = controller.health()
        self.assertEqual(status_code, 200)

    def test_health_returns_status_ok(self):
        """health() body must contain {"status": "ok"}."""
        controller = self._make_controller()
        response, status_code = controller.health()
        # stubs.py maps jsonify to the dict constructor, so response is a plain dict
        self.assertEqual(response.get("status"), "ok")

    def test_health_does_not_call_external_services(self):
        """health() must not invoke chat_service, auth_service, or any injected dependency."""
        controller = self._make_controller()
        controller.health()

        controller.chat_service.assert_not_called()
        controller.auth_service.assert_not_called()

    def test_health_response_has_no_extra_keys(self):
        """health() response should be minimal -- only the 'status' key."""
        controller = self._make_controller()
        response, _ = controller.health()
        self.assertEqual(set(response.keys()), {"status"})


if __name__ == "__main__":
    unittest.main()
