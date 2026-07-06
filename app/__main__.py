from __future__ import annotations

import os

from app import create_app


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=debug)
