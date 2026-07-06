from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def check_environment() -> bool:
    required_vars = ["PINECONE_API_KEY", "COHERE_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "FLASK_SECRET_KEY"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        return False
    return True


def index_data() -> bool:
    try:
        result = subprocess.run([sys.executable, "-m", "scripts.index_data"], capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError) as exc:
        logger.exception("Failed to index data")
        return False


def start_flask() -> None:
    redis_url = os.getenv("REDIS_URL")
    requested_workers = int(os.getenv("GUNICORN_WORKERS", "2"))

    if not redis_url and requested_workers > 1:
        print(
            f"⚠️  WARNING: REDIS_URL is not set but GUNICORN_WORKERS={requested_workers}. "
            "Rate-limit counts and conversation history are stored in-process and will "
            "NOT be shared across workers, causing inconsistent behaviour.\n"
            "Falling back to 1 worker for safety. "
            "Set REDIS_URL to enable multi-worker mode."
        )
        workers = 1
    else:
        workers = requested_workers

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "--bind",
            "0.0.0.0:8000",
            "--workers",
            str(workers),
            "app:app",
        ]
    )


def start_streamlit() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app_streamlit.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ]
    )


def main() -> None:
    print("MediChat Deployment Script")
    print("=" * 40)

    if not check_environment():
        raise SystemExit(1)

    data_dir = Path("Data")
    if data_dir.exists() and any(data_dir.glob("*.pdf")):
        response = input("Re-index data? (y/N): ").lower().strip()
        if response in {"y", "yes"} and not index_data():
            raise SystemExit(1)
    else:
        print("No PDF files found in Data directory. Skipping indexing.")

    print("1. Flask Web App (port 8000)")
    print("2. Streamlit App (port 8501)")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == "1":
        start_flask()
    elif choice == "2":
        start_streamlit()
    else:
        raise SystemExit("Invalid choice.")


if __name__ == "__main__":
    main()
