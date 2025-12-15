#!/usr/bin/env python3
"""
MediChat Deployment Script
Handles data indexing and application startup for deployment.
"""

import os
import sys
import subprocess
from pathlib import Path

# Disable LangChain tracing
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_HANDLER'] = 'false'
os.environ['LANGCHAIN_TELEMETRY'] = 'false'

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ['PINECONE_API_KEY', 'COHERE_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
    missing = []

    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set them in your .env file or deployment environment.")
        return False

    print("✅ All environment variables are set.")
    return True

def index_data():
    """Index the medical data into Pinecone."""
    try:
        result = subprocess.run([sys.executable, 'store_index.py'],
                              capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        return False

def start_flask():
    """Start the Flask application."""
    print("🚀 Starting Flask application...")
    try:
        # Use gunicorn for production
        cmd = [sys.executable, '-m', 'gunicorn',
               '--bind', '0.0.0.0:8000',
               '--workers', '2',
               'app:app']

        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("🛑 Flask server stopped.")
    except Exception as e:
        print(f"❌ Error starting Flask: {e}")

def start_streamlit():
    """Start the Streamlit application."""
    print("🚀 Starting Streamlit application...")
    try:
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'app_streamlit.py',
               '--server.port', '8501', '--server.address', '0.0.0.0']

        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("🛑 Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def main():
    """Main deployment function."""
    print("🧠 MediChat Deployment Script")
    print("=" * 40)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Check if data needs indexing
    data_dir = Path("Data")
    if data_dir.exists() and any(data_dir.glob("*.pdf")):
        print("📁 Found PDF files in Data directory.")

        # Ask user if they want to re-index
        response = input("Re-index data? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            if not index_data():
                sys.exit(1)
        else:
            print("⏭️ Skipping data indexing.")
    else:
        print("⚠️ No PDF files found in Data directory. Skipping indexing.")

    # Choose application type
    print("\nChoose application to run:")
    print("1. Flask Web App (port 8000)")
    print("2. Streamlit App (port 8501)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        start_flask()
    elif choice == '2':
        start_streamlit()
    else:
        print("❌ Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
