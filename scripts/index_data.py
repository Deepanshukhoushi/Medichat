from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config.settings import get_settings
from app.rag.document_loader import load_pdf_documents, split_documents
from app.rag.vector_store import rebuild_index
from pinecone import Pinecone


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs into Pinecone.")
    parser.add_argument("--force", action="store_true", help="Delete and recreate an existing index without prompting.")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path("Data")
    documents = split_documents(
        load_pdf_documents(data_dir),
        chunk_size=settings.document_chunk_size,
        chunk_overlap=settings.document_chunk_overlap,
    )
    if len(documents) > settings.max_indexed_chunks:
        discarded = len(documents) - settings.max_indexed_chunks
        print(f"⚠️ Truncated {discarded} chunks due to MAX_CHUNKS limit.")
        documents = documents[: settings.max_indexed_chunks]

    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    index_exists = settings.index_name in pinecone_client.list_indexes().names()
    if index_exists and not args.force:
        if sys.stdin.isatty():
            response = input("Delete and recreate the Pinecone index? (y/N): ").strip().lower()
            if response not in {"y", "yes"}:
                print("Skipping destructive index rebuild.")
                return
        else:
            print("Refusing to delete an existing index without --force in non-interactive mode.")
            return

    rebuild_index(settings, documents, force=True if index_exists else args.force)
    print("Data uploaded to Pinecone successfully.")


if __name__ == "__main__":
    main()
