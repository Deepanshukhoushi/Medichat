from __future__ import annotations

import io
import logging
import tempfile
import os

from app.core.config.settings import AppSettings
from app.core.security.exceptions import AppError
from app.rag.document_loader import load_pdf_documents, split_documents
from app.rag.vector_store import build_user_namespace, load_vector_store

logger = logging.getLogger(__name__)


class DocumentService:
    def process_upload(self, file_bytes: bytes, filename: str, settings: AppSettings, user_id: str) -> dict:
        if len(file_bytes) > settings.max_upload_size_bytes:
            raise AppError("File is too large", status_code=413, error_type="payload_too_large")
            
        if not file_bytes.startswith(b"%PDF-"):
            raise AppError("Invalid file format. Only PDFs are allowed.", status_code=415, error_type="unsupported_media_type")
            
        try:
            from werkzeug.utils import secure_filename
            safe_filename = secure_filename(filename) or "uploaded_doc.pdf"
            
            # We use a temp directory because pypdf requires a file path or file-like object
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, safe_filename)
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)
                
                # Load and split
                raw_docs = load_pdf_documents(temp_dir)
                if not raw_docs:
                    return {"chunks_indexed": 0, "filename": filename, "message": "No text extracted"}
                
                chunks = split_documents(
                    raw_docs, 
                    chunk_size=settings.document_chunk_size, 
                    chunk_overlap=settings.document_chunk_overlap
                )
                
                if not chunks:
                    return {"chunks_indexed": 0, "filename": filename, "message": "No text extracted after splitting"}
                    
                # Limit chunks if configured
                if settings.max_indexed_chunks and len(chunks) > settings.max_indexed_chunks:
                    chunks = chunks[:settings.max_indexed_chunks]

                namespace = build_user_namespace(user_id)
                if namespace:
                    for chunk in chunks:
                        metadata = dict(getattr(chunk, "metadata", {}) or {})
                        metadata["user_id"] = user_id
                        metadata["access_scope"] = "private"
                        chunk.metadata = metadata
                
                # Upsert to Pinecone
                vector_store = load_vector_store(settings, namespace=namespace)
                vector_store.add_documents(chunks, namespace=namespace)
                
                return {
                    "chunks_indexed": len(chunks),
                    "filename": safe_filename,
                    "message": "Document successfully indexed"
                }
        except Exception as exc:
            logger.exception("Failed to process document upload")
            raise AppError("Failed to process document upload", status_code=500, error_type="processing_error") from exc
