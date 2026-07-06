from __future__ import annotations

from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_documents(folder_path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Strip garbled non-printable characters and ensure valid unicode
            import re
            text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_number},
                    )
                )
    return documents


def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

