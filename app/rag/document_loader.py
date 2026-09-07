from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Control characters to remove: C0 controls (except HT/LF/CR), VT, FF, C1 controls, DEL.
# Printable Unicode — including µ, °, ±, ≥, ≤, α, β and all accented Latin — is preserved.
_CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')


def load_pdf_documents(folder_path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Fix #28: normalise composed Unicode forms (NFKC) then strip only genuine
            # control characters.  The previous r'[^\x20-\x7E\n\r\t]' stripped ALL
            # non-ASCII, turning "50 µg" into "50 g" (a 1,000,000× unit error) and
            # "β-blocker" into "-blocker" before the text reached the RAG context.
            text = unicodedata.normalize("NFKC", text)
            text = _CONTROL_CHARS_RE.sub("", text)
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

