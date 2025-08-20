from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
import os

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunks = text_splitter.split_documents(extracted_data)
    return texts_chunks

def download_embeddings():
    """Download embeddings from Cohere if API key is set, else Hugging Face."""
    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        print("✅ Using Cohere embeddings...")
        return CohereEmbeddings(
            model="embed-english-v3.0",  # 1024 dimensions
            cohere_api_key=cohere_key
        )
    else:
        print("✅ Using Hugging Face embeddings...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384 dimensions
