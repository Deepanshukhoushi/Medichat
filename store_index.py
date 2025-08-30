# from src.helper import load_pdf_file, text_split
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone.grpc import PineconeGRPC
# from pinecone import ServerlessSpec
# from langchain_pinecone import PineconeVectorStore  
# from langchain_cohere import CohereEmbeddings
# from dotenv import load_dotenv
# import os 

# # Load env variables
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# # Load and split PDFs
# extracted_data = load_pdf_file("data/")
# text_chunks = text_split(extracted_data)

# # Use Cohere for embeddings
# embeddings = CohereEmbeddings(
#     model="embed-english-v3.0",
#     cohere_api_key=COHERE_API_KEY
# )

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_name = "medichat"

# # Create Pinecone index if it doesn't exist
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1024,  # Cohere's embed-english-v3.0 outputs 1024 dimensions
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Create vector store
# vectorstore = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     embedding=embeddings,
#     index_name=index_name
# )


import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader

# ------------------------------
# 1. Load environment variables
# ------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ------------------------------
# 2. Helper Functions
# ------------------------------

def load_pdf_file(folder_path: str):
    """Read all PDFs inside a folder and extract text."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def text_split(documents, chunk_size=1000, chunk_overlap=200):
    """Split long documents into smaller chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


# ------------------------------
# 3. Load and Split PDFs
# ------------------------------
extracted_data = load_pdf_file("data/")
text_chunks = text_split(extracted_data)

print(f"✅ Loaded {len(extracted_data)} PDF(s)")
print(f"✅ Created {len(text_chunks)} text chunks")

# ------------------------------
# 4. Initialize Hugging Face embeddings (1024-d)
# ------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# ------------------------------
# 5. Initialize Pinecone
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medichat"

# Create index if it doesn’t exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # HuggingFace intfloat/multilingual-e5-large → 1024 dims
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Created new Pinecone index: {index_name}")
else:
    print(f"ℹ️ Pinecone index '{index_name}' already exists")

# ------------------------------
# 6. Upload Data to Pinecone
# ------------------------------
vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("🚀 Data uploaded to Pinecone successfully!")

