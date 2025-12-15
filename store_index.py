import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader

# Disable LangChain tracing
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_HANDLER'] = 'false'
os.environ['LANGCHAIN_TELEMETRY'] = 'false'

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
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():  # Only add non-empty pages
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    ))
    return documents


def text_split(documents, chunk_size=500, chunk_overlap=100):
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
extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)

# Limit chunks for free tier (increase for paid plans)
MAX_CHUNKS = 500
if len(text_chunks) > MAX_CHUNKS:
    text_chunks = text_chunks[:MAX_CHUNKS]

# ------------------------------
# 4. Initialize Cohere embeddings (1024-d)
# ------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY not found in environment!")

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# ------------------------------
# 5. Initialize Pinecone
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medichat"

# Delete existing index if it exists (to ensure clean re-indexing with correct embeddings)
if index_name in pc.list_indexes().names():
    print(f"🗑️ Deleting existing index '{index_name}'...")
    pc.delete_index(index_name)
    print(f"✅ Deleted existing index '{index_name}'")

# Create new index
pc.create_index(
    name=index_name,
    dimension=1024,  # Cohere embed-english-v3.0 → 1024 dims
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
print(f"✅ Created new Pinecone index: {index_name}")

# ------------------------------
# 6. Upload Data to Pinecone
# ------------------------------
vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("🚀 Data uploaded to Pinecone successfully!")
