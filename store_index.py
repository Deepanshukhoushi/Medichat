# from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec
# from langchain_pinecone import PineconeVectorStore  
# from dotenv import load_dotenv
# import os 

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Only if you'll use OpenAI later

# extracted_data = load_pdf_file("data/")
# text_chunks = text_split(extracted_data)
# embeddings = download_huggingface_embeddings()

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name= "medichat"

# pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )


# vectorstore = PineconeVectorStore.from_documents(
#     documents=text_chunks,             # your list of Document objects
#     embedding=embeddings,             # embedding object
#     index_name="medichat"             # your Pinecone index name

    
# )











from src.helper import load_pdf_file, text_split
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore  
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os 

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Load and split PDFs
extracted_data = load_pdf_file("data/")
text_chunks = text_split(extracted_data)

# Use Cohere for embeddings
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medichat"

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Cohere's embed-english-v3.0 outputs 1024 dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
