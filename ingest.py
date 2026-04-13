import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def ingest_multiple_pdfs(pdf_paths):
    #load and combine all the pdfs
    all_documents=[]
    for pdf_path in pdf_paths:
        loader= PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded:{pdf_path}")
    #splitting 
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks from all PDFs")
    #creating embeddings and store in chromaDB
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vectorstore"
    )
    print("vectorstore saved.")
    return vectorstore
if __name__ == "__main__":
    pdf_paths = ["data/sample.pdf"]
    ingest_multiple_pdfs(pdf_paths)