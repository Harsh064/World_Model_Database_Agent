from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Get Hugging Face token if needed (optional for most public models)
HF_TOKEN = os.getenv("HF_TOKEN")  # ✅ Note: Fixed typo from "HF_TOEKN"

# Set up the Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

def build_vectorstore():
    """Build FAISS vectorstore from sample questions using Hugging Face embeddings."""
    with open("sample_questions.json", "r") as f:
        sample_questions = json.load(f)["sample_questions"]

    documents = [Document(page_content=q) for q in sample_questions]
    return FAISS.from_documents(documents, embedding_model)
