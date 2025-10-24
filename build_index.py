# build_index.py
from pathlib import Path
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level="INFO")
logger = logging.getLogger()

MANUALS_DIR = Path("/app/backend/data/manuals")
INDEX_DIR = Path("/app/backend/vector_store/support_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build():
    logger.info(f"Pre-loading embedding model: {MODEL_NAME}")
    # This triggers download + caches in /root/.cache/huggingface
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    logger.info("Model cached successfully.")

    logger.info("Loading PDFs...")
    docs = []
    for pdf in MANUALS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = pdf.name
        docs.extend(pages)
    logger.info(f"Loaded {len(docs)} pages.")

    logger.info("Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    logger.info(f"Building FAISS index with {len(chunks)} chunks...")
    vs = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    logger.info(f"Index saved to {INDEX_DIR}")

if __name__ == "__main__":
    build()
