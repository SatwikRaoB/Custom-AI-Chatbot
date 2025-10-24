# build_index.py - Project root
from pathlib import Path
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level="INFO")
logger = logging.getLogger("build_index")

# Paths inside container (match Dockerfile)
MANUALS_DIR = Path("/app/backend/data/manuals")
INDEX_DIR = Path("/app/backend/vector_store/support_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    logger.info(f"Loading PDFs from {MANUALS_DIR}")
    docs = []
    for pdf_path in MANUALS_DIR.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_path.name
            docs.extend(pages)
            logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")

    if not docs:
        logger.error("No PDF content found. Exiting.")
        return

    logger.info(f"Chunking {len(docs)} pages...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")

    logger.info(f"Downloading and caching model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    logger.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    logger.info(f"FAISS index saved to {INDEX_DIR}")

if __name__ == "__main__":
    main()
