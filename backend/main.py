"""
FastAPI backend: PDF-backed RAG + optional Google Web Search + Gemini generation.
GCS-backed FAISS Index Loading for Cloud Run.
Loads index from GCS bucket to /tmp, resolving permission/path issues.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Type
import tempfile
import shutil # For cleanup

# --- Core FastAPI & Pydantic ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

# --- LangChain Core Imports ---
from langchain_core.documents import Document

# --- Google GenAI Core Imports ---
import google.generativeai as genai

# --- Google API Client Core Imports ---
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError

# --- Google Cloud Storage Import ---
try:
    from google.cloud import storage
except ImportError:
    storage = None
# --- END GCS Import ---


# --- Load environment early ---
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format="%(asctime)s [%(levelname)s] %(message)s", 
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("backend")

# --- GCS HARDCODED PATHS (Absolute Fix) ---
# NOTE: Using the exact values confirmed by the user.
GCS_BUCKET_NAME = "chatbot-store" 
GCS_INDEX_PREFIX = "vector_store/support_index" 
# --- END HARDCODED FIX ---

# --- PATH CALCULATIONS (Simplified for GCS) ---
# The Index files will be downloaded to this temporary, writable location.
TEMP_INDEX_DIR = Path(tempfile.gettempdir()) / "faiss_index_cache"
INDEX_DIR = TEMP_INDEX_DIR # RAG components will load from here

logger.info(f"Calculated TEMP_INDEX_DIR: {INDEX_DIR}")

# --- Configuration Values ---
# Fallback paths for RAG assets are now removed since we rely on GCS
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "3"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
API_KEY_TO_USE = GOOGLE_API_KEY or GEMINI_API_KEY 

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_TOKENS_STR = os.getenv("GEMINI_MAX_TOKENS", "512")
GEMINI_TEMPERATURE_STR = os.getenv("GEMINI_TEMPERATURE", None)

GEMINI_MAX_TOKENS: Optional[int] = None
try:
    GEMINI_MAX_TOKENS = int(GEMINI_MAX_TOKENS_STR)
except (ValueError, TypeError):
    GEMINI_MAX_TOKENS = 512 

GEMINI_TEMPERATURE: Optional[float] = None
if GEMINI_TEMPERATURE_STR is not None:
    try:
        GEMINI_TEMPERATURE = float(GEMINI_TEMPERATURE_STR)
    except (ValueError, TypeError):
        pass

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", API_KEY_TO_USE)
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

# --- Type Hinting for LangChain/GenAI components (loaded later) ---
RecursiveCharacterTextSplitter: Optional[Type[Any]] = None
PyPDFLoader: Optional[Type[Any]] = None
HuggingFaceEmbeddings: Optional[Type[Any]] = None
FAISS: Optional[Type[Any]] = None
GenerativeModel: Optional[Type[Any]] = None
GenerationConfig: Optional[Type[Any]] = None

# --- Global state dictionary ---
app_state: Dict[str, Any] = {
    "vector_store": None,
    "genai_model": None,
    "google_search": None,
}

# --- Lazy Import Function (LangChain remains the same) ---
_langchain_imported = False
def import_langchain_dependencies():
    global _langchain_imported, RecursiveCharacterTextSplitter, PyPDFLoader, HuggingFaceEmbeddings, FAISS
    if not _langchain_imported:
        logger.debug("Lazily importing LangChain dependencies...")
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter as RCSplitter
            from langchain_community.document_loaders import PyPDFLoader as PDFLoader
            from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
            from langchain_community.vectorstores import FAISS as FAISSStore

            RecursiveCharacterTextSplitter = RCSplitter
            PyPDFLoader = PDFLoader
            HuggingFaceEmbeddings = HFEmbeddings
            FAISS = FAISSStore
            _langchain_imported = True
            logger.debug("LangChain dependencies imported successfully.")
        except ImportError as e:
            logger.error(f"Failed to import LangChain dependencies: {e}. RAG features will be unavailable.", exc_info=True)

# --- GenAI Import remains the same ---
_genai_imported = False
def import_genai_dependencies():
    global _genai_imported, genai, GenerativeModel, GenerationConfig
    if not _genai_imported:
        logger.debug("Lazily importing Google GenAI dependencies...")
        try:
            import google.generativeai as genai_module
            genai = genai_module
            GenerativeModel = genai.GenerativeModel
            GenerationConfig = genai.types.GenerationConfig 
            _genai_imported = True
            logger.debug("Google GenAI dependencies imported.")
        except AttributeError:
             try:
                 GenerationConfig = genai.GenerationConfig
                 _genai_imported = True
                 logger.debug("Google GenAI dependencies imported (GenerationConfig found directly under genai).")
             except AttributeError as e_inner:
                  logger.error(f"Failed to import GenAI classes: {e_inner}", exc_info=True)
        except ImportError as e:
            logger.error(f"Failed to import Google GenAI dependencies: {e}", exc_info=True)


_google_search_imported = False
def import_google_search_dependencies():
     global _google_search_imported, google_build, HttpError
     if not _google_search_imported:
        logger.debug("Lazily importing Google Search dependencies...")
        try:
            from googleapiclient.discovery import build as google_api_build
            from googleapiclient.errors import HttpError as GoogleHttpError
            google_build = google_api_build
            HttpError = GoogleHttpError
            _google_search_imported = True
            logger.debug("Google Search dependencies imported.")
        except ImportError as e:
            logger.error(f"Failed to import Google Search dependencies: {e}", exc_info=True)

# --- NEW: GCS Download Utility ---
def download_gcs_index(bucket_name: str, source_prefix: str, destination_dir: Path) -> bool:
    """Downloads FAISS index files (index.faiss and index.pkl) from GCS."""
    if not storage:
        logger.error("GCS client not imported. Check 'google-cloud-storage' installation.")
        return False
    
    # Bucket name check is less critical now as it's hardcoded, but kept for client initiation check
    if not bucket_name:
        logger.error("GCS_BUCKET_NAME is empty. Cannot download index.")
        return False

    # Clean up and create the local destination directory (/tmp/faiss_index_cache)
    try:
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to clean/create local temporary directory {destination_dir}: {e}")
        return False
    
    try:
        # Authentication relies on the Cloud Run Service Account (default behavior)
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        
        # Files expected in a FAISS index directory
        required_files = ["index.faiss", "index.pkl"] 
        success_count = 0
        
        for filename in required_files:
            source_prefix_clean = source_prefix.strip('/')
            source_blob_name = f"{source_prefix_clean}/{filename}"
            destination_file = destination_dir / filename
            blob = bucket.blob(source_blob_name)
            
            if blob.exists():
                logger.info(f"Downloading {source_blob_name} to {destination_file}")
                # The service account needs Storage Object Viewer role .
                blob.download_to_filename(destination_file)
                success_count += 1
            else:
                logger.error(f"Required GCS index file not found: gs://{bucket_name}/{source_blob_name}")
        
        return success_count == len(required_files)

    except Exception as e:
        logger.error(f"Failed to download FAISS index from GCS: {e}", exc_info=True)
        return False

# --- Core RAG Loading Function (Rewritten for GCS) ---
def load_vector_store_sync():
    import_langchain_dependencies()
    if not HuggingFaceEmbeddings or not FAISS:
        logger.error("Cannot load vector store: LangChain dependencies failed to import.")
        return None

    # 1. Download index files from GCS to local /tmp
    logger.info(f"Attempting to download index from GCS bucket: {GCS_BUCKET_NAME} (Prefix: {GCS_INDEX_PREFIX})")
    download_success = download_gcs_index(GCS_BUCKET_NAME, GCS_INDEX_PREFIX, TEMP_INDEX_DIR)

    if not download_success:
        logger.error("Index download failed or missing required files. Cannot load vector store.")
        return None

    # 2. Initialize embedding model and load FAISS from the temporary local path
    logger.info(f"Initializing embedding model for loading: {EMBEDDING_MODEL}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        logger.info(f"Loading FAISS index from local path: {TEMP_INDEX_DIR.resolve()}")
        # Load from the guaranteed writable /tmp location
        vs = FAISS.load_local(str(TEMP_INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        logger.info("Successfully loaded FAISS index.")
        return vs
    except Exception as e:
        logger.error(f"Critical error during FAISS index loading from disk: {e}", exc_info=True)
        # Clean up the corrupted index files
        if TEMP_INDEX_DIR.exists():
            shutil.rmtree(TEMP_INDEX_DIR)
        return None

# --- Placeholder functions for build steps (now unused) ---
def load_pdfs(manuals_dir: Path) -> List[Document]:
    logger.warning("PDF loading logic is disabled.")
    return []
def chunk_documents(docs: List[Document]) -> List[Document]:
    return []
def build_faiss(docs: List[Document], index_dir: Path) -> Optional[Any]:
    logger.warning("FAISS building logic is disabled.")
    return None
    
# --- FastAPI App Initialization ---
app = FastAPI(title="Customer Support Chatbot Backend (GCS RAG + Web Search + Gemini)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ... (rest of the file remains the same, including endpoints)
@app.get("/")
def root():
    return {"message": "Customer Support Chatbot Backend", "index_ready": app_state.get("vector_store") is not None}

@app.get("/health")
def health():
    vs = app_state.get("vector_store")
    num_docs = 0
    if vs and hasattr(vs, 'index_to_docstore_id'):
        try:
            num_docs = len(vs.index_to_docstore_id)
        except Exception:
            logger.warning("Could not get num_docs_indexed from vector store.")
    return {
        "status": "ok",
        "index_ready": vs is not None,
        "num_docs_indexed": num_docs
    }

@app.post("/reindex")
async def reindex_endpoint():
    logger.info("Reindex endpoint called. Reloading FAISS index from GCS...")
    
    app_state["vector_store"] = None
    
    loop = asyncio.get_event_loop()
    vector_store = await loop.run_in_executor(None, load_vector_store_sync) 
    app_state["vector_store"] = vector_store
    
    vs = app_state.get("vector_store")
    if not vs:
        raise HTTPException(status_code=500, detail="Reindex failed. Index could not be downloaded/loaded from GCS.")
        
    logger.info("Reindex completed successfully from GCS.")
    num_docs = len(vs.index_to_docstore_id) if hasattr(vs, 'index_to_docstore_id') else 0
    return {"status": "ok", "num_docs_indexed": num_docs}

@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip() if req.question else ""
    if not q: raise HTTPException(status_code=400, detail="Question cannot be empty.")

    vs = app_state.get("vector_store")

    if vs is None and not req.web_search:
        logger.error("Request failed: Vector store not ready and web search disabled.")
        raise HTTPException(status_code=503, detail="Chatbot is not ready (vector store unavailable). Please try again later or enable web search.")

    all_sources: List[Dict[str, Any]] = []
    retrieved_context_parts = []

    # 1) RAG retrieval
    if vs:
        logger.info(f"Performing similarity search for query (first 50 chars): '{q[:50]}...'")
        try:
            import_langchain_dependencies()
            if FAISS:
                hits = vs.similarity_search(q, k=TOP_K)
                if hits:
                    logger.info(f"Found {len(hits)} relevant document chunks.")
                    for i, d in enumerate(hits, start=1):
                        src = d.metadata.get("source", "Unknown PDF") if hasattr(d, 'metadata') and d.metadata else "Unknown PDF"
                        content = getattr(d, 'page_content', '')
                        header = f"[PDF Result {i}] Source: {src}"
                        retrieved_context_parts.append(f"{header}\n{content}")
                        source_key = f"PDF_{src}"
                        if source_key not in [f"{s.get('type')}_{s.get('source')}" for s in all_sources]:
                            all_sources.append({"type": "PDF", "source": src, "rank": i})
                else: logger.info("No relevant document chunks found in vector store.")
            else: logger.error("FAISS library was not loaded correctly, cannot perform search.")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            retrieved_context_parts.append("[Error during document search]")
    else: logger.warning("Vector store not available for RAG retrieval.")

    # 2) Web search
    web_results_context_parts = []
    if req.web_search:
        search_service = get_google_search_service()
        if search_service:
            web_results = perform_google_search(search_service, q, GOOGLE_CSE_ID, num_results=req.web_num_results or 3)
            valid_web_results = [res for res in web_results if res.get("type") != "Error"]
            if valid_web_results:
                 logger.info(f"Found {len(valid_web_results)} web results.")
                 for res in valid_web_results:
                     header = f"[Web Result {res.get('rank', '?')}] Title: {res.get('title', 'N/A')}"
                     web_results_context_parts.append(f"{header}\n{res.get('snippet', '')}\nURL: {res.get('source', '#')}")
                     all_sources.append(res)
            else: logger.info("No valid web results found.")
            for res in web_results:
                 if res.get("type") == "Error": logger.error(f"Web search error encountered: {res.get('snippet')}")
        else: logger.warning("Web search requested but Google Search service is unavailable.")

    # 3) Compose context
    combined_contexts = []
    if retrieved_context_parts: combined_contexts.append("Relevant document sections:\n" + "\n\n---\n\n".join(retrieved_context_parts))
    if web_results_context_parts: combined_contexts.append("Web search results:\n" + "\n\n---\n\n".join(web_results_context_parts))

    if not combined_contexts:
        logger.warning(f"No context generated for query: {q}")
        answer = "Sorry, I couldn't find any relevant information."
        search_service_available = get_google_search_service() is not None
        if vs is None and req.web_search and not search_service_available: answer += " (Document store and web search are currently unavailable)."
        elif vs is None: answer += " (Document store is currently unavailable)."
        elif req.web_search and not search_service_available: answer += " (Web search is currently unavailable)."
        return {"answer": answer, "sources": all_sources, "source_context": ""}

    combined_context = "\n\n===\n\n".join(combined_contexts)
    source_context_for_display = combined_context[:4000]

    # 4) Generate answer
    genai_model_instance = get_genai_model()
    if genai_model_instance:
        try:
            prompt = (
                "You are a concise and helpful customer support assistant. Use ONLY the Context below (document sections and web search results) to answer the Question in 2-4 sentences. "
                "Prioritize information from the 'Relevant document sections' if available. If the Context does not contain the answer, state that clearly. Do not make information up.\n\n"
                f"Context:\n{combined_context}\n\nQuestion:\n{q}\n\nAnswer:"
            )
            gen_answer = generate_with_gemini(genai_model_instance, prompt)
            if gen_answer:
                return {"answer": gen_answer, "sources": all_sources, "source_context": source_context_for_display}
            else:
                logger.warning("Gemini generation returned None or was blocked. Falling back.")
        except Exception as e:
            logger.warning(f"Gemini generation raised an exception ({e}). Falling back.")
    else:
        logger.warning("Gemini model not available. Falling back.")

    # Fallback: return combined context
    logger.info("Falling back to returning combined context for query: %s", q[:50])
    fallback_answer = "Based on the retrieved information:\n\n" + combined_context
    if len(fallback_answer) > 4000: fallback_answer = fallback_answer[:3900] + "\n\n...[Context Truncated]"
    return {"answer": fallback_answer, "sources": all_sources, "source_context": source_context_for_display}

# --- Local Run ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally for testing...")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
