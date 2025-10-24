"""
FastAPI backend: PDF-backed RAG + optional Google Web Search + Gemini generation.
Docker-safe paths, corrected imports, FAISS loaded on startup, other clients lazy-loaded.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Type # Added Type for clarity

# --- Core FastAPI & Pydantic ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

# --- LangChain Core Imports (Safe to import early) ---
from langchain_core.documents import Document

# --- Google GenAI Core Imports (Safe to import early) ---
# Import the main module. Specific classes will be handled lazily if needed.
import google.generativeai as genai
# REMOVED: from google.generativeai.types import GenerativeModel, GenerationConfig

# --- Google API Client Core Imports (Safe to import early) ---
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError

# --- Load environment early ---
load_dotenv() # Reads .env if present

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Ensure logs go to stdout/stderr for Cloud Run
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("backend")

# --- CORRECTED PATH CALCULATIONS for Docker ---
# We ensure all paths resolve to absolute, non-relative paths inside /app
APP_ROOT = Path(__file__).resolve().parent # /app inside container

# Define fallback paths using clean absolute strings
DEFAULT_DATA_DIR = str(APP_ROOT / "data")
DEFAULT_MANUALS_DIR = str(APP_ROOT / "data" / "manuals")
DEFAULT_INDEX_DIR = str(APP_ROOT / "vector_store" / "support_index")

# Read from ENV or use the absolute fallback path
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR)).resolve()
MANUALS_DIR = Path(os.getenv("MANUALS_DIR", DEFAULT_MANUALS_DIR)).resolve()
INDEX_DIR = Path(os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)).resolve()

logger.info(f"Running from: {Path(__file__).resolve()}")
logger.info(f"Calculated APP_ROOT: {APP_ROOT}")
logger.info(f"Calculated DATA_DIR: {DATA_DIR}")
logger.info(f"Calculated MANUALS_DIR: {MANUALS_DIR}")
logger.info(f"Calculated INDEX_DIR: {INDEX_DIR}")

# --- Configuration Values ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "3"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
API_KEY_TO_USE = GOOGLE_API_KEY or GEMINI_API_KEY # Prefer GOOGLE_API_KEY if both set

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_TOKENS_STR = os.getenv("GEMINI_MAX_TOKENS", "512")
GEMINI_TEMPERATURE_STR = os.getenv("GEMINI_TEMPERATURE", None)

GEMINI_MAX_TOKENS: Optional[int] = None
try:
    GEMINI_MAX_TOKENS = int(GEMINI_MAX_TOKENS_STR)
except (ValueError, TypeError):
    logger.warning(f"Invalid GEMINI_MAX_TOKENS value: '{GEMINI_MAX_TOKENS_STR}'. Using default.")
    GEMINI_MAX_TOKENS = 512 # Fallback default

GEMINI_TEMPERATURE: Optional[float] = None
if GEMINI_TEMPERATURE_STR is not None:
    try:
        GEMINI_TEMPERATURE = float(GEMINI_TEMPERATURE_STR)
    except (ValueError, TypeError):
        logger.warning(f"Invalid GEMINI_TEMPERATURE value: '{GEMINI_TEMPERATURE_STR}'. Disabling temperature setting.")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", API_KEY_TO_USE) # Default to main key
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

# --- Type Hinting for LangChain components (loaded later) ---
RecursiveCharacterTextSplitter: Optional[Type[Any]] = None
PyPDFLoader: Optional[Type[Any]] = None
HuggingFaceEmbeddings: Optional[Type[Any]] = None
FAISS: Optional[Type[Any]] = None

# --- Type Hinting for GenAI components (Corrected) ---
GenerativeModel: Optional[Type[Any]] = None
GenerationConfig: Optional[Type[Any]] = None

# --- Global state dictionary ---
app_state: Dict[str, Any] = {
    "vector_store": None,
    "genai_model": None,
    "google_search": None,
}

# --- Lazy Import Function (Consolidated) ---
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

# Corrected GenAI lazy import - Import classes directly from genai
_genai_imported = False
def import_genai_dependencies():
    global _genai_imported, genai, GenerativeModel, GenerationConfig
    if not _genai_imported:
        logger.debug("Lazily importing Google GenAI dependencies...")
        try:
            import google.generativeai as genai_module
            genai = genai_module # Assign to global genai if needed elsewhere
            GenerativeModel = genai.GenerativeModel
            GenerationConfig = genai.types.GenerationConfig # GenerationConfig *might* still be under types
            _genai_imported = True
            logger.debug("Google GenAI dependencies imported.")
        except AttributeError:
             # Fallback if GenerationConfig is also directly under genai
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


# --- FastAPI App Initialization ---
app = FastAPI(title="Customer Support Chatbot Backend (RAG + Web Search + Gemini)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --- Request Model ---
class AskRequest(BaseModel):
    question: str
    web_search: Optional[bool] = False
    web_num_results: Optional[int] = 3

# --- PDF & FAISS Utilities ---
# (Functions load_pdfs, chunk_documents, build_faiss remain the same conceptually,
# relying on the updated import_langchain_dependencies)
def load_pdfs(manuals_dir: Path) -> List[Document]:
    import_langchain_dependencies()
    if not PyPDFLoader: return [] # Check if import failed

    docs: List[Document] = []
    # FIX: We rely on the Dockerfile to create the directory, but still ensure it exists here.
    # The PermissionError/FileNotFoundError suggests the initial path resolution was wrong.
    # We remove the old error-prone code that created the paths incorrectly, relying on the 
    # absolute paths defined at the top of the file.
    if not manuals_dir.exists():
        logger.warning(f"Manuals directory not found: {manuals_dir}. Attempting to create it (should be done by Dockerfile).")
        # Try to create it just in case, using the now absolute path
        try:
            manuals_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # We expect this to be fixed by the Dockerfile, log but continue gracefully if empty.
            logger.error(f"Failed to create directory {manuals_dir} during load: {e}")
            return docs
        
    logger.info(f"Looking for PDF files in: {manuals_dir.resolve()}")
    pdf_files = sorted(list(manuals_dir.glob("*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDF files found in {manuals_dir}")
        return docs

    logger.info(f"Found {len(pdf_files)} PDF files to load.")
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load_and_split()
            logger.debug(f"Loaded {len(pages)} pages from {pdf.name}")
            for p in pages:
                p.metadata = p.metadata or {}
                p.metadata["source"] = pdf.name
            docs.extend(pages)
        except Exception as e:
            logger.error(f"Failed to load or process PDF {pdf.name}: {e}", exc_info=True)
    logger.info(f"Successfully loaded a total of {len(docs)} pages from {len(pdf_files)} files.")
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    import_langchain_dependencies()
    if not RecursiveCharacterTextSplitter: return []
    if not docs: return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    try:
        chunks = splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
        return chunks
    except Exception as e:
        logger.error(f"Failed to split documents: {e}", exc_info=True)
        return []

def build_faiss(docs: List[Document], index_dir: Path) -> Optional[Any]:
    import_langchain_dependencies()
    if not HuggingFaceEmbeddings or not FAISS: return None
    if not docs:
        logger.error("Attempted to build FAISS index with no document chunks.")
        return None

    try:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info(f"Building FAISS index (this may take time)...")
        vs = FAISS.from_documents(docs, embeddings)
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_dir))
        logger.info(f"Successfully built and saved FAISS index to {index_dir.resolve()}")
        return vs
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}", exc_info=True)
        return None

def load_vector_store_sync():
    import_langchain_dependencies()
    if not HuggingFaceEmbeddings or not FAISS:
        logger.error("Cannot load vector store: LangChain dependencies failed to import.")
        return None

    logger.info(f"Attempting to load index from: {INDEX_DIR.resolve()}")
    vs = None
    try:
        index_file_path = INDEX_DIR / "index.faiss"
        
        # --- FIX: Ensure MANUALS_DIR and INDEX_DIR exist before proceeding, 
        # but rely on the Dockerfile for ownership/creation in a safe space (/app)
        MANUALS_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        # --- END FIX ---

        if index_file_path.exists():
            try:
                logger.info(f"Initializing embedding model for loading: {EMBEDDING_MODEL}")
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                logger.info(f"Attempting to load FAISS index from {INDEX_DIR.resolve()}...")
                # Note: allow_dangerous_deserialization=True is needed when loading from disk
                vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Successfully loaded FAISS index from {INDEX_DIR.resolve()}")
            except Exception as e:
                logger.warning(f"Failed to load existing index from {INDEX_DIR.resolve()}: {e}. Will rebuild.", exc_info=True)
                vs = None
        else:
            logger.info(f"Index file {index_file_path.resolve()} not found. Will build index.")

        if vs is None:
            logger.info(f"Building index from PDFs in {MANUALS_DIR.resolve()}")
            
            pages = load_pdfs(MANUALS_DIR)
            if not pages:
                logger.error("No PDF pages found to build index. Vector store cannot be created.")
                return None

            chunks = chunk_documents(pages)
            if not chunks:
                 logger.error("No chunks created from PDFs. Vector store cannot be created.")
                 return None

            vs = build_faiss(chunks, INDEX_DIR)

    except Exception as e:
        logger.error(f"Critical error during FAISS index loading/building: {e}", exc_info=True)
        vs = None

    return vs

# --- Gemini & Google Search Utilities (Lazy Loaded) ---
def get_genai_model() -> Optional[GenerativeModel]:
# ... (rest of the functions remain unchanged)
# The rest of your main.py file should remain unchanged from the previous version.
# Only the path resolution and error handling in load_vector_store_sync and load_pdfs were modified.
    import_genai_dependencies()
    if not GenerativeModel: # Check if import failed
        return None

    if app_state.get("genai_model") is None and API_KEY_TO_USE:
        logger.info("Initializing Google GenAI model...")
        try:
            genai.configure(api_key=API_KEY_TO_USE)
            # Use the globally imported (and potentially corrected) GenerativeModel
            model = GenerativeModel(GEMINI_MODEL_NAME)
            logger.info(f"Initialized genai.GenerativeModel: {GEMINI_MODEL_NAME}")
            app_state["genai_model"] = model
        except Exception as e:
            logger.error(f"Failed to initialize genai model: {e}", exc_info=True)
            app_state["genai_model"] = None
    elif not API_KEY_TO_USE:
         if app_state.get("genai_model", "not_set") == "not_set":
            logger.warning("Cannot initialize GenAI model: API key not set.")
            app_state["genai_model"] = None

    return app_state.get("genai_model")

def get_google_search_service() -> Optional[Any]:
    import_google_search_dependencies() # Ensure google_build is imported
    if not google_build: # Check if import failed
        return None

    if app_state.get("google_search") is None and GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID:
        logger.info("Initializing Google Custom Search service...")
        try:
            # Use the globally imported google_build
            service = google_build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
            logger.info("Initialized Google Custom Search service.")
            app_state["google_search"] = service
        except Exception as e:
            logger.error(f"Failed to initialize Google Search service: {e}", exc_info=True)
            app_state["google_search"] = None
    elif not (GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID):
        if app_state.get("google_search", "not_set") == "not_set":
             logger.warning("Cannot initialize Google Search: API Key or CSE ID not set.")
             app_state["google_search"] = None

    return app_state.get("google_search")

def extract_text_from_genai_response(response: Any) -> Optional[str]:
    # (Same robust extractor as before)
    try:
        if hasattr(response, 'text') and response.text: return response.text.strip()
        if hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
            return response.parts[0].text.strip()
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            logger.warning(f"Gemini response blocked. Reason: {response.prompt_feedback.block_reason}")
            return f"Error: Response blocked due to {response.prompt_feedback.block_reason}."
        logger.warning(f"Could not extract text using standard methods. Response: {str(response)[:500]}")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from Gemini response: {e}", exc_info=True)
        return None

def generate_with_gemini(model: GenerativeModel, prompt_text: str) -> Optional[str]:
    import_genai_dependencies() # Ensure GenerationConfig is imported and available
    if not GenerationConfig:
        logger.error("Cannot generate with Gemini: GenerationConfig not imported.")
        return None
    if model is None:
        logger.error("generate_with_gemini called but model is None.")
        return None

    generation_config_params = {}
    if GEMINI_MAX_TOKENS:
        generation_config_params["max_output_tokens"] = GEMINI_MAX_TOKENS
    if GEMINI_TEMPERATURE is not None:
        generation_config_params["temperature"] = GEMINI_TEMPERATURE

    try:
        logger.debug(f"Sending prompt to Gemini (first 100 chars): {prompt_text[:100]}...")
        # Use the globally imported GenerationConfig
        gen_config = GenerationConfig(**generation_config_params) if generation_config_params else None
        response = model.generate_content(
             prompt_text,
             generation_config=gen_config,
        )
        txt = extract_text_from_genai_response(response)
        if txt:
            logger.debug("Received non-empty response from Gemini.")
            return txt
        else:
            return None
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return None

def perform_google_search(service: Any, query: str, cse_id: str, num_results: int = 3) -> List[Dict[str, Any]]:
    import_google_search_dependencies() # Ensure HttpError is imported
    if not HttpError:
        logger.error("Cannot perform Google Search: HttpError not imported.")
        return [{"source": "#", "title": "Search Error", "snippet": "Google Search client library failed to import.", "rank": 1, "type": "Error"}]

    search_results: List[Dict[str, Any]] = []
    if not service or not cse_id:
        logger.warning("Google Search called but service/CSE ID not configured.")
        return search_results
    try:
        logger.info(f"Executing Google Search for query: {query}")
        # Use the globally imported HttpError
        results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        items = results.get("items", [])
        logger.info(f"Google Search returned {len(items)} results.")
        for i, item in enumerate(items):
            search_results.append({
                "source": item.get("link", "#"),
                "title": item.get("title", "No Title"),
                "snippet": item.get("snippet", "No Snippet"),
                "rank": i + 1,
                "type": "Web"
            })
        return search_results
    except HttpError as he:
        logger.error(f"Google Search HTTP error: {he.resp.status} - {he._get_reason()}", exc_info=False) # Log less verbosely for HTTP errors
        return [{"source": "#", "title": "Search Error", "snippet": f"Google Search failed (HTTP {he.resp.status}). Check API key/CSE ID.", "rank": 1, "type": "Error"}]
    except Exception as e:
        logger.error(f"Google Search error: {e}", exc_info=True)
        return [{"source": "#", "title": "Search Error", "snippet": f"Google Search failed: {e}", "rank": 1, "type": "Error"}]

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup initiated: Loading/building FAISS index...")
    loop = asyncio.get_event_loop()
    vector_store = await loop.run_in_executor(None, load_vector_store_sync)
    app_state["vector_store"] = vector_store
    if app_state["vector_store"]:
        logger.info("FAISS index loading/building completed successfully.")
    else:
        logger.error("FAISS index failed to load or build during startup. RAG features unavailable.")

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Customer Support Chatbot Backend", "index_ready": app_state.get("vector_store") is not None}

@app.get("/health")
def health():
    vs = app_state.get("vector_store")
    # Safely check length only if vs exists and has the attribute
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
    logger.info("Reindex endpoint called. Reloading/rebuilding FAISS index...")
    loop = asyncio.get_event_loop()
    vector_store = await loop.run_in_executor(None, load_vector_store_sync) # Re-run load/build
    app_state["vector_store"] = vector_store # Update state
    vs = app_state.get("vector_store")
    if not vs:
        raise HTTPException(status_code=500, detail="Reindex failed. Check application logs for errors.")
    logger.info("Reindex completed successfully.")
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
            import_langchain_dependencies() # Ensure FAISS etc. are available
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
        # Add more details if possible
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
    # reload=True can cause issues with startup events in some cases, monitor if needed
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
