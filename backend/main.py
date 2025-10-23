"""
FastAPI backend: PDF-backed RAG + optional Google Web Search + Gemini generation.
Corrected paths for Docker deployment and updated LangChain import.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain helpers
from langchain_core.documents import Document # CORRECTED IMPORT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google Gen AI (Gemini)
import google.generativeai as genai
from google.generativeai.types import GenerativeModel # For type hinting

# Google Custom Search
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError

# ---------------- config & logging ----------------
load_dotenv() # reads backend/.env if present

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backend")

# --- CORRECTED PATH CALCULATIONS for Docker ---
# Inside the container, the working directory is /app, and __file__ is /app/main.py
# We want paths relative to the working directory /app where the data was copied.
APP_ROOT = Path(__file__).resolve().parent # This will be /app inside the container
DATA_DIR = Path(os.getenv("DATA_DIR", str(APP_ROOT / "data")))
MANUALS_DIR = Path(os.getenv("MANUALS_DIR", str(DATA_DIR / "manuals")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(APP_ROOT / "vector_store" / "support_index")))

# --- Log the calculated paths to confirm ---
logger.info("Running from: %s", Path(__file__).resolve())
logger.info("Calculated APP_ROOT: %s", APP_ROOT)
logger.info("Calculated DATA_DIR: %s", DATA_DIR)
logger.info("Calculated MANUALS_DIR: %s", MANUALS_DIR)
logger.info("Calculated INDEX_DIR: %s", INDEX_DIR)
# --- End Logging ---

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Gemini / API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None) # some environments use this name
API_KEY_TO_USE = GOOGLE_API_KEY or GEMINI_API_KEY

# Google Custom Search (web search) config - set in .env
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", API_KEY_TO_USE) # Use same key by default
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Use a standard model name
GEMINI_MAX_TOKENS = os.getenv("GEMINI_MAX_TOKENS", None)
GEMINI_TEMPERATURE = os.getenv("GEMINI_TEMPERATURE", None)
if GEMINI_TEMPERATURE is not None:
    try:
        GEMINI_TEMPERATURE = float(GEMINI_TEMPERATURE)
    except Exception:
        GEMINI_TEMPERATURE = None

# ---------------- FastAPI ----------------
app = FastAPI(title="Customer Support Chatbot Backend (RAG + Web Search + Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------- Request model ----------------
class AskRequest(BaseModel):
    question: str
    web_search: Optional[bool] = False
    web_num_results: Optional[int] = 3 # Default to 3 web results


# ---------------- PDF loading & indexing ----------------
# (These functions remain the same as your previous correct versions)
def load_pdfs(manuals_dir: Path) -> List[Document]:
    docs: List[Document] = []
    if not manuals_dir.exists():
        logger.warning("Manuals dir not found: %s", manuals_dir)
        return docs
    logger.info("Looking for PDFs in: %s", manuals_dir)
    pdf_files = sorted(manuals_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s", manuals_dir)
        return docs
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load_and_split()
            logger.info("Loaded %d pages from %s", len(pages), pdf.name)
            for p in pages:
                p.metadata = p.metadata or {}
                p.metadata["source"] = pdf.name
            docs.extend(pages)
        except Exception as e:
            logger.exception("Failed to load PDF %s: %s", pdf, e)
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    logger.info("Split %d docs into %d chunks", len(docs), len(chunks))
    return chunks

def build_faiss(docs: List[Document], index_dir: Path) -> FAISS:
    if not docs:
        raise ValueError("No documents to index.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("Building FAISS index with embeddings %s", EMBEDDING_MODEL)
    vs = FAISS.from_documents(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    logger.info("Saved FAISS index to %s", index_dir)
    return vs

def load_or_build_index() -> Optional[FAISS]:
    logger.info("Attempting to load index from: %s", INDEX_DIR)
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if INDEX_DIR.exists() and any(f.name == 'index.faiss' for f in INDEX_DIR.iterdir()):
             # Check specifically for index.faiss
            try:
                # If index is local and trusted, allow deserialization
                vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded FAISS index from %s", INDEX_DIR)
                return vs
            except Exception as e:
                logger.warning("Failed to load existing index: %s. Will rebuild.", e)
        else:
             logger.info("Index directory %s does not exist or is empty. Will build.", INDEX_DIR)

    except Exception as e:
        logger.exception("Embeddings init error during load attempt: %s", e)

    # If loading failed or index doesn't exist, build it
    logger.info("Building index from PDFs in %s", MANUALS_DIR)
    pages = load_pdfs(MANUALS_DIR)
    if not pages:
        logger.warning("No PDF pages found to build index.")
        return None
    chunks = chunk_documents(pages)
    if not chunks:
         logger.warning("No chunks created from PDFs.")
         return None
    try:
        vs = build_faiss(chunks, INDEX_DIR)
        return vs
    except Exception as e:
        logger.exception("Error building FAISS index: %s", e)
        return None

logger.info("Loading/building FAISS index (may take a while)...")
VECTOR_STORE: Optional[FAISS] = load_or_build_index()
if VECTOR_STORE is None:
    logger.error("Failed to load or build vector store. RAG will be unavailable.")
else:
    logger.info("Vector store loaded successfully.")


# ---------------- Google Services Initialization ----------------

# --- Use GenerativeModel (Standard Way) ---
def init_genai_model() -> Optional[GenerativeModel]:
    if not API_KEY_TO_USE:
        logger.warning("No Gemini/Google API key set. Generation disabled.")
        return None
    try:
        genai.configure(api_key=API_KEY_TO_USE)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info("Initialized genai.GenerativeModel: %s", GEMINI_MODEL_NAME)
        return model
    except Exception as e:
        logger.exception("Failed to init genai model: %s", e)
        return None

GENAI_MODEL: Optional[GenerativeModel] = init_genai_model()

# --- Fallback: genai.Client (If needed for specific older SDK) ---
# def init_genai_client():
#     if not API_KEY_TO_USE:
#         logger.warning("No Gemini/Google API key set. Generation disabled.")
#         return None
#     try:
#         # Ensure 'genai' refers to the correct library if using Client
#         from google import genai as google_genai_sdk
#         client = google_genai_sdk.Client(api_key=API_KEY_TO_USE)
#         logger.info("Initialized genai.Client")
#         return client
#     except ImportError:
#          logger.error("Attempted to use genai.Client, but 'google.genai' structure seems different.")
#          return None
#     except Exception as e:
#         logger.exception("Failed to init genai client: %s", e)
#         return None
# GENAI_CLIENT = init_genai_client() # Uncomment if using Client

# --- Google Search Service ---
def init_google_search():
     if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("Google Search API Key or CSE ID not set. Web search disabled.")
        return None
     try:
         service = google_build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
         logger.info("Initialized Google Custom Search service.")
         return service
     except Exception as e:
         logger.exception("Failed to init Google Search service: %s", e)
         return None

GOOGLE_SEARCH_SERVICE = init_google_search()


# ---------------- Web search helper ----------------
def google_search(service, query: str, cse_id: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Performs Google search and returns structured results."""
    search_results: List[Dict[str, str]] = []
    if not service or not cse_id:
        logger.warning("Google Search called but service/CSE ID not configured.")
        return search_results

    try:
        results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        items = results.get("items", [])
        for i, item in enumerate(items):
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "No Snippet")
            link = item.get("link", "#")
            search_results.append({
                "source": link,
                "title": title,
                "snippet": snippet,
                "rank": i + 1,
                "type": "Web"
            })
        logger.info("Google Search returned %d results for query: %s", len(items), query)
        return search_results
    except HttpError as he:
        logger.exception("Google Search HTTP error: %s", he)
        return [{"source": "#", "title": "Search Error", "snippet": f"Google Search HTTP error: {he}", "rank": 1, "type": "Error"}]
    except Exception as e:
        logger.exception("Google Search error: %s", e)
        return [{"source": "#", "title": "Search Error", "snippet": f"Google Search error: {e}", "rank": 1, "type": "Error"}]


# ---------------- Response extraction & generator ----------------

def extract_text_from_genai_response(response: Any) -> Optional[str]:
    """Extracts text from google.generativeai response object."""
    try:
        # Standard access for GenerativeModel response
        if hasattr(response, 'text'):
            return response.text.strip()
        # Fallback for potential variations or older client structures
        if hasattr(response, 'parts') and response.parts:
             return response.parts[0].text.strip()
        logger.warning("Could not extract text using standard methods. Response: %s", str(response)[:500])
        return str(response).strip() # Final fallback
    except Exception as e:
        logger.exception("Error extracting text from Gemini response: %s", e)
        return None

# --- Generate function using GenerativeModel ---
def generate_with_gemini_model(model: GenerativeModel, prompt_text: str) -> str:
    if model is None:
        raise RuntimeError("Gemini model not initialized")

    generation_config = {}
    if GEMINI_MAX_TOKENS:
        try:
             generation_config["max_output_tokens"] = int(GEMINI_MAX_TOKENS)
        except ValueError:
             logger.warning("Invalid GEMINI_MAX_TOKENS value: %s", GEMINI_MAX_TOKENS)
    if GEMINI_TEMPERATURE is not None:
        generation_config["temperature"] = GEMINI_TEMPERATURE

    try:
        # Use safety_settings='block_none' cautiously for less filtering, remove if not needed
        response = model.generate_content(
             prompt_text,
             generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None,
             # safety_settings={'HARASSMENT':'block_none'} # Example, adjust as needed
        )
        txt = extract_text_from_genai_response(response)
        if txt:
            return txt
        else:
            logger.error("Gemini returned empty text. Prompt: %s | Response: %s", prompt_text[:500], response)
            raise RuntimeError("Gemini returned empty text.")
    except Exception as e:
        logger.exception("Gemini API call failed: %s", e)
        raise RuntimeError(f"Gemini generation failed: {e}")


# --- Generate function using genai.Client (Fallback if needed) ---
# def generate_with_gemini_client(client: Any, prompt_text: str) -> str:
#     if client is None:
#         raise RuntimeError("Gemini client not initialized")
#     base_kwargs: Dict[str, Any] = {"model": GEMINI_MODEL_NAME, "contents": prompt_text}
#     optional_sets: List[Dict[str, Any]] = []
#     # ... [logic to try different kwargs like max_tokens/temperature] ...
#     optional_sets.append({}) # Bare call last
#     last_exc = None
#     for opts in optional_sets:
#         kwargs = {**base_kwargs, **opts}
#         try:
#             resp = client.models.generate_content(**kwargs)
#             txt = extract_text_from_genai_response(resp) # Adjust extractor if Client response is different
#             if txt: return txt
#             last_exc = RuntimeError("Empty text returned by Client")
#         except Exception as e:
#             logger.debug("Client generate_content failed for kwargs %s: %s", opts, e)
#             last_exc = e
#             continue
#     raise RuntimeError(f"Gemini Client generation failed. Last error: {last_exc}")


# ---------------- API endpoints ----------------
@app.get("/")
def root():
    return {"message": "Customer Support Chatbot Backend", "index_ready": VECTOR_STORE is not None}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_ready": VECTOR_STORE is not None,
        "num_docs_indexed": (len(VECTOR_STORE.index_to_docstore_id) if VECTOR_STORE else 0)
    }

@app.post("/reindex")
def reindex():
    global VECTOR_STORE
    logger.info("Reindex requested.")
    # Ensure directories exist before trying to build
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    VECTOR_STORE = load_or_build_index() # Re-run the load/build logic
    if not VECTOR_STORE:
        raise HTTPException(status_code=500, detail="Reindex failed. Check logs for errors (e.g., PDF loading, FAISS build).")
    return {"status": "ok", "num_docs_indexed": len(VECTOR_STORE.index_to_docstore_id)}

@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip() if req.question else ""
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")

    if VECTOR_STORE is None and not req.web_search:
        # If web search is disabled AND vector store isn't ready, fail fast.
        logger.error("Request failed: Vector store not ready and web search disabled.")
        raise HTTPException(status_code=500, detail="Vector store not ready and web search is disabled. Add PDFs and /reindex or enable web search.")

    all_sources: List[Dict[str, Any]] = []
    retrieved_context_parts = []

    # 1) RAG retrieval (if vector store exists)
    if VECTOR_STORE:
        logger.info("Performing similarity search...")
        try:
            hits = VECTOR_STORE.similarity_search(q, k=TOP_K)
            if hits:
                logger.info("Found %d relevant document chunks.", len(hits))
                for i, d in enumerate(hits, start=1):
                    src = (d.metadata or {}).get("source", "Unknown PDF")
                    header = f"[PDF Result {i}] Source: {src}"
                    retrieved_context_parts.append(f"{header}\n{d.page_content}")
                    # Add unique sources
                    if not any(s.get("source") == src and s.get("type") == "PDF" for s in all_sources):
                       all_sources.append({"type": "PDF", "source": src, "rank": i})
            else:
                 logger.info("No relevant document chunks found.")
        except Exception as e:
            logger.exception("Error during similarity search: %s", e)
            # Optionally add an error marker to context or sources
    else:
        logger.warning("Vector store not available for RAG retrieval.")

    # 2) Optionally run web search
    web_results_context_parts = []
    if req.web_search:
        logger.info("Performing web search...")
        web_results = google_search(GOOGLE_SEARCH_SERVICE, q, GOOGLE_CSE_ID, num_results=req.web_num_results or 3)
        if web_results:
             logger.info("Found %d web results.", len(web_results))
             for res in web_results:
                 header = f"[Web Result {res.get('rank', '?')}] Title: {res.get('title', 'N/A')}"
                 web_results_context_parts.append(f"{header}\n{res.get('snippet', '')}\nURL: {res.get('source', '#')}")
                 all_sources.append(res) # Add web results directly to sources
        else:
             logger.info("No web results found.")


    # 3) Compose final context for Gemini
    combined_contexts = []
    # Prioritize web results if available
    if web_results_context_parts:
        combined_contexts.append("Web search results:\n" + "\n\n---\n\n".join(web_results_context_parts))
    if retrieved_context_parts:
        combined_contexts.append("Relevant document sections:\n" + "\n\n---\n\n".join(retrieved_context_parts))

    if not combined_contexts:
        # Should only happen if RAG fails/finds nothing AND web search is off or finds nothing
        logger.warning("No context available from RAG or web search.")
        return {"answer": "Sorry, I couldn't find any relevant information in the documents or on the web to answer your question.", "sources": all_sources, "source_context": ""}

    combined_context = "\n\n===\n\n".join(combined_contexts)
    source_context_for_display = combined_context[:4000] # Truncate for response

    # 4) Ask Gemini to generate concise answer (or fallback)
    if GENAI_MODEL: # Change to GENAI_CLIENT if using that pattern
        try:
            prompt = (
                "You are a concise and helpful customer support assistant. Use ONLY the Context below (which may include document sections and web search results) to answer the Question in 2-4 sentences. "
                "If the Context does not contain the answer, state that clearly. Do not make information up.\n\n"
                f"Context:\n{combined_context}\n\nQuestion:\n{q}\n\nAnswer:"
            )
            logger.info("Generating response with Gemini...")
            gen_answer = generate_with_gemini_model(GENAI_MODEL, prompt) # Use generate_with_gemini_client if needed
            logger.info("Gemini generation successful.")
            return {"answer": gen_answer, "sources": all_sources, "source_context": source_context_for_display}
        except Exception as e:
            # Error is logged within generate_with_gemini_...
            logger.warning("Gemini generation failed. Falling back to context-only response.")
            # Fall through to fallback
    else:
        logger.warning("Gemini client/model not initialized. Falling back.")


    # Fallback: return combined context (truncated)
    fallback_answer = "Based on the retrieved information:\n\n" + combined_context
    if len(fallback_answer) > 8000: # Limit fallback length
        fallback_answer = fallback_answer[:8000] + "\n\n...[truncated]"
    return {"answer": fallback_answer, "sources": all_sources, "source_context": source_context_for_display}

# log routes at startup
logger.info("Registered routes:")
for r in app.routes:
    logger.info("  %s %s", list(getattr(r, "methods", [])), getattr(r, "path", ""))

# Add this block for running locally with uvicorn for testing
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally for testing...")
    # Use environment variable for port or default to 8000
    port = int(os.getenv("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
