import os
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

from langchain_core.documents import Document
import google.generativeai as genai
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag-chatbot")

# --- Paths ---
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = (APP_ROOT / "data").resolve()
MANUALS_DIR = (DATA_DIR / "manuals").resolve()
INDEX_DIR = (APP_ROOT / "vector_store" / "support_index").resolve()

logger.info(f"APP_ROOT: {APP_ROOT}")
logger.info(f"MANUALS_DIR: {MANUALS_DIR}")
logger.info(f"INDEX_DIR: {INDEX_DIR}")

# --- Config (NO _safe_int) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "3"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY_TO_USE = GOOGLE_API_KEY or GEMINI_API_KEY
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Max tokens
try:
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "512"))
except (ValueError, TypeError):
    logger.warning("Invalid GEMINI_MAX_TOKENS. Using 512")
    GEMINI_MAX_TOKENS = 512

# Temperature
GEMINI_TEMPERATURE = None
temp_str = os.getenv("GEMINI_TEMPERATURE")
if temp_str:
    try:
        GEMINI_TEMPERATURE = float(temp_str)
    except (ValueError, TypeError):
        logger.warning(f"Invalid GEMINI_TEMPERATURE: '{temp_str}'. Ignoring.")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", API_KEY_TO_USE)
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# --- Lazy-loaded type hints ---
RecursiveCharacterTextSplitter: Optional[Any] = None
PyPDFLoader: Optional[Any] = None
HuggingFaceEmbeddings: Optional[Any] = None
FAISS: Optional[Any] = None
GenerativeModel: Optional[Any] = None
GenerationConfig: Optional[Any] = None

# --- Global state ---
app_state: Dict[str, Any] = {
    "vector_store": None,
    "genai_model": None,
    "google_search": None,
}

# --- Lazy import locks ---
_langchain_lock = asyncio.Lock()
_genai_lock = asyncio.Lock()

# --- Lazy imports ---
async def import_langchain():
    global RecursiveCharacterTextSplitter, PyPDFLoader, HuggingFaceEmbeddings, FAISS
    async with _langchain_lock:
        if FAISS is not None:
            return
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter as RCS
            from langchain_community.document_loaders import PyPDFLoader as PDFL
            from langchain_huggingface import HuggingFaceEmbeddings as HFE
            from langchain_community.vectorstores import FAISS as FAISSStore

            RecursiveCharacterTextSplitter = RCS
            PyPDFLoader = PDFL
            HuggingFaceEmbeddings = HFE
            FAISS = FAISSStore
            logger.debug("LangChain loaded.")
        except ImportError as e:
            logger.error(f"LangChain import failed: {e}. RAG disabled.")

async def import_genai():
    global GenerativeModel, GenerationConfig
    async with _genai_lock:
        if GenerativeModel is not None:
            return
        try:
            GenerativeModel = genai.GenerativeModel
            GenerationConfig = genai.GenerationConfig
            logger.debug("Gemini classes loaded.")
        except Exception as e:
            logger.error(f"GenAI import failed: {e}")

# --- FastAPI App ---
app = FastAPI(title="RAG Support Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class AskRequest(BaseModel):
    question: str
    web_search: Optional[bool] = False
    web_num_results: Optional[int] = 3

# --- PDF & FAISS ---
async def load_pdfs() -> List[Document]:
    await import_langchain()
    if not PyPDFLoader:
        return []

    if not MANUALS_DIR.exists():
        logger.warning(f"Manuals dir missing: {MANUALS_DIR}")
        return []

    pdf_files = sorted(MANUALS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDFs found.")
        return []

    docs: List[Document] = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for p in pages:
                p.metadata["source"] = pdf.name
            docs.extend(pages)
        except Exception as e:
            logger.error(f"Failed to load {pdf.name}: {e}")
    logger.info(f"Loaded {len(docs)} pages from {len(pdf_files)} PDFs.")
    return docs

async def chunk_documents(docs: List[Document]) -> List[Document]:
    await import_langchain()
    if not RecursiveCharacterTextSplitter or not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

async def build_faiss(docs: List[Document]) -> Optional[Any]:
    await import_langchain()
    if not HuggingFaceEmbeddings or not FAISS or not docs:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(INDEX_DIR))
        logger.info(f"FAISS index saved to {INDEX_DIR}")
        return vs
    except Exception as e:
        logger.error(f"FAISS build failed: {e}")
        return None

async def load_vector_store() -> Optional[Any]:
    await import_langchain()
    if not FAISS or not HuggingFaceEmbeddings:
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index_file = INDEX_DIR / "index.faiss"

    if index_file.exists():
        try:
            vs = FAISS.load_local(
                str(INDEX_DIR), embeddings,
                allow_dangerous_deserialization=True  # Safe: built in CI
            )
            logger.info("Loaded FAISS index.")
            return vs
        except Exception as e:
            logger.warning(f"Index load failed: {e}. Rebuilding...")

    logger.info("Building FAISS index from PDFs...")
    pages = await load_pdfs()
    if not pages:
        logger.error("No PDF content.")
        return None

    chunks = await chunk_documents(pages)
    if not chunks:
        return None

    return await build_faiss(chunks)

# --- Gemini & Search ---
async def get_genai_model() -> Optional[Any]:
    if app_state["genai_model"]:
        return app_state["genai_model"]

    if not API_KEY_TO_USE:
        logger.warning("Gemini API key missing.")
        return None

    await import_genai()
    if not GenerativeModel:
        return None

    try:
        genai.configure(api_key=API_KEY_TO_USE)
        model = GenerativeModel(GEMINI_MODEL_NAME)
        app_state["genai_model"] = model
        logger.info(f"Gemini initialized: {GEMINI_MODEL_NAME}")
        return model
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return None

def get_google_search() -> Optional[Any]:
    if app_state["google_search"]:
        return app_state["google_search"]

    if not (GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID):
        return None

    try:
        service = google_build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        app_state["google_search"] = service
        return service
    except Exception as e:
        logger.error(f"Google Search init failed: {e}")
        return None

def google_search(query: str, num: int) -> List[Dict]:
    service = get_google_search()
    if not service:
        return []

    try:
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num).execute()
        items = res.get("items", [])
        return [
            {
                "type": "Web",
                "source": i.get("link", "#"),
                "title": i.get("title", "No Title"),
                "snippet": i.get("snippet", ""),
                "rank": idx + 1
            }
            for idx, i in enumerate(items)
        ]
    except HttpError as e:
        logger.error(f"Search HTTP {e.resp.status}")
        return [{"type": "Error", "snippet": f"HTTP {e.resp.status}"}]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return [{"type": "Error", "snippet": "Search failed"}]

# --- Startup ---
@app.on_event("startup")
async def startup():
    logger.info("Starting up: loading FAISS index...")
    vs = await load_vector_store()
    app_state["vector_store"] = vs
    if vs:
        logger.info("RAG ready.")
    else:
        logger.error("RAG failed to load.")

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "RAG Chatbot", "index_ready": app_state["vector_store"] is not None}

@app.get("/health")
def health():
    vs = app_state["vector_store"]
    num = len(getattr(vs, "index_to_docstore_id", {})) if vs else 0
    return {"status": "ok", "index_ready": vs is not None, "docs": num}

@app.post("/reindex")
async def reindex():
    logger.info("Reindexing...")
    vs = await load_vector_store()
    app_state["vector_store"] = vs
    if not vs:
        raise HTTPException(500, "Reindex failed")
    return {"status": "ok", "docs": len(vs.index_to_docstore_id)}

@app.post("/ask")
async def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Question required")

    vs = app_state["vector_store"]
    sources: List[Dict] = []
    context_parts: List[str] = []

    # RAG
    if vs:
        try:
            docs = vs.similarity_search(q, k=TOP_K)
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "Unknown")
                context_parts.append(f"[PDF {i}] {src}\n{d.page_content}")
                sources.append({"type": "PDF", "source": src, "rank": i})
        except Exception as e:
            logger.error(f"RAG failed: {e}")
            context_parts.append("[RAG Error]")

    # Web
    if req.web_search:
        results = google_search(q, req.web_num_results or 3)
        for r in results:
            if r["type"] == "Web":
                context_parts.append(f"[Web {r['rank']}] {r['title']}\n{r['snippet']}\nURL: {r['source']}")
                sources.append(r)

    if not context_parts:
        return {
            "answer": "No relevant information found.",
            "sources": sources,
            "source_context": ""
        }

    context = "\n\n---\n\n".join(context_parts)
    display_ctx = context[:4000]

    # Gemini
    model = await get_genai_model()
    if model:
        prompt = (
            "You are a concise support agent. Use ONLY the context below to answer in 2–4 sentences. "
            "If unsure, say so. Do not hallucinate.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        )
        try:
            config = {}
            if GEMINI_MAX_TOKENS: config["max_output_tokens"] = GEMINI_MAX_TOKENS
            if GEMINI_TEMPERATURE is not None: config["temperature"] = GEMINI_TEMPERATURE

            resp = model.generate_content(prompt, generation_config=config or None)
            answer = resp.text.strip() if hasattr(resp, "text") and resp.text else None
            if answer:
                return {"answer": answer, "sources": sources, "source_context": display_ctx}
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

    # Fallback
    fallback = "Based on sources:\n\n" + context
    if len(fallback) > 4000:
        fallback = fallback[:3900] + "\n\n...[truncated]"
    return {"answer": fallback, "sources": sources, "source_context": display_ctx}

# --- Local dev ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

