# backend/main.py
"""
FastAPI backend: PDF-backed RAG + optional Google Web Search + Gemini generation.
Save as customer-support-bot/backend/main.py
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
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Google Gen AI (Gemini)
from google import genai

# Google Custom Search
from googleapiclient.discovery import build as google_build
from googleapiclient.errors import HttpError

# ---------------- config & logging ----------------
load_dotenv()  # reads backend/.env if present

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backend")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
MANUALS_DIR = Path(os.getenv("MANUALS_DIR", str(DATA_DIR / "manuals")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(ROOT / "vector_store" / "support_index")))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Gemini / API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)  # some environments use this name
API_KEY_TO_USE = GOOGLE_API_KEY or GEMINI_API_KEY

# Google Custom Search (web search) config - set in .env
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", None)
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", None)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------- Request model ----------------
class AskRequest(BaseModel):
    question: str
    web_search: Optional[bool] = False
    web_num_results: Optional[int] = 5

# ---------------- PDF loading & indexing ----------------
def load_pdfs(manuals_dir: Path) -> List[Document]:
    docs: List[Document] = []
    if not manuals_dir.exists():
        logger.warning("Manuals dir not found: %s", manuals_dir)
        return docs
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
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
            try:
                # If index is local and trusted, allow deserialization
                vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded FAISS index from %s", INDEX_DIR)
                return vs
            except Exception as e:
                logger.warning("Failed to load existing index: %s. Will rebuild.", e)
    except Exception as e:
        logger.exception("Embeddings init error: %s", e)

    pages = load_pdfs(MANUALS_DIR)
    if not pages:
        logger.warning("No PDF pages found to index.")
        return None
    chunks = chunk_documents(pages)
    try:
        vs = build_faiss(chunks, INDEX_DIR)
        return vs
    except Exception as e:
        logger.exception("Error building FAISS index: %s", e)
        return None

logger.info("Loading/building FAISS index (may take a while)...")
VECTOR_STORE = load_or_build_index()

# ---------------- Genie (genai) client ----------------
def init_genai_client():
    if not API_KEY_TO_USE:
        logger.warning("No Gemini/Google API key set. Generation disabled.")
        return None
    try:
        client = genai.Client(api_key=API_KEY_TO_USE)
        logger.info("Initialized genai.Client")
        return client
    except Exception as e:
        logger.exception("Failed to init genai client: %s", e)
        return None

GENAI_CLIENT = init_genai_client()

# ---------------- Web search helper ----------------
def google_search(query: str, num_results: int = 5) -> str:
    """
    Return concatenated snippets from Google Custom Search API.
    Requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID in env/.env.
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        return "❌ Web search not configured (set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID)."

    try:
        service = google_build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        results = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
        items = results.get("items", [])
        snippets = []
        for it in items:
            title = it.get("title", "")
            snippet = it.get("snippet", "")
            link = it.get("link", "")
            snippets.append(f"{title} — {snippet} ({link})")
        if not snippets:
            return "No results found."
        return "\n\n".join(snippets)
    except HttpError as he:
        logger.exception("Google Search HTTP error: %s", he)
        return f"❌ Google Search HTTP error: {str(he)}"
    except Exception as e:
        logger.exception("Google Search error: %s", e)
        return f"❌ Google Search error: {str(e)}"

# ---------------- Response extraction & generator ----------------
def extract_text_from_response(response: Any) -> str:
    try:
        txt = getattr(response, "text", None)
        if txt:
            return txt.strip()
        out = getattr(response, "output", None) or (response.get("output") if isinstance(response, dict) else None)
        if out:
            first = out[0]
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if content:
                first_c = content[0]
                t = getattr(first_c, "text", None) or (first_c.get("text") if isinstance(first_c, dict) else None)
                if t:
                    return str(t).strip()
        txt = getattr(response, "output_text", None)
        if txt:
            return str(txt).strip()
        if isinstance(response, dict):
            for k in ("text", "output_text"):
                if k in response and response[k]:
                    return str(response[k]).strip()
    except Exception as e:
        logger.exception("Error extracting text: %s", e)
    try:
        logger.debug("Gemini raw response (snippet): %s", str(response)[:3000])
    except Exception:
        pass
    return str(response).strip()

def generate_with_gemini(client: genai.Client, prompt_text: str) -> str:
    """
    Tolerant call to client.models.generate_content with different kwarg combos.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    base_kwargs: Dict[str, Any] = {"model": GEMINI_MODEL, "contents": prompt_text}
    # Collect candidates for optional kwargs
    optional_sets: List[Dict[str, Any]] = []
    if GEMINI_TEMPERATURE is not None and GEMINI_MAX_TOKENS:
        optional_sets.append({"temperature": GEMINI_TEMPERATURE, "max_output_tokens": int(GEMINI_MAX_TOKENS)})
        optional_sets.append({"temperature": GEMINI_TEMPERATURE, "max_tokens": int(GEMINI_MAX_TOKENS)})
    if GEMINI_MAX_TOKENS:
        optional_sets.append({"max_output_tokens": int(GEMINI_MAX_TOKENS)})
        optional_sets.append({"max_tokens": int(GEMINI_MAX_TOKENS)})
    if GEMINI_TEMPERATURE is not None:
        optional_sets.append({"temperature": GEMINI_TEMPERATURE})
    optional_sets.append({})  # bare call as final fallback

    last_exc = None
    for opts in optional_sets:
        kwargs = {**base_kwargs, **opts}
        try:
            resp = client.models.generate_content(**kwargs)
            txt = extract_text_from_response(resp)
            if txt:
                return txt
            last_exc = RuntimeError("Empty text returned")
        except TypeError as te:
            logger.debug("generate_content TypeError for kwargs %s: %s", opts, te)
            last_exc = te
            continue
        except Exception as e:
            logger.exception("Gemini call failed for kwargs %s: %s", opts, e)
            last_exc = e
            continue
    raise RuntimeError(f"Gemini generation failed. Last error: {last_exc}")

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
    VECTOR_STORE = load_or_build_index()
    if not VECTOR_STORE:
        raise HTTPException(status_code=500, detail="Reindex failed.")
    return {"status": "ok", "num_docs_indexed": len(VECTOR_STORE.index_to_docstore_id)}

@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip() if req.question else ""
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")
    if VECTOR_STORE is None and not req.web_search:
        raise HTTPException(status_code=500, detail="Vector store not ready. Add PDFs and /reindex.")
    # 1) Optionally run web search
    web_results_text = None
    if req.web_search:
        try:
            web_results_text = google_search(q, num_results=req.web_num_results or 5)
        except Exception as e:
            web_results_text = f"❌ Web search error: {e}"
    # 2) RAG retrieval (if vector store exists)
    retrieved_context = ""
    hits = []
    if VECTOR_STORE:
        hits = VECTOR_STORE.similarity_search(q, k=TOP_K)
        if hits:
            parts = []
            for i, d in enumerate(hits, start=1):
                src = (d.metadata or {}).get("source", "")
                header = f"[Result {i}] {src}" if src else f"[Result {i}]"
                parts.append(f"{header}\n{d.page_content}")
            retrieved_context = "\n\n---\n\n".join(parts)
    # 3) Compose final context for Gemini: put web results first if present, then retrieved context
    combined_contexts = []
    if web_results_text:
        combined_contexts.append(f"Web search results:\n{web_results_text}")
    if retrieved_context:
        combined_contexts.append(f"Retrieved docs:\n{retrieved_context}")
    if not combined_contexts:
        # nothing to base on: return a friendly fallback
        return {"answer": "No data found (no indexed PDFs and web_search disabled). Please add PDFs or enable web_search.", "web_results": web_results_text}
    combined_context = "\n\n===\n\n".join(combined_contexts)
    # 4) Ask Gemini to generate concise answer (or fallback to returning combined context)
    if GENAI_CLIENT:
        try:
            prompt = (
                "You are a concise and helpful customer support assistant. Use ONLY the Context below to answer the Question in 2-4 sentences. "
                "If the Context does not contain the answer, say you couldn't find it and suggest contacting support if contact info is present.\n\n"
                f"Context:\n{combined_context}\n\nQuestion:\n{q}\n\nAnswer:"
            )
            gen_answer = generate_with_gemini(GENAI_CLIENT, prompt)
            return {"answer": gen_answer, "web_results": web_results_text, "source_context": combined_context[:4000]}
        except Exception as e:
            logger.warning("Gemini call failed: %s. Falling back to context-only response.", e)
    # fallback: return combined context (truncated)
    fallback = "Answer based on available content:\n\n" + combined_context
    if len(fallback) > 8000:
        fallback = fallback[:8000] + "\n\n...[truncated]"
    return {"answer": fallback, "web_results": web_results_text, "source_context": combined_context[:4000]}

# log routes at startup
logger.info("Registered routes:")
for r in app.routes:
    logger.info("  %s %s", list(getattr(r, "methods", [])), getattr(r, "path", ""))

