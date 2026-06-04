import json
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "clinic_knowledge")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CLEAN_THINK = os.getenv("CLEAN_THINK", "true").lower() == "true"

BASE_DIR = Path(__file__).resolve().parent
DEMO_PAGE = BASE_DIR / "static" / "index.html"

app = FastAPI(title="Clinic RAG Support API")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


class IngestItem(BaseModel):
    text: str = Field(..., min_length=1)
    meta: Dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=10)
    debug: bool = Field(default=False)
    filters: Optional[Dict[str, Any]] = None


def _ollama_url(path: str) -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return f"{base}{path}"


def _ollama_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _ollama_url(path),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is Ollama running?",
        ) from exc


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not EMBEDDING_MODEL:
        raise HTTPException(status_code=500, detail="EMBEDDING_MODEL is not configured.")

    data = _ollama_post("/api/embed", {"model": EMBEDDING_MODEL, "input": texts})
    embeddings = data.get("embeddings")
    if not embeddings:
        raise HTTPException(status_code=500, detail="Ollama did not return embeddings.")
    return embeddings


def chat_with_context(question: str, contexts: List[str]) -> str:
    system_prompt = (
        "You are a clinic management software support assistant. "
        "Answer only from the provided knowledge base context. "
        "Be clear, practical, and concise. "
        "Do not infer workflow details that are not explicitly supported by the context. "
        "If the knowledge base does not support the answer, say you do not know."
    )
    context_block = "\n\n".join(f"Source snippet:\n{context}" for context in contexts) if contexts else "No relevant knowledge found."
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Clinic support question: {question}\n\n"
                    f"Knowledge base context:\n{context_block}\n\n"
                    "Answer with a direct support response."
                ),
            },
        ],
    }
    data = _ollama_post("/api/chat", payload)
    message = data.get("message", {})
    raw_answer = message.get("content", "").strip()
    if not raw_answer:
        raise HTTPException(status_code=500, detail="Ollama did not return an answer.")
    if CLEAN_THINK:
        return re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    return raw_answer


def build_filter(filter_values: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filter_values:
        return None
    return Filter(
        must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filter_values.items()]
    )


def _get_collection_dim(info: Any) -> Optional[int]:
    try:
        return info.config.params.vectors.size
    except Exception:
        try:
            return info["config"]["params"]["vectors"]["size"]
        except Exception:
            return None


def ensure_collection() -> None:
    target_dim = len(embed_texts(["clinic support health check"])[0])
    try:
        info = qdrant.get_collection(COLLECTION)
        current_dim = _get_collection_dim(info)
        if current_dim != target_dim:
            qdrant.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
            )
    except Exception:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
        )


@app.on_event("startup")
def startup_event() -> None:
    ensure_collection()


@app.get("/")
def demo_page() -> FileResponse:
    if not DEMO_PAGE.exists():
        raise HTTPException(status_code=404, detail="Demo page not found.")
    return FileResponse(DEMO_PAGE)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "clinic-rag-support-api"}


@app.post("/ingest")
def ingest(items: List[IngestItem]) -> Dict[str, Any]:
    vectors = embed_texts([item.text for item in items])
    points = []
    for vector, item in zip(vectors, items):
        payload = {"text": item.text, **(item.meta or {})}
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"ok": True, "message": "Knowledge chunks stored.", "count": len(points)}


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    query_vector = embed_texts([req.question])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=req.k,
        query_filter=build_filter(req.filters),
    )
    results = response.points

    contexts = [result.payload.get("text", "") for result in results]
    sources = [
        {
            "source": result.payload.get("source", "unknown"),
            "title": result.payload.get("title", result.payload.get("source", "unknown")),
            "chunk_index": result.payload.get("chunk_index"),
            "score": result.score,
        }
        for result in results
    ]
    answer = chat_with_context(req.question, contexts)
    response: Dict[str, Any] = {
        "answer": answer,
        "sources": sources,
        "retrieval_scores": [result.score for result in results],
    }
    if req.debug:
        response["contexts"] = contexts
    return response


@app.post("/chat")
def chat(req: AskRequest) -> Dict[str, Any]:
    return ask(req)


@app.get("/debug/env")
def debug_env() -> Dict[str, Any]:
    return {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY_SET": bool(QDRANT_API_KEY),
        "COLLECTION": COLLECTION,
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "CLEAN_THINK": CLEAN_THINK,
    }


@app.get("/debug/collection")
def debug_collection() -> Dict[str, Any]:
    try:
        info = qdrant.get_collection(COLLECTION)
        return {"ok": True, "dim": _get_collection_dim(info), "raw": str(info)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"get_collection failed: {exc}") from exc
