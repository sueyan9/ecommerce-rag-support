import os
import re
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue,
    VectorParams, Distance
)
from openai import OpenAI

# ---- Env & init -------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ecom_knowledge")

# Ollama (OpenAI 兼容 API)
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY", "ollama")  # 本地随便填
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:7b")
CLEAN_THINK = os.getenv("CLEAN_THINK", "true").lower() == "true"

app = FastAPI(title="Ecom RAG Support API (Ollama)")
qdrant = QdrantClient(url=QDRANT_URL)
ollama = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

# ---- Schemas ----------------------------------------------------------------
class IngestItem(BaseModel):
    text: str
    meta: dict = {}

class ChatRequest(BaseModel):
    query: str
    k: int = 5
    filters: Optional[dict] = None

# ---- Helpers ----------------------------------------------------------------
def embed(texts: List[str]) -> List[List[float]]:
    """Create embeddings via Ollama (OpenAI-compatible)."""
    resp = ollama.embeddings.create(model=OLLAMA_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]

def build_filter(d: Optional[dict]):
    if not d:
        return None
    return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in d.items()])

def _get_collection_dim(info) -> Optional[int]:
    """Safely read current vector size from Qdrant get_collection() result."""
    # qdrant-client 不同版本返回对象/字典，这里都兼容
    try:
        return info.config.params.vectors.size  # 新版对象
    except Exception:
        try:
            return info["config"]["params"]["vectors"]["size"]  # 旧版字典
        except Exception:
            return None

def ensure_collection():
    """Create collection if missing; recreate only if dim mismatch."""
    target_dim = len(embed(["ping"])[0])
    try:
        info = qdrant.get_collection(COLLECTION)
        current_dim = _get_collection_dim(info)
        if current_dim != target_dim:
            qdrant.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
            )
    except Exception:
        # 不存在则创建
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
        )

@app.on_event("startup")
def startup_event():
    ensure_collection()

# ---- Routes -----------------------------------------------------------------
@app.post("/ingest")
def ingest(items: List[IngestItem]):
    vectors = embed([it.text for it in items])
    points = []
    for v, it in zip(vectors, items):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=v, payload={"text": it.text, **(it.meta or {})}))
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"ok": True, "count": len(points)}

@app.post("/chat")
def chat(req: ChatRequest):
    # retrieve
    qvec = embed([req.query])[0]
    flt = build_filter(req.filters)
    hits = qdrant.search(collection_name=COLLECTION, query_vector=qvec, limit=req.k, query_filter=flt)
    contexts = [h.payload.get("text", "") for h in hits]

    # prompt
    system_prompt = (
        "You are an e-commerce support assistant. "
        "Use the provided knowledge when possible. "
        "Answer directly without showing your reasoning process. "
        "If unsure, say you don't know."
    )
    context_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no relevant knowledge found)"
    user_msg = f"Customer question: {req.query}\n\nRelevant knowledge:\n{context_block}\n\nAnswer:"

    try:
        resp = ollama.chat.completions.create(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        choice = resp.choices[0].message
        raw_answer = choice["content"] if isinstance(choice, dict) else choice.content
        clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip() if CLEAN_THINK else raw_answer

        return {"answer": clean_answer, "contexts": contexts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

@app.get("/debug/env")
def debug_env():
    return {
        "QDRANT_URL": QDRANT_URL,
        "COLLECTION": COLLECTION,
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "OLLAMA_EMBED_MODEL": OLLAMA_EMBED_MODEL,
        "OLLAMA_LLM_MODEL": OLLAMA_LLM_MODEL,
        "CLEAN_THINK": CLEAN_THINK,
    }

@app.get("/debug/collection")
def debug_collection():
    try:
        info = qdrant.get_collection(COLLECTION)
        return {"ok": True, "dim": _get_collection_dim(info), "raw": str(info)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_collection failed: {e}")
