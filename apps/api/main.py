from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List
import os
import uuid
import openai

load_dotenv()
app = FastAPI(title="Ecom RAG Support API")

# env
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ecom_knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

# init
embedder = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL)
openai.api_key = OPENAI_API_KEY
if OPENAI_BASE_URL:
    openai.base_url = OPENAI_BASE_URL

class IngestItem(BaseModel):
    text: str
    meta: dict = {}

class ChatRequest(BaseModel):
    query: str
    k: int = 5
    filters: dict | None = None   # 例如 {"type": "faq"} 或 {"sku": "123"}

def embed(texts: List[str]):
    return embedder.encode(texts, normalize_embeddings=True).tolist()

@app.post("/ingest")
def ingest(items: List[IngestItem]):
    vectors = embed([it.text for it in items])
    points = []
    for v, it in zip(vectors, items):
        pid = str(uuid.uuid4())
        payload = {"text": it.text, **(it.meta or {})}
        points.append(PointStruct(id=pid, vector=v, payload=payload))
    client.upsert(collection_name=COLLECTION, points=points)
    return {"ok": True, "count": len(points)}

def build_filter(d: dict | None):
    if not d: return None
    conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in d.items()]
    return Filter(must=conditions)

@app.post("/chat")
def chat(req: ChatRequest):
    qvec = embed([req.query])[0]
    flt = build_filter(req.filters)
    search = client.search(collection_name=COLLECTION, query_vector=qvec, limit=req.k, query_filter=flt)
    contexts = [hit.payload["text"] for hit in search]

    system_prompt = (
        "You are an e-commerce support assistant. "
        "Answer with concise, accurate information. If unsure, say you don't know."
    )
    context_block = "\n\n".join([f"- {c}" for c in contexts])
    user_msg = f"Customer question: {req.query}\n\nRelevant knowledge:\n{context_block}\n\nAnswer:"

    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return {"answer": answer, "contexts": contexts}
