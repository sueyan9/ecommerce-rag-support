import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.api.main import COLLECTION, EMBEDDING_MODEL, QDRANT_API_KEY, QDRANT_URL, embed_texts

load_dotenv()

KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge-base"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def chunk_markdown(text: str, chunk_size: int = 900) -> List[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        while len(paragraph) > chunk_size:
            chunks.append(paragraph[:chunk_size].strip())
            paragraph = paragraph[chunk_size:].strip()
        current = paragraph

    if current:
        chunks.append(current)

    return chunks


def load_markdown_documents() -> List[Dict[str, Any]]:
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {KNOWLEDGE_BASE_DIR}")

    documents = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_markdown(text), start=1):
            documents.append(
                {
                    "text": chunk,
                    "meta": {
                        "source": path.name,
                        "title": path.stem.replace("-", " ").title(),
                        "chunk_index": index,
                        "document_type": "markdown",
                    },
                }
            )
    return documents


def recreate_collection() -> None:
    if not EMBEDDING_MODEL:
        raise RuntimeError("EMBEDDING_MODEL is empty. Configure a local Ollama embedding model in .env.")
    dim = len(embed_texts(["clinic ingestion setup"])[0])
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def ingest_documents() -> int:
    docs = load_markdown_documents()
    if not docs:
        return 0

    recreate_collection()
    vectors = embed_texts([doc["text"] for doc in docs])
    points = [
        PointStruct(id=idx, vector=vector, payload={"text": doc["text"], **doc["meta"]})
        for idx, (doc, vector) in enumerate(zip(docs, vectors), start=1)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


if __name__ == "__main__":
    total = ingest_documents()
    print(f"Ingested {total} clinic knowledge chunks from {KNOWLEDGE_BASE_DIR}.")
