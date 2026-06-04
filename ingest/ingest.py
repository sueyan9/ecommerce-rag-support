import re
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

KNOWLEDGE_BASE_DIR = ROOT_DIR / "Demo-knowledge-base"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

DOMAIN_BY_FILENAME = {
    "booking-workflow.md": "booking",
    "invoice-rules.md": "invoice",
    "pricing-policy.md": "pricing",
    "privacy-policy.md": "privacy",
    "practitioner-schedule.md": "practitioner_schedule",
    "branch-management.md": "branch_management",
    "troubleshooting.md": "troubleshooting",
    "user-guide.md": "general",
}


def clean_markdown_for_embedding(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def _split_paragraph(text: str, chunk_size: int) -> List[str]:
    parts: List[str] = []
    remaining = text.strip()
    while len(remaining) > chunk_size:
        split_at = remaining[:chunk_size].rsplit(" ", 1)[0].strip()
        if not split_at:
            split_at = remaining[:chunk_size].strip()
        parts.append(split_at)
        remaining = remaining[len(split_at):].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_markdown_with_sections(text: str, default_section: str, chunk_size: int = 900) -> List[Dict[str, str]]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    chunks: List[Dict[str, str]] = []
    current_section = default_section
    current_chunk = ""
    current_chunk_section = default_section

    for paragraph in paragraphs:
        heading_match = re.match(r"^#{1,6}\s+(.+)$", paragraph)
        if heading_match:
            current_section = clean_markdown_for_embedding(heading_match.group(1)) or default_section
            continue

        cleaned_paragraph = clean_markdown_for_embedding(paragraph)
        if not cleaned_paragraph:
            continue

        for part in _split_paragraph(cleaned_paragraph, chunk_size):
            if not current_chunk:
                current_chunk = part
                current_chunk_section = current_section
                continue

            if current_chunk_section == current_section:
                candidate = f"{current_chunk}\n\n{part}"
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                    continue

            chunks.append({"text": current_chunk, "section": current_chunk_section})
            current_chunk = part
            current_chunk_section = current_section

    if current_chunk:
        chunks.append({"text": current_chunk, "section": current_chunk_section})

    return chunks


def load_markdown_documents() -> List[Dict[str, Any]]:
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {KNOWLEDGE_BASE_DIR}")

    documents = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        domain = DOMAIN_BY_FILENAME.get(path.name, "general")
        title = path.stem.replace("-", " ").title()
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_markdown_with_sections(text, default_section=title), start=1):
            documents.append(
                {
                    "text": chunk["text"],
                    "meta": {
                        "domain": domain,
                        "source": path.name,
                        "title": title,
                        "section": chunk["section"] or title,
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
