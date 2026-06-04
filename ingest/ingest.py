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


def parse_markdown_sections(text: str, fallback_title: str) -> Dict[str, Any]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return {"title": fallback_title, "sections": []}

    title = fallback_title
    current_section: str | None = None
    current_lines: List[str] = []
    sections: List[Dict[str, str]] = []

    def flush_section() -> None:
        nonlocal current_lines
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            current_lines = []
            return
        sections.append(
            {
                "section": current_section or title,
                "text": section_text,
            }
        )
        current_lines = []

    for line in normalized.split("\n"):
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if not heading_match:
            current_lines.append(line)
            continue

        level = len(heading_match.group(1))
        heading_text = clean_markdown_for_embedding(heading_match.group(2)) or title

        if level == 1:
            flush_section()
            title = heading_text
            current_section = None
            continue

        if level in (2, 3):
            flush_section()
            current_section = heading_text
            continue

    flush_section()
    return {"title": title, "sections": sections}


def chunk_section_text(section_text: str, chunk_size: int = 900) -> List[str]:
    paragraphs = [part.strip() for part in section_text.split("\n\n") if part.strip()]
    if not paragraphs:
        cleaned = clean_markdown_for_embedding(section_text)
        return [cleaned] if cleaned else []

    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        cleaned_paragraph = clean_markdown_for_embedding(paragraph)
        if not cleaned_paragraph:
            continue

        if len(cleaned_paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(_split_paragraph(cleaned_paragraph, chunk_size))
            continue

        candidate = cleaned_paragraph if not current_chunk else f"{current_chunk}\n\n{cleaned_paragraph}"
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
        current_chunk = cleaned_paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def load_markdown_documents() -> List[Dict[str, Any]]:
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {KNOWLEDGE_BASE_DIR}")

    documents = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        domain = DOMAIN_BY_FILENAME.get(path.name, "general")
        fallback_title = path.stem.replace("-", " ").title()
        parsed = parse_markdown_sections(path.read_text(encoding="utf-8"), fallback_title=fallback_title)
        title = parsed["title"] or fallback_title

        for section in parsed["sections"]:
            section_name = section["section"] or title
            for index, chunk_text in enumerate(chunk_section_text(section["text"]), start=1):
                documents.append(
                    {
                        "text": chunk_text,
                        "meta": {
                            "domain": domain,
                            "source": path.name,
                            "title": title,
                            "section": section_name,
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
