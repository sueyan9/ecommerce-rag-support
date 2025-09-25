import os, pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
import uuid, glob

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ecom_knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

client = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)

# 确保 collection 存在
dim = embedder.get_sentence_embedding_dimension()
client.recreate_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def add_texts(texts, meta):
    vecs = embedder.encode(texts, normalize_embeddings=True).tolist()
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload={"text": t, **meta}) for t, v in zip(texts, vecs)]
    client.upsert(COLLECTION, points=points)

# 1) 载入 FAQ（markdown 简单按段落分割）
with open("../data/samples/faq.md", "r", encoding="utf-8") as f:
    faq_text = f.read()
faq_chunks = [seg.strip() for seg in faq_text.split("\n\n") if seg.strip()]
add_texts(faq_chunks, {"type": "faq"})

# 2) 载入商品 CSV
df = pd.read_csv("../data/samples/products.csv")  # 列包含: sku,title,desc,policy 等
for _, row in df.iterrows():
    txt = f"SKU: {row['sku']}\nTitle: {row['title']}\nDescription: {row['desc']}\nPolicy: {row.get('policy','')}"
    add_texts([txt], {"type": "product", "sku": row["sku"]})

print("Ingest done.")
