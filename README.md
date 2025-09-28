# Ecom RAG Support API

A lightweight **E-commerce Retrieval-Augmented Generation (RAG) API** powered by **FastAPI + Qdrant + Ollama (DeepSeek local models)**.  
It lets you ingest product knowledge (FAQs, SKUs, policies) and answer customer queries with local LLMs.

---

## 🚀 Quick Start

### 1. Requirements
- Python **3.11+**
- [Docker](https://www.docker.com/) (for Qdrant)
- [Ollama](https://ollama.com) (for DeepSeek local models)
- macOS / Linux (Apple Silicon or GPU recommended)

### 2. Run the API
```bash
python -m uvicorn apps.api.main:app --reload --port 8000
```

---

## 📖 References
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Ollama Docs](https://github.com/ollama/ollama)
- [DeepSeek Models](https://huggingface.co/deepseek-ai)
