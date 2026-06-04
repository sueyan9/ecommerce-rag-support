# Clinic RAG Support API

Clinic RAG Support API is a lightweight local-first Retrieval-Augmented Generation API for clinic management software, powered by FastAPI, Qdrant, and Ollama local models.

It is designed to answer support questions about a clinic management system using demo knowledge-base documents. Typical topics include product usage, booking workflow, practitioner schedule rules, branch management, invoicing policy, pricing policy, privacy policy, and general business rules.

## Why local-first AI for clinic software

Clinic software often contains operationally sensitive information. A local-first setup helps teams prototype AI support workflows without sending prompts, embeddings, or retrieved context to external AI vendors. This makes it easier to:

- keep early experiments private and developer-controlled
- reduce data exposure risk while testing support scenarios
- work offline or within restricted internal environments
- prepare an open-source baseline without shipping any real clinic or patient data

## What the project does

The project implements a simple RAG flow:

1. Load markdown files from the local knowledge base.
2. Split documents into small chunks.
3. Generate embeddings locally with Ollama.
4. Store chunks and metadata in Qdrant.
5. Retrieve relevant chunks for a support question.
6. Send the retrieved context to an Ollama chat model.
7. Return a grounded answer with source references.

## Tech stack

- FastAPI for the API and local demo page
- Qdrant for vector storage and retrieval
- Ollama for local embeddings and answer generation
- Python standard-library HTTP client for calling Ollama locally

## Project structure

- [README.md](/Users/xuyan/vs_code/ecommerce-rag-support/README.md)
- [apps/api/main.py](/Users/xuyan/vs_code/ecommerce-rag-support/apps/api/main.py): FastAPI app, retrieval, and answer generation
- [apps/api/static/index.html](/Users/xuyan/vs_code/ecommerce-rag-support/apps/api/static/index.html): local demo UI
- [ingest/ingest.py](/Users/xuyan/vs_code/ecommerce-rag-support/ingest/ingest.py): markdown ingestion script
- [knowledge-base/](/Users/xuyan/vs_code/ecommerce-rag-support/knowledge-base): demo clinic support documents only
- [docker-compose.yml](/Users/xuyan/vs_code/ecommerce-rag-support/docker-compose.yml): local Qdrant service
- [.env.example](/Users/xuyan/vs_code/ecommerce-rag-support/.env.example): safe local configuration template

## How to run locally

### 1. Requirements

- Python 3.11+
- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/)

### 2. Create your local environment file

```bash
cp .env.example .env
```

Set `EMBEDDING_MODEL` to a local Ollama embedding model you have pulled, for example `nomic-embed-text`.

### 3. Start Qdrant

```bash
docker compose up -d
```

### 4. Start Ollama and pull your models

Example:

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

### 5. Install Python dependencies

```bash
pip install fastapi uvicorn python-dotenv qdrant-client
```

### 6. Ingest the knowledge base

```bash
python ingest/ingest.py
```

### 7. Run the API

```bash
python -m uvicorn apps.api.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) to use the local demo page.

## How to ingest knowledge-base files

The ingestion script reads every `*.md` file from [knowledge-base/](/Users/xuyan/vs_code/ecommerce-rag-support/knowledge-base), splits them into chunks, creates embeddings locally with Ollama, and stores the chunks in Qdrant.

You can add more demo markdown files to that folder and rerun:

```bash
python ingest/ingest.py
```

Each stored chunk includes simple source metadata so answers can cite where the retrieved context came from.

## How to ask questions

Use the demo page or call the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does branch-level invoicing work for a practitioner who works at two locations?","k":3}'
```

The response includes:

- `answer`: grounded clinic support answer
- `sources`: source file and chunk metadata
- `retrieval_scores`: similarity scores for the returned chunks
- `contexts`: retrieved chunk text when `debug` is `true`

Example:

```json
{
  "answer": "Branch-level invoicing follows the rules defined for the practitioner's working location.",
  "sources": [
    {
      "source": "invoice-rules.md",
      "title": "Invoice Rules",
      "chunk_index": 1,
      "score": 0.83
    }
  ],
  "retrieval_scores": [0.83]
}
```

## What should not be committed to GitHub

Do not commit:

- `.env`
- API keys
- production credentials
- real database files
- real Qdrant storage
- uploads and logs
- real patient data
- real clinic data
- private business rules that are not intended for open-source publication

This repository should contain demo content only.

## Milestones

### v0.1

- `Local Ollama integration`
- `Qdrant vector database`
- `Markdown knowledge base`
- `Knowledge ingestion pipeline`
- `FastAPI support API`
- `Source-grounded answers`

### Sprint 2

- `Source Citation`
- return a direct citation field in the API response, for example `"source": "invoice-rules.md"`
- keep chunk-level metadata so the UI and downstream services can show richer evidence later

### Sprint 3

- `Feedback`
- add `👍` and `👎` feedback capture
- store feedback in a database for prompt and retrieval tuning

### Sprint 4

- `Evaluation`
- add a simple evaluation table with `Question`, `Expected`, `Actual`, and `Score`
- use this to measure retrieval and answer quality more systematically
- this is the point where the project starts moving from app prototype toward AI engineering workflow

### Sprint 5

- `Agent Router`
- route incoming questions to specialized agents such as `Booking Agent`, `Invoice Agent`, `Policy Agent`, and `Support Agent`

## References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Ollama Docs](https://ollama.com/)
