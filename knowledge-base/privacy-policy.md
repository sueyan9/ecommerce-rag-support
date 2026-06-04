# Privacy Policy

This repository contains demo content only and must not be used with real patient data in public source control.

## Data handling

Only fake or anonymized operational examples should be stored in the local knowledge base. Real patient records, practitioner personal details, and production exports must remain outside the repository.

## Local-first guidance

Use local Ollama models and a local Qdrant instance for development. Before publishing changes, verify that `.env`, database files, logs, uploads, and vector storage folders are excluded from Git.

## Support boundaries

The assistant is intended for clinic software support and business-rule questions. It is not a diagnostic, medical, or emergency-response system.
