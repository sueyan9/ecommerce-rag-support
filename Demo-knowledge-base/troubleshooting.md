# Troubleshooting

## No answer returned

If the assistant returns an uncertain answer, first verify that the knowledge base has been ingested and that Ollama is running with both chat and embedding models available locally.

## Empty retrieval results

If retrieval returns no useful chunks, add clearer markdown documentation, rerun ingestion, and ask a more specific question that matches the documented workflow language.

## Schedule conflicts

If the system reports schedule conflicts, review practitioner overrides, branch assignment, and existing confirmed appointments before retrying the booking.

## Invoice validation problems

If an invoice cannot be finalized, confirm that the required fields are present and that the invoice is still in draft status.
