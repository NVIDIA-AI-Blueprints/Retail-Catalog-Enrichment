# Policy Compliance

This document explains how the policy compliance feature works in the Catalog Enrichment System. Policy compliance allows you to upload PDF policy documents and automatically check enriched product listings against them before publishing.

## Overview

The policy compliance system is a persistent RAG (Retrieval-Augmented Generation) pipeline. You upload policy PDFs once, and every subsequent product analysis automatically checks the product against all loaded policies. The system returns a pass/fail decision with matched rules, reasons, and evidence.

**Key characteristics:**
- **Persistent** -- Uploaded policies remain loaded across product analyses until explicitly cleared
- **Automatic** -- Once policies are loaded, every call to `/vlm/analyze` includes a compliance check
- **Shared** -- Policies apply to all products equally (they are not per-product)

## How It Works

### Step 1: Upload Policy PDFs

Upload one or more PDF policy documents through the UI (Advanced Options > Policy Library) or via the API.

```
POST /policies
```

When a PDF is uploaded, the system processes it through this pipeline:

1. **Text extraction** -- `pypdf` extracts raw text from each page of the PDF
2. **Normalization** -- The Nemotron LLM converts the raw text into a structured JSON summary containing:
   - `policy_title` -- Short title for the policy
   - `summary` -- 2-3 sentence summary
   - `blocking_rules` -- Rules that would cause a product to fail (each with title, conditions, and observable signals)
   - `permitted_rules` -- Rules that describe allowed products (each with title and conditions)
   - `required_evidence` -- What the evaluator must confirm in a product listing
   - `notes` -- Important nuances
3. **Chunking** -- The normalized policy records (overview + individual rules) are formatted as text entries
4. **Embedding** -- Each entry is embedded using NVIDIA `nv-embedqa-e5-v5`
5. **Vector storage** -- Embeddings are stored in a Milvus collection (`policy_chunks`) with metadata (document hash, filename, policy title, summary, chunk text)
6. **Metadata storage** -- Document metadata (hash, filename, size, chunk count, summary) is persisted in a SQLite database at `data/policies/library.db`

Duplicate uploads are detected by SHA-256 content hash and skipped.

### Step 2: Automatic Retrieval During Analysis

When you call `POST /vlm/analyze` and policies are loaded, the system automatically:

1. **Builds a retrieval query** from the VLM observation (title, description, categories, tags, colors)
2. **Semantic search** against the Milvus collection using cosine similarity (top-k=8, min relevance score=0.3)
3. If no policy chunks score above the minimum relevance threshold, the product is treated as not covered by any loaded policy and receives a pass

### Step 3: Compliance Classification

When relevant policy chunks are retrieved, the system runs a compliance classifier:

1. **Product snapshot** is assembled from:
   - Primary evidence: VLM-observed title, description, categories, tags, colors
   - Secondary context: generated catalog fields from the enrichment pipeline
   - User-provided product data (if any)
2. **Policy context** is assembled from the retrieved chunks plus their parent document summaries
3. **LLM classification** -- Nemotron evaluates the product against the retrieved policy rules and returns a structured decision
4. **Consistency check** -- The system verifies the decision is internally consistent (e.g., `status: "fail"` must have at least one matched policy). If inconsistent, a repair call is attempted
5. **Fallback** -- If classification or repair fails, the system falls back to a pass with a warning

### Decision Output

The compliance decision is returned as part of the `/vlm/analyze` response:

```json
{
  "policy_decision": {
    "status": "pass | fail",
    "label": "Policy Check Passed",
    "summary": "No loaded policy appears applicable to this product.",
    "matched_policies": [
      {
        "document_name": "restricted-items.pdf",
        "policy_title": "Restricted Items Policy",
        "rule_title": "Prohibited electronics",
        "reason": "Product matches the blocked category",
        "evidence": ["observed title", "product function"]
      }
    ],
    "warnings": [],
    "evidence_note": "Decision based on the uploaded image and the generated catalog evidence."
  }
}
```

- `status: "pass"` -- No loaded policy blocks this product. `matched_policies` is always empty.
- `status: "fail"` -- At least one policy rule blocks this product. `matched_policies` contains the specific rules that were violated.

## Infrastructure

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| Milvus | Vector database for policy embeddings | `milvus.host`, `milvus.port`, `milvus.collection` in `config.yaml` |
| SQLite | Metadata storage for policy documents | `policy_library.db_path` in `config.yaml` (default: `data/policies/library.db`) |
| NVIDIA nv-embedqa-e5-v5 | Text embedding model | `embeddings.url`, `embeddings.model` in `config.yaml` |
| NVIDIA Nemotron LLM | Policy normalization and compliance classification | `llm.url`, `llm.model` in `config.yaml` |

### Milvus Collection Schema

The `policy_chunks` collection stores:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | VARCHAR (PK) | `{document_hash}:{index}` |
| `document_hash` | VARCHAR | SHA-256 hash of the PDF content |
| `document_name` | VARCHAR | Original filename |
| `policy_title` | VARCHAR | LLM-extracted policy title |
| `summary` | VARCHAR | LLM-extracted summary |
| `chunk_text` | VARCHAR | The normalized policy entry text |
| `chunk_index` | INT64 | Position within the document's entries |
| `embedding` | FLOAT_VECTOR | nv-embedqa-e5-v5 embedding |

### Configuration

```yaml
# shared/config/config.yaml

embeddings:
  url: "http://localhost:8005/v1"
  model: "nvidia/nv-embedqa-e5-v5"

milvus:
  host: "localhost"
  port: 19530
  collection: "policy_chunks"
  alias: "policy_library"

policy_library:
  storage_dir: "data/policies"
  db_path: "data/policies/library.db"
  top_k: 8
  min_relevance_score: 0.3
```

## UI

In the frontend, the policy library is managed under **Advanced Options**:

- **Upload PDFs** -- Click to select one or more PDF files. A staged progress indicator shows upload, parsing, embedding, and indexing stages.
- **Files loaded** -- Each loaded policy document shows its filename and number of indexed records.
- **Start from scratch** -- Clears all loaded policies, embeddings, and stored artifacts.
- **Compliance result** -- After analysis, the Details tab shows a pass/fail card with the policy decision summary and matched rules.

## API Reference

See [API Documentation](API.md) for full endpoint details:

- `GET /policies` -- List loaded policy documents
- `POST /policies` -- Upload policy PDFs
- `DELETE /policies` -- Clear all loaded policies

## Limitations

- Policy text is truncated to 12,000 characters during normalization. Very long policy documents may lose content from later sections.
- Scanned/image-based PDFs with no extractable text are not supported.
- The compliance classifier is an LLM-based judgment call, not a deterministic rule engine. Edge cases may produce inconsistent results.
- The system errs on the side of passing when the model response is malformed or inconsistent, logging a warning.

## Source Files

| File | Purpose |
|------|---------|
| `src/backend/policy.py` | PDF text extraction, policy normalization, compliance classification |
| `src/backend/policy_library.py` | Persistent policy library (SQLite + Milvus), embedding, retrieval |
| `src/backend/main.py` | `/policies` and `/vlm/analyze` endpoints |
| `src/ui/components/AdvancedOptionsCard.tsx` | Policy upload UI |
| `src/ui/components/FieldsCard.tsx` | Compliance result display |
