# Product Manual PDF for FAQ Enrichment

This document explains how product manual PDFs are used to generate richer FAQs in the Catalog Enrichment System.

## Overview

By default, FAQs are generated from the enriched product title and description, which often produces shallow results that duplicate the description. When a product manual PDF is provided, the system extracts detailed knowledge (specifications, care instructions, safety warnings, warranty, etc.) and feeds it into the FAQ generation prompt alongside the product data. This produces up to 10 FAQs that surface information the description does not cover.

**Key characteristics:**
- **Stateless** -- The server stores nothing between requests. The PDF is processed, knowledge is extracted and returned to the client, and all server-side resources (embeddings, vectors) are freed immediately.
- **Per-product** -- Each manual is associated with one product. Unlike the policy library (which is persistent and shared across all products), manual knowledge is scoped to a single enrichment session.
- **Scalable** -- Because the architecture is stateless, concurrent requests for different products are fully independent. This supports batch processing of thousands of products in parallel.

## How It Works

### Step 1: Upload and Extract Knowledge

Upload a product manual PDF through the UI (Advanced Options > Product manual for FAQs) or via the API.

```
POST /vlm/manual/extract
```

The extraction pipeline runs entirely within a single request:

1. **PDF validation** -- File must be a PDF, non-empty, and under 50 MB
2. **Text extraction** -- `pypdf` extracts raw text from each page
3. **Chunking** -- Text is split into overlapping word-based chunks (default: 250 words per chunk, 50-word overlap between consecutive chunks)
4. **Embedding** -- All chunks are embedded using NVIDIA `nv-embedqa-e5-v5`, batched at 128 chunks per API call
5. **Dynamic query generation** -- The Nemotron LLM generates 5-8 product-type-specific questions based on the product **title and categories only** (not the description -- this is intentional to avoid retrieving content that duplicates the description)
6. **Targeted retrieval** -- For each generated query, the system embeds the query and retrieves the top-3 most similar chunks from the manual via in-memory cosine similarity (numpy). Chunks are deduplicated across queries.
7. **Knowledge returned** -- The structured knowledge (a JSON object mapping topic labels to relevant extracted text) is returned to the client
8. **Cleanup** -- All vectors and the `ProductManualContext` object are garbage-collected when the response is sent

### Step 2: Generate FAQs with Manual Knowledge

Pass the extracted knowledge to the FAQ generation endpoint along with the enriched product data.

```
POST /vlm/faqs  (with manual_knowledge parameter)
```

The FAQ prompt receives two distinct inputs:
- **PRODUCT** -- The enriched title, description, categories, tags, and colors (same as without a manual)
- **PRODUCT MANUAL KNOWLEDGE** -- The extracted knowledge organized by topic section

The prompt includes an explicit rule: *"The description already covers certain details. Generate FAQs about information FROM THE MANUAL that adds to or expands on the description. Do NOT create questions whose answers are fully contained in the description."*

**Without manual:** 3-5 basic FAQs, max 2048 output tokens
**With manual:** Up to 10 FAQs, max 4096 output tokens

## Why Queries Use Title + Categories, Not Description

This is the core design decision that prevents FAQ duplication. If the system generated retrieval queries from the description, it would pull back chunks that restate what the description already says, leading to FAQs that just rephrase the description.

By using only the title and categories, the LLM generates questions based on the **product type** (e.g., "What is the battery life?" for electronics, "What are the washing instructions?" for clothing) rather than echoing description content. The FAQ generation prompt then sees both the description and the manual knowledge and is instructed to surface what the description does not cover.

## Data Flow

```
Without manual (unchanged):
  enriched product data (title + desc) --> FAQ prompt --> 3-5 basic FAQs

With manual (stateless two-step):
  Step 1: POST /vlm/manual/extract + PDF + title + categories
    PDF --> extract text --> chunk --> embed in memory
    --> LLM generates 5-8 product-type-specific queries
    --> retrieve relevant chunks per topic via cosine similarity
    --> return knowledge JSON to client
    --> all vectors freed

  Step 2: POST /vlm/faqs + product data + manual_knowledge JSON
    FAQ prompt gets BOTH:
      1. enriched product data (title/desc/tags)
      2. manual_knowledge (extracted details per topic)
    --> up to 10 rich FAQs drawing from BOTH sources
```

## Configuration

```yaml
# shared/config/config.yaml

product_manual:
  chunk_size_words: 250       # Words per text chunk
  chunk_overlap_words: 50     # Overlap between consecutive chunks
  top_k_per_query: 3          # Chunks retrieved per query
  min_relevance_score: 0.25   # Minimum cosine similarity to include a chunk

embeddings:
  url: "http://localhost:8005/v1"
  model: "nvidia/nv-embedqa-e5-v5"
```

## UI

In the frontend, the product manual upload is under **Advanced Options**:

- **Upload PDF** -- Click to select a single PDF file. A staged progress indicator shows upload, text extraction, embedding, query generation, and knowledge retrieval stages.
- **When loaded** -- Shows the filename, number of indexed chunks, and a Remove button.
- **Effect on FAQs** -- When the manual is loaded and analysis runs (or has already run), the FAQ generation automatically includes the extracted manual knowledge. The FAQs tab shows the enriched results.

## API Usage

### From the UI

1. Upload a product image and run analysis (FAQs generate from title + description)
2. Upload a product manual PDF under Advanced Options
3. Run analysis again -- FAQs now include manual-enriched content

### From curl

```bash
# Step 1: Extract knowledge from the manual
curl -s -X POST \
  -F "file=@speaker-manual.pdf;type=application/pdf" \
  -F "title=JBL Flip 6 Portable Speaker" \
  -F 'categories=["electronics"]' \
  -F "locale=en-US" \
  http://localhost:8000/vlm/manual/extract

# Response:
# {
#   "filename": "speaker-manual.pdf",
#   "chunk_count": 42,
#   "knowledge": {
#     "battery_life": "The speaker provides up to 12 hours of continuous playback...",
#     "waterproof_rating": "IPX7 rated, can be submerged up to 1 meter...",
#     "bluetooth_specs": "Bluetooth 5.1, range up to 30 meters...",
#     ...
#   }
# }

# Step 2: Generate FAQs with manual knowledge
KNOWLEDGE=$(curl -s -X POST \
  -F "file=@speaker-manual.pdf;type=application/pdf" \
  -F "title=JBL Flip 6 Portable Speaker" \
  -F 'categories=["electronics"]' \
  http://localhost:8000/vlm/manual/extract | jq -c '.knowledge')

curl -X POST \
  -F "title=JBL Flip 6 Portable Speaker" \
  -F "description=A portable Bluetooth speaker with bold sound..." \
  -F 'categories=["electronics"]' \
  -F 'tags=["bluetooth","speaker","portable","waterproof"]' \
  -F 'colors=["black"]' \
  -F "locale=en-US" \
  -F "manual_knowledge=$KNOWLEDGE" \
  http://localhost:8000/vlm/faqs
```

### Batch Processing

Because the architecture is stateless, batch scripts can process many products concurrently:

```bash
for product in products/*.json; do
  TITLE=$(jq -r '.title' "$product")
  CATS=$(jq -c '.categories' "$product")
  PDF=$(jq -r '.manual_pdf' "$product")
  DESC=$(jq -r '.description' "$product")

  # Extract knowledge (each request is independent)
  KNOWLEDGE=$(curl -s -X POST \
    -F "file=@$PDF" \
    -F "title=$TITLE" \
    -F "categories=$CATS" \
    http://localhost:8000/vlm/manual/extract | jq -c '.knowledge')

  # Generate FAQs
  curl -s -X POST \
    -F "title=$TITLE" \
    -F "description=$DESC" \
    -F "categories=$CATS" \
    -F "manual_knowledge=$KNOWLEDGE" \
    http://localhost:8000/vlm/faqs
done
```

## Comparison with Policy Compliance

| Aspect | Policy Compliance | Product Manual for FAQs |
|--------|-------------------|------------------------|
| **Scope** | Shared across all products | Per-product |
| **Persistence** | Persistent (Milvus + SQLite) | Stateless (in-memory, freed after response) |
| **Storage** | Milvus vector DB + SQLite metadata | None -- knowledge returned to client |
| **Triggered by** | Automatically during `/vlm/analyze` | Explicitly via `/vlm/manual/extract` |
| **Purpose** | Pass/fail compliance decision | Richer FAQ generation |
| **PDF type** | Policy/regulation documents | Product manuals/spec sheets |
| **Embedding model** | nv-embedqa-e5-v5 (via Milvus) | nv-embedqa-e5-v5 (in-memory numpy) |

## Limitations

- Scanned/image-based PDFs with no extractable text will return an error. The system requires machine-readable text.
- PDF file size is limited to 50 MB.
- The minimum extractable text threshold is 50 characters. Very short PDFs are rejected.
- Dynamic query generation depends on the LLM understanding the product type from the title and categories. Generic or vague titles may produce less targeted queries.
- The quality of extracted knowledge depends on how well the manual text matches the generated queries. Highly technical or domain-specific manuals may benefit from more specific product categories.

## Source Files

| File | Purpose |
|------|---------|
| `src/backend/product_manual.py` | PDF processing, chunking, embedding, query generation, knowledge extraction |
| `src/backend/vlm.py` | FAQ generation with optional manual knowledge (`_call_nemotron_generate_faqs`, `_format_manual_knowledge`) |
| `src/backend/main.py` | `/vlm/manual/extract` and `/vlm/faqs` endpoints |
| `src/backend/config.py` | `get_product_manual_config()` |
| `shared/config/config.yaml` | `product_manual` configuration section |
| `src/ui/components/AdvancedOptionsCard.tsx` | Manual upload UI |
| `src/ui/app/page.tsx` | Manual state management and FAQ integration |
| `tests/test_product_manual_unit.py` | Unit tests (32 tests) |
