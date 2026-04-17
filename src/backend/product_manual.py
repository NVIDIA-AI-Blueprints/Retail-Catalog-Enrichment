# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Product manual PDF processing for enhanced FAQ generation.

Stateless targeted RAG: extract text from a product manual PDF, chunk it,
embed chunks in memory, generate product-specific queries via LLM, and
retrieve relevant chunks per topic.  All vectors are transient -- they
live only for the duration of a single request.
"""

import json
import logging
import os
from typing import Dict, List, Sequence

import numpy as np
from openai import OpenAI

from backend.config import get_config
from backend.policy import extract_text_from_pdf_bytes

logger = logging.getLogger("catalog_enrichment.product_manual")

NGC_API_KEY_NOT_SET_ERROR = "NGC_API_KEY or NVIDIA_API_KEY is not set"

LOCALE_CONFIG = {
    "en-US": {"language": "English", "region": "United States", "context": "American English"},
    "en-GB": {"language": "English", "region": "United Kingdom", "context": "British English"},
    "en-AU": {"language": "English", "region": "Australia", "context": "Australian English"},
    "en-CA": {"language": "English", "region": "Canada", "context": "Canadian English"},
    "es-ES": {"language": "Spanish", "region": "Spain", "context": "Peninsular Spanish"},
    "es-MX": {"language": "Spanish", "region": "Mexico", "context": "Mexican Spanish"},
    "es-AR": {"language": "Spanish", "region": "Argentina", "context": "Argentinian Spanish"},
    "es-CO": {"language": "Spanish", "region": "Colombia", "context": "Colombian Spanish"},
    "fr-FR": {"language": "French", "region": "France", "context": "Metropolitan French"},
    "fr-CA": {"language": "French", "region": "Canada", "context": "Quebec French"},
}


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size_words: int = 250, overlap_words: int = 50) -> List[str]:
    """Split *text* into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")
    if not key:
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)
    return key


EMBEDDING_BATCH_SIZE = 128


def _embed_texts(texts: Sequence[str], input_type: str) -> List[List[float]]:
    """Embed a list of texts using the configured NVIDIA embedding model.

    Batches requests to avoid overloading the embedding endpoint with
    very large payloads (e.g. a 200-page manual may produce 500+ chunks).
    """
    if not texts:
        return []
    embedding_config = get_config().get_embeddings_config()
    client = OpenAI(api_key=_get_api_key(), base_url=embedding_config["url"])

    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[i : i + EMBEDDING_BATCH_SIZE])
        response = client.embeddings.create(
            input=batch,
            model=embedding_config["model"],
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"},
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


# ---------------------------------------------------------------------------
# ProductManualContext — transient per-request context
# ---------------------------------------------------------------------------

class ProductManualContext:
    """Holds chunked text and its embeddings for a single product manual.

    This object is transient — it is created during a single request,
    used for targeted retrieval, and then garbage-collected.
    """

    def __init__(self, filename: str, chunks: List[str], vectors: List[List[float]]) -> None:
        self.filename = filename
        self.chunks = chunks
        self._vectors = np.array(vectors, dtype=np.float32)  # (N, dim)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def retrieve(self, query_vector: List[float], top_k: int = 3, min_score: float = 0.25) -> List[str]:
        """Return the top-k chunks most similar to *query_vector*."""
        if self.chunk_count == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        # Cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = self._vectors @ q / norms
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            self.chunks[i]
            for i in top_indices
            if similarities[i] >= min_score
        ]


# ---------------------------------------------------------------------------
# PDF → ProductManualContext
# ---------------------------------------------------------------------------

def process_manual_pdf(pdf_bytes: bytes, filename: str) -> "ProductManualContext":
    """Extract text from a PDF, chunk it, embed it, and return a context object."""
    config = get_config().get_product_manual_config()
    chunk_size = config["chunk_size_words"]
    overlap = config["chunk_overlap_words"]

    logger.info("[Manual] Extracting text from PDF: %s", filename)
    text = extract_text_from_pdf_bytes(pdf_bytes)
    if not text or len(text.strip()) < 50:
        raise ValueError(
            f"Unable to extract meaningful text from PDF: {filename}. "
            "The file may be scanned/image-based."
        )

    logger.info("[Manual] Extracted %d chars, chunking (size=%d, overlap=%d)",
                len(text), chunk_size, overlap)
    chunks = _chunk_text(text, chunk_size_words=chunk_size, overlap_words=overlap)
    if not chunks:
        raise ValueError(f"No text chunks produced from PDF: {filename}")

    logger.info("[Manual] Embedding %d chunks", len(chunks))
    vectors = _embed_texts(chunks, input_type="passage")
    if not vectors:
        raise RuntimeError(f"Embedding endpoint returned no vectors for {filename}")

    logger.info("[Manual] ProductManualContext ready: %d chunks, dim=%d",
                len(chunks), len(vectors[0]))
    return ProductManualContext(filename=filename, chunks=chunks, vectors=vectors)


# ---------------------------------------------------------------------------
# Dynamic query generation via LLM
# ---------------------------------------------------------------------------

def generate_manual_queries(
    title: str,
    categories: List[str],
    locale: str = "en-US",
) -> List[Dict[str, str]]:
    """Ask the LLM to generate 5-8 shopper-relevant questions for this product type.

    Uses only title + categories (NOT the description) so retrieved
    content does not duplicate what the description already covers.
    """
    api_key = _get_api_key()
    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config["url"], api_key=api_key)
    info = LOCALE_CONFIG.get(locale, {"language": "English", "region": "United States", "context": "American English"})

    categories_str = ", ".join(categories) if categories else "general"

    prompt = f"""/no_think You are a product manual analyst. Given a product title and category, generate questions a shopper would ask that are typically answered by a product manual or technical documentation.

PRODUCT TITLE: {title}
PRODUCT CATEGORIES: {categories_str}
TARGET LANGUAGE: {info['language']} ({info['region']})

RULES:
- Generate between 5 and 8 questions.
- Each question must have a short "topic" label (snake_case, in English) and the full "query" text.
- Questions should target details found in manuals: specifications, dimensions, care instructions, safety warnings, compatibility, materials, warranty, setup/assembly, troubleshooting.
- Do NOT ask questions that could be answered just by looking at a product photo or reading a basic product listing.
- Write the query text in {info['language']} appropriate for {info['region']}.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown, no commentary.
Example: [{{"topic": "battery_life", "query": "..."}}]"""

    logger.info("[Manual] Generating queries for: title=%r, categories=%s", title, categories_str)

    completion = client.chat.completions.create(
        model=llm_config["model"],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.3, top_p=0.9, max_tokens=1024, stream=True,
        extra_body={"reasoning_budget": 16384, "chat_template_kwargs": {"enable_thinking": False}},
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )

    # Parse JSON array
    cleaned = text.strip()
    for marker in ("```json", "```"):
        if marker in cleaned:
            start = cleaned.find(marker) + len(marker)
            end = cleaned.find("```", start)
            if end > start:
                cleaned = cleaned[start:end].strip()
                break
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        cleaned = cleaned[first_bracket : last_bracket + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("[Manual] Failed to parse query JSON: %s", exc)
        return []

    queries = [
        {"topic": q["topic"], "query": q["query"]}
        for q in parsed
        if isinstance(q, dict) and "topic" in q and "query" in q
    ]
    logger.info("[Manual] Generated %d queries", len(queries))
    return queries


# ---------------------------------------------------------------------------
# Targeted knowledge extraction
# ---------------------------------------------------------------------------

def extract_manual_knowledge(
    ctx: ProductManualContext,
    queries: List[Dict[str, str]],
) -> Dict[str, str]:
    """Retrieve relevant manual chunks for each query and return structured knowledge."""
    config = get_config().get_product_manual_config()
    top_k = config["top_k_per_query"]
    min_score = config["min_relevance_score"]

    knowledge: Dict[str, str] = {}
    seen_chunks: set = set()

    for q in queries:
        topic = q["topic"]
        query_text = q["query"]

        query_vector = _embed_texts([query_text], input_type="query")
        if not query_vector:
            logger.warning("[Manual] No embedding for query: %s", topic)
            knowledge[topic] = ""
            continue

        chunks = ctx.retrieve(query_vector[0], top_k=top_k, min_score=min_score)
        # Deduplicate across queries
        new_chunks = [c for c in chunks if c not in seen_chunks]
        seen_chunks.update(new_chunks)

        knowledge[topic] = "\n\n".join(new_chunks) if new_chunks else ""
        logger.info("[Manual] Topic %r: retrieved %d chunks (%d new)",
                    topic, len(chunks), len(new_chunks))

    return knowledge
