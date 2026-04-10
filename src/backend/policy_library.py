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

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import OpenAI
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend.config import get_config
from backend.policy import extract_text_from_pdf_bytes, summarize_policy_document

logger = logging.getLogger("catalog_enrichment.policy_library")

EMBEDDING_API_KEY_ERROR = "NVIDIA_API_KEY or NGC_API_KEY is not set"
MAX_QUERY_WORDS = 160
MAX_EMBED_TOTAL_WORDS = 190


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def build_policy_query(product_snapshot: Dict[str, Any]) -> str:
    """Build a compact retrieval query from analyzed product evidence."""
    parts = [
        f"Title: {product_snapshot.get('title', '')}",
        f"Description: {product_snapshot.get('description', '')}",
        f"Categories: {', '.join(product_snapshot.get('categories', []))}",
        f"Tags: {', '.join(product_snapshot.get('tags', []))}",
        f"Colors: {', '.join(product_snapshot.get('colors', []))}",
    ]
    return _limit_words("\n".join(part for part in parts if part.strip()), MAX_QUERY_WORDS)


class PolicyLibrary:
    """Persistent single-user policy document library backed by SQLite and Milvus."""

    def __init__(self) -> None:
        config = get_config()
        self._policy_config = config.get_policy_library_config()
        self._milvus_config = config.get_milvus_config()
        self._embedding_config = config.get_embeddings_config()
        self._storage_dir = Path(self._policy_config["storage_dir"])
        self._db_path = Path(self._policy_config["db_path"])
        self._top_k = int(self._policy_config["top_k"])
        self._min_relevance_score = float(self._policy_config["min_relevance_score"])
        self._collection_name = str(self._milvus_config["collection"])
        self._milvus_alias = str(self._milvus_config["alias"])
        self._connected = False

    def initialize(self) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_documents (
                    document_hash TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    text_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect_db() as conn:
            rows = conn.execute(
                """
                SELECT document_hash, filename, file_size, chunk_count, created_at, updated_at
                FROM policy_documents
                ORDER BY updated_at DESC
                """
            ).fetchall()

        return [
            {
                "document_hash": row["document_hash"],
                "filename": row["filename"],
                "file_size": row["file_size"],
                "chunk_count": row["chunk_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def ingest_documents(self, uploads: Sequence[Dict[str, Any]], locale: str = "en-US") -> List[Dict[str, Any]]:
        results = []
        for upload in uploads:
            filename = upload["filename"]
            pdf_bytes = upload["bytes"]
            document_hash = hashlib.sha256(pdf_bytes).hexdigest()
            existing = self._get_document(document_hash)
            if existing is not None:
                self._touch_document(document_hash)
                results.append(
                    {
                        "document_hash": document_hash,
                        "filename": existing["filename"],
                        "chunk_count": existing["chunk_count"],
                        "already_loaded": True,
                        "processed": False,
                    }
                )
                continue

            extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
            if not extracted_text:
                raise ValueError(f"Unable to extract text from PDF: {filename}")

            normalized_text = extracted_text.strip()
            summary = summarize_policy_document(filename, normalized_text, locale)
            records = self._build_policy_entries(filename, summary)
            embedding_inputs = [
                self._format_policy_entry_for_embedding(entry_text)
                for entry_text in records
            ]
            vectors = self._embed_texts(embedding_inputs, input_type="passage")
            if not vectors:
                raise RuntimeError(f"No embeddings were returned for {filename}")

            self._ensure_collection(len(vectors[0]))
            self._replace_document_vectors(document_hash, filename, summary, records, vectors)
            self._persist_document(document_hash, filename, len(pdf_bytes), len(records), summary, normalized_text)
            results.append(
                {
                    "document_hash": document_hash,
                    "filename": filename,
                    "chunk_count": len(records),
                    "already_loaded": False,
                    "processed": True,
                }
            )

        return results

    def retrieve_context(self, product_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.list_documents():
            return []

        if not self._collection_exists():
            return []

        query_text = build_policy_query(product_snapshot)
        if not query_text.strip():
            return []

        query_vector = self._embed_texts([query_text], input_type="query")[0]
        collection = self._get_collection(load=True)
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=self._top_k,
            output_fields=["document_hash", "document_name", "policy_title", "summary", "chunk_text", "chunk_index"],
        )

        raw_hits = []
        document_hashes = set()
        for hit in results[0]:
            entity = hit.entity
            document_hash = entity.get("document_hash")
            if document_hash:
                document_hashes.add(document_hash)
            raw_hits.append((hit, entity))

        if raw_hits:
            logger.info(
                "Policy retrieval candidate scores: %s",
                ", ".join(f"{float(hit.score):.4f}" for hit, _ in raw_hits[: min(5, len(raw_hits))]),
            )
            top_score = float(raw_hits[0][0].score)
            if top_score < self._min_relevance_score:
                logger.info(
                    "Policy retrieval skipped classification: top score %.4f below min_relevance_score %.4f",
                    top_score,
                    self._min_relevance_score,
                )
                return []

        document_summaries = self._get_document_summaries(document_hashes)
        retrieved = []
        for hit, entity in raw_hits:
            document_hash = entity.get("document_hash")
            retrieved.append(
                {
                    "document_hash": document_hash,
                    "document_name": entity.get("document_name"),
                    "policy_title": entity.get("policy_title"),
                    "summary": entity.get("summary"),
                    "chunk_text": entity.get("chunk_text"),
                    "chunk_index": entity.get("chunk_index"),
                    "score": float(hit.score),
                    "document_summary": document_summaries.get(document_hash, {}),
                }
            )
        return retrieved

    def clear(self) -> None:
        if self._collection_exists():
            utility.drop_collection(self._collection_name, using=self._milvus_alias)
        if self._db_path.exists():
            with self._connect_db() as conn:
                conn.execute("DELETE FROM policy_documents")
                conn.commit()
        if self._storage_dir.exists():
            for child in self._storage_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_document(self, document_hash: str) -> sqlite3.Row | None:
        with self._connect_db() as conn:
            row = conn.execute(
                """
                SELECT document_hash, filename, chunk_count
                FROM policy_documents
                WHERE document_hash = ?
                """,
                (document_hash,),
            ).fetchone()
        return row

    def _touch_document(self, document_hash: str) -> None:
        now = int(time.time())
        with self._connect_db() as conn:
            conn.execute(
                """
                UPDATE policy_documents
                SET updated_at = ?
                WHERE document_hash = ?
                """,
                (now, document_hash),
            )
            conn.commit()

    def _persist_document(
        self,
        document_hash: str,
        filename: str,
        file_size: int,
        chunk_count: int,
        summary: Dict[str, Any],
        extracted_text: str,
    ) -> None:
        now = int(time.time())
        document_dir = self._storage_dir / document_hash
        document_dir.mkdir(parents=True, exist_ok=True)
        text_path = document_dir / "text.txt"
        summary_path = document_dir / "summary.json"
        text_path.write_text(extracted_text, encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        with self._connect_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO policy_documents (
                    document_hash, filename, file_size, chunk_count, summary_json, text_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_hash,
                    filename,
                    file_size,
                    chunk_count,
                    json.dumps(summary, ensure_ascii=False),
                    str(text_path),
                    now,
                    now,
                ),
            )
            conn.commit()

    def _get_document_summaries(self, document_hashes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        unique_hashes = [document_hash for document_hash in dict.fromkeys(document_hashes) if document_hash]
        if not unique_hashes:
            return {}

        placeholders = ",".join("?" for _ in unique_hashes)
        with self._connect_db() as conn:
            rows = conn.execute(
                f"""
                SELECT document_hash, summary_json
                FROM policy_documents
                WHERE document_hash IN ({placeholders})
                """,
                unique_hashes,
            ).fetchall()

        summaries: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            try:
                summaries[row["document_hash"]] = json.loads(row["summary_json"])
            except json.JSONDecodeError:
                logger.warning("Failed to decode stored summary_json for %s", row["document_hash"])
                summaries[row["document_hash"]] = {}
        return summaries

    def _build_policy_entries(self, filename: str, summary: Dict[str, Any]) -> List[str]:
        entries: List[str] = []

        overview_lines = [
            f"Document: {filename}",
            f"Policy Title: {summary.get('policy_title', filename)}",
            f"Summary: {summary.get('summary', '')}",
            "Rule Type: Overview",
        ]
        required_evidence = [str(item) for item in summary.get("required_evidence", []) if str(item).strip()]
        notes = [str(item) for item in summary.get("notes", []) if str(item).strip()]
        if required_evidence:
            overview_lines.append(f"Required Evidence: {'; '.join(required_evidence)}")
        if notes:
            overview_lines.append(f"Notes: {'; '.join(notes)}")
        entries.append("\n".join(overview_lines))

        for rule_type, rules in (
            ("Blocking", summary.get("blocking_rules", [])),
            ("Permitted", summary.get("permitted_rules", [])),
        ):
            for rule in rules:
                lines = [
                    f"Document: {filename}",
                    f"Policy Title: {summary.get('policy_title', filename)}",
                    f"Summary: {summary.get('summary', '')}",
                    f"Rule Type: {rule_type}",
                    f"Rule Title: {rule.get('title', '')}",
                    f"Conditions: {'; '.join(str(item) for item in rule.get('conditions', []) if str(item).strip())}",
                ]
                signals = [str(item) for item in rule.get("signals", []) if str(item).strip()]
                if signals:
                    lines.append(f"Signals: {'; '.join(signals)}")
                entries.append("\n".join(line for line in lines if line.strip()))

        return [_limit_words(entry, MAX_EMBED_TOTAL_WORDS) for entry in entries if entry.strip()]

    def _format_policy_entry_for_embedding(self, entry_text: str) -> str:
        return _limit_words(entry_text, MAX_EMBED_TOTAL_WORDS)

    def _embed_texts(self, texts: Sequence[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")
        if not api_key:
            raise RuntimeError(EMBEDDING_API_KEY_ERROR)
        client = OpenAI(api_key=api_key, base_url=self._embedding_config["url"])
        response = client.embeddings.create(
            input=list(texts),
            model=self._embedding_config["model"],
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"},
        )
        return [item.embedding for item in response.data]

    def _connect_milvus(self) -> None:
        if self._connected:
            return
        connections.connect(
            alias=self._milvus_alias,
            host=self._milvus_config["host"],
            port=self._milvus_config["port"],
        )
        self._connected = True

    def _collection_exists(self) -> bool:
        self._connect_milvus()
        return utility.has_collection(self._collection_name, using=self._milvus_alias)

    def _ensure_collection(self, dimension: int) -> None:
        self._connect_milvus()
        if utility.has_collection(self._collection_name, using=self._milvus_alias):
            return

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="document_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="policy_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields=fields, description="Persistent normalized policy records")
        collection = Collection(name=self._collection_name, schema=schema, using=self._milvus_alias)
        collection.create_index("embedding", {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}})
        collection.load()

    def _get_collection(self, load: bool = False) -> Collection:
        self._connect_milvus()
        collection = Collection(name=self._collection_name, using=self._milvus_alias)
        if load:
            collection.load()
        return collection

    def _replace_document_vectors(
        self,
        document_hash: str,
        filename: str,
        summary: Dict[str, Any],
        records: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        collection = self._get_collection(load=True)
        collection.delete(expr=f'document_hash == "{document_hash}"')
        entities = [
            [f"{document_hash}:{index}" for index in range(len(records))],
            [document_hash] * len(records),
            [filename] * len(records),
            [str(summary.get("policy_title", filename))] * len(records),
            [str(summary.get("summary", ""))] * len(records),
            list(records),
            list(range(len(records))),
            [list(vector) for vector in vectors],
        ]
        collection.insert(entities)
        collection.flush()
