# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the persistent policy library.
"""
from types import SimpleNamespace
from unittest.mock import patch

from backend.policy_library import PolicyLibrary, build_policy_query


class TestBuildPolicyQuery:
    def test_build_policy_query_includes_product_evidence(self):
        query = build_policy_query(
            {
                "title": "Catalog Item",
                "description": "Demo description",
                "categories": ["bags"],
                "tags": ["premium", "structured"],
                "colors": ["black"],
            }
        )

        assert "Catalog Item" in query
        assert "bags" in query
        assert "black" in query

    def test_build_policy_query_is_trimmed_for_embedding_limit(self):
        long_description = "word " * 400
        query = build_policy_query(
            {
                "title": "Catalog Item",
                "description": long_description,
                "categories": ["bags"],
                "tags": ["premium"],
                "colors": ["black"],
            }
        )

        assert len(query.split()) <= 160


class TestPolicyLibrary:
    @patch("backend.policy_library.summarize_policy_document")
    @patch("backend.policy_library.extract_text_from_pdf_bytes")
    def test_ingest_documents_deduplicates_by_hash(
        self,
        mock_extract_text,
        mock_summarize,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))

        library = PolicyLibrary()
        library.initialize()

        mock_extract_text.return_value = "Policy text for dedupe test"
        mock_summarize.return_value = {"policy_title": "Policy", "summary": "Summary"}

        with (
            patch.object(library, "_embed_texts", return_value=[[0.1, 0.2]]),
            patch.object(library, "_ensure_collection"),
            patch.object(library, "_replace_document_vectors"),
        ):
            first = library.ingest_documents([{"filename": "policy-a.pdf", "bytes": b"same-content"}])
            second = library.ingest_documents([{"filename": "policy-b.pdf", "bytes": b"same-content"}])

        assert first[0]["already_loaded"] is False
        assert second[0]["already_loaded"] is True
        assert len(library.list_documents()) == 1

    @patch("backend.policy_library.utility")
    def test_clear_removes_metadata_and_storage(self, mock_utility, monkeypatch, tmp_path):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))

        library = PolicyLibrary()
        library.initialize()
        with library._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO policy_documents (
                    document_hash, filename, file_size, chunk_count, summary_json, text_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("abc", "policy.pdf", 123, 1, "{}", str(tmp_path / "policies" / "abc" / "text.txt"), 1, 1),
            )
            conn.commit()

        (tmp_path / "policies" / "abc").mkdir(parents=True, exist_ok=True)
        (tmp_path / "policies" / "abc" / "text.txt").write_text("hello", encoding="utf-8")

        with patch.object(library, "_collection_exists", return_value=True):
            library.clear()

        assert library.list_documents() == []
        assert (tmp_path / "policies").exists()
        mock_utility.drop_collection.assert_called_once()

    def test_retrieve_context_returns_search_results(self, monkeypatch, tmp_path):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))

        library = PolicyLibrary()
        library.initialize()
        summary_json = {"policy_title": "Policy", "summary": "Summary", "blocking_rules": []}
        with library._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO policy_documents (
                    document_hash, filename, file_size, chunk_count, summary_json, text_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("abc", "policy.pdf", 123, 1, '{"policy_title":"Policy","summary":"Summary","blocking_rules":[]}', str(tmp_path / "policies" / "abc" / "text.txt"), 1, 1),
            )
            conn.commit()

        fake_hit = SimpleNamespace(
            score=0.99,
            entity={
                "document_hash": "abc",
                "document_name": "policy.pdf",
                "policy_title": "Policy",
                "summary": "Summary",
                "chunk_text": "Chunk body",
                "chunk_index": 0,
            },
        )
        fake_collection = SimpleNamespace(search=lambda **kwargs: [[fake_hit]])

        with (
            patch.object(library, "_collection_exists", return_value=True),
            patch.object(library, "_embed_texts", return_value=[[0.1, 0.2]]),
            patch.object(library, "_get_collection", return_value=fake_collection),
        ):
            result = library.retrieve_context({"title": "Catalog Item", "description": "Demo"})

        assert result[0]["document_name"] == "policy.pdf"
        assert result[0]["chunk_text"] == "Chunk body"
        assert result[0]["document_summary"] == summary_json

    def test_retrieve_context_skips_low_relevance_candidates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))
        monkeypatch.setenv("POLICY_LIBRARY_MIN_RELEVANCE_SCORE", "0.3")

        library = PolicyLibrary()
        library.initialize()
        with library._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO policy_documents (
                    document_hash, filename, file_size, chunk_count, summary_json, text_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("abc", "policy.pdf", 123, 1, '{"policy_title":"Policy","summary":"Summary"}', str(tmp_path / "policies" / "abc" / "text.txt"), 1, 1),
            )
            conn.commit()

        low_score_hit = SimpleNamespace(
            score=0.24,
            entity={
                "document_hash": "abc",
                "document_name": "policy.pdf",
                "policy_title": "Policy",
                "summary": "Summary",
                "chunk_text": "Chunk body",
                "chunk_index": 0,
            },
        )
        fake_collection = SimpleNamespace(search=lambda **kwargs: [[low_score_hit]])

        with (
            patch.object(library, "_collection_exists", return_value=True),
            patch.object(library, "_embed_texts", return_value=[[0.1, 0.2]]),
            patch.object(library, "_get_collection", return_value=fake_collection),
        ):
            result = library.retrieve_context({"title": "Catalog Item", "description": "Demo"})

        assert result == []

    def test_retrieve_context_keeps_candidates_above_relevance_threshold(self, monkeypatch, tmp_path):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))
        monkeypatch.setenv("POLICY_LIBRARY_MIN_RELEVANCE_SCORE", "0.3")

        library = PolicyLibrary()
        library.initialize()
        with library._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO policy_documents (
                    document_hash, filename, file_size, chunk_count, summary_json, text_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("abc", "policy.pdf", 123, 1, '{"policy_title":"Policy","summary":"Summary"}', str(tmp_path / "policies" / "abc" / "text.txt"), 1, 1),
            )
            conn.commit()

        hit = SimpleNamespace(
            score=0.41,
            entity={
                "document_hash": "abc",
                "document_name": "policy.pdf",
                "policy_title": "Policy",
                "summary": "Summary",
                "chunk_text": "Chunk body",
                "chunk_index": 0,
            },
        )
        fake_collection = SimpleNamespace(search=lambda **kwargs: [[hit]])

        with (
            patch.object(library, "_collection_exists", return_value=True),
            patch.object(library, "_embed_texts", return_value=[[0.1, 0.2]]),
            patch.object(library, "_get_collection", return_value=fake_collection),
        ):
            result = library.retrieve_context({"title": "Catalog Item", "description": "Demo"})

        assert len(result) == 1
        assert result[0]["score"] == 0.41

    def test_build_policy_entries_uses_normalized_summary(self, monkeypatch, tmp_path):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))

        library = PolicyLibrary()
        entries = library._build_policy_entries(
            "policy.pdf",
            {
                "policy_title": "Policy",
                "summary": "Normalized policy summary",
                "blocking_rules": [
                    {
                        "title": "Blocking Rule",
                        "conditions": ["Condition A", "Condition B"],
                        "signals": ["Signal A"],
                    }
                ],
                "permitted_rules": [
                    {
                        "title": "Permitted Rule",
                        "conditions": ["Condition C"],
                    }
                ],
                "required_evidence": ["Evidence A"],
                "notes": ["Note A"],
            },
        )

        assert len(entries) == 3
        assert any("Rule Type: Blocking" in entry for entry in entries)
        assert any("Rule Type: Permitted" in entry for entry in entries)

    def test_format_policy_entry_for_embedding_trims_text(self, monkeypatch, tmp_path):
        monkeypatch.setenv("POLICY_LIBRARY_STORAGE_DIR", str(tmp_path / "policies"))
        monkeypatch.setenv("POLICY_LIBRARY_DB_PATH", str(tmp_path / "policies" / "library.db"))

        library = PolicyLibrary()
        formatted = library._format_policy_entry_for_embedding("entry " * 400)

        assert len(formatted.split()) <= 190
