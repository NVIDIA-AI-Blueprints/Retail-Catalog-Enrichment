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

import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, List

from openai import OpenAI
from pypdf import PdfReader

from backend.config import get_config
from backend.utils import parse_llm_json

logger = logging.getLogger("catalog_enrichment.policy")

MAX_POLICY_TEXT_CHARS = 12000
MAX_POLICY_SUMMARY_CHARS = 6000
NGC_API_KEY_NOT_SET_ERROR = "NGC_API_KEY is not set"
LOCALE_CONFIG = {
    "en-US": {"language": "English", "region": "United States", "country": "United States", "context": "American English with US terminology"},
    "en-GB": {"language": "English", "region": "United Kingdom", "country": "United Kingdom", "context": "British English with UK terminology"},
    "en-AU": {"language": "English", "region": "Australia", "country": "Australia", "context": "Australian English"},
    "en-CA": {"language": "English", "region": "Canada", "country": "Canada", "context": "Canadian English"},
    "es-ES": {"language": "Spanish", "region": "Spain", "country": "Spain", "context": "Peninsular Spanish"},
    "es-MX": {"language": "Spanish", "region": "Mexico", "country": "Mexico", "context": "Mexican Spanish"},
    "es-AR": {"language": "Spanish", "region": "Argentina", "country": "Argentina", "context": "Argentinian Spanish"},
    "es-CO": {"language": "Spanish", "region": "Colombia", "country": "Colombia", "context": "Colombian Spanish"},
    "fr-FR": {"language": "French", "region": "France", "country": "France", "context": "Metropolitan French"},
    "fr-CA": {"language": "French", "region": "Canada", "country": "Canada", "context": "Quebec French"},
}

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte stream."""
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            parts.append(page_text)

    return "\n\n".join(parts).strip()


def summarize_policy_document(document_name: str, document_text: str, locale: str = "en-US") -> Dict[str, Any]:
    """Convert a policy PDF into compact structured rules for indexing and retrieval."""
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config["url"], api_key=api_key)
    info = LOCALE_CONFIG.get(locale, LOCALE_CONFIG["en-US"])
    truncated_text = document_text[:MAX_POLICY_TEXT_CHARS]

    prompt = f"""/no_think You are a policy normalization assistant for an e-commerce catalog team.

Convert the policy document below into concise structured JSON for downstream compliance checks.

DOCUMENT NAME:
{document_name}

TARGET MARKET CONTEXT:
{info["region"]} ({info["context"]})

POLICY DOCUMENT TEXT:
{truncated_text}

Return ONLY valid JSON with this schema:
{{
  "document_name": "{document_name}",
  "policy_title": "<short title>",
  "summary": "<2-3 sentence summary>",
  "blocking_rules": [
    {{
      "title": "<short rule title>",
      "conditions": ["<condition>", "<condition>"],
      "signals": ["<observable signal>", "<observable signal>"]
    }}
  ],
  "permitted_rules": [
    {{
      "title": "<short rule title>",
      "conditions": ["<condition>", "<condition>"]
    }}
  ],
  "required_evidence": ["<what the evaluator must confirm>", "<...>"],
  "notes": ["<important nuance>", "<...>"]
}}

Rules:
- Keep the output compact and focused on classifying products against pass/fail policy checks.
- Prefer observable signals, packaging text, listing text, and ingredient/regulatory markers.
- If the document contains examples, convert them into explicit rules/signals.
- Do not quote long passages verbatim.
"""

    completion = client.chat.completions.create(
        model=llm_config["model"],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1,
        top_p=0.9,
        max_tokens=1600,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        parsed.setdefault("document_name", document_name)
        parsed.setdefault("policy_title", document_name)
        parsed.setdefault("summary", "")
        parsed.setdefault("blocking_rules", [])
        parsed.setdefault("permitted_rules", [])
        parsed.setdefault("required_evidence", [])
        parsed.setdefault("notes", [])
        return parsed

    logger.warning("Policy summary parse failed for %s; falling back to minimal summary", document_name)
    return {
        "document_name": document_name,
        "policy_title": document_name,
        "summary": truncated_text[:400],
        "blocking_rules": [],
        "permitted_rules": [],
        "required_evidence": [],
        "notes": ["Automatic policy summary fallback was used for this document."],
    }


def _prepare_policy_context(policy_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reduce duplicate document-level context while preserving retrieved policy records."""
    prepared: List[Dict[str, Any]] = []
    document_hashes_with_summary: set[str] = set()

    for item in policy_context:
        document_hash = str(item.get("document_hash", ""))
        prepared_item = {
            "document_hash": document_hash,
            "document_name": item.get("document_name"),
            "policy_title": item.get("policy_title"),
            "chunk_index": item.get("chunk_index"),
            "score": item.get("score"),
            "chunk_text": item.get("chunk_text"),
        }
        if document_hash and document_hash not in document_hashes_with_summary and item.get("document_summary"):
            prepared_item["document_summary"] = item.get("document_summary")
            document_hashes_with_summary.add(document_hash)
        elif item.get("document_summary"):
            prepared_item["document_summary"] = item.get("document_summary")
        elif any(
            key in item
            for key in ("summary", "blocking_rules", "permitted_rules", "required_evidence", "notes")
        ):
            prepared_item["document_summary"] = {
                key: item.get(key)
                for key in ("document_name", "policy_title", "summary", "blocking_rules", "permitted_rules", "required_evidence", "notes")
                if key in item
            }
        prepared.append(prepared_item)

    return prepared


def _format_product_snapshot_for_policy(product_snapshot: Dict[str, Any]) -> str:
    primary_lines = [
        f"Observed title: {product_snapshot.get('title', '')}",
        f"Observed description: {product_snapshot.get('description', '')}",
        f"Observed categories: {', '.join(product_snapshot.get('categories', []))}",
        f"Observed tags: {', '.join(product_snapshot.get('tags', []))}",
        f"Observed colors: {', '.join(product_snapshot.get('colors', []))}",
    ]

    generated = product_snapshot.get("generated_catalog_fields") or {}
    secondary_lines = []
    if generated:
        secondary_lines = [
            f"Generated title: {generated.get('title', '')}",
            f"Generated description: {generated.get('description', '')}",
            f"Generated categories: {', '.join(generated.get('categories', []))}",
            f"Generated tags: {', '.join(generated.get('tags', []))}",
        ]

    sections = [
        "PRIMARY PRODUCT EVIDENCE:",
        "\n".join(line for line in primary_lines if line.strip()),
    ]
    if secondary_lines:
        sections.extend(
            [
                "SECONDARY GENERATED CATALOG CONTEXT:",
                "\n".join(line for line in secondary_lines if line.strip()),
            ]
        )
    return "\n\n".join(section for section in sections if section.strip())


def _format_policy_context_for_policy(prepared_policy_context: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for item in prepared_policy_context:
        document_summary = item.get("document_summary") or {}
        blocking_rules = document_summary.get("blocking_rules") or []
        permitted_rules = document_summary.get("permitted_rules") or []
        required_evidence = document_summary.get("required_evidence") or []
        blocking_titles = ", ".join(
            str(rule.get("title", "")).strip()
            for rule in blocking_rules
            if str(rule.get("title", "")).strip()
        )
        permitted_titles = ", ".join(
            str(rule.get("title", "")).strip()
            for rule in permitted_rules
            if str(rule.get("title", "")).strip()
        )
        section_lines = [
            f"Document: {item.get('document_name', '')}",
            f"Policy title: {item.get('policy_title', '')}",
            f"Chunk index: {item.get('chunk_index', '')}",
            f"Similarity score: {item.get('score', '')}",
            f"Policy summary: {document_summary.get('summary') or item.get('summary', '')}",
            f"Blocking rules: {blocking_titles}",
            f"Permitted rules: {permitted_titles}",
            f"Required evidence: {', '.join(str(entry) for entry in required_evidence if str(entry).strip())}",
            f"Retrieved chunk: {item.get('chunk_text', '')}",
        ]
        sections.append("\n".join(line for line in section_lines if line.strip()))
    return "\n\n---\n\n".join(sections)


def _is_policy_decision_consistent(decision: Dict[str, Any]) -> bool:
    status = str(decision.get("status", "pass"))
    matched_policies = decision.get("matched_policies")
    if not isinstance(matched_policies, list):
        return False
    if status == "pass" and matched_policies:
        return False
    if status == "fail" and not matched_policies:
        return False
    return True


def _repair_policy_decision(
    client: OpenAI,
    model: str,
    locale_info: Dict[str, str],
    product_json: str,
    policy_json: str,
    product_evidence_text: str,
    policy_evidence_text: str,
    candidate_decision: Dict[str, Any],
) -> Dict[str, Any] | None:
    candidate_json = json.dumps(candidate_decision, ensure_ascii=False)
    prompt = f"""/no_think You are repairing a malformed catalog compliance decision.

The candidate JSON below is internally inconsistent. Rewrite it so the final JSON is both accurate and structurally valid.

TARGET MARKET CONTEXT:
{locale_info["region"]} ({locale_info["context"]})

PRODUCT SNAPSHOT:
{product_json}

RETRIEVED POLICY CONTEXT:
{policy_json}

FOCUSED PRODUCT EVIDENCE:
{product_evidence_text}

FOCUSED POLICY EVIDENCE:
{policy_evidence_text}

INCONSISTENT CANDIDATE DECISION:
{candidate_json}

Return ONLY valid JSON with this schema:
{{
  "status": "pass" | "fail",
  "label": "<short label>",
  "summary": "<one sentence>",
  "matched_policies": [
    {{
      "document_name": "<pdf filename>",
      "policy_title": "<policy title>",
      "rule_title": "<matched rule>",
      "reason": "<why it matched>",
      "evidence": ["<evidence item>", "<evidence item>"]
    }}
  ],
  "warnings": ["<uncertainty or missing evidence>", "<...>"],
  "evidence_note": "<brief note describing what evidence was used>"
}}

Rules:
- Keep the decision faithful to the supplied product and policy context.
- If status is "pass", matched_policies must be empty.
- If status is "fail", matched_policies must contain at least one supporting rule match.
- Keep the response concise and internally consistent.
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1,
        top_p=0.9,
        max_tokens=900,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )
    return parse_llm_json(text, extract_braces=True, strip_comments=True)


def evaluate_policy_compliance(
    product_snapshot: Dict[str, Any],
    policy_context: List[Dict[str, Any]],
    locale: str = "en-US",
) -> Dict[str, Any]:
    """Classify the analyzed product against retrieved policy context."""
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise RuntimeError(NGC_API_KEY_NOT_SET_ERROR)

    llm_config = get_config().get_llm_config()
    client = OpenAI(base_url=llm_config["url"], api_key=api_key)
    info = LOCALE_CONFIG.get(locale, LOCALE_CONFIG["en-US"])

    prepared_policy_context = _prepare_policy_context(policy_context)
    policy_json = json.dumps(prepared_policy_context, ensure_ascii=False)[:MAX_POLICY_SUMMARY_CHARS * max(len(prepared_policy_context), 1)]
    product_json = json.dumps(product_snapshot, ensure_ascii=False)
    product_evidence_text = _format_product_snapshot_for_policy(product_snapshot)
    policy_evidence_text = _format_policy_context_for_policy(prepared_policy_context)

    prompt = f"""/no_think You are a catalog compliance reviewer.

Review the product below against the uploaded policy summaries. The UI supports two statuses:
- pass
- fail

Choose the best-fit classification based on the observed product title, description, and retrieved policy records.

TARGET MARKET CONTEXT:
{info["region"]} ({info["context"]})

PRODUCT SNAPSHOT:
{product_json}

RETRIEVED POLICY CONTEXT:
{policy_json}

FOCUSED PRODUCT EVIDENCE:
{product_evidence_text}

FOCUSED POLICY EVIDENCE:
{policy_evidence_text}

Return ONLY valid JSON with this schema:
{{
  "status": "pass" | "fail",
  "label": "<short label>",
  "summary": "<one sentence>",
  "matched_policies": [
    {{
      "document_name": "<pdf filename>",
      "policy_title": "<policy title>",
      "rule_title": "<matched rule>",
      "reason": "<why it matched>",
      "evidence": ["<evidence item>", "<evidence item>"]
    }}
  ],
  "warnings": ["<uncertainty or missing evidence>", "<...>"],
  "evidence_note": "<brief note describing what evidence was used>"
}}

Rules:
- Use "fail" if any policy clearly disallows the product.
- matched_policies must be empty when status is "pass".
- Be specific and short.
- Base the decision only on the supplied product snapshot and policies.
- Treat the top-level product fields as the primary evidence source. Those fields represent the raw product observation.
- Treat generated_catalog_fields as secondary context only.
- Prefer direct product evidence from the title, visible text, form, components, and retrieved policy records over polished marketing language.
- Do not require exact literal keyword equality when close lexical variants, inflections, or obvious wording variants point to the same product type and the product's form or function also aligns with the policy.
- Prefer "fail" when the product's observed title, visible text, or described function clearly names or strongly implies a blocked product family in the policy and there is no stronger allowed-category match.
- Treat blocking-rule conditions, listed keywords, and listed signals as alternative supporting indicators unless the policy explicitly says all of them are required together.
- Do not require every example component or every listed signal to be present when the product already strongly matches a blocked product family through title, visible text, or described purpose.
- Do not assume a product passes just because the listing does not explicitly state an end use if the retrieved policies define blocking by function, form, components, or keywords.
- Use the retrieved policy records as the policy source of truth.
- Before returning JSON, verify that status, summary, matched_policies, warnings, and evidence_note are internally consistent.
- If status is "pass", summary must clearly say that no retrieved policy blocks the product.
- If status is "fail", summary must clearly say that the product does not comply and matched_policies must contain the supporting rule matches.
"""

    completion = client.chat.completions.create(
        model=llm_config["model"],
        messages=[{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}],
        temperature=0.1,
        top_p=0.9,
        max_tokens=1200,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in completion
        if chunk.choices[0].delta and chunk.choices[0].delta.content
    )

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if parsed is not None:
        parsed_status = str(parsed.get("status", "pass"))
        if parsed_status not in {"pass", "fail"}:
            parsed_status = "pass"
        parsed["status"] = parsed_status
        parsed.setdefault(
            "label",
            "Policy Check Failed" if parsed["status"] == "fail" else "Policy Check Passed",
        )
        parsed.setdefault("summary", "")
        parsed.setdefault("matched_policies", [])
        parsed.setdefault("warnings", [])
        parsed.setdefault("evidence_note", "")
        if parsed["status"] == "pass":
            parsed["matched_policies"] = []
        if not _is_policy_decision_consistent(parsed):
            logger.warning(
                "Policy decision was internally inconsistent; attempting repair. status=%s matched=%d",
                parsed.get("status"),
                len(parsed.get("matched_policies", [])) if isinstance(parsed.get("matched_policies"), list) else -1,
            )
            repaired = _repair_policy_decision(
                client,
                llm_config["model"],
                info,
                product_json,
                policy_json,
                product_evidence_text,
                policy_evidence_text,
                parsed,
            )
            if repaired is not None:
                repaired_status = str(repaired.get("status", "pass"))
                if repaired_status not in {"pass", "fail"}:
                    repaired_status = "pass"
                repaired["status"] = repaired_status
                repaired.setdefault(
                    "label",
                    "Policy Check Failed" if repaired["status"] == "fail" else "Policy Check Passed",
                )
                repaired.setdefault("summary", "")
                repaired.setdefault("matched_policies", [])
                repaired.setdefault("warnings", [])
                repaired.setdefault("evidence_note", "")
                if repaired["status"] == "pass":
                    repaired["matched_policies"] = []
                if _is_policy_decision_consistent(repaired):
                    return repaired
            logger.warning("Policy decision repair failed; using fallback pass result")
            return {
                "status": "pass",
                "label": "Policy Check Passed",
                "summary": "No retrieved policy blocks this product.",
                "matched_policies": [],
                "warnings": ["Policy evaluation used a fallback pass result because the model response was internally inconsistent."],
                "evidence_note": "Fallback decision based on inconsistent model output.",
            }
        return parsed

    logger.warning("Policy compliance parse failed; falling back to pass result")
    return {
        "status": "pass",
        "label": "Policy Check Passed",
        "summary": "No retrieved policy blocks this product.",
        "matched_policies": [],
        "warnings": ["Policy evaluation used a fallback pass result because the model response was malformed."],
        "evidence_note": "Fallback decision based on parser failure.",
    }
