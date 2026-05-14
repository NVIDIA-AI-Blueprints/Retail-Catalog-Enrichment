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
import re
from typing import Any, Optional

from deepagents import create_deep_agent
from exa_py import Exa
from langchain_openai import ChatOpenAI

from backend.config import get_config
from backend.utils import parse_llm_json
from backend.vlm import LOCALE_CONFIG, NGC_API_KEY_NOT_SET_ERROR

logger = logging.getLogger("catalog_enrichment.web_insights")

WEB_INSIGHTS_LIST_FIELDS = (
    "pros",
    "cons",
    "use_cases",
    "customer_insights",
    "purchase_considerations",
    "search_queries",
    "warnings",
)

REPORT_METRIC_DEFAULTS = {
    "customer_sentiment": {"label": "Not enough data", "scale": "percent"},
    "build_quality": {"label": "Not enough data", "scale": "label"},
    "price_segment": {"label": "Not enough data", "scale": "label"},
    "retail_confidence": {"label": "Not enough data", "scale": "rating_10"},
}
PLACEHOLDER_METRIC_LABELS = {
    "",
    "qualitative",
    "qualitative label",
    "qualitative positive",
    "qualitative negative",
    "qualitative neutral",
    "label",
    "not enough data",
    "not enough source data",
    "unknown",
    "n/a",
    "none",
    "varies",
    "varies by retailer",
    "varies by retailers",
    "retailer dependent",
}
SOURCE_BOILERPLATE_PATTERNS = (
    "your browser is no longer supported",
    "please use an alternative browser",
    "skip to main content",
    "chat360",
    "addingaddingadd to cart",
    "successsuccessadded",
    "errorerror",
)
SOURCE_BOILERPLATE_CLAIM_PATTERNS = (
    "browser compatibility",
    "browser is no longer supported",
    "unsupported browser",
    "alternative browser",
    "limits access to full product details",
    "limiting access to full product details",
)

RESEARCH_SCOPES = {"product_specific", "brand_level", "category_level", "insufficient_identity"}
IDENTITY_CONFIDENCE_LEVELS = {"high", "medium", "low", "none"}
SCOPE_DEFAULTS = {
    "product_specific": {
        "identity_confidence": "high",
        "scope_note": "Sources appear to match a specific product identity.",
    },
    "brand_level": {
        "identity_confidence": "medium",
        "scope_note": "A likely brand was identified, but no exact model was confirmed; insights may mix nearby brand products.",
    },
    "category_level": {
        "identity_confidence": "low",
        "scope_note": "No reliable brand or model was identified; insights are scoped to the broader product category.",
    },
    "insufficient_identity": {
        "identity_confidence": "none",
        "scope_note": "The product identity is too broad for reliable web research; use these insights only as general category context.",
    },
}
MISSING_SCOPE_NOTE = (
    "The research agent did not return a supported identity scope; insights are treated as category-level context."
)
EXA_API_KEY_NOT_CONFIGURED_MESSAGE = (
    "Web Insights are unavailable because EXA_API_KEY is not configured. "
    "Add EXA_API_KEY to the backend environment to enable product web research."
)


class WebInsightsDependencyError(RuntimeError):
    """Raised when the web insights external dependencies are unavailable."""


def build_web_insights_disabled_response(
    *,
    locale: str = "en-US",
    reason: str = EXA_API_KEY_NOT_CONFIGURED_MESSAGE,
) -> dict[str, Any]:
    """Return a non-error Web Insights payload for optional dependency gaps."""
    return {
        "status": "disabled",
        "disabled_reason": reason,
        "summary": "",
        "pros": [],
        "cons": [],
        "use_cases": [],
        "customer_insights": [],
        "purchase_considerations": [],
        "search_queries": [],
        "sources": [],
        "warnings": [reason],
        "locale": locale,
        "research_scope": "insufficient_identity",
        "identity_confidence": "none",
        "detected_brand": None,
        "detected_model": None,
        "scope_note": reason,
        "identity_evidence": [],
        "report": {
            "executive_summary": "",
            "positioning_tags": [],
            "metrics": _default_metrics(),
            "retail_insights": [],
            "primary_use_cases": [],
            "customer_sentiment_summary": "",
        },
    }


def build_product_web_insights(
    *,
    title: str,
    description: str = "",
    categories: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    locale: str = "en-US",
    max_results: Optional[int] = None,
) -> dict[str, Any]:
    """Run a Deep Agents product research pass and return normalized insights."""
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        logger.info("[Web Insights] disabled because EXA_API_KEY is not configured")
        return build_web_insights_disabled_response(locale=locale)
    if not (api_key := os.getenv("NGC_API_KEY")):
        raise WebInsightsDependencyError(NGC_API_KEY_NOT_SET_ERROR)

    config = get_config()
    llm_config = config.get_llm_config()
    insights_config = config.get_web_insights_config()
    effective_max_results = _clamp_int(max_results or insights_config["max_results"], minimum=1, maximum=20)
    max_sources = _clamp_int(insights_config["max_sources"], minimum=1, maximum=20)
    min_sources = _clamp_int(insights_config["min_sources"], minimum=0, maximum=max_sources)
    highlights_max_characters = _clamp_int(
        insights_config["highlights_max_characters"], minimum=500, maximum=12000
    )
    search_type = str(insights_config["search_type"] or "auto")

    exa = Exa(api_key=exa_api_key)
    collected_sources: list[dict[str, Any]] = []
    search_queries: list[str] = []

    def web_search(query: str, max_results: Optional[int] = None) -> str:
        """Search public web sources for product information and return source highlights."""
        sources = _search_exa_sources(
            exa,
            query,
            search_type=search_type,
            max_results=max_results or effective_max_results,
            highlights_max_characters=highlights_max_characters,
            query_log=search_queries,
        )
        collected_sources.extend(sources)
        return json.dumps({"results": sources}, ensure_ascii=False)

    model = ChatOpenAI(
        model=llm_config["model"],
        base_url=llm_config["url"],
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
        timeout=60,
        max_retries=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    agent = create_deep_agent(
        model=model,
        tools=[web_search],
        system_prompt=_build_system_prompt(locale),
    )

    try:
        user_prompt = _build_user_prompt(
            title=title,
            description=description,
            categories=categories or [],
            tags=tags or [],
            locale=locale,
            max_results=effective_max_results,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        final_text = _extract_final_message_text(result)
        parsed = _parse_agent_json(final_text)
        if not collected_sources:
            fallback_queries = _fallback_search_queries(
                title=title,
                categories=categories or [],
                tags=tags or [],
                parsed=parsed if isinstance(parsed, dict) else {},
            )
            for query in fallback_queries:
                collected_sources.extend(
                    _search_exa_sources(
                        exa,
                        query,
                        search_type=search_type,
                        max_results=effective_max_results,
                        highlights_max_characters=highlights_max_characters,
                        query_log=search_queries,
                    )
                )
                if len(_dedupe_sources(collected_sources)) >= max_sources:
                    break
            collected_sources = _dedupe_sources(collected_sources)[:max_sources]
            if collected_sources:
                logger.info("[Web Insights] using fallback source packet with %d sources", len(collected_sources))
                grounded_prompt = _build_source_grounded_prompt(
                    title=title,
                    description=description,
                    categories=categories or [],
                    tags=tags or [],
                    locale=locale,
                    sources=collected_sources,
                )
                grounded_result = agent.invoke({"messages": [{"role": "user", "content": grounded_prompt}]})
                grounded_text = _extract_final_message_text(grounded_result)
                grounded_parsed = _parse_agent_json(grounded_text)
                if isinstance(grounded_parsed, dict):
                    final_text = grounded_text
                    parsed = grounded_parsed
        collected_sources = _ensure_price_sources(
            exa,
            title=title,
            parsed=parsed if isinstance(parsed, dict) else {},
            collected_sources=collected_sources,
            search_type=search_type,
            max_results=effective_max_results,
            highlights_max_characters=highlights_max_characters,
            query_log=search_queries,
            max_sources=max_sources,
        )
    except Exception as exc:
        logger.exception("[Web Insights] agent invocation failed: %s", exc)
        raise WebInsightsDependencyError(str(exc)) from exc

    return normalize_web_insights_output(
        parsed if parsed is not None else {},
        fallback_summary=final_text,
        collected_sources=collected_sources,
        search_queries=search_queries,
        locale=locale,
        min_sources=min_sources,
        max_sources=max_sources,
    )


def normalize_web_insights_output(
    parsed: dict[str, Any],
    *,
    fallback_summary: str = "",
    collected_sources: Optional[list[dict[str, Any]]] = None,
    search_queries: Optional[list[str]] = None,
    locale: str = "en-US",
    min_sources: int = 2,
    max_sources: int = 8,
) -> dict[str, Any]:
    """Coerce agent output into the public API schema."""
    if not isinstance(parsed, dict):
        parsed = {}
    sources = _limit_sources(collected_sources or _coerce_sources(parsed.get("sources", [])), max_sources=max_sources)
    warnings = _clean_string_list(parsed.get("warnings", []), limit=8, filter_source_boilerplate=True)
    if len(sources) < min_sources:
        warnings.append(
            f"Only {len(sources)} relevant source{'s' if len(sources) != 1 else ''} found; review coverage before using these insights."
        )

    has_sources = len(sources) > 0
    has_claim_evidence = any(_clean_string(source.get("snippet")) for source in sources)
    if has_sources and not has_claim_evidence:
        warnings.append(
        "Found matching source URLs, but Exa returned no snippets or text excerpts; report claims were suppressed."
        )
    identity = _normalize_identity(parsed, has_sources=has_sources)
    report_source = parsed.get("report", {}) if isinstance(parsed.get("report", {}), dict) else {}
    report_summary = _clean_string(report_source.get("executive_summary"))

    output = {
        "summary": (
            _clean_string(parsed.get("summary")) or report_summary or _clean_string(fallback_summary)
            if has_claim_evidence
            else "Found matching web sources, but no source snippets were available to generate a source-backed report."
            if has_sources
            else "No source-backed web insights were generated for this product."
        ),
        "pros": _clean_string_list(parsed.get("pros", []), limit=8, filter_source_boilerplate=True)
        if has_claim_evidence
        else [],
        "cons": _clean_string_list(parsed.get("cons", []), limit=8, filter_source_boilerplate=True)
        if has_claim_evidence
        else [],
        "use_cases": _clean_string_list(parsed.get("use_cases", []), limit=8, filter_source_boilerplate=True)
        if has_claim_evidence
        else [],
        "customer_insights": _clean_string_list(
            parsed.get("customer_insights", []), limit=8, filter_source_boilerplate=True
        )
        if has_claim_evidence
        else [],
        "purchase_considerations": (
            _clean_string_list(parsed.get("purchase_considerations", []), limit=8, filter_source_boilerplate=True)
            if has_claim_evidence
            else []
        ),
        "search_queries": _clean_string_list(parsed.get("search_queries", []) or search_queries or [], limit=10),
        "sources": sources,
        "warnings": warnings,
        "locale": locale,
        "research_scope": identity["research_scope"],
        "identity_confidence": identity["identity_confidence"],
        "detected_brand": identity["detected_brand"],
        "detected_model": identity["detected_model"],
        "scope_note": identity["scope_note"],
        "identity_evidence": identity["identity_evidence"],
    }

    if not output["summary"] and not any(output[field] for field in WEB_INSIGHTS_LIST_FIELDS if field != "warnings"):
        output["summary"] = "No source-backed web insights were generated for this product."
    output["report"] = _normalize_report(report_source, output, has_sources=has_claim_evidence, identity=identity)
    return output


def _build_system_prompt(locale: str) -> str:
    info = LOCALE_CONFIG.get(locale, LOCALE_CONFIG["en-US"])
    return f"""/no_think You are a retail product web research analyst.

Use the web_search tool to research public web sources about the product and brand. Use broad public web coverage, including official product pages, retailer pages, reviews, forums, support pages, and relevant articles when useful.

Rules:
- Use the product title as the primary identity signal.
- Use description, categories, and tags only to reduce ambiguity.
- Classify the research scope before making claims:
  - product_specific: exact brand/model or unique product identity is supported by the title and sources.
  - brand_level: a likely brand is supported, but no exact model is confirmed.
  - category_level: no reliable brand or model is present; research only the broader product category.
  - insufficient_identity: the title is too vague for useful source-backed research.
- Brand/model detection must be source-evidence-based, not token-position-based. Do not assume the first word is or is not a brand.
- Treat a title token as a brand only when source evidence identifies it as a brand/manufacturer or uses it on an official or retailer product page for the same product type.
- Do not infer a brand or model from a source unless the source identity clearly matches the provided title, description, categories, or tags.
- Source titles and URLs can support identity classification, but review sentiment, complaints, ratings, price, quality, and use-case claims require source snippets, highlights, or text excerpts that explicitly support them.
- Ignore website boilerplate and access chrome such as unsupported-browser messages, skip links, add-to-cart status text, cookie banners, navigation, and generic page errors. These are source extraction artifacts, not product or retail insights.
- For category_level or insufficient_identity, do not return product-specific customer sentiment percentages or retail confidence scores. Use null scores and labels such as "Category signal", "Unknown", or "Limited".
- Do not invent claims.
- Every user-visible claim about pros, cons, customer feedback, usage, comparisons, complaints, or purchase considerations must be grounded in the returned sources.
- Scores must be evidence-backed. If sources are too thin or contradictory for a score, set score to null and use a specific label such as "Review signal unavailable", "Build signal unavailable", "Price unavailable", or "Limited".
- Never return literal placeholder labels such as "qualitative label" or "label".
- Keep the output concise and useful for catalog managers.
- Write the summary and bullets in {info["language"]} for {info["region"]}.
- Return ONLY valid JSON. No markdown, no comments.

Return JSON with these keys:
summary, pros, cons, use_cases, customer_insights, purchase_considerations, search_queries, sources, warnings, research_scope, identity_confidence, detected_brand, detected_model, scope_note, identity_evidence, report.

pros, cons, use_cases, customer_insights, purchase_considerations, search_queries, warnings, and identity_evidence must be arrays of strings.
research_scope must be one of product_specific, brand_level, category_level, or insufficient_identity.
identity_confidence must be one of high, medium, low, or none.
detected_brand and detected_model must be strings or null.
identity_evidence must briefly state what source evidence supports the brand/model/scope decision.

The report object must use this shape:
{{
  "executive_summary": "two concise source-backed paragraphs",
  "positioning_tags": ["short merchandising tags"],
  "metrics": {{
    "customer_sentiment": {{"label": "Positive, Mixed, Neutral, Negative, or Review signal unavailable", "score": 0-100 or null, "rationale": "short source-backed reason"}},
    "build_quality": {{"label": "Premium, Durable, Standard, Mixed, or Build signal unavailable", "score": 0-10 or null, "rationale": "short source-backed reason"}},
    "price_segment": {{"label": "Budget, Mid-range, Premium, High-end, or Price unavailable", "score": null, "rationale": "short source-backed reason with price range when available"}},
    "retail_confidence": {{"label": "High, Moderate, Low, or Limited", "score": 0-10 or null, "rationale": "short source-backed reason"}}
  }},
  "retail_insights": [{{"type": "positive or negative", "title": "short title", "detail": "source-backed detail"}}],
  "primary_use_cases": [{{"title": "short use case", "detail": "source-backed detail"}}],
  "customer_sentiment_summary": "short source-backed sentiment narrative"
}}."""


def _build_user_prompt(
    *,
    title: str,
    description: str,
    categories: list[str],
    tags: list[str],
    locale: str,
    max_results: int,
) -> str:
    return f"""Research public web insights for this product.

TITLE:
{title}

DESCRIPTION:
{description}

CATEGORIES:
{json.dumps(categories, ensure_ascii=False)}

TAGS:
{json.dumps(tags, ensure_ascii=False)}

LOCALE:
{locale}

Search requirements:
- First search the full title exactly enough to identify whether public sources confirm a brand, manufacturer, model, or exact product family.
- Run targeted searches for exact product identity, reviews, retailer listings, official brand or manufacturer pages, price positioning, complaints or common problems, customer sentiment, and practical use cases.
- If full-title searches do not confirm a brand or model, broaden to category-level evidence using title attributes, categories, and tags. Label the result as category_level or insufficient_identity and phrase insights as general category patterns.
- Use at most {max_results} results per search call.
- Look for pros, cons, product information, customer insights, pricing context, purchase risks, and how people use the product.
- Prefer source diversity and avoid duplicate pages from the same result set.
- Prefer official, retailer, review, support, and forum sources when relevant.
- Include identity_evidence entries that explain why the scope and detected_brand/detected_model were chosen.

Return only the JSON object."""


def _normalize_exa_result(result: Any) -> dict[str, Any]:
    snippet_parts: list[str] = []
    existing_snippet = _clean_string(_get_value(result, "snippet"))
    if existing_snippet:
        snippet_parts.append(existing_snippet)
    highlights = _get_value(result, "highlights", []) or []
    if isinstance(highlights, str):
        highlights = [highlights]
    snippet_parts.extend(_clean_string(item) for item in highlights if _clean_string(item))
    text = _clean_string(_get_value(result, "text"))
    if text:
        snippet_parts.append(text)
    snippet = _clean_source_snippet(" ".join(snippet_parts))
    return {
        "title": _clean_string(_get_value(result, "title")),
        "url": _clean_string(_get_value(result, "url")),
        "published_date": _clean_string(_get_value(result, "published_date")) or None,
        "snippet": snippet[:700],
    }


def _search_exa_sources(
    exa: Exa,
    query: str,
    *,
    search_type: str,
    max_results: int,
    highlights_max_characters: int,
    query_log: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    query = _clean_string(query)
    if not query:
        return []

    requested_results = _clamp_int(max_results, minimum=1, maximum=20)
    if query_log is not None:
        query_log.append(query)
    logger.info("[Web Insights] Exa search query=%s max_results=%d", query, requested_results)

    response = exa.search(
        query,
        type=search_type,
        num_results=requested_results,
        contents={
            "highlights": {"query": query, "max_characters": highlights_max_characters},
            "text": {"max_characters": highlights_max_characters},
            "livecrawl": "fallback",
        },
        moderation=True,
    )
    sources = [_normalize_exa_result(result) for result in getattr(response, "results", [])]
    return [source for source in sources if source["url"]]


def _ensure_price_sources(
    exa: Exa,
    *,
    title: str,
    parsed: dict[str, Any],
    collected_sources: list[dict[str, Any]],
    search_type: str,
    max_results: int,
    highlights_max_characters: int,
    query_log: list[str],
    max_sources: int,
) -> list[dict[str, Any]]:
    sources = _dedupe_sources(collected_sources)
    if not sources or _extract_dollar_prices(_source_evidence_text(sources)):
        return collected_sources

    for query in _price_search_queries(title=title, parsed=parsed):
        collected_sources.extend(
            _search_exa_sources(
                exa,
                query,
                search_type=search_type,
                max_results=max_results,
                highlights_max_characters=highlights_max_characters,
                query_log=query_log,
            )
        )
        sources = _dedupe_sources(collected_sources)
        if _extract_dollar_prices(_source_evidence_text(_limit_sources(sources, max_sources=max_sources))):
            break
    return _limit_sources(collected_sources, max_sources=max_sources)


def _price_search_queries(*, title: str, parsed: dict[str, Any]) -> list[str]:
    candidates = [
        f'"{title}" price',
        f'"{title}" current retail price',
        f'"{title}" price range',
    ]
    detected_model = _clean_optional_identity(parsed.get("detected_model")) if isinstance(parsed, dict) else None
    detected_brand = _clean_optional_identity(parsed.get("detected_brand")) if isinstance(parsed, dict) else None
    if detected_model:
        candidates.append(f'"{detected_model}" price')
    if detected_brand:
        candidates.append(f'"{detected_brand}" "{title}" price')

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        query = _clean_string(candidate)
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= 5:
            break
    return queries


def _fallback_search_queries(
    *,
    title: str,
    categories: list[str],
    tags: list[str],
    parsed: dict[str, Any],
) -> list[str]:
    candidates = [
        title,
        f'"{title}"',
        f'"{title}" reviews',
        f'"{title}" retailer',
        f'"{title}" official',
        f'"{title}" complaints',
    ]
    candidates.extend(_clean_string_list(parsed.get("search_queries", []), limit=8))
    if tags:
        candidates.append(f'{title} {" ".join(tags[:3])} reviews')
    if categories:
        candidates.append(f'{title} {" ".join(categories[:2])} buying guide')

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        query = _clean_string(candidate)
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= 8:
            break
    return queries


def _build_source_grounded_prompt(
    *,
    title: str,
    description: str,
    categories: list[str],
    tags: list[str],
    locale: str,
    sources: list[dict[str, Any]],
) -> str:
    source_packet = json.dumps(sources[:8], ensure_ascii=False)
    return f"""Create product web insights from the provided source packet.

The initial research pass did not return tool sources. Use only the source packet below for source-backed claims. Source titles and URLs may support identity classification, but review sentiment, complaints, ratings, price, quality, and use-case claims require source snippets, highlights, or text excerpts that explicitly support them. Ignore unsupported-browser messages, navigation, add-to-cart status text, cookie banners, and other website boilerplate because they are not product insights. Do not invent missing review counts, ratings, model numbers, prices, or sentiment scores.

TITLE:
{title}

DESCRIPTION:
{description}

CATEGORIES:
{json.dumps(categories, ensure_ascii=False)}

TAGS:
{json.dumps(tags, ensure_ascii=False)}

LOCALE:
{locale}

SOURCE PACKET:
{source_packet}

Return the same JSON schema requested by the system prompt. Include identity_evidence entries that cite why the scope and detected_brand/detected_model were chosen. Return only the JSON object."""


def _parse_agent_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return loaded
    if isinstance(loaded, str):
        text = loaded

    parsed = parse_llm_json(text, extract_braces=True, strip_comments=True)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, str):
        try:
            loaded_nested = json.loads(parsed)
        except json.JSONDecodeError:
            loaded_nested = None
        if isinstance(loaded_nested, dict):
            return loaded_nested
        if isinstance(loaded_nested, str):
            parsed = loaded_nested
        nested = parse_llm_json(parsed, extract_braces=True, strip_comments=True)
        return nested if isinstance(nested, dict) else None
    return None


def _normalize_identity(parsed: dict[str, Any], *, has_sources: bool) -> dict[str, Any]:
    parsed_scope = _normalize_scope(parsed.get("research_scope"))
    if parsed_scope:
        scope = parsed_scope
    elif has_sources:
        scope = "category_level"
    else:
        scope = "insufficient_identity"
    defaults = SCOPE_DEFAULTS[scope]
    if parsed_scope:
        confidence = _normalize_identity_confidence(parsed.get("identity_confidence")) or defaults["identity_confidence"]
        scope_note = _clean_string(parsed.get("scope_note")) or defaults["scope_note"]
    else:
        confidence = defaults["identity_confidence"]
        scope_note = MISSING_SCOPE_NOTE if has_sources else defaults["scope_note"]

    detected_brand = _clean_optional_identity(parsed.get("detected_brand"))
    detected_model = _clean_optional_identity(parsed.get("detected_model"))
    if scope in {"category_level", "insufficient_identity"}:
        detected_brand = None
        detected_model = None

    return {
        "research_scope": scope,
        "identity_confidence": confidence,
        "detected_brand": detected_brand,
        "detected_model": detected_model,
        "scope_note": scope_note,
        "identity_evidence": _clean_string_list(parsed.get("identity_evidence", []), limit=6),
    }


def _normalize_scope(value: Any) -> str | None:
    scope = _clean_string(value).lower()
    return scope if scope in RESEARCH_SCOPES else None


def _normalize_identity_confidence(value: Any) -> str | None:
    confidence = _clean_string(value).lower()
    return confidence if confidence in IDENTITY_CONFIDENCE_LEVELS else None


def _clean_optional_identity(value: Any) -> str | None:
    text = _clean_string(value)
    if not text or text.lower() in {"none", "null", "unknown", "generic", "unbranded", "brandless"}:
        return None
    return text


def _coerce_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_normalize_exa_result(item) for item in value]


def _normalize_report(
    report: dict[str, Any],
    output: dict[str, Any],
    *,
    has_sources: bool,
    identity: dict[str, Any],
) -> dict[str, Any]:
    if not has_sources:
        return {
            "executive_summary": "",
            "positioning_tags": [],
            "metrics": _default_metrics(),
            "retail_insights": [],
            "primary_use_cases": [],
            "customer_sentiment_summary": "",
        }

    executive_summary = _clean_string(report.get("executive_summary")) or output["summary"]
    positioning_tags = _clean_string_list(report.get("positioning_tags", []), limit=8)
    if not positioning_tags:
        positioning_tags = _derive_positioning_tags(output)

    retail_insights = _normalize_retail_insights(report.get("retail_insights", []))
    if not retail_insights:
        retail_insights = _derive_retail_insights(output)

    primary_use_cases = _normalize_use_cases(report.get("primary_use_cases", []))
    if not primary_use_cases:
        primary_use_cases = _derive_use_cases(output)

    customer_sentiment_summary = _clean_string(report.get("customer_sentiment_summary"))
    if not customer_sentiment_summary and output["customer_insights"]:
        customer_sentiment_summary = " ".join(output["customer_insights"][:2])

    metrics = _normalize_metrics(report.get("metrics", {}))
    metrics = _backfill_report_metrics(metrics, output, identity)
    metrics = _scope_adjust_metrics(metrics, identity)

    return {
        "executive_summary": executive_summary,
        "positioning_tags": positioning_tags,
        "metrics": metrics,
        "retail_insights": retail_insights,
        "primary_use_cases": primary_use_cases,
        "customer_sentiment_summary": customer_sentiment_summary,
    }


def _default_metrics() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "label": defaults["label"],
            "score": None,
            "scale": defaults["scale"],
            "rationale": "",
        }
        for key, defaults in REPORT_METRIC_DEFAULTS.items()
    }


def _normalize_metrics(value: Any) -> dict[str, dict[str, Any]]:
    raw_metrics = value if isinstance(value, dict) else {}
    normalized = _default_metrics()
    for key, defaults in REPORT_METRIC_DEFAULTS.items():
        metric = raw_metrics.get(key, {})
        if not isinstance(metric, dict):
            metric = {"label": metric}
        score = _normalize_metric_score(key, metric.get("score"))
        raw_label = _clean_string(metric.get("label")) or _clean_string(metric.get("value")) or defaults["label"]
        label = _normalize_metric_label(key, raw_label, score)
        normalized[key] = {
            "label": label,
            "score": score,
            "scale": defaults["scale"],
            "rationale": _clean_string(metric.get("rationale")),
        }
    return normalized


def _normalize_metric_label(key: str, label: str, score: float | None) -> str:
    normalized_label = _clean_string(label)
    if key == "price_segment":
        price_segment = _normalize_price_segment_label(normalized_label)
        if price_segment:
            return price_segment
        return "Price unavailable"
    if not _is_placeholder_metric_label(normalized_label):
        return normalized_label
    if key == "customer_sentiment":
        return _customer_sentiment_label(score) if score is not None else "Review signal unavailable"
    if key == "build_quality":
        return _build_quality_label(score) if score is not None else "Build signal unavailable"
    if key == "retail_confidence":
        return _retail_confidence_label(score) if score is not None else "Limited"
    return "Unavailable"


def _normalize_price_segment_label(label: str) -> str | None:
    normalized = _clean_string(label).lower()
    if normalized in {"budget", "entry-level", "entry level", "value"}:
        return "Budget"
    if normalized in {"mid-range", "mid range", "midrange", "mid-market", "mid market", "mainstream"}:
        return "Mid-range"
    if normalized in {"premium", "upper-mid", "upper mid"}:
        return "Premium"
    if normalized in {"high-end", "high end", "luxury"}:
        return "High-end"
    return None


def _is_placeholder_metric_label(label: str) -> bool:
    normalized = _clean_string(label).lower()
    return normalized in PLACEHOLDER_METRIC_LABELS or normalized.startswith("qualitative ")


def _normalize_metric_score(key: str, value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if key == "customer_sentiment":
        return round(max(0.0, min(score, 100.0)))
    if key in {"build_quality", "retail_confidence"}:
        return round(max(0.0, min(score, 10.0)), 1)
    return None


def _backfill_report_metrics(
    metrics: dict[str, dict[str, Any]],
    output: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    scope = identity.get("research_scope")
    if scope in {"category_level", "insufficient_identity"}:
        return metrics

    sources = output.get("sources", [])
    source_text = _source_evidence_text(sources)
    if not source_text:
        return metrics

    adjusted = {key: dict(value) for key, value in metrics.items()}
    _backfill_customer_sentiment(adjusted, source_text)
    _backfill_build_quality(adjusted, source_text)
    _backfill_price_segment(adjusted, source_text)
    _backfill_retail_confidence(adjusted, sources, identity)
    return adjusted


def _source_evidence_text(sources: Any) -> str:
    if not isinstance(sources, list):
        return ""
    parts: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        snippet = _clean_string(source.get("snippet"))
        if snippet:
            parts.append(snippet)
    return "\n".join(parts)


def _metric_needs_backfill(metric: dict[str, Any]) -> bool:
    return metric.get("score") is None or _is_placeholder_metric_label(metric.get("label", ""))


def _backfill_customer_sentiment(metrics: dict[str, dict[str, Any]], source_text: str) -> None:
    metric = metrics["customer_sentiment"]
    if not _metric_needs_backfill(metric):
        return

    ratings = _extract_five_star_ratings(source_text)
    if ratings:
        average_rating = sum(ratings) / len(ratings)
        score = round(max(0.0, min(100.0, average_rating / 5 * 100)))
        metric.update(
            {
                "label": _customer_sentiment_label(score),
                "score": score,
                "rationale": f"Source snippets include {average_rating:.1f}/5 average rating evidence.",
            }
        )
        return

    text = source_text.lower()
    positive_hits = _count_term_hits(
        text,
        ("positive", "rated 5", "recommend", "love", "easy to use", "works well", "great", "excellent"),
    )
    negative_hits = _count_term_hits(
        text,
        ("negative", "complaint", "poor", "broken", "defect", "does not work", "hard to use", "disappointed"),
    )
    if positive_hits or negative_hits:
        score = 70 + min(20, positive_hits * 3) - min(30, negative_hits * 5)
        score = round(max(0.0, min(100.0, score)))
        metric.update(
            {
                "label": _customer_sentiment_label(score),
                "score": score,
                "rationale": "Source snippets include customer sentiment language.",
            }
        )
        return

    metric.update(
        {
            "label": "Review signal unavailable",
            "score": None,
            "rationale": "Source snippets do not include enough rating or review sentiment evidence.",
        }
    )


def _backfill_build_quality(metrics: dict[str, dict[str, Any]], source_text: str) -> None:
    metric = metrics["build_quality"]
    if not _metric_needs_backfill(metric):
        return

    text = source_text.lower()
    positive_hits = _count_term_hits(
        text,
        (
            "stainless steel",
            "durable",
            "scratch resistant",
            "scratch-resistant",
            "nonstick",
            "non-stick",
            "solid",
            "sturdy",
            "premium",
            "construction",
        ),
    )
    negative_hits = _count_term_hits(
        text,
        (
            "durability concern",
            "wear",
            "worn",
            "broken",
            "defect",
            "flimsy",
            "crack",
            "peel",
            "coating",
            "seal",
        ),
    )
    if not positive_hits and not negative_hits:
        metric.update(
            {
                "label": "Build signal unavailable",
                "score": None,
                "rationale": "Source snippets do not include enough material or durability evidence.",
            }
        )
        return

    score = 6.2 + min(2.4, positive_hits * 0.45) - min(2.0, negative_hits * 0.4)
    score = round(max(0.0, min(10.0, score)), 1)
    metric.update(
        {
            "label": _build_quality_label(score),
            "score": score,
            "rationale": "Source snippets include material, construction, or durability evidence.",
        }
    )


def _backfill_price_segment(metrics: dict[str, dict[str, Any]], source_text: str) -> None:
    metric = metrics["price_segment"]
    label = _clean_string(metric.get("label")).lower()
    if not (_is_placeholder_metric_label(label) or label == "price unavailable"):
        return

    prices = _extract_dollar_prices(source_text)
    if prices:
        average_price = sum(prices) / len(prices)
        label = _price_segment_from_average(average_price)
        price_min = min(prices)
        price_max = max(prices)
        price_range = (
            f"${price_min:.0f}"
            if round(price_min, 2) == round(price_max, 2)
            else f"${price_min:.0f}-${price_max:.0f}"
        )
        metric.update(
            {
                "label": label,
                "score": None,
                "rationale": f"Source snippets include retail prices in the {price_range} range.",
            }
        )
        return

    metric.update(
        {
            "label": "Price unavailable",
            "score": None,
            "rationale": "Source snippets do not include reliable price evidence.",
        }
    )


def _price_segment_from_average(average_price: float) -> str:
    if average_price < 50:
        return "Budget"
    if average_price < 250:
        return "Mid-range"
    if average_price < 600:
        return "Premium"
    return "High-end"


def _backfill_retail_confidence(
    metrics: dict[str, dict[str, Any]],
    sources: Any,
    identity: dict[str, Any],
) -> None:
    metric = metrics["retail_confidence"]
    if not _metric_needs_backfill(metric):
        return
    if not isinstance(sources, list) or not sources:
        metric.update({"label": "Limited", "score": None, "rationale": "No source coverage was available."})
        return

    snippet_count = sum(1 for source in sources if isinstance(source, dict) and _clean_string(source.get("snippet")))
    source_count = len(sources)
    confidence_bonus = {"high": 1.2, "medium": 0.7, "low": 0.2, "none": 0.0}.get(
        _clean_string(identity.get("identity_confidence")).lower(),
        0.0,
    )
    scope = identity.get("research_scope")
    score = 4.0 + min(source_count, 8) * 0.35 + min(snippet_count, 8) * 0.35 + confidence_bonus
    score = min(score, 8.6 if scope == "brand_level" else 9.4)
    score = round(max(0.0, min(10.0, score)), 1)
    metric.update(
        {
            "label": _retail_confidence_label(score),
            "score": score,
            "rationale": (
                f"Based on {snippet_count} source snippet{'s' if snippet_count != 1 else ''} "
                f"and {identity.get('identity_confidence', 'unknown')} identity confidence."
            ),
        }
    )


def _extract_five_star_ratings(text: str) -> list[float]:
    ratings: list[float] = []
    patterns = (
        r"\b([0-5](?:\.\d+)?)\s*out\s*of\s*5\b",
        r"\brated\s+([0-5](?:\.\d+)?)\s*(?:out\s*of\s*5|stars?)\b",
        r"\b([0-5](?:\.\d+)?)\s*/\s*5\b",
        r"\b([0-5](?:\.\d+)?)\s*stars?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            rating = float(match.group(1))
            if 0 <= rating <= 5:
                ratings.append(rating)
    return ratings[:8]


def _extract_dollar_prices(text: str) -> list[float]:
    prices: list[float] = []
    pattern = r"\$\s*((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,4})(?:\.[0-9]{2})?)(?![0-9])"
    for match in re.finditer(pattern, text):
        price = float(match.group(1).replace(",", ""))
        if 1 <= price <= 10000:
            prices.append(price)
    return prices[:8]


def _count_term_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def _customer_sentiment_label(score: float | None) -> str:
    if score is None:
        return "Review signal unavailable"
    if score >= 85:
        return "Positive"
    if score >= 65:
        return "Mostly positive"
    if score >= 45:
        return "Mixed"
    return "Negative"


def _build_quality_label(score: float | None) -> str:
    if score is None:
        return "Build signal unavailable"
    if score >= 8:
        return "Premium"
    if score >= 6.5:
        return "Durable"
    if score >= 5:
        return "Mixed"
    return "Needs review"


def _retail_confidence_label(score: float | None) -> str:
    if score is None:
        return "Limited"
    if score >= 8:
        return "High"
    if score >= 5.5:
        return "Moderate"
    return "Low"


def _scope_adjust_metrics(metrics: dict[str, dict[str, Any]], identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scope = identity.get("research_scope")
    if scope not in {"category_level", "insufficient_identity"}:
        return metrics

    scope_note = _clean_string(identity.get("scope_note"))
    limitation = scope_note or "No reliable brand or exact model was identified, so metrics cannot be product-specific."
    adjusted = {key: dict(value) for key, value in metrics.items()}
    if scope == "category_level":
        adjusted["customer_sentiment"].update(
            {"label": "Category signal", "score": None, "rationale": limitation}
        )
        adjusted["build_quality"].update(
            {"label": "Category pattern", "score": None, "rationale": limitation}
        )
        adjusted["price_segment"].update(
            {"label": "Category range", "score": None, "rationale": limitation}
        )
        adjusted["retail_confidence"].update(
            {"label": "Limited", "score": None, "rationale": limitation}
        )
        return adjusted

    adjusted["customer_sentiment"].update({"label": "Not enough data", "score": None, "rationale": limitation})
    adjusted["build_quality"].update({"label": "Unknown", "score": None, "rationale": limitation})
    adjusted["price_segment"].update({"label": "Unknown", "score": None, "rationale": limitation})
    adjusted["retail_confidence"].update({"label": "Limited", "score": None, "rationale": limitation})
    return adjusted


def _normalize_retail_insights(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    insights: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            insight = {"type": "positive", "title": item, "detail": ""}
        elif isinstance(item, dict):
            insight_type = _clean_string(item.get("type")).lower()
            insight = {
                "type": "negative" if insight_type == "negative" else "positive",
                "title": _clean_string(item.get("title")),
                "detail": _clean_string(item.get("detail")),
            }
        else:
            continue
        if _is_source_boilerplate_claim(f'{insight["title"]} {insight["detail"]}'):
            continue
        if insight["title"] or insight["detail"]:
            insights.append(insight)
        if len(insights) >= 8:
            break
    return insights


def _normalize_use_cases(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    use_cases: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            use_case = {"title": item, "detail": ""}
        elif isinstance(item, dict):
            use_case = {
                "title": _clean_string(item.get("title")),
                "detail": _clean_string(item.get("detail")),
            }
        else:
            continue
        if _is_source_boilerplate_claim(f'{use_case["title"]} {use_case["detail"]}'):
            continue
        if use_case["title"] or use_case["detail"]:
            use_cases.append(use_case)
        if len(use_cases) >= 6:
            break
    return use_cases


def _derive_retail_insights(output: dict[str, Any]) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    for item in output.get("pros", [])[:4]:
        insights.append(_string_to_titled_item(item, item_type="positive"))
    for item in output.get("cons", [])[:4]:
        insights.append(_string_to_titled_item(item, item_type="negative"))
    return insights[:8]


def _derive_use_cases(output: dict[str, Any]) -> list[dict[str, str]]:
    return [_string_to_titled_item(item) for item in output.get("use_cases", [])[:6]]


def _derive_positioning_tags(output: dict[str, Any]) -> list[str]:
    candidates = output.get("pros", []) + output.get("use_cases", []) + output.get("purchase_considerations", [])
    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        title = _split_title_and_detail(candidate)[0]
        if not title or title in seen:
            continue
        seen.add(title)
        tags.append(title[:48])
        if len(tags) >= 6:
            break
    return tags


def _string_to_titled_item(text: str, *, item_type: str = "positive") -> dict[str, str]:
    title, detail = _split_title_and_detail(text)
    item = {"title": title, "detail": detail}
    if item_type:
        item["type"] = item_type
    return item


def _split_title_and_detail(text: str) -> tuple[str, str]:
    cleaned = _clean_string(text)
    if not cleaned:
        return "", ""
    for separator in (":", " - ", ". "):
        if separator in cleaned:
            title, detail = cleaned.split(separator, 1)
            return title.strip()[:80], detail.strip()
    words = cleaned.split()
    title = " ".join(words[:5])
    return title[:80], cleaned


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for source in sources:
        normalized = _normalize_exa_result(source)
        url = normalized["url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(normalized)
    return deduped


def _limit_sources(sources: list[dict[str, Any]], *, max_sources: int) -> list[dict[str, Any]]:
    deduped = _dedupe_sources(sources)
    if len(deduped) <= max_sources:
        return deduped

    selected = deduped[:max_sources]
    if any(_extract_dollar_prices(source.get("snippet", "")) for source in selected):
        return selected

    for source in deduped[max_sources:]:
        if _extract_dollar_prices(source.get("snippet", "")):
            selected[-1] = source
            break
    return selected


def _extract_final_message_text(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    messages = result.get("messages") or []
    if not messages:
        return ""
    final_message = messages[-1]
    content = getattr(final_message, "content", None)
    if content is None and isinstance(final_message, dict):
        content = final_message.get("content")
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return _clean_string(content)


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_string_list(value: Any, *, limit: int, filter_source_boilerplate: bool = False) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _clean_string(item)
        if filter_source_boilerplate and _is_source_boilerplate_claim(text):
            continue
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _clean_source_snippet(text: str) -> str:
    snippet = _clean_string(text)
    if not snippet:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in snippet.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
        if any(pattern in normalized for pattern in SOURCE_BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\b(?:Error|Close|Menu|Quantity):?\s*", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _is_source_boilerplate_claim(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", _clean_string(text).lower()).strip()
    return any(pattern in normalized for pattern in SOURCE_BOILERPLATE_CLAIM_PATTERNS)


def _clamp_int(value: Any, *, minimum: int, maximum: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = minimum
    return max(minimum, min(integer, maximum))
