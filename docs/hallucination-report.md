# LLM Enhancement Hallucination Report

**Date:** 2026-04-15
**Reported by:** Antonio Martinez
**Status:** Open — Separate task pending
**Affected component:** `src/backend/vlm.py` — `_call_nemotron_enhance_vlm()` (Step 1 enhancement)

---

## Summary

The VLM model (`nemotron-nano-12b-v2-vl`, 12B parameters) introduces hallucinations at the source — misreading visible text, fabricating materials and features, and drawing from training data rather than strictly describing the image. The LLM enhancement step (`_call_nemotron_enhance_vlm`) then compounds these errors by rewriting them into confident marketing copy. Both layers contribute, but the root cause is the VLM.

---

## Root Cause Analysis

### Pipeline Flow

```
Image Upload
  |
  v
[VLM] _call_vlm()                         <-- Accurate visual analysis
  |   Model: nemotron-nano-12b-v2-vl
  |   Output: title, description, categories, tags, colors
  v
[LLM] _call_nemotron_enhance_vlm()        <-- Hallucinations introduced HERE
  |   Model: nemotron-3-nano
  |   Task: "Write rich, persuasive product description"
  v
[LLM] _call_nemotron_apply_branding()     <-- Inherits errors from Step 1
  |   (only runs if brand_instructions provided)
  v
[LLM] _call_nemotron_generate_faqs()      <-- Consumes VLM output directly,
      (runs in parallel with Step 1)           but FAQs still affected if
                                               VLM has minor OCR issues
```

### Where the Problem Lives

**Layer 1 — VLM** (`src/backend/vlm.py`, `_call_vlm()`):
The 12B VLM model misreads text, fabricates materials/features, and fills in details from training data. This happens regardless of prompt complexity — even "describe this product" triggers hallucinations. Longer prompts produce *more* hallucinations, not fewer. This is confirmed by the NVIDIA research team: longer system prompts degrade VLM output quality for this model class.

**Layer 2 — LLM Enhancement** (`src/backend/vlm.py`, `_call_nemotron_enhance_vlm()`):
The LLM takes the already-hallucinated VLM output and rewrites it into confident marketing copy, compounding errors and adding its own fabrications. Skipping this step when no user data is provided eliminates the second layer.

---

## Evidence: Craftsman 2XV20 Lawn Mower

### Test Image

`mower.jpeg` — Craftsman battery-powered lawn mower with "2XV20" printed on the deck (indicating dual V20 battery platform).

### VLM Direct Testing (2026-04-15)

Three prompts were tested against the same VLM endpoint (`nemotron-nano-12b-v2-vl`) with `mower.jpeg`:

**Prompt 1 — Minimal: "describe this product"**

> "This product is a Craftsman 20-inch 20V MAX Lithium Ion Cordless Lawn Mower. It's a compact, electric lawn mower designed for residential use. The mower features a 20-inch cutting deck [...] The 20V MAX Lithium Ion battery provides cordless convenience [...] includes a grass collection bag [...] equipped with a safety key to prevent accidental startups."

- Gets closest to reality: correctly identifies it as cordless/battery-powered ("20V MAX Lithium Ion")
- Still fabricates: "20-inch cutting deck", "safety key"
- Clearly pulling from Craftsman training data rather than reading "2XV20" text

**Prompt 2 — Detailed descriptive: "In detail, give a description of this image, include everything you see including texts. Be extremely descriptive."**

> "The cutting deck itself is marked with the text '20' indicating the width of the cutting blade in inches [...] a clear plastic cover over the cutting deck, allowing a view of the blades inside."

- Misreads "2XV20" as "20" and reinterprets it as cutting width
- Fabricates "clear plastic cover over the cutting deck"
- More hallucinations than the minimal prompt

**Prompt 3 — Catalog enrichment structured prompt (our production prompt)**

> `"title": "Craftsman 20-Inch Electric Lawn Mower"` ... `"clear plastic front cover"` ... `"control panel on the handlebar"` ... `"model number '20' is visible on the front"`

- Same hallucinations as prompt 2, now in JSON format
- Fabricates: "clear plastic front cover", "control panel on the handlebar"
- Misreads "2XV20" as "20" and calls it a model number

### Key Finding: Hallucinations Originate in the VLM

Initial analysis attributed hallucinations to the LLM enhancement step. **Direct VLM testing disproved this.** The VLM itself:
1. Misreads "2XV20" as "20" across all prompt styles
2. Fabricates materials ("clear plastic") and features ("control panel", "safety key") not visible in the image
3. Draws from training data about Craftsman products rather than strictly describing the image
4. Performs *worse* with longer, more detailed prompts — the minimal prompt produced the fewest hallucinations

### Hallucination Inventory (VLM output, all prompts combined)

| Claim | Reality | Type | Source |
|-------|---------|------|--------|
| "20-Inch" cutting width | "2XV20" is Craftsman's dual V20 battery platform | Text misread | VLM |
| "clear plastic cutting deck/cover" | Deck is opaque black | Fabricated material | VLM |
| "control panel on the handlebar" | Only a safety lever is visible | Fabricated feature | VLM |
| "safety key" | No safety key visible | Fabricated feature | VLM |
| "Electric Lawn Mower" (prompt 2/3) | Battery-powered (cordless) | Training data inference | VLM |
| "silver accents" on wheels | Wheels are entirely black | Fabricated detail | LLM enhancement |
| "red power button" | Not visible | Fabricated feature | LLM enhancement |

The LLM enhancement step compounded the VLM's errors (adding "silver accents", "red power button"), but the root cause is the 12B VLM model's vision limitations.

---

## Proposed Solution

### Fix 1 (Implemented): Skip LLM Enhancement When Unnecessary

**Status: Done** — merged in this branch.

The LLM enhancement step is now skipped when no user product data is provided. This eliminates the second layer of hallucinations.

| Scenario | Current Behavior | New Behavior |
|----------|-----------------|--------------|
| Image only (no user data, no brand instructions) | VLM -> LLM enhance -> output | VLM -> output directly (skip LLM) |
| Image + user product data | VLM -> LLM enhance (merge) -> output | VLM -> LLM enhance (merge) -> output (keep) |
| Image + brand instructions | VLM -> LLM enhance -> LLM brand -> output | VLM -> LLM brand -> output |
| Image + user data + brand instructions | VLM -> LLM enhance -> LLM brand -> output | VLM -> LLM enhance -> LLM brand -> output (keep) |

### Fix 2 (Future): Shorten the VLM Prompt

The current VLM prompt in `_call_vlm()` is ~30 lines with detailed rules, category lists, formatting instructions, and output constraints. Testing showed that a minimal prompt ("describe this product") produced the fewest hallucinations — the VLM correctly identified the mower as "20V MAX Lithium Ion Cordless" with that prompt, while the long structured prompt caused it to misread "2XV20" as "20" and fabricate features.

This is confirmed by the NVIDIA research team: longer system prompts degrade output quality for this VLM model class. The model spends capacity following formatting rules rather than focusing on accurate visual analysis.

**Proposed approach:**
- Strip the VLM prompt down to a short, focused instruction — prioritize visual accuracy over output formatting
- Move structural concerns (JSON format, category validation, tag count) to a lightweight post-processing step or a separate LLM call
- Test iteratively: compare hallucination rates across prompt lengths using a set of test images (mower, shoes, skincare, etc.)

**Trade-off:** A shorter VLM prompt may return unstructured text instead of clean JSON. This would require parsing the free-text output into structured fields, either with regex/heuristics or a fast LLM call. The benefit is more accurate visual descriptions at the source.

### Fix 3 (Future): Upgrade VLM Model

The `nemotron-nano-12b-v2-vl` (12B parameters) has fundamental vision limitations with stylized text and detail accuracy. A larger VLM (72B+) would likely improve OCR accuracy and reduce training-data hallucinations. This is a infrastructure/cost trade-off rather than a code change.

---

## Impact on FAQ Feature

The FAQ generation feature (`_call_nemotron_generate_faqs`) consumes the raw VLM observation directly (not the enhanced output), which reduces but does not eliminate the risk:

- FAQs generated from accurate VLM output will be factually grounded
- Minor VLM OCR errors (e.g., "2x20" vs "2XV20") can still propagate into FAQ answers
- If the proposed fix (skip enhancement) is implemented, the Details tab and FAQ tab will both be grounded in the same factual VLM observation, creating consistency

---

## Reproduction Steps

1. Start the backend and frontend services
2. Upload `mower.jpeg` (Craftsman 2XV20 lawn mower)
3. Click Generate with default settings (no product data, no brand instructions)
4. Observe the enriched description in the Details tab
5. Compare against the VLM's raw output (visible in backend logs at `[VLM]` level)

---

## Files Referenced

| File | Relevance |
|------|-----------|
| `src/backend/vlm.py:128-205` | `_call_nemotron_enhance_vlm()` — where hallucinations are introduced |
| `src/backend/vlm.py:167-186` | Enhancement prompt with insufficient anti-hallucination rules |
| `src/backend/vlm.py:175` | Current anti-hallucination rule (too narrow — numbers only) |
| `src/backend/vlm.py:397-439` | `_call_nemotron_enhance()` — orchestrator where the skip logic would go |
| `src/backend/vlm.py:441-510` | `_call_vlm()` — VLM analysis (produces accurate output) |
