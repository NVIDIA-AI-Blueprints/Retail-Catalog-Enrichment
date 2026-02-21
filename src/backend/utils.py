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
import re
from typing import Optional


def parse_llm_json(
    text: str,
    *,
    extract_braces: bool = False,
    strip_comments: bool = False,
) -> Optional[dict]:
    """Parse a JSON dict from an LLM response, handling common formatting issues.

    Returns the parsed dict, or None on any failure.
    """
    text = text.strip()

    # Strip markdown fences
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
                break

    if extract_braces:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    if strip_comments:
        text = re.sub(r"//.*?(?=\n|$)", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
