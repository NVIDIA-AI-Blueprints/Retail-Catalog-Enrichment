# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static checks for Nemotron chat-completion latency controls."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "src" / "backend"


def _attribute_chain(node):
    chain = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        chain.append(node.id)
    return list(reversed(chain))


def _dict_value(dict_node, key_name):
    if not isinstance(dict_node, ast.Dict):
        return None
    for key, value in zip(dict_node.keys, dict_node.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return value
    return None


def _call_keyword(node, key_name):
    return next((kw.value for kw in node.keywords if kw.arg == key_name), None)


def _uses_vlm_model(node):
    model = _call_keyword(node, "model")
    return (
        isinstance(model, ast.Subscript)
        and isinstance(model.value, ast.Name)
        and model.value.id == "vlm_config"
    )


def _contains_no_think(node):
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and "/no_think" in child.value
        for child in ast.walk(node)
    )


def test_backend_chat_completion_calls_disable_thinking():
    failures = []

    for path in sorted(BACKEND_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _attribute_chain(node.func)[-3:] != ["chat", "completions", "create"]:
                continue

            extra_body = _call_keyword(node, "extra_body")

            if _uses_vlm_model(node):
                if extra_body is not None:
                    failures.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} VL call sets extra_body")
                messages = _call_keyword(node, "messages")
                if messages is None or not _contains_no_think(messages):
                    failures.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} VL call missing /no_think")
                continue

            if extra_body is None:
                failures.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} missing extra_body")
                continue

            if _dict_value(extra_body, "reasoning_budget") is not None:
                failures.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} still sets reasoning_budget")

            chat_template_kwargs = _dict_value(extra_body, "chat_template_kwargs")
            enable_thinking = _dict_value(chat_template_kwargs, "enable_thinking")
            if not (isinstance(enable_thinking, ast.Constant) and enable_thinking.value is False):
                failures.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} does not disable thinking")

    assert failures == []
