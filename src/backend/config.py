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

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("catalog_enrichment.config")


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent.parent / "shared" / "config" / "config.yaml"
        self._config_data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config_data or {}
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            raise
    def _get_section_config(self, section: str, required_fields: list) -> Dict[str, str]:
        config = self._config_data.get(section, {})
        if not config:
            raise ValueError(f"{section.upper()} configuration section not found in config file")
        
        result = {}
        for field in required_fields:
            value = config.get(field)
            if not value:
                raise ValueError(f"{section.upper()} {field} not configured")
            result[field] = value
        return result

    def _get_optional_section_config(self, section: str) -> Dict[str, Any]:
        return self._config_data.get(section, {}) or {}
        
    def get_vlm_config(self) -> Dict[str, str]:
        return self._get_section_config('vlm', ['url', 'model'])
        
    def get_llm_config(self) -> Dict[str, str]:
        return self._get_section_config('llm', ['url', 'model'])
        
    def get_flux_config(self) -> Dict[str, str]:
        return self._get_section_config('flux', ['url'])
        
    def get_trellis_config(self) -> Dict[str, str]:
        return self._get_section_config('trellis', ['url'])

    def get_embeddings_config(self) -> Dict[str, str]:
        config = self._get_optional_section_config('embeddings')
        return {
            "url": os.getenv("NVIDIA_API_BASE_URL") or config.get("url") or "https://integrate.api.nvidia.com/v1",
            "model": config.get("model") or "nvidia/nv-embedqa-e5-v5",
        }

    def get_milvus_config(self) -> Dict[str, Any]:
        config = self._get_optional_section_config('milvus')
        return {
            "host": os.getenv("MILVUS_HOST") or config.get("host") or "localhost",
            "port": os.getenv("MILVUS_PORT") or str(config.get("port") or "19530"),
            "collection": os.getenv("MILVUS_COLLECTION") or config.get("collection") or "policy_chunks",
            "alias": config.get("alias") or "policy_library",
        }

    def get_policy_library_config(self) -> Dict[str, Any]:
        config = self._get_optional_section_config('policy_library')
        return {
            "storage_dir": os.getenv("POLICY_LIBRARY_STORAGE_DIR") or config.get("storage_dir") or "data/policies",
            "db_path": os.getenv("POLICY_LIBRARY_DB_PATH") or config.get("db_path") or "data/policies/library.db",
            "top_k": int(os.getenv("POLICY_LIBRARY_TOP_K") or config.get("top_k") or 8),
            "min_relevance_score": float(
                os.getenv("POLICY_LIBRARY_MIN_RELEVANCE_SCORE") or config.get("min_relevance_score") or 0.3
            ),
        }

    def get_product_manual_config(self) -> Dict[str, Any]:
        config = self._get_optional_section_config('product_manual')
        return {
            "chunk_size_words": int(config.get("chunk_size_words") or 250),
            "chunk_overlap_words": int(config.get("chunk_overlap_words") or 50),
            "top_k_per_query": int(config.get("top_k_per_query") or 3),
            "min_relevance_score": float(config.get("min_relevance_score") or 0.25),
        }


_config_instance: Optional[Config] = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
