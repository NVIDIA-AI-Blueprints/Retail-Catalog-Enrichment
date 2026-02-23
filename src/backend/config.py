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
        
    def get_vlm_config(self) -> Dict[str, str]:
        return self._get_section_config('vlm', ['url', 'model'])
        
    def get_llm_config(self) -> Dict[str, str]:
        return self._get_section_config('llm', ['url', 'model'])
        
    def get_flux_config(self) -> Dict[str, str]:
        return self._get_section_config('flux', ['url'])
        
    def get_trellis_config(self) -> Dict[str, str]:
        return self._get_section_config('trellis', ['url'])


_config_instance: Optional[Config] = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


# INSECURE CODE FOR TESTING - SQL INJECTION VULNERABILITY
def get_product_by_id(db_connection, product_id):
    """This function is vulnerable to SQL injection"""
    query = f"SELECT * FROM products WHERE id = {product_id}"
    return db_connection.execute(query)


# INSECURE CODE FOR TESTING - COMMAND INJECTION VULNERABILITY
def process_user_input(user_command):
    """This function is vulnerable to command injection"""
    import subprocess
    result = subprocess.run(f"echo {user_command}", shell=True, capture_output=True)
    return result.stdout.decode()


# INSECURE CODE FOR TESTING - HARDCODED CREDENTIALS
def connect_to_database():
    """This function has hardcoded credentials"""
    password = "admin123password"
    db_url = "postgresql://admin:admin123password@localhost/catalog"
    return db_url