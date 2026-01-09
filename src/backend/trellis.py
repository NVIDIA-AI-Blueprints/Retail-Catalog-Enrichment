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

"""3D Asset Generation using Microsoft TRELLIS model"""
import json
import base64
import logging
from typing import Dict, Any
import httpx

from backend.config import get_config

logger = logging.getLogger("catalog_enrichment.trellis")


async def generate_3d_asset(
    image_bytes: bytes,
    content_type: str,
    slat_cfg_scale: float = 5.0,
    ss_cfg_scale: float = 10.0,
    slat_sampling_steps: int = 50,
    ss_sampling_steps: int = 50,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Generate a 3D GLB asset from a 2D product image using TRELLIS model.
    
    Args:
        image_bytes: Binary image data
        content_type: MIME type of the image
        slat_cfg_scale: SLAT configuration scale
        ss_cfg_scale: SS configuration scale
        slat_sampling_steps: SLAT sampling steps
        ss_sampling_steps: SS sampling steps
        seed: Random seed for reproducibility
        
    Returns:
        dict with glb_data, artifact_id, and metadata
    """
    logger.info(
        f"Generating 3D asset: slat_cfg={slat_cfg_scale}, ss_cfg={ss_cfg_scale}, "
        f"slat_steps={slat_sampling_steps}, ss_steps={ss_sampling_steps}, seed={seed}"
    )
    
    endpoint_url = get_config().get_trellis_config()["url"]
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_format = content_type.split("/")[-1] if "/" in content_type else "png"
    
    payload = {
        "image": f"data:image/{image_format};base64,{image_b64}",
        "slat_cfg_scale": slat_cfg_scale,
        "ss_cfg_scale": ss_cfg_scale,
        "slat_sampling_steps": slat_sampling_steps,
        "ss_sampling_steps": ss_sampling_steps,
        "seed": seed
    }
    
    headers = {"Accept": "application/octet-stream", "Content-Type": "application/json"}
    logger.info(f"Calling TRELLIS endpoint: {endpoint_url}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
        
        try:
            response_json = response.json()
            if "artifacts" in response_json and response_json["artifacts"]:
                glb_data = base64.b64decode(response_json["artifacts"][0]["base64"])
                artifact_id = response_json.get("id", f"trellis_{seed}")
                logger.info(f"Decoded GLB from artifacts: {len(glb_data)} bytes")
            else:
                glb_data = response.content
                artifact_id = f"trellis_{seed}"
        except json.JSONDecodeError:
            glb_data = response.content
            artifact_id = f"trellis_{seed}"
        
        logger.info(f"Successfully generated 3D asset: {len(glb_data)} bytes")
        
        return {
            "glb_data": glb_data,
            "artifact_id": artifact_id,
            "metadata": {
                "slat_cfg_scale": slat_cfg_scale,
                "ss_cfg_scale": ss_cfg_scale,
                "slat_sampling_steps": slat_sampling_steps,
                "ss_sampling_steps": ss_sampling_steps,
                "disable_safety_checker": 1,
                "seed": seed,
                "size_bytes": len(glb_data)
            }
        }
        
    except httpx.TimeoutException:
        logger.error("TRELLIS request timed out after 120 seconds")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"TRELLIS request failed with status {e.response.status_code}: {e}")
        raise
    except httpx.RequestError as e:
        logger.error(f"TRELLIS request failed: {e}")
        raise

