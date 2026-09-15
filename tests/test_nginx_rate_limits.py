# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


NGINX_CONFIG = Path(__file__).parents[1] / "nginx.conf"


def test_expensive_api_routes_share_rate_and_connection_limits():
    config = NGINX_CONFIG.read_text()

    assert "~^/api/(vlm/analyze|generate/(variation|3d))$ $binary_remote_addr;" in config
    assert "limit_req_zone $expensive_api_client zone=expensive_api_rate:10m rate=5r/m;" in config
    assert "limit_conn_zone $expensive_api_client zone=expensive_api_connections:10m;" in config
    assert "limit_req zone=expensive_api_rate burst=2 nodelay;" in config
    assert "limit_req_status 429;" in config
    assert "limit_conn expensive_api_connections 2;" in config
    assert "limit_conn_status 429;" in config
