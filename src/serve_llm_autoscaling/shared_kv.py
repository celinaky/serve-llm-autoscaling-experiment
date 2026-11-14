"""Validate an external shared cache without changing its lifecycle or router."""
from __future__ import annotations

import hashlib
import json
import socket
from typing import Any
from urllib.parse import urlsplit

import requests

from .config import ExperimentConfig


def check_shared_kv_support(config: ExperimentConfig) -> dict[str, Any] | None:
    shared = config.deployment.engine.shared_kv_cache
    if shared is None or not shared.enabled:
        return None
    path = shared.config_path.resolve()
    raw = path.read_bytes()
    requester = json.loads(raw)
    if requester.get("mode") != "standalone-store" or requester.get("global_segment_size") != 0:
        raise ValueError("shared_kv_cache requires standalone-store mode and global_segment_size: 0; the external client owns the pool")
    for field in ("metadata_server", "master_server_address", "protocol"):
        if not requester.get(field):
            raise ValueError(f"Mooncake requester JSON is missing {field}")
    address = requester["master_server_address"]
    host, separator, port = address.rpartition(":")
    if not separator or not host:
        raise ValueError("Mooncake master_server_address must be HOST:PORT")
    if host in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        raise ValueError("Use the head node's routable internal address for Mooncake")
    with socket.create_connection((host, int(port)), timeout=5):
        pass
    health = None
    if shared.health_url:
        response = requests.get(shared.health_url, timeout=5)
        response.raise_for_status()
        health = response.json()
        if health.get("status") != "healthy":
            raise RuntimeError(f"Mooncake storage client is not healthy: {health}")
    for endpoint in (shared.master_metrics_url, shared.client_metrics_url):
        if endpoint and urlsplit(endpoint).scheme not in ("http", "https"):
            raise ValueError("Mooncake metric endpoints must be HTTP(S) URLs")
    return {
        "backend": "mooncake", "connector": "MooncakeStoreConnector",
        "config_path": str(path), "config_sha256": hashlib.sha256(raw).hexdigest(),
        "requester": requester, "cache_prefix": shared.cache_prefix,
        "package": shared.package, "health": health,
        "router_modified": False,
    }
