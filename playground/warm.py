"""Start the deployed model servers for the Modal playground.

Each server starts its container when it receives the first request. The
capabilities route returns 401 without a bearer token, but the response still
confirms that the server is ready. The warm request therefore needs no secret.
"""

from __future__ import annotations

import concurrent.futures
import time
import urllib.error
import urllib.request
from collections.abc import Mapping

from playground.demos import MODELS

WARM_TIMEOUT_S = 20 * 60.0


def ping(endpoint: str, timeout_s: float = WARM_TIMEOUT_S) -> int:
    """Return the HTTP status of the server's capabilities route."""
    try:
        with urllib.request.urlopen(f"{endpoint}/v1/capabilities",
                                    timeout=timeout_s) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def warm_servers(servers: Mapping[str, str | None]) -> list[str]:
    """Start one server per model and return one status line per model."""
    started = time.time()
    lines = []
    with concurrent.futures.ThreadPoolExecutor(len(servers) or 1) as pool:
        futures = {model: pool.submit(ping, endpoint)
                   for model, endpoint in servers.items() if endpoint}
        for model in MODELS:
            future = futures.get(model)
            if future is None:
                lines.append(f"{model}: no server deployed")
                continue
            try:
                status = future.result()
            except Exception as error:  # noqa: BLE001, report the error below
                lines.append(f"{model}: failed: {error}")
                continue
            lines.append(f"{model}: HTTP {status} after "
                         f"{time.time() - started:.0f} s")
    return lines
