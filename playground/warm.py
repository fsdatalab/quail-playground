"""Start the deployed servers with one HTTP request each.

    python -m playground.warm https://<page url>

A server's container starts on its first request; the first container
after a deploy loads the model and takes the snapshot. A request to
``/v1/capabilities`` without the bearer token is answered 401 as soon
as the server is up, so no secret is needed here.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
import urllib.error
import urllib.request

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


def warm_servers(servers: dict) -> list:
    """Ping every server at once; return one line per server."""
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
            except Exception as error:  # noqa: BLE001 - reported, not raised
                lines.append(f"{model}: failed: {error}")
                continue
            lines.append(f"{model}: HTTP {status} after "
                         f"{time.time() - started:.0f} s")
    return lines


def main(argv=None) -> None:
    import json

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        raise SystemExit(__doc__)
    with urllib.request.urlopen(f"{args[0].rstrip('/')}/config",
                                timeout=120) as response:
        servers = json.load(response)["servers"]
    for line in warm_servers(servers):
        print(line, flush=True)


if __name__ == "__main__":
    main()
