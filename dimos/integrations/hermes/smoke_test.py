"""Smoke-test the DIMOS MCP server via raw JSON-RPC.

Run this *while* run_dimos_mcp.py is up in another terminal:

    .venv/bin/python -m dimos.integrations.hermes.smoke_test

It exercises the same JSON-RPC calls Hermes will make:
  1. initialize
  2. tools/list
  3. tools/call -> server_status
  4. tools/call -> speak (a short sentence)

Pass criteria: all 4 requests get a 200 response with no JSON-RPC error.
"""

import json
import sys
import urllib.request

URL = "http://127.0.0.1:9990/mcp"


def _post(payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _call(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        payload["params"] = params
    return _post(payload)


def _check(label: str, resp: dict) -> bool:
    if "error" in resp:
        print(f"  FAIL  {label}: {resp['error']}")
        return False
    print(f"  PASS  {label}")
    return True


def main() -> int:
    print("DIMOS MCP smoke test")
    print(f"Endpoint: {URL}")
    print()

    ok = True

    print("1) initialize")
    try:
        resp = _call("initialize", {"protocolVersion": "2025-11-25"})
        ok &= _check("initialize", resp)
        print(f"     server: {resp.get('result', {}).get('serverInfo')}")
    except Exception as e:
        print(f"  FAIL  initialize: {e}")
        return 1

    print()
    print("2) tools/list")
    resp = _call("tools/list", {}, req_id=2)
    ok &= _check("tools/list", resp)
    tools = resp.get("result", {}).get("tools", [])
    print(f"     {len(tools)} tools registered:")
    for t in tools[:15]:
        print(f"       - {t['name']}")
    if len(tools) > 15:
        print(f"       ... and {len(tools) - 15} more")

    print()
    print("3) tools/call -> server_status")
    resp = _call("tools/call", {"name": "server_status", "arguments": {}}, req_id=3)
    ok &= _check("server_status", resp)
    content = resp.get("result", {}).get("content", [])
    if content:
        print(f"     {content[0].get('text', '')[:200]}")

    print()
    print("4) tools/call -> speak")
    resp = _call(
        "tools/call",
        {"name": "speak", "arguments": {"text": "Hermes is now connected to Dimos."}},
        req_id=4,
    )
    ok &= _check("speak", resp)
    content = resp.get("result", {}).get("content", [])
    if content:
        print(f"     {content[0].get('text', '')[:200]}")

    print()
    print("=" * 40)
    if ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
