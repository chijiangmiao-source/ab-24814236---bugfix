#!/usr/bin/env python3
"""HTTP smoke test against a running ledger instance (LEDGER_BASE_URL).

Posts one small batch at the currently-advertised next sequence and reads it
back via the cursor API, then commits a maximal 32-record batch of doses
near the signed-integer upper bound and replays it with small pages --
the compose-service half of the restart-durability scenario (the restart
itself is proven against real server subprocesses by tests/e2e_smoke.py).
Retries briefly on 409 so parallel smoke runs do not fail spuriously.
Exits non-zero on any violation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

BASE = os.environ.get("LEDGER_BASE_URL", "http://127.0.0.1:8080")


def call(method: str, path: str, body: object = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def commit(records: list[int]) -> tuple[int, dict]:
    """Commit ``records`` at the advertised head; retry transient 409s."""
    status, health = call("GET", "/healthz")
    assert status == 200, f"healthz: {status} {health}"
    next_seq = health["next_seq"]
    for _attempt in range(10):
        status, body = call("POST", "/api/batches", {
            "expected_seq": next_seq, "records": records})
        if status == 201:
            assert body["seq"] == next_seq, body
            assert body["next_seq"] == next_seq + len(records), body
            return next_seq, body
        assert status == 409, f"unexpected POST status {status}: {body}"
        next_seq = body["current_seq"]
        time.sleep(0.2)
    raise AssertionError("could not commit smoke batch after retries")


def read_all(cursor: int, limit: int = 1000) -> list[dict]:
    """Replay every record after ``cursor`` via the paged cursor API."""
    out: list[dict] = []
    while True:
        query = urlencode({"cursor": cursor, "limit": limit})
        status, page = call("GET", f"/api/records?{query}")
        assert status == 200, page
        out.extend(page["records"])
        if not page["records"]:
            return out
        cursor = page["next_cursor"]


def main() -> int:
    status, health = call("GET", "/healthz")
    assert status == 200, f"healthz: {status} {health}"
    print(f"smoke: service healthy at {BASE}, next_seq={health['next_seq']}")

    records = [int(time.time()) % 100_000, -17, 42]
    seq, _body = commit(records)
    print(f"smoke: committed batch seq={seq} -> {seq + len(records)}")
    got = [(r["seq"], r["dose"]) for r in read_all(seq - 1)]
    assert got == [(seq + i, d) for i, d in enumerate(records)], got
    print(f"smoke: read back {got}")

    # Full 32-record batch of int64-scale doses, replayed in small pages:
    # no skipped or duplicated sequence numbers.
    big = [2**63 - 1 - i for i in range(32)]
    seq, _body = commit(big)
    print(f"smoke: committed full 32-record batch seq={seq} -> {seq + 32}")
    page_size = 7
    got = read_all(seq - 1, limit=page_size)
    assert got == [{"seq": seq + i, "dose": big[i]} for i in range(32)], got
    print(f"smoke: paged replay of 32 records ok (limit={page_size})")
    print("HTTP SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"HTTP SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
