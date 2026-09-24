#!/usr/bin/env python3
"""HTTP verification of the restart-persistence scenario against a running
ledger instance (LEDGER_BASE_URL) -- the same chain the unit tests and the
subprocess e2e prove locally, executed against the compose service once it
reports healthy:

  1. commit a full 32-record batch of doses near the int64 upper bound
  2. restart the ledger container (Docker API over the mounted socket when
     LEDGER_RESTART_CONTAINER is set and the socket is present; otherwise
     this leg is skipped with a notice)
  3. health must still advertise the advanced next_seq, and the committed
     batch must page back continuously -- no gaps, no duplicates
  4. a follow-up batch at the new head commits and continues the sequence
  5. the whole log reads back gap-free from seq 1

Exits non-zero on any violation.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import stat
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

BASE = os.environ.get("LEDGER_BASE_URL", "http://127.0.0.1:8080")
RESTART_CONTAINER = os.environ.get("LEDGER_RESTART_CONTAINER", "")
DOCKER_SOCKET = os.environ.get("LEDGER_DOCKER_SOCKET", "/var/run/docker.sock")
PAGE_LIMIT = 7
FULL_BATCH = [2**63 - 1 - i for i in range(32)]
FOLLOW_UP = [-5, 2**63 - 1]


def call(method: str, path: str, body: object = None,
         timeout: float = 5.0) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_healthy(timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last: object = "no attempt"
    while time.time() < deadline:
        try:
            status, body = call("GET", "/healthz")
            if status == 200:
                return body
            last = f"{status} {body}"
        except (OSError, urllib.error.URLError) as exc:
            last = str(exc)
        time.sleep(0.2)
    raise AssertionError(f"{BASE} never became healthy: {last}")


def commit(expected_seq: int, records: list[int]) -> dict:
    """Commit a batch, re-reading the advertised head on 409 races."""
    for _ in range(10):
        status, body = call("POST", "/api/batches", {
            "expected_seq": expected_seq, "records": records})
        if status == 201:
            return body
        assert status == 409, f"unexpected POST status {status}: {body}"
        expected_seq = body["current_seq"]
        time.sleep(0.2)
    raise AssertionError("could not commit batch after retries")


def read_all_from(cursor: int) -> list[dict]:
    """Page through every record after ``cursor`` with a small page size."""
    out: list[dict] = []
    while True:
        query = urlencode({"cursor": cursor, "limit": PAGE_LIMIT})
        status, page = call("GET", f"/api/records?{query}")
        assert status == 200, f"read cursor={cursor}: {status} {page}"
        out.extend(page["records"])
        cursor = page["next_cursor"]
        if not page["records"]:
            return out


class _DockerConnection(http.client.HTTPConnection):
    """Minimal Docker API client over the mounted unix socket."""

    def __init__(self, socket_path: str) -> None:
        super().__init__("docker-socket", timeout=30)
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect(self._socket_path)


def docker_api(method: str, path: str) -> tuple[int, dict]:
    conn = _DockerConnection(DOCKER_SOCKET)
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


def maybe_restart_container() -> bool:
    """Restart the ledger container via the Docker API, if available."""
    if not RESTART_CONTAINER:
        print("smoke: LEDGER_RESTART_CONTAINER unset; restart leg skipped")
        return False
    try:
        is_socket = stat.S_ISSOCK(os.stat(DOCKER_SOCKET).st_mode)
    except OSError:
        is_socket = False
    if not is_socket:
        print(f"smoke: {DOCKER_SOCKET} unavailable; restart leg skipped")
        return False
    status, info = docker_api("GET", f"/containers/{RESTART_CONTAINER}/json")
    assert status == 200, f"inspect {RESTART_CONTAINER}: {status} {info}"
    started_before = info["State"]["StartedAt"]
    status, body = docker_api(
        "POST", f"/containers/{RESTART_CONTAINER}/restart?t=5")
    assert status == 204, f"restart {RESTART_CONTAINER}: {status} {body}"
    status, info = docker_api("GET", f"/containers/{RESTART_CONTAINER}/json")
    assert status == 200 and info["State"]["Running"], \
        f"{RESTART_CONTAINER} not running after restart: {status} {info}"
    assert info["State"]["StartedAt"] != started_before, \
        "container was not actually restarted"
    print(f"smoke: restarted container {RESTART_CONTAINER} via Docker API")
    return True


def main() -> int:
    health = wait_healthy()
    print(f"smoke: service healthy at {BASE}, next_seq={health['next_seq']}")

    # 1. A full 32-record batch of doses near the signed-int64 upper bound.
    body = commit(health["next_seq"], FULL_BATCH)
    seq0 = body["seq"]
    assert body["count"] == len(FULL_BATCH), body
    head_after = seq0 + len(FULL_BATCH)
    assert body["next_seq"] == head_after, body
    print(f"smoke: committed full batch seq={seq0}..{head_after - 1}")

    # 2. A service restart must not lose the batch that already got a 201.
    if maybe_restart_container():
        health = wait_healthy()
        assert health["next_seq"] == head_after, \
            f"next_seq regressed after restart: {health}"
        print(f"smoke: after restart next_seq={health['next_seq']} (kept)")

    # 3. The committed batch pages back continuously from its first seq.
    page_records = read_all_from(seq0 - 1)
    got = page_records[: len(FULL_BATCH)]
    assert [r["seq"] for r in got] == list(range(seq0, head_after)), got
    assert [r["dose"] for r in got] == FULL_BATCH, got
    seqs = [r["seq"] for r in page_records]
    assert seqs == list(range(seq0, seq0 + len(seqs))), \
        f"gaps/duplicates after restart: {seqs}"
    print(f"smoke: paged back {len(got)} records continuously from {seq0}")

    # 4. A follow-up batch commits at the advertised head.
    body = commit(head_after, FOLLOW_UP)
    assert body["seq"] == head_after, body
    tail = read_all_from(head_after - 1)
    assert [(r["seq"], r["dose"]) for r in tail] == [
        (head_after + i, d) for i, d in enumerate(FOLLOW_UP)], tail
    print(f"smoke: appended at {head_after}, sequence continuous")

    # 5. The whole log is gap-free from seq 1.
    everything = read_all_from(0)
    all_seqs = [r["seq"] for r in everything]
    assert all_seqs == list(range(1, len(all_seqs) + 1)), all_seqs
    print(f"smoke: full log continuous, {len(all_seqs)} records, "
          f"head={len(all_seqs) + 1}")
    print("HTTP SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"HTTP SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
