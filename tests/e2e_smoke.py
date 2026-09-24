"""End-to-end proof scenario, runnable inside the container (stdlib only).

It drives a *real* server subprocess through the full lifecycle:

  1. healthy start on an empty log
  2. batch commit + cursor pagination over HTTP
  3. torn-write recovery: a writer process is killed (``os._exit``) after
     emitting only part of a frame; restart must truncate the tail and
     rebuild the continuous sequence
  4. corruption isolation: a byte is flipped inside a *complete* frame;
     restart must serve 503 on health, reads and appends (poisoned)
  5. restart durability: a maximal 32-record batch of int64-scale doses
     must still read back 1..32 with next_seq 33 after a restart, and the
     follow-up batch appends at 33 without pagination gaps or duplicates
  6. concurrent preemption over HTTP: N clients submit the same
     ``expected_seq``; exactly one gets 201, all others 409, and the WAL
     grows by exactly one frame -- no loser bytes, no consumed numbers
  7. if LEDGER_BASE_URL is set (docker compose), also smoke that instance

Exit status is non-zero if any constraint is violated, so a CI/compose
"verify" service's exit code is the evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.wal import (  # noqa: E402
    DIGEST_LEN,
    HEADER_LEN,
    canonical_payload,
    encode_frame,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def request(method: str, url: str, body: object = None,
            timeout: float = 10.0) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else {}


def wait_ready(base: str, expect_status: int = 200,
               timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            status, body = request("GET", f"{base}/healthz")
            if status == expect_status:
                return body
        except (OSError, urllib.error.URLError) as exc:
            last_err = exc
        time.sleep(0.1)
    raise AssertionError(
        f"server at {base} never reported {expect_status}: {last_err}"
    )


def start_server(wal_path: str, expect_status: int = 200
                 ) -> tuple[subprocess.Popen, str]:
    # Bind an ephemeral port ourselves to avoid races.
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = dict(os.environ)
    env.update({
        "LEDGER_HOST": "127.0.0.1",
        "LEDGER_PORT": str(port),
        "LEDGER_WAL_PATH": wal_path,
        "PYTHONUNBUFFERED": "1",
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.server"],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        wait_ready(base, expect_status=expect_status)
    except AssertionError:
        proc.terminate()
        out = proc.stdout.read() if proc.stdout else ""
        raise AssertionError(
            f"server failed to reach status {expect_status}:\n{out}")
    return proc, base


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="ledger-e2e-")
    wal_path = os.path.join(tmp, "wal.bin")

    section("1. empty log starts healthy")
    proc, base = start_server(wal_path)
    try:
        status, body = request("GET", f"{base}/healthz")
        assert status == 200 and body["next_seq"] == 1, body
        print("health:", body)

        section("2. batch commits and cursor pagination")
        status, body = request(
            "POST", f"{base}/api/batches",
            {"expected_seq": 1, "records": [10, 20, 30]})
        assert status == 201, body
        assert body == {"status": "committed", "seq": 1, "count": 3,
                        "next_seq": 4}, body
        status, body = request(
            "POST", f"{base}/api/batches",
            {"expected_seq": 4, "records": [40, 50]})
        assert status == 201 and body["next_seq"] == 6, body

        all_records = []
        cursor = 0
        while True:
            q = urlencode({"cursor": cursor, "limit": 2})
            status, page = request("GET", f"{base}/api/records?{q}")
            assert status == 200, page
            all_records.extend(page["records"])
            cursor = page["next_cursor"]
            if not page["records"]:
                break
        assert [(r["seq"], r["dose"]) for r in all_records] == [
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)], all_records
        print("read back:", all_records)
        size_after_commits = os.path.getsize(wal_path)
    finally:
        stop_server(proc)

    section("3. writer killed mid frame -> torn tail truncated on restart")
    # A real crashed append always carried the correct next frame ordinal
    # (two committed frames -> frame 3); only its trailing bytes are missing.
    killer_code = (
        "import os,sys;"
        "import app.wal as w;"
        "f=open(os.environ['LEDGER_WAL_PATH'],'ab');"
        "f.write(w.encode_frame(3, w.canonical_payload([7,8,9]))[:40]);"
        "f.flush(); os._exit(1)"
    )
    killed = subprocess.run(
        [sys.executable, "-c", killer_code],
        cwd=REPO_ROOT, env={**os.environ, "LEDGER_WAL_PATH": wal_path},
    )
    assert killed.returncode != 0
    torn_size = os.path.getsize(wal_path)
    assert torn_size == size_after_commits + 40
    proc, base = start_server(wal_path)
    try:
        assert os.path.getsize(wal_path) == size_after_commits, \
            "torn bytes must be physically removed"
        status, body = request("GET", f"{base}/healthz")
        assert status == 200 and body["next_seq"] == 6, body
        # Numbering continues seamlessly; no half-batch visible.
        status, page = request("GET", f"{base}/api/records?cursor=0")
        assert len(page["records"]) == 5, page
        status, body = request(
            "POST", f"{base}/api/batches",
            {"expected_seq": 6, "records": [60]})
        assert status == 201 and body["seq"] == 6 and body["next_seq"] == 7, body
        print("recovered next_seq=6, torn 40 bytes removed, seq 6 reused-free")
    finally:
        stop_server(proc)

    section("4. corruption inside a complete frame -> poisoned (503 only)")
    data = bytearray(open(wal_path, "rb").read())
    # Flip a payload byte of the very first frame, well before EOF.
    data[HEADER_LEN + 1] ^= 0xFF
    with open(wal_path, "wb") as fh:
        fh.write(data)
    poisoned_size = os.path.getsize(wal_path)
    proc, base = start_server(wal_path, expect_status=503)
    try:
        checks = (
            ("GET", "/healthz", None),
            ("GET", "/api/records", None),
            ("POST", "/api/batches",
             {"expected_seq": 1, "records": [1]}),
        )
        for method, path, payload in checks:
            status, body = request(method, f"{base}{path}", payload)
            assert status == 503, (path, status, body)
            if path == "/healthz":
                assert body["status"] == "poisoned", body
            else:
                assert body["error"] == "ledger_poisoned", body
        print("health/records/batches all return 503 poisoned")
    finally:
        stop_server(proc)
    assert os.path.getsize(wal_path) == poisoned_size, \
        "poisoned log must not be silently truncated/rewritten"
    # Restore a clean log for later phases: rebuild from scratch.
    os.unlink(wal_path)

    section("5. full 32-record batch survives a restart, append at 33")
    # The durability incident over a real process restart: a maximal batch
    # of doses near the signed-integer upper bound is acknowledged, the
    # service is restarted, and the batch must still read back 1..32 with
    # next_seq 33; the next batch appends at 33 and pagination neither
    # skips nor duplicates a sequence.
    durable_path = os.path.join(tmp, "durable.bin")
    big_records = [2**63 - 1 - i for i in range(32)]
    proc, base = start_server(durable_path)
    try:
        status, body = request(
            "POST", f"{base}/api/batches",
            {"expected_seq": 1, "records": big_records})
        assert status == 201 and body == {
            "status": "committed", "seq": 1, "count": 32,
            "next_seq": 33}, body
        committed_size = os.path.getsize(durable_path)
    finally:
        stop_server(proc)

    proc, base = start_server(durable_path)
    try:
        # The acknowledged frame must be physically and logically intact.
        assert os.path.getsize(durable_path) == committed_size, \
            "restart must not truncate committed frames"
        status, body = request("GET", f"{base}/healthz")
        assert status == 200 and body == {"status": "ok", "next_seq": 33}, body

        replay: list[dict] = []
        cursor = 0
        while True:
            q = urlencode({"cursor": cursor, "limit": 7})
            status, page = request("GET", f"{base}/api/records?{q}")
            assert status == 200, page
            replay.extend(page["records"])
            if not page["records"]:
                break
            cursor = page["next_cursor"]
        assert replay == [{"seq": i + 1, "dose": big_records[i]}
                          for i in range(32)], replay
        assert cursor == 32

        # Follow-up batch at the advertised head succeeds, stays continuous.
        follow = [-1, 0, 2**31]
        status, body = request(
            "POST", f"{base}/api/batches",
            {"expected_seq": 33, "records": follow})
        assert status == 201 and body["next_seq"] == 36, body
        status, page = request("GET", f"{base}/api/records?cursor=32")
        assert [(r["seq"], r["dose"]) for r in page["records"]] == [
            (33, -1), (34, 0), (35, 2**31)], page
        print("restart kept next_seq=33; 1..32 replayed; append at 33 ok")
    finally:
        stop_server(proc)

    # A second restart must still show the complete chain.
    proc, base = start_server(durable_path)
    try:
        status, body = request("GET", f"{base}/healthz")
        assert status == 200 and body["next_seq"] == 36, body
        _, page = request("GET", f"{base}/api/records?cursor=0")
        seqs = [r["seq"] for r in page["records"]]
        assert seqs == list(range(1, 36)), seqs
        assert [r["dose"] for r in page["records"]] == big_records + follow
        print("second restart still shows 1..35 continuous")
    finally:
        stop_server(proc)

    section("6. HTTP concurrent preemption: one winner, zero loser bytes")
    proc, base = start_server(wal_path)
    try:
        n = 16
        request("POST", f"{base}/api/batches",
                {"expected_seq": 1, "records": [0]})  # head -> 2
        size_before = os.path.getsize(wal_path)
        # All worker payloads are [1000..1015], i.e. "[10xx]" = 6 bytes.
        expected_delta = (HEADER_LEN
                          + len(canonical_payload([1000])) + DIGEST_LEN)
        barrier = threading.Barrier(n)
        results: list[tuple[int, dict]] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            barrier.wait()
            res = request("POST", f"{base}/api/batches",
                          {"expected_seq": 2, "records": [1000 + i]})
            with lock:
                results.append(res)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r[0] == 201]
        losers = [r for r in results if r[0] == 409]
        assert len(winners) == 1, results
        assert len(losers) == n - 1, results
        assert all(r[1]["current_seq"] == 3 for r in losers), losers
        assert os.path.getsize(wal_path) - size_before == expected_delta, (
            os.path.getsize(wal_path) - size_before, expected_delta)
        _, page = request("GET", f"{base}/api/records")
        seqs = [r["seq"] for r in page["records"]]
        assert seqs == list(range(1, len(seqs) + 1)), seqs
        assert len(seqs) == 2, seqs
        print(f"{n} clients, 1 commit + {n-1} 409; file grew by exactly "
              f"{expected_delta} bytes; seqs={seqs}")
    finally:
        stop_server(proc)

    external = os.environ.get("LEDGER_BASE_URL")
    if external:
        section(f"7. external smoke against {external}")
        body = wait_ready(external, expect_status=200)
        status, body = request("POST", f"{external}/api/batches", {
            "expected_seq": body["next_seq"],
            "records": [9001, 9002],
        })
        assert status == 201, body
        q = urlencode({"cursor": body["seq"] - 1})
        status, page = request("GET", f"{external}/api/records?{q}")
        assert status == 200 and len(page["records"]) == 2, page
        print("external instance committed:", body, "read:", page["records"])

    print("\nALL E2E CONSTRAINTS VERIFIED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nE2E FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
