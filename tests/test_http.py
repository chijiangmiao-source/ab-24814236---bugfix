"""End-to-end tests against a real HTTP server in a background thread."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app.server import build_server


def _request(method: str, url: str, body: object = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.wal_path = os.path.join(self.dir, "wal.bin")
        self.httpd: ThreadingHTTPServer = build_server(
            "127.0.0.1", 0, self.wal_path
        )
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def test_health_and_info(self) -> None:
        status, body = _request("GET", f"{self.base}/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "next_seq": 1})

    def test_post_batch_then_cursor_read(self) -> None:
        status, body = _request(
            "POST", f"{self.base}/api/batches",
            {"expected_seq": 1, "records": [10, 20, 30]},
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["seq"], 1)
        self.assertEqual(body["next_seq"], 4)

        status, body = _request(
            "GET", f"{self.base}/api/records?cursor=0&limit=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["records"],
                         [{"seq": 1, "dose": 10}, {"seq": 2, "dose": 20}])
        self.assertEqual(body["next_cursor"], 2)

        status, body = _request(
            "GET", f"{self.base}/api/records?cursor=2"
        )
        self.assertEqual(body["records"], [{"seq": 3, "dose": 30}])
        self.assertEqual(body["next_cursor"], 3)

    def test_conflict_loser_keeps_log_clean(self) -> None:
        s1, _ = _request("POST", f"{self.base}/api/batches",
                         {"expected_seq": 1, "records": [1]})
        self.assertEqual(s1, 201)
        s2, body = _request("POST", f"{self.base}/api/batches",
                            {"expected_seq": 1, "records": [2]})
        self.assertEqual(s2, 409)
        self.assertEqual(body["current_seq"], 2)

        _, body = _request("GET", f"{self.base}/api/records")
        self.assertEqual(body["records"], [{"seq": 1, "dose": 1}])

    def test_bad_requests(self) -> None:
        for payload in (
            {"expected_seq": 1, "records": []},
            {"expected_seq": 1, "records": [1] * 33},
            {"expected_seq": 0, "records": [1]},
            {"expected_seq": 1, "records": "nope"},
        ):
            status, body = _request(
                "POST", f"{self.base}/api/batches", payload
            )
            self.assertEqual(status, 400, payload)
            self.assertEqual(body["error"], "bad_request")

        status, body = _request("GET", f"{self.base}/api/records?cursor=x")
        self.assertEqual(status, 400)

    def test_rebuilds_sequence_after_restart(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [5, 6]})
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 3, "records": [7]})
        self._restart()

        status, health = _request("GET", f"{self.base}/healthz")
        self.assertEqual(health["next_seq"], 4)
        _, body = _request("GET", f"{self.base}/api/records")
        self.assertEqual(
            body["records"],
            [{"seq": 1, "dose": 5}, {"seq": 2, "dose": 6},
             {"seq": 3, "dose": 7}],
        )
        # A stale expected_seq is rejected; correct one succeeds.
        self.assertEqual(
            _request("POST", f"{self.base}/api/batches",
                     {"expected_seq": 3, "records": [8]})[0],
            409,
        )
        self.assertEqual(
            _request("POST", f"{self.base}/api/batches",
                     {"expected_seq": 4, "records": [8]})[0],
            201,
        )

    def test_full_batch_restart_then_append_and_pagination(self) -> None:
        # The reported chain over HTTP: a full 32-record batch of doses
        # near the int64 upper bound commits, the service restarts, and
        # the committed batch must page back continuously; the next batch
        # appends at the advertised head without gaps or duplicates.
        records = [2**63 - 1 - i for i in range(32)]
        status, body = _request(
            "POST", f"{self.base}/api/batches",
            {"expected_seq": 1, "records": records},
        )
        self.assertEqual(status, 201)
        self.assertEqual(body, {"status": "committed", "seq": 1,
                                "count": 32, "next_seq": 33})

        self._restart()

        status, health = _request("GET", f"{self.base}/healthz")
        self.assertEqual((status, health["next_seq"]), (200, 33))

        seen: list[dict] = []
        cursor = 0
        while True:
            status, page = _request(
                "GET", f"{self.base}/api/records?cursor={cursor}&limit=7"
            )
            self.assertEqual(status, 200)
            seen.extend(page["records"])
            cursor = page["next_cursor"]
            if not page["records"]:
                break
        self.assertEqual([r["seq"] for r in seen], list(range(1, 33)))
        self.assertEqual([r["dose"] for r in seen], records)

        status, body = _request(
            "POST", f"{self.base}/api/batches",
            {"expected_seq": 33, "records": [-1]},
        )
        self.assertEqual((status, body["seq"], body["next_seq"]),
                         (201, 33, 34))
        _, page = _request("GET", f"{self.base}/api/records?cursor=32")
        self.assertEqual(page["records"], [{"seq": 33, "dose": -1}])

    def _restart(self) -> None:
        """Bounce the service on the same WAL file (a service restart)."""
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.httpd = build_server("127.0.0.1", 0, self.wal_path)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def test_poisoned_log_returns_503_for_everything(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1, 2, 3]})
        # Corrupt a payload byte of the committed frame, then restart: the
        # damage is only visible to a fresh scan of the file.
        data = bytearray(open(self.wal_path, "rb").read())
        data[24] ^= 0xFF
        with open(self.wal_path, "wb") as fh:
            fh.write(data)
        self._restart()

        self.assertEqual(_request("GET", f"{self.base}/healthz")[0], 503)
        self.assertEqual(
            _request("GET", f"{self.base}/api/records")[0], 503
        )
        self.assertEqual(
            _request("POST", f"{self.base}/api/batches",
                     {"expected_seq": 1, "records": [9]})[0],
            503,
        )


if __name__ == "__main__":
    unittest.main()
