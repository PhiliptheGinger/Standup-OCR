"""Handwriting OCR API client.

Small, focused client for https://www.handwritingocr.com API.
- Uploads a single PNG/PDF with action=transcribe
- Polls status until finished
- Returns result JSON

Features:
- Rate limiting to ~2 RPS (min_interval)
- Exponential backoff on 429 and 5xx
- Raises HandwritingOCRError on unrecoverable errors
"""
from __future__ import annotations
from pathlib import Path
import time
import logging
from typing import Any, Dict, Optional
import requests
import sys

# Ensure repo root is on sys.path so "from src..." works when script is run directly.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

DEFAULT_BASE = "https://api.handwritingocr.com/v1"


class HandwritingOCRError(RuntimeError):
    pass


class HandwritingOCRClient:
    def __init__(self, token: str, *, base_url: str = DEFAULT_BASE, rps: float = 2.0, session: Optional[requests.Session] = None):
        """
        token: Bearer token string
        base_url: API base URL (default to known endpoint)
        rps: requests-per-second target (min interval = 1 / rps)
        """
        if not token:
            raise ValueError("Handwriting OCR token required")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.min_interval = 1.0 / max(0.1, float(rps))

    def _throttle(self):
        # Simple spacing to respect rate limit
        time.sleep(self.min_interval)

    def _request_with_backoff(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        backoff = 1.0
        for attempt in range(6):
            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.RequestException as e:
                logging.warning("Handwriting OCR request exception, retrying: %s", e)
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 429:
                logging.warning("Handwriting OCR rate limited (429), backing off %.1fs", backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            if 500 <= resp.status_code < 600:
                logging.warning("Handwriting OCR server error %d, backing off %.1fs", resp.status_code, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            return resp
        raise HandwritingOCRError(f"Request to {url} failed after retries (last status {resp.status_code})")

    def transcribe_image(self, image_path: Path, *, poll_interval: float = 1.0, timeout: int = 120) -> Dict[str, Any]:
        """
        Upload image_path (PNG/PDF) and poll until transcription completes.
        Returns transcription JSON as returned by the API.

        Raises HandwritingOCRError on failure/timeout.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise HandwritingOCRError(f"Image not found: {image_path}")

        files = {"file": (image_path.name, image_path.open("rb"), "image/png")}
        data = {"action": "transcribe"}
        # upload
        self._throttle()
        resp = self._request_with_backoff("POST", "/documents", files=files, data=data)
        if resp.status_code not in (200, 201):
            raise HandwritingOCRError(f"Upload failed: {resp.status_code} {resp.text}")
        doc = resp.json()
        doc_id = doc.get("id")
        if not doc_id:
            raise HandwritingOCRError(f"Upload response missing document id: {doc}")

        # poll
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise HandwritingOCRError("Transcription polling timed out")
            self._throttle()
            status_resp = self._request_with_backoff("GET", f"/documents/{doc_id}/status")
            if status_resp.status_code != 200:
                raise HandwritingOCRError(f"Status check failed: {status_resp.status_code} {status_resp.text}")
            status_json = status_resp.json()
            status = (status_json.get("status") or "").lower()
            if status in ("done", "completed", "finished", "success"):
                break
            if status in ("failed", "error"):
                raise HandwritingOCRError(f"Transcription failed: {status_json}")
            time.sleep(poll_interval)

        # fetch results
        self._throttle()
        res = self._request_with_backoff("GET", f"/documents/{doc_id}/results")
        if res.status_code != 200:
            raise HandwritingOCRError(f"Results fetch failed: {res.status_code} {res.text}")
        return res.json()