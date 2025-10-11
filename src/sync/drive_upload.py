"""Upload helper for pushing transcripts to Google Drive."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import google.auth
from google.auth.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from ..ocr.main import READY_ZIP

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
FOLDER_NAME = "Stand-up Notes"


def _load_credentials(credentials_path: Optional[Path]) -> Credentials:
    if credentials_path:
        return service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=SCOPES
        )
    creds, _ = google.auth.default(scopes=SCOPES)
    return creds


def _build_service(creds: Credentials):
    return build("drive", "v3", credentials=creds)


def _ensure_folder(service, name: str) -> str:
    query = (
        "name = '{}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    ).format(name.replace("'", "\'"))
    response = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1)
        .execute()
    )
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    folder_metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder["id"]


def _share_publicly(service, file_id: str) -> None:
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
        fields="id",
    ).execute()


def upload_ready_zip(credentials_path: Optional[Path] = None) -> str:
    if not READY_ZIP.exists():
        raise FileNotFoundError(
            "ready_for_review.zip not found. Run 'python -m src.ocr.main --agentic' first."
        )

    creds = _load_credentials(credentials_path)
    service = _build_service(creds)

    folder_id = _ensure_folder(service, FOLDER_NAME)

    media = MediaFileUpload(str(READY_ZIP), mimetype="application/zip", resumable=True)
    metadata = {"name": READY_ZIP.name, "parents": [folder_id]}

    file = (
        service.files()
        .create(body=metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )
    file_id = file["id"]
    _share_publicly(service, file_id)
    info = service.files().get(fileId=file_id, fields="webViewLink, webContentLink").execute()
    return info.get("webViewLink") or info.get("webContentLink") or ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Upload ready_for_review.zip to Google Drive")
    parser.add_argument(
        "--credentials",
        type=Path,
        help="Path to a service account JSON file. Defaults to application default credentials.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        link = upload_ready_zip(args.credentials)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return
    except HttpError as exc:  # pragma: no cover - network interactions
        logging.error("Drive API error: %s", exc)
        return

    if not link:
        logging.warning("Upload succeeded but no share link was returned.")
    else:
        logging.info("Uploaded ready_for_review.zip. Share link: %s", link)


if __name__ == "__main__":
    main()
