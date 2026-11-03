"""Pinned public HF model download, resumable and hash checked before activation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import requests

REPO = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def atomic(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def verified(path, entry):
    if not path.exists() or path.stat().st_size != entry["size"]:
        return False
    if "lfs" in entry:
        h = hashlib.sha256()
    else:
        h = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest() == entry.get("lfs", {}).get("sha256", entry["blobId"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    target = root / "models/Qwen2.5-3B-Instruct"
    audit = root / "artifacts/audit"
    target.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if args.direct:
        session.trust_env = False
    response = session.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=45)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION:
        raise ValueError("revision mismatch")
    entries = [x for x in metadata["siblings"] if x["rfilename"].endswith((".json", ".safetensors", ".txt", ".md")) and "/" not in x["rfilename"]]
    atomic(audit / "model_source_manifest.json", {"repo":REPO, "revision":REVISION, "files":entries})

    def download(entry):
        dest = target / entry["rfilename"]
        if verified(dest, entry):
            return {"name":dest.name, "state":"VERIFIED_EXISTING"}
        part = dest.with_name(dest.name + ".partial")
        for attempt in range(6):
            try:
                size = part.stat().st_size if part.exists() else 0
                if size == entry["size"]:
                    if not verified(part, entry):
                        raise ValueError("Full partial hash mismatch; preserved for review")
                    os.replace(part, dest)
                    return {"name":dest.name, "state":"VERIFIED"}
                if size > entry["size"]:
                    raise ValueError("Oversize partial")
                headers = {"Range":f"bytes={size}-"} if size else {}
                with session.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{dest.name}", headers=headers, stream=True, timeout=(30,60)) as r:
                    r.raise_for_status()
                    if size and r.status_code == 200:
                        size = 0
                    if size and not r.headers.get("Content-Range", "").startswith(f"bytes {size}-"):
                        raise ValueError("Range response mismatch")
                    with part.open("ab" if size else "wb") as out:
                        for block in r.iter_content(2*1024*1024):
                            if block:
                                out.write(block)
                if not verified(part, entry):
                    raise ValueError("Downloaded file size/hash mismatch")
                os.replace(part, dest)
                print(json.dumps({"file":dest.name,"status":"VERIFIED"}), flush=True)
                return {"name":dest.name, "state":"VERIFIED"}
            except (requests.RequestException, OSError) as exc:
                print(json.dumps({"file":dest.name,"attempt":attempt+1,"error":type(exc).__name__}), flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 30))
        raise RuntimeError("unreachable")

    atomic(audit / "model_download_status.json", {"status":"RUNNING", "pid":os.getpid(), "revision":REVISION})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(download, entries))
        atomic(audit / "model_download_status.json", {"status":"PASS", "revision":REVISION, "receipts":receipts})
    except Exception as exc:
        atomic(audit / "model_download_status.json", {"status":"FAILED", "revision":REVISION, "error_type":type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
