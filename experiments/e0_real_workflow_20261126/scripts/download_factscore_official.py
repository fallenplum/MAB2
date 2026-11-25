from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

root = Path('experiments/e0_real_workflow_20261126/external/factscore_official')
root.mkdir(parents=True, exist_ok=True)
items = {
    'data.zip': '155exEdKs7R21gZF4G-x54-XN3qswBcPo',
    'demos.zip': '1sbW6pkYl6cc9gooD4WLaeoFKcAj3poZu',
}
manifest = {'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'source': 'https://github.com/shmsw25/FActScore/blob/main/factscore/download_data.py', 'items': {}}
for name, file_id in items.items():
    target = root / name
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    urllib.request.urlretrieve(url, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
        info = {'status': 'DOWNLOADED_AND_ZIP_VALID', 'file_id': file_id, 'url': url, 'bytes': target.stat().st_size, 'sha256': digest, 'members': names}
        extract_dir = root / name[:-4]
        extract_dir.mkdir(exist_ok=True)
        zf.extractall(extract_dir)
    target.unlink()
    manifest['items'][name] = info
(root / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))

