from pathlib import Path
import urllib.request, zipfile, hashlib, json
root=Path('experiments/e0_real_workflow_20261126/external/factscore_official')
archive=root/'FActScore-main.zip'
url='https://github.com/shmsw25/FActScore/archive/refs/heads/main.zip'
urllib.request.urlretrieve(url, archive)
with zipfile.ZipFile(archive) as zf:
    members=zf.namelist()
    zf.extractall(root/'source')
record={'status':'DOWNLOADED_AND_ZIP_VALID','url':url,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'member_count':len(members),'has_download_script':any(n.endswith('factscore/download_data.py') for n in members)}
archive.unlink()
(root/'SOURCE_DOWNLOAD_MANIFEST.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))

