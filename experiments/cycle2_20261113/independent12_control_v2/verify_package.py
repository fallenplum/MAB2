"""Verify archived bytes before CPU tests or a separately admitted qualification."""
from pathlib import Path
import argparse,hashlib,json

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
root=args.root.resolve(strict=True)
manifest=json.loads((root/'SHA256SUMS.json').read_text('utf-8'))
for row in manifest['files']:
    path=(root/row['path']).resolve(strict=True)
    assert path.is_relative_to(root),'Path escaped package root'
    assert path.stat().st_size==row['bytes'],'Byte length mismatch: '+row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],'SHA256 mismatch: '+row['path']
print(json.dumps({'status':'PASS_ARCHIVED_BYTES','files_verified':len(manifest['files']),'scientific_or_GPU_admission':False}))
