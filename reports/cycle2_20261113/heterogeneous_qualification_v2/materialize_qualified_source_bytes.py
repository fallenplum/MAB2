"""Reconstruct qualified deployment bytes from the immutable public Git commit."""
from pathlib import Path
import argparse,hashlib,json,subprocess

def main():
 p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists(),'Use a fresh output directory'
 m=json.loads(a.manifest.read_bytes());files=[]
 for row in m['files']:
  relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
  raw=subprocess.check_output(['git','-c','safe.directory='+a.repository.resolve().as_posix(),'-C',str(a.repository),'show',m['git_commit']+':'+row['path']])
  assert hashlib.sha256(raw).hexdigest()==row['git_blob_content_sha256']
  if row['byte_transform']=='LF_TO_CRLF':
   assert b'\r' not in raw;raw=raw.replace(b'\n',b'\r\n')
  else:assert row['byte_transform']=='IDENTITY'
  assert len(raw)==row['deployed_bytes'] and hashlib.sha256(raw).hexdigest()==row['deployed_sha256']
  files.append((relative,raw))
 a.output.mkdir(parents=True)
 for rel,raw in files:
  target=a.output/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 result={'status':'EXACT_QUALIFIED_SOURCE_BYTES_RECONSTRUCTED','git_commit':m['git_commit'],'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'files':len(files),'models_loaded':False,'inference_calls':0,'private_runtime_plan_included':False}
 (a.output/'RECONSTRUCTION_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
