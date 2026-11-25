from pathlib import Path
import urllib.request, hashlib, json, time
url='https://drive.usercontent.google.com/download?id=1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I&export=download&confirm=t&uuid=a36be8e3-1915-433f-9d64-8bc32a9f0a18'
target=Path(r'E:\图结构演化\data\raw\factscore\enwiki-20230401.db')
target.parent.mkdir(parents=True,exist_ok=True)
part=target.with_suffix('.db.part')
if part.exists():
    part.unlink()
req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=120) as r, part.open('wb') as f:
    total=int(r.headers.get('Content-Length') or 0)
    done=0
    last=0
    start=time.time()
    while True:
        b=r.read(4*1024*1024)
        if not b: break
        f.write(b); done+=len(b)
        if done-last>=256*1024*1024:
            elapsed=max(time.time()-start,1)
            print(json.dumps({'event':'progress','bytes':done,'total':total,'percent':round(done*100/total,2) if total else None,'mbps':round(done/elapsed/1024/1024,2)},ensure_ascii=False),flush=True)
            last=done
part.replace(target)
h=hashlib.sha256()
with target.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
print(json.dumps({'event':'complete','path':str(target),'bytes':target.stat().st_size,'sha256':h.hexdigest()},ensure_ascii=False),flush=True)

