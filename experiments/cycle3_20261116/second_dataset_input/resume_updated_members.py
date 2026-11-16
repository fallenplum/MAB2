"""Resumable compressed-member transport; preserve and validate existing partials."""
from pathlib import Path
import io,json,hashlib,requests,zipfile,struct,zlib,time
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');p=root/'artifacts/private/cycle3_20261116/2wiki_alias_source_probe';out=root/'data/2wiki_author_ids_april7_v1';probe=json.loads((p/'RECEIPT.json').read_bytes());size=int(probe['content_range'].split('/')[-1]);tail=(p/'ZIP_TAIL.bin').read_bytes();url=probe['source']
class Metadata(io.RawIOBase):
 def __init__(self):self.pos=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,x,w=0):self.pos=x if w==0 else self.pos+x if w==1 else size+x;return self.pos
 def read(self,n=-1):
  end=min(size,self.pos+n) if n>=0 else size;assert self.pos>=size-len(tail);v=tail[self.pos-size+len(tail):end-size+len(tail)];self.pos=end;return v
session=requests.Session();session.trust_env=False;session.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'};ranges=[];failures=[];started=time.time()
def fetch(start,end):
 for attempt in range(5):
  try:
   res=session.get(url,headers={'Range':f'bytes={start}-{end-1}'},timeout=(20,35));assert res.status_code==206 and res.headers['Content-Range']==f'bytes {start}-{end-1}/{size}'
   content=res.content;assert len(content)==end-start;res.close();ranges.append(dict(start=start,end=end-1,bytes=len(content)));return content
  except (requests.RequestException,AssertionError) as e:
   failures.append(dict(start=start,end=end-1,attempt=attempt+1,error_type=type(e).__name__));print(json.dumps(dict(transport_retry=attempt+1,error_type=type(e).__name__)),flush=True)
   if attempt==4:raise
   time.sleep(min(2**attempt,8))
receipt=dict(source=url,archive_bytes=size,archive_tail_sha256=probe['sha256'],started_unix=started,status='IN_PROGRESS',labels_parsed=False,dev_test_members_opened=False,full_archive_downloaded=False,downloaded=[],prior_partial_preserved=True)
try:
 with zipfile.ZipFile(Metadata()) as z:
  for info in z.infolist():
   if info.filename not in ['train.json','id_aliases.json']:continue
   header=fetch(info.header_offset,info.header_offset+30);fields=struct.unpack('<4s5H3I2H',header);assert fields[0]==b'PK\x03\x04' and fields[3]==8 and not fields[2]&1
   start=info.header_offset+30+fields[-2]+fields[-1];cache=out/(info.filename+'.deflate');cachepart=out/(info.filename+'.deflate.part')
   if not cache.exists():
    have=cachepart.stat().st_size if cachepart.exists() else 0;assert have<=info.compress_size
    with cachepart.open('ab') as dst:
     while have<info.compress_size:
      stop=min(have+4*1024*1024,info.compress_size);raw=fetch(start+have,start+stop);dst.write(raw);dst.flush();have=stop
      if have%(32*1024*1024)==0 or have==info.compress_size:print(json.dumps(dict(member=info.filename,compressed_bytes=have,total=info.compress_size)),flush=True)
    cachepart.rename(cache)
   assert cache.stat().st_size==info.compress_size
   target=out/info.filename;part=out/(info.filename+'.part');prior=target if target.exists() else part;prior_size=prior.stat().st_size if prior.exists() else 0;assert prior_size<=info.file_size
   check=prior.open('rb') if prior.exists() else io.BytesIO();dst=None if target.exists() else part.open('ab');decoder=zlib.decompressobj(-15);crc=0;count=0;h=hashlib.sha256()
   try:
    with cache.open('rb') as src:
     while True:
      compressed=src.read(4*1024*1024)
      if not compressed:break
      raw=decoder.decompress(compressed);n=min(len(raw),max(0,prior_size-count));assert check.read(n)==raw[:n],'Retained partial differs; never overwrite'
      if len(raw)>n:
       assert dst is not None;dst.write(raw[n:])
      crc=zlib.crc32(raw,crc);h.update(raw);count+=len(raw)
    assert decoder.eof and not decoder.unused_data and count==info.file_size and crc==info.CRC
   finally:
    check.close()
    if dst:dst.close()
   if not target.exists():part.rename(target)
   receipt['downloaded'].append(dict(member=info.filename,bytes=count,sha256=h.hexdigest(),zip_CRC_verified=True,validated_prior_partial_bytes=prior_size));print(json.dumps(receipt['downloaded'][-1]),flush=True)
 receipt['status']='SELECTED_AUTHOR_TRAIN_AND_ALIAS_HASHED_UNOPENED'
finally:
 receipt.update(finished_unix=time.time(),elapsed_seconds=time.time()-started,successful_range_bytes=sum(x['bytes'] for x in ranges),retry_failures=len(failures),failed_transfer_bytes='UNKNOWN_NOT_IMPUTED')
 (out/'SOURCE_DESCRIPTOR_RESUMED.json').write_text(json.dumps(receipt,indent=2)+'\n');(p/'RESUMED_RANGE_LOG.json').write_text(json.dumps(dict(ranges=ranges,failures=failures),indent=2)+'\n')
