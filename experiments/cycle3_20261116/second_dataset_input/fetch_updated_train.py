"""Fetch only train/alias ZIP members by HTTP ranges; never open dev/test members."""
from pathlib import Path
import io,json,hashlib,requests,zipfile,shutil,time
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');p=root/'artifacts/private/cycle3_20261116/2wiki_alias_source_probe';out=root/'data/2wiki_author_ids_april7_v1';out.mkdir(exist_ok=False)
probe=json.loads((p/'RECEIPT.json').read_bytes());url=probe['source'];size=int(probe['content_range'].split('/')[-1]);tail=(p/'ZIP_TAIL.bin').read_bytes();assert hashlib.sha256(tail).hexdigest()==probe['sha256']
session=requests.Session();session.trust_env=False;session.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
class RangeReader(io.RawIOBase):
    def __init__(self):self.position=0;self.ranges=[];self.etag=None
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else size+offset;assert 0<=self.position<=size;return self.position
    def read(self,n=-1):
        if n<0:n=size-self.position
        end=min(size,self.position+n)
        if end==self.position:return b''
        start=self.position
        if start>=size-len(tail):raw=tail[start-(size-len(tail)):end-(size-len(tail))]
        else:
            headers={'Range':f'bytes={start}-{end-1}'}
            if self.etag:headers['If-Match']=self.etag
            res=session.get(url,headers=headers,timeout=(20,40));assert res.status_code==206 and res.headers['Content-Range']==f'bytes {start}-{end-1}/{size}',(res.status_code,res.headers.get('Content-Range'));raw=res.content;assert len(raw)==end-start
            etag=res.headers.get('ETag')
            if self.etag and etag:assert self.etag==etag
            if etag:self.etag=etag
            self.ranges.append(dict(start=start,end=end-1,bytes=len(raw)));res.close()
        self.position=end;return raw
reader=RangeReader();receipt=dict(source=url,archive_bytes=size,archive_tail_sha256=probe['sha256'],retrieved_unix=time.time(),members=[],downloaded=[],labels_parsed=False,dev_test_members_opened=False,full_archive_downloaded=False)
try:
    with zipfile.ZipFile(reader) as z:
        info=z.infolist();receipt['members']=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,CRC=x.CRC) for x in info];print(json.dumps(dict(members=receipt['members'])),flush=True)
        selected=[x for x in info if Path(x.filename).name in ['train.json','id_aliases.json']];assert sum(Path(x.filename).name=='train.json' for x in selected)==1
        assert sum(x.file_size for x in selected)<10*1024**3 and shutil.disk_usage(out).free>sum(x.file_size for x in selected)+1024**3
        for item in selected:
            target=out/Path(item.filename).name;part=target.with_suffix('.json.part');h=hashlib.sha256();count=0
            with z.open(item) as src,part.open('xb') as dst:
                while True:
                    chunk=src.read(8*1024*1024)
                    if not chunk:break
                    h.update(chunk);dst.write(chunk);count+=len(chunk)
            assert count==item.file_size;part.rename(target);receipt['downloaded'].append(dict(member=item.filename,filename=target.name,bytes=count,sha256=h.hexdigest(),zip_CRC_verified=True));print(json.dumps(receipt['downloaded'][-1]),flush=True)
        receipt['status']='SELECTED_AUTHOR_TRAIN_MEMBERS_HASHED_UNOPENED'
finally:
    receipt.update(range_requests=len(reader.ranges),network_member_range_bytes=sum(x['bytes'] for x in reader.ranges),entity_tag_present=reader.etag is not None)
    (out/'SOURCE_DESCRIPTOR.json').write_text(json.dumps(receipt,indent=2)+'\n');(p/'UPDATED_ARCHIVE_RANGE_LOG.json').write_text(json.dumps(reader.ranges,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','range_requests','network_member_range_bytes','labels_parsed','dev_test_members_opened']}),flush=True)
