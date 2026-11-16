"""Probe official updated archive metadata; no question or label extraction."""
from pathlib import Path
import requests,json,hashlib
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');p=root/'artifacts/private/cycle3_20261116/2wiki_alias_source_probe';p.mkdir(exist_ok=False)
url='https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1';s=requests.Session();s.trust_env=False;s.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
try:
    response=s.get(url,headers={'Range':'bytes=-65536'},stream=True,timeout=(20,40));info=dict(source=url,status=response.status_code,content_type=response.headers.get('Content-Type'),content_range=response.headers.get('Content-Range'),content_length=response.headers.get('Content-Length'),label_fields_extracted=False)
    if response.status_code==206:
        raw=response.raw.read(65537);assert len(raw)<=65536;(p/'ZIP_TAIL.bin').write_bytes(raw);info.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),range_supported=True)
    else:info['range_supported']=False
    response.close()
except Exception as e:info=dict(source=url,status='NETWORK_ERROR',error_type=type(e).__name__,label_fields_extracted=False)
(p/'RECEIPT.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
