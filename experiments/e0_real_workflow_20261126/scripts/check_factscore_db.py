import sqlite3,json,hashlib
from pathlib import Path
p=Path(r'E:\图结构演化\data\raw\factscore\enwiki-20230401.db')
uri='file:'+p.as_posix()+'?mode=ro'
con=sqlite3.connect(uri,uri=True)
tables=con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
count=con.execute('SELECT COUNT(*) FROM documents').fetchone()[0] if ('documents',) in tables else None
sel=json.loads(Path('experiments/e0_real_workflow_20261126/TOPIC_SELECTION_MANIFEST.json').read_text(encoding='utf-8'))
# Read topic names only from the private downloaded data, then test first four in DB.
source=Path('experiments/e0_real_workflow_20261126/external/factscore_official/data/data/labeled/InstructGPT.jsonl')
ids={x['group_id'] for x in sel['groups'][:4]}
found=[]
for line in source.open(encoding='utf-8'):
 row=json.loads(line); gid='e0g_'+hashlib.sha256((sel['salt']+'|'+row['topic']).encode()).hexdigest()[:16]
 if gid in ids: found.append((gid,row['topic']))
checks=[{'group_id':g,'topic':t,'exists':con.execute('SELECT 1 FROM documents WHERE title=?',(t,)).fetchone() is not None} for g,t in found]
print(json.dumps({'bytes':p.stat().st_size,'tables':tables,'document_count':count,'checks':checks},ensure_ascii=False))
con.close()

