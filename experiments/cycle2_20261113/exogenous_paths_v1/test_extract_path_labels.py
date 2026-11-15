import tempfile,unittest
from pathlib import Path
import extract_path_labels as target
class Tests(unittest.TestCase):
    def test_unapproved_release_fails_closed(self):
        with self.assertRaises(AssertionError):target.check_release({'decision':'PENDING'})
    def test_source_tamper_fails_before_data(self):
        with self.assertRaises(AssertionError):target.check_release({'decision':'APPROVED_FIXED_PATH_TRAIN_LABEL_EXTRACTION','source_sha256':'0'*64})
    def test_only_selected_ids_materialized(self):
        import pyarrow as pa,pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'synthetic.parquet'
            pq.write_table(pa.table({'id':['not_selected','a','b'],'answer':['hidden','x','y'],'supporting_facts':['never_read']*3}),p)
            vault,n=target.extract(['b','a'],{'parquet_files':[{'path':str(p),'num_rows':3}]})
            self.assertEqual(vault,{'a':'x','b':'y'});self.assertEqual(n,3)
    def test_duplicate_selected_ids_rejected(self):
        import pyarrow as pa,pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'synthetic.parquet';pq.write_table(pa.table({'id':['a','a'],'answer':['x','x']}),p)
            with self.assertRaises(AssertionError):target.extract(['a'],{'parquet_files':[{'path':str(p),'num_rows':2}]})
if __name__=='__main__':unittest.main()
