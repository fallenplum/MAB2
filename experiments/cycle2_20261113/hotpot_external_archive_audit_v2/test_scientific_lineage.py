import copy,json,unittest
import archive_audit as a
FIELDS=('scientific_scope','population','seeds','execution','failure_policy','statistics','evidence_rules','analysis_pins','selection_pins')
def enc(x):return (a.canonical(x)+'\n').encode()
class LineageTests(unittest.TestCase):
    def setUp(self):
        self.design={'status':'FROZEN',**{k:{'fixed':'sentinel_'+k} for k in FIELDS}}
        self.raw=enc(self.design);self.pin=a.sha(self.raw)
        self.execution={**copy.deepcopy(self.design),'original_scientific_design_sha256':self.pin,'resource':{'worker_caps':[43020]*4}}
        self.eraw=enc(self.execution);self.selection={'pins':{'protocol_sha256':self.pin}}
        self.bindings={'scientific_protocol_sha256':self.pin}
        self.contract={'scientific_design_key':'DESIGN','execution_protocol_sha256':a.sha(self.eraw)}
    def test_exact_original_single_stage(self):
        a.check_scientific_lineage(self.raw,self.selection,self.bindings,{}, {})
    def test_resource_binding_preserves_science(self):
        a.check_scientific_lineage(self.eraw,self.selection,self.bindings,self.contract,{'DESIGN':self.raw})
    def test_each_changed_scientific_field_refused_even_with_updated_execution_pin(self):
        for field in FIELDS:
            changed=copy.deepcopy(self.execution);changed[field]={'changed':True};raw=enc(changed)
            with self.assertRaisesRegex(a.AuditError,'scientific design changed'):
                a.check_scientific_lineage(raw,self.selection,self.bindings,{**self.contract,'execution_protocol_sha256':a.sha(raw)},{'DESIGN':self.raw})
    def test_missing_or_changed_original_bytes_refused(self):
        for aux in ({},{'DESIGN':b'{}'}):
            with self.assertRaises(a.AuditError):a.check_scientific_lineage(self.eraw,self.selection,self.bindings,self.contract,aux)
    def test_execution_hash_cannot_substitute_original_selection_hash(self):
        with self.assertRaises(a.AuditError):a.check_scientific_lineage(self.eraw,self.selection,{'scientific_protocol_sha256':a.sha(self.eraw)},self.contract,{'DESIGN':self.raw})
    def test_equal_python_values_with_different_json_types_refused(self):
        for original,altered in ((False,0),(True,1),(3,3.0)):
            design=copy.deepcopy(self.design);design['statistics']={'sentinel':original};draw=enc(design);pin=a.sha(draw)
            execution={**copy.deepcopy(design),'original_scientific_design_sha256':pin};execution['statistics']={'sentinel':altered};eraw=enc(execution)
            with self.assertRaisesRegex(a.AuditError,'scientific design changed'):
                a.check_scientific_lineage(eraw,{'pins':{'protocol_sha256':pin}},{'scientific_protocol_sha256':pin},
                    {'scientific_design_key':'DESIGN','execution_protocol_sha256':a.sha(eraw)},{'DESIGN':draw})
if __name__=='__main__':unittest.main(verbosity=2)
