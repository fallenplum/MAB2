"""Offline CPU tokenizer witness. No model class or tensor/device is loaded."""
import hashlib
import json
from pathlib import Path

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class OfflineTokenizerWitness:
    def __init__(self,directory,identity):
        import transformers
        directory=Path(directory)
        if transformers.__version__!=identity['transformers']:
            raise ValueError('Chat-template audit requires qualified Transformers version')
        names=('tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','tokenizer.model','chat_template.jinja','config.json')
        verified={}
        for name in names:
            path=directory/name
            if name in identity['model_files']:
                raw=path.read_bytes();pin=identity['model_files'][name]
                if hashlib.sha256(raw).hexdigest()!=pin['sha256'] or ('bytes' in pin and len(raw)!=pin['bytes']):
                    raise ValueError('Tokenizer/config bytes differ')
                verified[name]=pin['sha256']
            elif path.exists():raise ValueError('Unpinned tokenizer/config file')
        if not {'tokenizer.json','tokenizer_config.json','config.json'}<=set(verified):raise ValueError('Tokenizer identity incomplete')
        config=json.loads((directory/'config.json').read_bytes())
        self.position_limit=config.get('max_position_embeddings',8192+256)
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
        if type(self.tokenizer).__name__!=identity['tokenizer_class'] or hashlib.sha256((self.tokenizer.chat_template or '').encode()).hexdigest()!=identity['tokenizer_chat_template_sha256']:
            raise ValueError('Loaded tokenizer class/chat template differs')
        # Match the native backend's explicit pad fallback.
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        self.provenance={'backend_identity_sha256':digest(identity),'files_sha256':verified,
                         'transformers':transformers.__version__,'chat_template_reexecuted':True,'synthetic_only':False}
    def encode(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
    def decode(self,tokens):return self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    def chat_count(self,messages):
        encoded=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if isinstance(encoded,dict):encoded=encoded['input_ids']
        if encoded and isinstance(encoded[0],list):
            if len(encoded)!=1:raise ValueError('Unexpected tokenizer batch')
            encoded=encoded[0]
        return len(encoded)
