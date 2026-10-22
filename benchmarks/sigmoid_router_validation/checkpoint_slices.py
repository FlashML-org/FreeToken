"""Pinned safetensors byte-range reader for bounded prefix experiments."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request

import torch

MODEL = 'zai-org/GLM-4.5-Air'
REVISION = 'a24ceef6ce4f3536971efe9b778bdaa1bab18daa'
DIRECTORY = Path(os.environ.get('FREETOKEN_VALIDATION_CACHE',
                               str(Path.home()/'.cache/freetoken-sigmoid-prefix')))


class SliceReader:
    def __init__(self):
        self.root = DIRECTORY
        self.root.mkdir(parents=True, exist_ok=True)
        self.base = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/'
        self.config = self.json_file('config.json')
        self.index = self.json_file('model.safetensors.index.json')['weight_map']
        self.headers = {}
        self.manifest = {}

    def json_file(self, name):
        path = self.root / name
        if not path.exists():
            with urllib.request.urlopen(self.base+name, timeout=60) as response:
                raw = response.read()
            path.write_bytes(raw)
        return json.loads(path.read_bytes())

    def byte_range(self, shard, start, end):
        for attempt in range(3):
            try:
                request = urllib.request.Request(self.base+shard+f'?range={start}-{end}',
                    headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                        raise RuntimeError('Exact HTTP range not honored; refusing whole shard')
                    data = response.read(end-start+2)
                assert len(data) == end-start+1
                return data
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(1)

    def describe(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            path = self.root/(shard+'.header.json')
            if path.exists():
                raw = path.read_bytes()
            else:
                length = struct.unpack('<Q', self.byte_range(shard,0,7))[0]
                assert length < 64*1024*1024
                raw = self.byte_range(shard,8,7+length)
                path.write_bytes(raw)
            self.headers[shard] = (json.loads(raw),8+len(raw))
        header, offset = self.headers[shard]
        return shard, header[key], offset

    def tensor(self, key):
        shard, desc, base = self.describe(key)
        path = self.root/(key+'.bin')
        start, end = desc['data_offsets']
        if not path.exists():
            print(f'Downloading {key}: {(end-start)/2**20:.1f} MiB', flush=True)
            temp = path.with_suffix('.part')
            with temp.open('wb') as out:
                for pos in range(start,end,8*1024*1024):
                    out.write(self.byte_range(shard,base+pos,base+min(pos+8*1024*1024,end)-1))
            temp.replace(path)
        raw = path.read_bytes()
        assert len(raw) == end-start
        self.manifest[key] = dict(shard=shard, shape=desc['shape'], dtype=desc['dtype'],
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        dtype = {'BF16':torch.bfloat16, 'F32':torch.float32, 'F16':torch.float16}[desc['dtype']]
        return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(desc['shape']).clone()

    def embedding_rows(self, ids):
        from concurrent.futures import ThreadPoolExecutor
        key = 'model.embed_tokens.weight'
        shard, desc, base = self.describe(key)
        assert desc['dtype'] == 'BF16'
        row_bytes = desc['shape'][1]*2
        data_start = base+desc['data_offsets'][0]
        def fetch(token):
            path = self.root/f'embedding-row-{token}.bin'
            if not path.exists():
                pos = data_start+token*row_bytes
                path.write_bytes(self.byte_range(shard,pos,pos+row_bytes-1))
            raw = path.read_bytes()
            assert len(raw) == row_bytes
            return token, raw
        unique = sorted(set(ids))
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = dict(pool.map(fetch, unique))
        for token, raw in rows.items():
            self.manifest[f'{key}[{token}]'] = dict(shard=shard, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())
        return torch.stack([torch.frombuffer(bytearray(rows[token]),dtype=torch.bfloat16).clone() for token in ids])

    def save_manifest(self):
        (self.root/'tensor-manifest.json').write_text(json.dumps(dict(model=MODEL,
            revision=REVISION,tensors=self.manifest),indent=2),encoding='utf-8')
