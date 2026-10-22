"""Inspect immutable GGUF headers; stubs are never inference weights."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from collections import Counter

from huggingface_hub import HfApi, hf_hub_url

repo_root = Path('/home/freetoken/freetoken-work/FreeToken')
out = Path('/mnt/d/AI/FreeToken-validation/full-model')
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('fetch_headers', repo_root/'scripts/gguf_check/fetch_headers.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)
api = HfApi()
reports = []
for repo, suffix in [
    ('unsloth/GLM-4.5-Air-GGUF', 'UD-IQ1_S.gguf'),
    ('bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF', 'IQ1_M.gguf'),
]:
    info = api.model_info(repo, files_metadata=True)
    chosen = next(f for f in info.siblings if f.rfilename.endswith(suffix))
    fetcher.hf_hub_url = lambda r, p, rev=info.sha: hf_hub_url(r, p, revision=rev)
    record = fetcher.fetch(repo, chosen.rfilename, chosen.size, '/home/freetoken/freetoken-work/header-stubs')
    record['revision'] = info.sha
    record['lfs'] = chosen.lfs.__dict__ if chosen.lfs else None
    if record['ok']:
        import gguf
        reader = gguf.GGUFReader(record['stub'])
        experts = [t for t in reader.tensors if '_exps.' in t.name]
        record['expert_bytes'] = sum(t.n_bytes for t in experts)
        record['expert_types'] = dict(Counter(t.tensor_type.name for t in experts))
        record['expert_layer_type_pairs'] = sorted(set(
            (next(t.tensor_type.name for t in experts if t.name == f'blk.{layer}.ffn_gate_exps.weight'),
             next(t.tensor_type.name for t in experts if t.name == f'blk.{layer}.ffn_down_exps.weight'))
            for layer in sorted({int(t.name.split('.')[1]) for t in experts})
        ))
        record['scope'] = 'header metadata only; sparse zero-filled stub, NOT a full checkpoint'
    reports.append(record)
    (out/'candidates.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(record), flush=True)
print('linux_memtotal_bytes', os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
