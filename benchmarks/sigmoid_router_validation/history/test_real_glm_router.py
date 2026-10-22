"""Real checkpoint router weights with synthetic hidden states, not model inference."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'FreeToken/python'))
from freetoken.layers.linear import Fp32RouterLinear
from freetoken.models.glm4_moe.moe import Glm4MoeSparseBlock
from types import SimpleNamespace

MODEL = 'zai-org/GLM-4.5-Air'
REV = 'a24ceef6ce4f3536971efe9b778bdaa1bab18daa'
OUT = Path('D:/AI/FreeToken-validation/GLM-4.5-Air-router')
OUT.mkdir(parents=True, exist_ok=True)
BASE = f'https://huggingface.co/{MODEL}/resolve/{REV}/'


def get_json(filename):
    with urllib.request.urlopen(BASE+filename, timeout=60) as response:
        raw = response.read()
    (OUT/filename).write_bytes(raw)
    return json.loads(raw)


def get_range(filename, start, end):
    request = urllib.request.Request(BASE+filename+f'?range={start}-{end}',
                                    headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('Server did not honor exact byte range; refusing full-shard download')
        data = response.read(end-start+2)
    assert len(data) == end-start+1
    return data


config = get_json('config.json')
weight_map = get_json('model.safetensors.index.json')['weight_map']
headers = {}
provenance = []


def load_tensor(key):
    shard = weight_map[key]
    if shard not in headers:
        size = struct.unpack('<Q', get_range(shard, 0, 7))[0]
        assert size < 64*1024*1024
        raw = get_range(shard, 8, 7+size)
        (OUT/(shard+'.header.json')).write_bytes(raw)
        headers[shard] = (json.loads(raw), 8+size)
    header, offset = headers[shard]
    entry = header[key]
    a, b = entry['data_offsets']
    path = OUT/(key+'.bin')
    raw = get_range(shard, offset+a, offset+b-1)
    path.write_bytes(raw)
    dtype = {'BF16':torch.bfloat16, 'F32':torch.float32, 'F16':torch.float16}[entry['dtype']]
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry['shape']).clone()
    provenance.append(dict(key=key, shard=shard, dtype=entry['dtype'], shape=entry['shape'],
                           byte_range=[offset+a, offset+b-1], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    print(f'Loaded {key}: {tuple(tensor.shape)}, {tensor.dtype}', flush=True)
    return tensor


torch.manual_seed(20261020)
layers = [1, config['num_hidden_layers']//2, config['num_hidden_layers']-1]
results = []
for layer in layers:
    prefix = f'model.layers.{layer}.mlp.gate'
    original_weight = load_tensor(prefix+'.weight').to(device='cuda', dtype=torch.bfloat16)
    bias = load_tensor(prefix+'.e_score_correction_bias').to(device='cuda', dtype=torch.float32)
    with torch.device('cuda'):
        gate = Fp32RouterLinear(config['hidden_size'], config['n_routed_experts'], has_bias=False)
    gate.weight = torch.empty_like(original_weight)
    gate.load_state_dict({'weight':original_weight})
    torch.testing.assert_close(gate.weight, original_weight.float(), rtol=0, atol=0)
    block = SimpleNamespace(gate=gate, e_score_correction_bias=bias,
                            top_k=config['num_experts_per_tok'], norm_topk_prob=config['norm_topk_prob'],
                            n_group=config['n_group'], topk_group=config['topk_group'],
                            routed_scaling_factor=config['routed_scaling_factor'])
    assert block.n_group == 1
    for tokens in (1, 8, 256):
        for std in (0.1, 1.0, 3.0):
            x = (torch.randn(tokens, config['hidden_size'], device='cuda') * std).bfloat16()
            logits = torch.nn.functional.linear(x.float(), original_weight.float())
            scores = logits.sigmoid()
            choice = scores + bias
            ref_ids = choice.topk(block.top_k, dim=-1).indices
            ref_w = scores.gather(1, ref_ids)
            if block.norm_topk_prob:
                ref_w = ref_w/(ref_w.sum(-1,keepdim=True)+1e-20)
            ref_w *= block.routed_scaling_factor
            w, ids = Glm4MoeSparseBlock._route(block, x)
            same = (ids.sort(-1).values == ref_ids.sort(-1).values).all(-1)
            aw = w.gather(1, ids.argsort(-1))
            rw = ref_w.gather(1, ref_ids.argsort(-1))
            result = dict(layer=layer, tokens=tokens, hidden_std=std,
                          expert_set_mismatch_rows=int((~same).sum()),
                          order_mismatch_rows=int((ids != ref_ids).any(-1).sum()),
                          max_abs_weight_error_matching_sets=float((aw[same]-rw[same]).abs().max()) if same.any() else None,
                          fp32_gate_logits_bitwise_equal=bool(torch.equal(gate.forward(x),logits)))
            results.append(result)
            print(json.dumps(result), flush=True)
report = dict(model=MODEL, revision=REV, layers=layers, checkpoint_tensors=provenance,
              input='synthetic hidden states; not real prompt activations',
              scope='actual GLM _route and actual router weights; not full model or language quality',
              gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=results)
(OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(ROOT/'validation/real-glm-router-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'Saved results to {OUT}', flush=True)
