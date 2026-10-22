"""Cached-checkpoint tie audit and full-router 2x2 ablation, not model inference."""
import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

REPO = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location('sigmoid_audit_kernel', REPO/'python/freetoken/kernel/triton/sigmoid_router.py')
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)


def tensor(cache, key, dtype, shape):
    return torch.frombuffer(bytearray((cache/(key+'.bin')).read_bytes()), dtype=dtype).reshape(shape).clone().cuda()


def eager(logits, bias):
    scores = logits.sigmoid()
    ids = (scores + bias.float()).topk(8, dim=-1).indices
    weights = scores.gather(-1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    # The original GLM route launches this multiply even when its scale is one.
    weights = weights * 1.0
    return weights.float().contiguous(), ids.to(torch.int32).contiguous()


def fused(logits, bias):
    return kernel.fused_topk_sigmoid(logits, bias, 8, renormalize=True)


def profile(fn):
    try:
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        events = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        counts = Counter(e.name for e in events)
        return {'status': 'captured' if events else 'CUDA events unavailable',
                    'scope': 'one eager invocation, GPU events including any copies',
                    'cuda_event_count': len(events) if events else None, 'names': dict(counts)}
    except (RuntimeError, ImportError, AttributeError) as exc:
        return {'status': 'unavailable', 'error': str(exc)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--router-cache', type=Path, required=True)
    parser.add_argument('--prefix-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261020)
    mismatches = []
    cases = []
    for layer in (1, 23, 45):
        key = f'model.layers.{layer}.mlp.gate'
        weight = tensor(args.router_cache, key+'.weight', torch.bfloat16, (128,4096))
        bias = tensor(args.router_cache, key+'.e_score_correction_bias', torch.float32, (128,))
        for n in (1,8,256):
            for std in (0.1,1.0,3.0):
                x = (torch.randn(n,4096,device='cuda')*std).bfloat16()
                logits = F.linear(x.float(),weight.float())
                _old_w, old_ids = eager(logits,bias)
                _w, ids = fused(logits,bias)
                same = (ids.sort(-1).values == old_ids.sort(-1).values).all(-1)
                rows = (ids != old_ids).any(-1).nonzero().flatten().tolist()
                cases.append({'layer': layer,'tokens': n,'std': std,'order_mismatch': len(rows),
                                  'set_mismatch': int((~same).sum())})
                choice = logits.sigmoid()+bias
                for row in rows:
                    previous = old_ids[row].long()
                    current = ids[row].long()
                    changed = previous != current
                    boundary = choice[row].topk(9).values
                    mismatches.append({'layer': layer,'tokens': n,'std': std,'row': row,
                        'old_ids': previous.tolist(),'new_ids': current.tolist(),
                        'old_selection_scores': choice[row,previous].tolist(),
                        'new_selection_scores': choice[row,current].tolist(),
                        'changed_slots_equal_eager_scores': bool(torch.equal(choice[row,previous[changed]],choice[row,current[changed]])),
                        'top8_top9_gap': float(boundary[7]-boundary[8]),
                        'same_expert_set': bool(same[row])})
    (args.output/'tie-audit.json').write_text(json.dumps({'seed': 20261020,'cases': cases,
        'mismatches': mismatches,'scope': 'replay of real gate weights with synthetic hidden states'},indent=2))
    print('Tie audit:',len(mismatches),'order differences',flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(args.prefix_cache/'prompt-1-activations.pt',weights_only=True)
    weight = tensor(args.router_cache,'model.layers.1.mlp.gate.weight',torch.bfloat16,(128,4096))
    resident = weight.float()
    bias = tensor(args.router_cache,'model.layers.1.mlp.gate.e_score_correction_bias',torch.float32,(128,))
    results = []
    for n in (1,93):
        x = saved['hidden_states'][-n:].cuda()
        def run(cache, fusion, x=x):
            logits = F.linear(x.float(),resident if cache else weight.float())
            return (fused if fusion else eager)(logits,bias)
        variants = {'baseline': partial(run,False,False),'fusion_only': partial(run,False,True),
                        'fp32_only': partial(run,True,False),'both': partial(run,True,True)}
        expected = variants['baseline']()
        for fn in variants.values():
            actual = fn()
            torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)
            torch.testing.assert_close(actual[0],expected[0],rtol=1e-5,atol=1e-6)
        samples = {name:[] for name in variants}
        for repeat in range(5):
            names = list(variants)
            names = names[repeat % 4:]+names[:repeat % 4]
            for name in names:
                samples[name].append(do_bench_cudagraph(variants[name],rep=50)*1000)
        for name,fn in variants.items():
            values = samples[name]
            row = {'tokens': n,'variant': name,'samples_us': values,'median_us': statistics.median(values),
                       'min_us': min(values),'max_us': max(values),'sample_stdev_us': statistics.stdev(values),
                       'eager_profile': profile(fn)}
            results.append(row)
            print(n,name,row['median_us'],flush=True)
    report = {'scope': 'full router (input cast + gate GEMM + routing), no experts or model generation',
        'gpu': torch.cuda.get_device_name(),'torch': torch.__version__,'tf32': False,'repeats': 5,
        'extra_resident_weight_bytes': resident.numel()*resident.element_size()-weight.numel()*weight.element_size(),
        'results': results}
    (args.output/'ablation.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
