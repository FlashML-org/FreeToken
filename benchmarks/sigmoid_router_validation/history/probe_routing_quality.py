"""Synthetic sensitivity probe, NOT a full-model quality evaluation."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'FreeToken/python/freetoken/kernel/triton/sigmoid_router.py'
spec = importlib.util.spec_from_file_location('quality_probe_kernel', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

torch.manual_seed(20261020)
rows, experts, k = 1024, 128, 4
cases = [
    ('normal_random', torch.randn(rows, experts, device='cuda'), torch.randn(experts, device='cuda') * 0.1),
    ('near_ties', torch.randn(rows, experts, device='cuda') * 1e-6, torch.zeros(experts, device='cuda')),
    ('sigmoid_saturation', 20 + torch.randn(rows, experts, device='cuda'), torch.zeros(experts, device='cuda')),
    ('exact_ties', torch.zeros(rows, experts, device='cuda'), torch.zeros(experts, device='cuda')),
]
results = []
for name, x, bias in cases:
    scores = x.float().sigmoid()
    ranked = scores + bias.float()
    ref_ids = ranked.topk(k, dim=-1).indices
    ref_w = scores.gather(1, ref_ids)
    ref_w /= ref_w.sum(-1, keepdim=True) + 1e-20
    weights, ids = module.fused_topk_sigmoid(x, bias, k, renormalize=True)
    same = (ids.sort(-1).values == ref_ids.sort(-1).values).all(-1)
    aligned = weights.gather(1, ids.argsort(-1))
    ref_aligned = ref_w.gather(1, ref_ids.argsort(-1))
    cutoff = ranked.topk(k + 1, dim=-1).values
    tied = cutoff[:, k-1] == cutoff[:, k]
    selected_ref_scores = ranked.gather(1, ids.long())
    below_cutoff = (selected_ref_scores < cutoff[:, k-1:k]).any(-1)
    result = dict(case=name, rows=rows, expert_set_mismatch_rows=int((~same).sum()),
                  order_mismatch_rows=int((ids != ref_ids).any(-1).sum()),
                  reference_cutoff_tie_rows=int(tied.sum()),
                  candidate_below_reference_cutoff_rows=int(below_cutoff.sum()),
                  max_abs_weight_error_matching_sets=float((aligned[same]-ref_aligned[same]).abs().max()) if same.any() else None,
                  reference_first_ids=ref_ids[0].tolist(), candidate_first_ids=ids[0].tolist())
    results.append(result)

# Existing tie tests check the new deterministic policy, not old/new equivalence.
reference = runpy.run_path(str(ROOT/'FreeToken/tests/kernels/test_sigmoid_router.py'))['reference']
x = torch.zeros(1024, 192, device='cuda')
b = torch.zeros(192, device='cuda')
kwargs = dict(renormalize=True, n_group=8, topk_group=4)
wr, ir = reference(x, b, 8, **kwargs)
w, ids = module.fused_topk_sigmoid(x, b, 8, **kwargs)
same = (ids.sort(-1).values == ir.sort(-1).values).all(-1)
results.append(dict(case='grouped_exact_ties', rows=1024, groups=8, keep_groups=4,
                    expert_set_mismatch_rows=int((~same).sum()),
                    reference_first_ids=ir[0].tolist(), candidate_first_ids=ids[0].tolist(),
                    note='All biased scores equal; both satisfy top-k, but expert identities may differ.'))
report = dict(seed=20261020, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
              purpose='synthetic sensitivity; not model quality or real-data mismatch frequency', results=results)
output=ROOT/'validation/quality-probe-results.json'
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
