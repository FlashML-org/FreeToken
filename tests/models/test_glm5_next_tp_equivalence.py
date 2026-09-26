"""GLM-5.3-Flash under real TP: the sharded model must compute what the TP=1 model computes.

The shape test (test_glm5_next_tp.py) pins how tensors are cut; this runs the cut model. Each rank
loads the same tiny checkpoint through the TP-aware reader onto its own GPU and runs a hybrid
KDA + DSA prefill and a decode with real NCCL collectives; rank 0's logits must match a TP=1 run of
the same weights. A wrong head order, a missed all-reduce or a double-counted residual moves the
logits far outside bf16 round-off.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

REQUIRED_GPUS = 2
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(torch.cuda.device_count() < REQUIRED_GPUS, reason=f"needs {REQUIRED_GPUS} GPUs"),
]

_HERE = os.path.dirname(os.path.abspath(__file__))


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_rank(rank: int, tp: int, ckpt: str, init_file: str, out: str) -> None:
    from types import SimpleNamespace

    import torch.distributed as dist

    import freetoken.attention.dsa as dsa_mod
    import freetoken.layers.embedding as emb_mod
    import freetoken.models.glm5_next.attention as attn_mod
    import freetoken.models.glm5_next.kda as kda_mod
    import freetoken.models.glm5_next.model as model_mod
    from freetoken.attention.dsa_indexer_kpool import Glm5NextDSABackend
    from freetoken.distributed import set_tp_info
    from freetoken.kvcache.dsa_pool import KpoolDSAKVCache
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.glm5_next.weight import iter_weights

    rig = _sibling("test_glm5_next_model")
    shapes = _sibling("test_glm5_next_tp")

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=tp)
    set_tp_info(rank=rank, size=tp)
    dev = torch.device("cuda", rank)
    config = shapes._config()

    prev = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(dev)
    try:
        model = model_mod.Glm5NextForCausalLM(config)
    finally:
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])
    weights = dict(iter_weights(ckpt, dev, include_moe_experts=False, include_non_moe=True, include_vision=False))
    state = model.state_dict()
    model.load_state_dict({k: v.to(dev, state[k].dtype) for k, v in weights.items()})

    kv = KpoolDSAKVCache(
        latent_dim=shapes.KVLORA, num_layers=2, num_pages=4, page_size=64, dtype=torch.bfloat16, device=dev,
        index_head_dim=shapes.IDX_D, num_index_layers=1, index_ratio=shapes.KPOOL, num_req_slots=4,
    )
    page_table = torch.full((2, 256), -1, dtype=torch.int32, device=dev)
    page_table[0] = torch.arange(256, dtype=torch.int32, device=dev)
    linear_pool = LinearStatePool(config.linear_attention_group(), num_slots=4, dtype=torch.bfloat16, device=dev, tp_size=tp)
    ctx = SimpleNamespace(kv_cache=kv, page_table=page_table, linear_state_pool=linear_pool, attn_backend=None, batch=None)
    for mod in (dsa_mod, kda_mod, attn_mod, model_mod, emb_mod):
        mod.get_global_ctx = lambda: ctx
    ctx.attn_backend = Glm5NextDSABackend(config)

    rig.DEV, rig.VOCAB = dev, shapes.VOCAB
    torch.manual_seed(7)
    ids = torch.randint(0, shapes.VOCAB, (24,)).tolist()
    rig._batch(ctx, ids[:-1], 0, "prefill")
    prefill = model.forward().float().cpu()
    rig._batch(ctx, ids[-1:], len(ids) - 1, "decode")
    decode = model.forward().float().cpu()
    if rank == 0:
        torch.save({"prefill": prefill, "decode": decode}, out)
    dist.barrier()
    dist.destroy_process_group()


def _run(tp: int, ckpt: str, tmp: str) -> dict[str, torch.Tensor]:
    import torch.multiprocessing as mp

    out = os.path.join(tmp, f"logits-tp{tp}.pt")
    init_file = os.path.join(tmp, f"init-tp{tp}")
    mp.start_processes(_run_rank, args=(tp, ckpt, init_file, out), nprocs=tp, start_method="spawn", join=True)
    return torch.load(out)


@pytest.mark.parametrize("tp", [t for t in (2, 4) if t <= torch.cuda.device_count()])
def test_sharded_logits_match_single_rank(tmp_path, tp):
    shapes = _sibling("test_glm5_next_tp")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    shapes._write_checkpoint(str(ckpt))

    ref = _run(1, str(ckpt), str(tmp_path))
    got = _run(tp, str(ckpt), str(tmp_path))
    for phase in ("prefill", "decode"):
        err = (got[phase] - ref[phase]).abs().max().item()
        scale = ref[phase].abs().max().item() + 1e-8
        assert err / scale < 3e-2, f"TP={tp} {phase} logits diverge from TP=1: {err} (scale {scale})"
        assert torch.equal(got[phase].argmax(-1), ref[phase].argmax(-1)), f"TP={tp} {phase} greedy token differs"
