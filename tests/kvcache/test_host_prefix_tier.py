"""Degrau 1 — teste standalone do HostPrefixTier (roda sem servidor; GPU tiny)."""
import torch
from types import SimpleNamespace

from freetoken.kvcache.host_prefix_tier import HostPrefixTier

def main():
    import os
    dev = "cuda" if (os.environ.get("TIER_TEST_CPU") is None and torch.cuda.is_available()) else "cpu"
    # pool fake: mesmas geometrias, mini
    L, S = 4, 3   # 4 "layers", 3 slots GPU
    pool = SimpleNamespace(
        conv_states=torch.randn(L, S, 8, 3, dtype=torch.bfloat16, device=dev),
        recurrent_states=torch.randn(L, S, 2, 4, 4, dtype=torch.float32, device=dev),
        slot_states={"extra": torch.randn(L, S, 5, dtype=torch.bfloat16, device=dev)},
    )
    tier = HostPrefixTier(pool, gdn_slots_host=4, device=dev)

    # 1) round-trip D2H->H2D preserva bytes
    original = [t.clone() for t in (pool.conv_states, pool.recurrent_states, pool.slot_states["extra"])]
    hs = tier.write_gdn(1)                    # snapshot do slot 1 -> host
    # suja o slot GPU pra provar que o restore vem do host
    for t in (pool.conv_states, pool.recurrent_states, pool.slot_states["extra"]):
        t[:, 1].zero_()
    tier.read_gdn(hs, 2)                      # host -> slot 2
    ok = all(torch.equal(t[:, 1].cpu(), u[:, 2].cpu())
             for t, u in zip(original, (pool.conv_states, pool.recurrent_states, pool.slot_states["extra"])))
    assert ok, "snapshot divergiu no round-trip!"
    print("PASS 1: snapshot GDN round-trip D2H->H2D bit-exato")

    # 2) índice: put + lookup longest-prefix page-aligned
    toks = torch.arange(64 * 5, dtype=torch.int64)      # 5 páginas
    pages = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    tier.put_prefix(toks[: 64 * 4], pages[:4], hs)       # registro 4 páginas
    m = tier.lookup(torch.arange(64 * 5, dtype=torch.int64))  # query 5 páginas
    assert m is not None and m.host_len == 64 * 4 and m.gdn_slot == hs
    assert torch.equal(m.kv_pages, pages[:4])
    print("PASS 2: lookup acha o maior prefixo page-aligned")

    # 3) miss quando nada bate
    assert tier.lookup(torch.full((64 * 3,), 999999, dtype=torch.int64)) is None
    print("PASS 3: lookup retorna None em miss total")

    # 4) evict_host libera o slot GDN host
    tier.evict_host(toks[: 64 * 4])
    assert hs in tier._free
    print("PASS 4: evict_host libera slot host")

    # 5) proteção: tier novo com 2 slots, 3ª escrita levanta erro claro
    tier2 = HostPrefixTier(pool, gdn_slots_host=2, device=dev)
    tier2.write_gdn(0); tier2.write_gdn(0)
    try:
        tier2.write_gdn(0)
        raise SystemExit("ERA PRA TER FALHADO")
    except RuntimeError:
        print("PASS 5: capacidade cheia levanta RuntimeError claro")

    print("\nTODOS OS TESTES PASSARAM ✓")

main()
