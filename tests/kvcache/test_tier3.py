"""Degrau 3 — teste standalone do KV store host do HostPrefixTier (CPU).

Cobre: round-trip D2H->H2D de páginas KV (códigos + scales), a serpente dos ids lógicos
(a página lógica reescrita depois do evict NÃO contamina a cópia host), merge de entrada
(gdn depois kv), e drop-oldest quando o budget enche.
"""
import torch
from types import SimpleNamespace

from freetoken.kvcache.host_prefix_tier import HostPrefixTier


def main():
    # pool fake: [2, L=2, P=8, page=4, H=2, D=3] códigos bf16 + scales [2, L, P*4, 2] fp32
    L, P, PAGE, H, D = 2, 8, 4, 2, 3
    kv_pool = SimpleNamespace(
        _kv_buffer=torch.randn(2, L, P, PAGE, H, D, dtype=torch.bfloat16),
        _scale_buffer=torch.randn(2, L, P * PAGE, H, dtype=torch.float32),
    )
    state_pool = SimpleNamespace(
        conv_states=torch.randn(L, 3, 8, 3, dtype=torch.bfloat16),
        recurrent_states=torch.randn(L, 3, 2, 4, 4, dtype=torch.float32),
        slot_states={},
    )
    page_bytes = 2 * L * PAGE * H * D * 2 + 2 * L * PAGE * H * 4
    # budget pra 3 páginas (força drop-oldest cedo)
    tier = HostPrefixTier(state_pool, kv_pool, gdn_slots_host=2,
                          kv_budget_bytes=3 * page_bytes, device="cpu")
    assert tier.has_kv and len(tier._kv_free) == 3

    # 1) write/read round-trip: prefixo de 2 páginas (páginas lógicas 3 e 5)
    kv_idx = torch.cat([torch.arange(3 * PAGE, 4 * PAGE), torch.arange(5 * PAGE, 6 * PAGE)]).to(torch.int32)
    orig_codes = kv_pool._kv_buffer[:, :, 3].clone(), kv_pool._kv_buffer[:, :, 5].clone()
    orig_scales = (kv_pool._scale_buffer[:, :, 3 * PAGE:4 * PAGE].clone(),
                   kv_pool._scale_buffer[:, :, 5 * PAGE:6 * PAGE].clone())
    hs = tier.write_kv(kv_idx)
    assert hs is not None and hs.numel() == 2

    # 2) A SERPENTE: sobrescreve as páginas lógicas 3 e 5 (novo inquilino)
    kv_pool._kv_buffer[:, :, 3].fill_(99)
    kv_pool._kv_buffer[:, :, 5].fill_(99)
    kv_pool._scale_buffer[:, :, 3 * PAGE:6 * PAGE].fill_(99)
    # lê de volta pra páginas 0 e 7
    tier.read_kv(hs, torch.tensor([0, 7], dtype=torch.int32))
    assert torch.equal(kv_pool._kv_buffer[:, :, 0], orig_codes[0])
    assert torch.equal(kv_pool._kv_buffer[:, :, 7], orig_codes[1])
    assert torch.equal(kv_pool._scale_buffer[:, :, 0 * PAGE:1 * PAGE], orig_scales[0])
    assert torch.equal(kv_pool._scale_buffer[:, :, 7 * PAGE:8 * PAGE], orig_scales[1])
    print("PASS 1: round-trip KV host + ids lógicos reescritos não contaminam")

    # 3) merge: entrada com gdn primeiro, kv depois
    toks = torch.arange(64 * 2, dtype=torch.int64)
    g = tier.write_gdn(1)
    tier.put_prefix(toks, None, g)
    tier.put_prefix(toks, hs, None)  # merge: adota kv, mantém gdn
    m = tier.lookup(toks)
    assert m is not None and m.gdn_slot == g and torch.equal(m.kv_pages, hs)
    print("PASS 2: merge gdn+kv na mesma entrada")

    # 4) drop-oldest quando o budget enche: escreve 2 páginas (resta 1), depois 2 de novo
    #    -> a entrada mais velha (toks) é dropada INTEIRA (KV slots + slot GDN)
    kv2 = torch.cat([torch.arange(0 * PAGE, 1 * PAGE), torch.arange(2 * PAGE, 3 * PAGE)]).to(torch.int32)
    toks2 = torch.arange(64 * 2, 64 * 3, dtype=torch.int64)
    s2 = tier.write_kv(kv2)   # precisa de 2 slots, tem 1 -> drop de toks (inteiro)
    tier.put_prefix(toks2, s2, None)
    assert tier.lookup(toks) is None, "o mais velho devia ter sido dropado inteiro"
    assert g in tier._free, "o slot GDN do dropado devia voltar pro free-list"
    kv3 = torch.arange(1 * PAGE, 3 * PAGE).to(torch.int32)
    toks3 = torch.arange(64 * 3, 64 * 4, dtype=torch.int64)
    s3 = tier.write_kv(kv3)   # precisa de 2, tem 1 -> drop de toks2
    tier.put_prefix(toks3, s3, None)
    assert tier.lookup(toks2) is None
    m3 = tier.lookup(toks3)
    assert m3 is not None and torch.equal(m3.kv_pages, s3)
    print("PASS 3: drop-oldest remove a entrada mais fria inteira (KV + GDN)")

    # 5) evict_host libera tudo
    tier.evict_host(toks3)
    assert tier.lookup(toks3) is None and len(tier._kv_free) == 3
    print("PASS 4: evict_host remove e libera")

    print("\nTODOS OS TESTES PASSARAM ✓")


main()
