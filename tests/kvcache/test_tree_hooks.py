"""Degrau 2 — teste standalone dos hooks do HybridRadixCache (CPU, sem servidor)."""
import torch
from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache


def main():
    tree = HybridRadixCache(torch.device("cpu"), 64)
    evicted = []       # (prefix_len, mamba_slot) capturados pelo hook de evicção
    rehydrated = []    # prefix_len reidratado no match
    host_index = set()

    tree.on_evict_node = lambda node: evicted.append(
        (len(tree._collect_key(node)), node.mamba_value))
    next_slot = [100]

    def fake_rehydrate(node):
        n = len(tree._collect_key(node))
        if n in host_index:
            rehydrated.append(n)
            s = next_slot[0]
            next_slot[0] += 1
            return s
        return None

    tree.on_match_dead_snapshot = fake_rehydrate

    # A (4096 toks, snapshot 7) + filho B (mais 2048, snapshot 8) -> A é nó INTERNO
    idsA = torch.arange(4096, dtype=torch.int32)
    kvA = torch.arange(4096, dtype=torch.int32) * 10
    assert tree.insert(idsA, kvA, mamba_value=7) == (0, False)
    idsB = torch.cat([idsA, torch.full((2048,), 555, dtype=torch.int32)])
    kvB = torch.arange(6144, dtype=torch.int32) * 10
    assert tree.insert(idsB, kvB, mamba_value=8) == (4096, False)

    # match enquanto tudo vivo: ganha o snapshot mais profundo (B, 6144)
    m = tree.match_prefix(idsB)
    assert m.cached_len == 6144 and m.mamba_value == 8

    # evicta os 2 snapshots: A (interno -> tombstone) e B (folha -> com tier, tombstone
    # in-place: o KV da folha FICA no tree, não é liberado)
    er = tree.evict_mamba(2)
    assert sorted(er.mamba_slots) == [7, 8], er
    assert sorted(evicted) == [(4096, 7), (6144, 8)], evicted
    assert er.kv_indices.numel() == 0, "com o tier, nenhuma página KV é liberada"
    print("PASS 1: on_evict_node dispara; folha vira tombstone e mantém o KV:", evicted)

    # sem cópia host: match trunca pra 0 (A e B viraram tombstones)
    m = tree.match_prefix(idsA)
    assert m.cached_len == 0 and m.mamba_value is None
    print("PASS 2: match trunca sem tier host")

    # host tem A: match(idsA) reidrata A
    host_index.add(4096)
    m = tree.match_prefix(idsA)
    assert m.cached_len == 4096 and m.mamba_value == 100, (m.cached_len, m.mamba_value)
    assert rehydrated == [4096]
    print("PASS 3: match reidrata tombstone interno via tier host (slot 100)")

    # host tem B também: match(idsB) reidrata a FOLHA tombstone (mais profunda)
    host_index.add(6144)
    m = tree.match_prefix(idsB)
    assert m.cached_len == 6144 and m.mamba_value == 101, (m.cached_len, m.mamba_value)
    print("PASS 4: match reidrata folha-tombstone (KV preservado) -> 6144")

    # nós reidratados ficam vivos: segundo match não chama o hook
    m2 = tree.match_prefix(idsB)
    assert m2.mamba_value == 101 and rehydrated == [4096, 6144]
    print("PASS 5: nós reidratados ficam vivos (sem H2D repetido)")

    # integridade de contagem: os dois snapshots revividos voltam a contar como evictable
    assert tree.mamba_evictable == 2, tree.mamba_evictable
    print("PASS 6: contagem mamba_evictable consistente")

    print("\nTODOS OS TESTES PASSARAM ✓")


main()
