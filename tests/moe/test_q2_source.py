"""Exercise a real miniature GSQ Q2_0 GGUF, including expert order and corruption."""
import hashlib
import struct
from types import SimpleNamespace

import pytest

from freetoken.models.qwen4_exp.q2_source import iter_q2_expert_pieces, read_q2_directory


def _string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def _source(path, *, kind=42, overlap=False):
    """Write a real GGUF with distinct role/expert bytes as the expected payload."""
    metadata = {'general.architecture': 'qwen4exp', 'qwen4exp.block_count': 2,
                'qwen4exp.expert_count': 3, 'qwen4exp.embedding_length': 64,
                'qwen4exp.expert_feed_forward_length': 128}
    header = struct.pack('<IIQQ', 0x46554747, 3, 6, len(metadata))
    for name, value in metadata.items():
        header += _string(name)
        header += struct.pack('<I', 8) + _string(value) if isinstance(value, str) else struct.pack('<II', 4, value)
    payload, expected = bytearray(), {}
    for layer in range(2):
        for role_index, role in enumerate(('gate', 'up', 'down')):
            payload.extend(b'\0' * (-len(payload) % 32))
            offset = 0 if overlap else len(payload)
            shape = (128, 64, 3) if role == 'down' else (64, 128, 3)
            header += (_string(f'blk.{layer}.ffn_{role}_exps.weight') + struct.pack('<I', 3)
                       + struct.pack('<QQQIQ', *shape, kind, offset))
            for expert in range(3):
                data = bytes([1 + layer * 20 + role_index * 4 + expert]) * 2304
                payload.extend(data)
                expected[layer, role, expert] = data
    header += b'\0' * (-len(header) % 32)
    path.write_bytes(header + payload)
    source = {'format': 'q2_0', 'scope': 'routed_experts_only', 'path': path.name,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': path.stat().st_size,
              'verified_mtime_ns': path.stat().st_mtime_ns, 'layers': 2, 'experts_per_layer': 3,
              'hidden': 64, 'intermediate': 128, 'expert_bytes': 41472}
    config = SimpleNamespace(expert_source=source, num_layers=2, num_experts=3,
                             hidden_size=64, moe_intermediate_size=128)
    return config, expected


def test_all_experts_stream_with_model_geometry_and_alternate_identity(tmp_path):
    path = tmp_path / 'small.gguf'
    config, expected = _source(path)
    batches = list(iter_q2_expert_pieces(tmp_path, config, chunk=15000))
    assert [(layer, start, end) for layer, start, end, _ in batches] == [(0, 0, 2), (0, 2, 3), (1, 0, 2), (1, 2, 3)]
    for layer, start, end, pieces in batches:
        for role, tensor in pieces.items():
            for index, expert in enumerate(range(start, end)):
                assert tensor[index].numpy().tobytes() == expected[layer, role, expert]


@pytest.mark.parametrize('change,match', [('kind', 'incompatible'), ('overlap', 'overlapping'),
                                        ('truncated', 'out-of-bounds'), ('geometry', 'geometry mismatch')])
def test_invalid_directory_is_rejected(tmp_path, change, match):
    path = tmp_path / 'small.gguf'
    _source(path, kind=10 if change == 'kind' else 42, overlap=change == 'overlap')
    if change == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match=match):
        read_q2_directory(path, layers=2, experts=4 if change == 'geometry' else 3, hidden=64, intermediate=128)


def test_source_change_requires_reverification(tmp_path):
    path = tmp_path / 'small.gguf'
    config, _ = _source(path)
    config.expert_source['verified_mtime_ns'] -= 1
    with pytest.raises(ValueError, match='identity changed'):
        list(iter_q2_expert_pieces(tmp_path, config))


def test_receipt_geometry_cannot_override_model(tmp_path):
    config, _ = _source(tmp_path / 'small.gguf')
    config.expert_source['experts_per_layer'] = 2
    with pytest.raises(ValueError, match='model geometry'):
        list(iter_q2_expert_pieces(tmp_path, config))


def test_bad_source_is_rejected_before_bank_allocation_can_start(tmp_path):
    config, _ = _source(tmp_path / 'small.gguf', kind=10)
    with pytest.raises(ValueError, match='incompatible'):
        iter_q2_expert_pieces(tmp_path, config)
