# Unified ROCm path shape and gate matrix

## Shared route contract

```text
RocmDispatchRequest:
  backend: cuda | rocm | cpu
  target: normalized gfx/sm target or null
  phase: prefill | decode
  operation: dense | moe_gate_up | moe_down | router | attention | copy
  quant_type: Q4_0 | Q4_K | Q5_K | Q6_K | Q8_0 | other
  rows: positive integer
  cols: positive integer
  tokens: positive integer
  requested_impl: legacy | auto | rdna3_mmid | rdna3_mmvdq | grouped_mmq
  gate_up_quant_type: quant type or null
  down_quant_type: quant type or null
  expert_stride_bytes: positive integer or null
  row_stride_bytes: positive integer or null
  id_space: raw | slot | null
  shape_key: canonical exact-shape string or null

RocmDispatchReport:
  route: generic | candidate | unsupported
  fallback_route: generic | null
  capability_status: unavailable | compile-only | served | correctness | performance
  target, operation, phase, quant_type, rows, cols, tokens
  gate_up_quant_type, down_quant_type, expert_stride_bytes, row_stride_bytes
  id_space, shape_key
  source_version, abi, implementation, reason
```

Unknown fields must be explicit. Reports are diagnostic data, not permission to
select a candidate.

## Target policy

| Target | Generic ROCm | Native candidate | Default | Evidence required |
| --- | --- | --- | --- | --- |
| `gfx1100` | allowed when generic contract passes | current candidate only for exact supported op/quant/shape/ABI | generic | self-test, finite logits, route/token parity, 3+ run A/B |
| `gfx1101`, `gfx1102`, `gfx1103` | allowed if generic contract passes | unavailable unless separately registered | generic | compile/import/contract only until candidate exists |
| `gfx1150`, `gfx1151` | allowed if generic contract passes | unavailable unless separately registered | generic | compile/import/contract only until candidate exists |
| `gfx1200`, `gfx1201` | allowed if generic contract passes | unavailable unless separately registered | generic | compile/import/contract only until candidate exists |
| unknown ROCm target | conservative generic/reference path or explicit unsupported | rejected | generic/fail loud | target matrix decision required |
| CUDA `sm_*` | existing CUDA route | no ROCm candidate import/probe | existing CUDA default | NVIDIA compile/import/runtime gates |

## Primary Qwen decode shape

```text
batch: 1
hidden: 2048
intermediate: 512
top_k: 8
gate_up: Q4_K packed GGUF rows
down: Q8_0 or source-compatible declared bank
phase: decode
mtp: false
```

Candidate code must reject prefill or a shape/quant/stride/ID-space mismatch
before extension loading unless separately registered. Each candidate record
must list canonical `shape_key` values; generic route may support a broader
declared matrix.

Canonical primary candidate key:

```text
decode:moe_gate_up+down:h2048:i512:k8:gate=Q4_K:down=Q8_0:id=<raw-or-slot>:expert_stride=<bytes>:row_stride=<bytes>:abi=moe-abi-v2
```

`raw` and `slot` are separate registrations when their kernels differ. Shape
checks must use actual tensor dimensions and strides, not only model family.

## Benchmark acceptance

```text
same model file/hash
same tokenizer/prompt/continuation identity
same decode token count
same quant/KV/graph/cache/backend configuration
same Torch/HIP/ROCm/driver/toolchain identity
same route and fallback count
>= 3 fresh-process runs per side after warmup
finite logits + exact completion count + independent oracle
median and raw spread reported
```

Missing identity, route, oracle, or completion evidence means `NO-PROMOTION`,
not zero or inferred throughput.
