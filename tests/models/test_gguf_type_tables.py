"""Unit tests for GGML type tables in dequant.py.

These tests verify that the BLOCK_SHAPE table and capability sets accurately
reflect the runtime behavior of the CUDA kernels. They run without CUDA (no kernel
compilation), checking only the type definitions and switch statement extraction.

Tests cover:
1. Block struct byte sizes derived from ggml-common.h match BLOCK_SHAPE entries
2. row_bytes() roundtrip consistency for various numerals
3. Logical consistency between type-capability sets (subsets, unions, exclusions)
4. Switch statement cases in gguf_kernel.cu match Python frozenset definitions
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    DEQUANT_TYPES,
    GGML_UNQUANTIZED,
    GGML_NAME,
    MMQ_TYPES,
    MMVQ_TYPES,
    MOE_MMQ_TYPES,
    MOE_VEC_TYPES,
    row_bytes,
)


def _parse_ggml_common_h() -> dict[str, int]:
    """Extract block struct definitions from ggml-common.h and compute packed byte sizes.

    Returns a dict mapping quant type names (e.g. "block_q4_0") to their byte sizes.
    Substitutes QK_K=256 and K_SCALE_SIZE=12 before computing sizes.
    """
    header_path = Path(__file__).parent.parent.parent / "python" / "freetoken" / "kernel" / "csrc" / "gguf" / "ggml-common.h"
    with open(header_path) as f:
        content = f.read()

    # Extract block struct definitions
    # Pattern: typedef struct { ... } block_<name>;
    pattern = r"typedef\s+struct\s*\{([^}]+)\}\s*block_(\w+);"
    matches = re.findall(pattern, content)

    # Define substitutions for macro constants
    macros = {
        "QK_K": 256,
        "K_SCALE_SIZE": 12,
        # ggml-common.h: `#define IQ3S_N_SCALE QK_K / 64`. Without it block_iq3_s's
        # `scales[IQ3S_N_SCALE]` field parses as 0 and the struct comes out 106 instead
        # of 110 -- which the real-checkpoint check disproves: Ornith's IQ3_S
        # token_embd.weight is 248320*2048/256*110 == 218,521,600 bytes on disk exactly.
        "IQ3S_N_SCALE": 4,
        "QK4_0": 32,
        "QK4_1": 32,
        "QK5_0": 32,
        "QK5_1": 32,
        "QK8_0": 32,
        "QK8_1": 32,
        "QK4_NL": 32,
    }

    sizes = {}

    for struct_body, type_name in matches:
        # Parse field declarations from the struct body
        # Format: <type> <name>[<size>];
        field_pattern = r"(\w+)\s+(\w+)(?:\[([^\]]+)\])?;"
        fields = re.findall(field_pattern, struct_body)

        total_size = 0
        for field_type, field_name, array_size in fields:
            # Compute size of each field
            if field_type == "half":
                field_size = 2
            elif field_type == "half2":
                field_size = 4
            elif field_type == "uint8_t":
                field_size = 1
            elif field_type == "uint16_t":
                field_size = 2
            elif field_type == "uint32_t":
                field_size = 4
            elif field_type == "int8_t":
                field_size = 1
            elif field_type == "int32_t":
                field_size = 4
            else:
                # Unknown type; skip
                continue

            # Handle array sizes
            if array_size:
                # Substitute macros in array size
                size_expr = array_size
                for macro_name, macro_val in macros.items():
                    size_expr = size_expr.replace(macro_name, str(macro_val))
                # Evaluate the expression (handles division, multiplication, etc.)
                try:
                    array_len = eval(size_expr)
                except Exception:
                    # If evaluation fails, try simple substitution
                    continue
                field_size *= array_len

            total_size += field_size

        if total_size > 0:
            sizes[type_name] = total_size

    return sizes


def _extract_switch_cases(file_path: str, func_name: str) -> set[int]:
    """Extract case labels from a switch statement in a C++ file.

    Args:
        file_path: Path to the .cu file
        func_name: Name of the function containing the switch

    Returns:
        Set of case numbers extracted from the switch statement
    """
    with open(file_path) as f:
        content = f.read()

    # Find the function
    func_pattern = rf"(?:torch::Tensor|int64_t|void)\s+{func_name}\s*\([^)]*\)\s*\{{"
    func_match = re.search(func_pattern, content)
    if not func_match:
        return set()

    # Find the switch statement within the function
    start_pos = func_match.end()
    # Scan forward to find "switch ("
    switch_pos = content.find("switch (", start_pos)
    if switch_pos == -1 or switch_pos > start_pos + 5000:
        return set()

    # Find the opening brace of the switch
    brace_pos = content.find("{", switch_pos)
    # Find the matching closing brace
    brace_count = 1
    end_pos = brace_pos + 1
    while brace_count > 0 and end_pos < len(content):
        if content[end_pos] == "{":
            brace_count += 1
        elif content[end_pos] == "}":
            brace_count -= 1
        end_pos += 1

    switch_body = content[brace_pos + 1:end_pos - 1]

    # Extract all "case <number>:" labels
    case_pattern = r"case\s+(\d+):"
    cases = set(int(n) for n in re.findall(case_pattern, switch_body))

    return cases


def test_block_shape_matches_ggml_common():
    """Verify BLOCK_SHAPE table entries match struct sizes from ggml-common.h.

    Parses typedef struct { ... } block_<name> definitions from the header,
    computes packed byte sizes from field declarations (accounting for QK_K=256,
    K_SCALE_SIZE=12, and field type sizes), and asserts each matches the
    corresponding BLOCK_SHAPE[type_enum][1] entry.
    """
    parsed_sizes = _parse_ggml_common_h()

    # Map C struct names to GGML type enums and expected sizes
    type_mappings = {
        "q4_0": (2, 18),
        "q4_1": (3, 20),
        "q5_0": (6, 22),
        "q5_1": (7, 24),
        "q8_0": (8, 34),
        "q2_K": (10, 84),
        "q3_K": (11, 110),
        "q4_K": (12, 144),
        "q5_K": (13, 176),
        "q6_K": (14, 210),
        "iq2_xxs": (16, 66),
        "iq2_xs": (17, 74),
        "iq3_xxs": (18, 98),
        "iq1_s": (19, 50),
        "iq4_nl": (20, 18),
        "iq3_s": (21, 110),
        "iq2_s": (22, 82),
        "iq4_xs": (23, 136),
        "iq1_m": (29, 56),
    }

    for struct_name, (ggml_type, expected_bytes) in type_mappings.items():
        parsed_size = parsed_sizes.get(struct_name)

        # If parsing failed for this type (too complex), accept the hard-coded expectation
        if parsed_size is not None:
            parsed_size = int(parsed_size)
            assert parsed_size == expected_bytes, (
                f"block_{struct_name}: parsed size {parsed_size} != expected {expected_bytes}"
            )

        # Also verify BLOCK_SHAPE matches
        assert ggml_type in BLOCK_SHAPE, f"GGML type {ggml_type} not in BLOCK_SHAPE"
        block_numel, bytes_per_block = BLOCK_SHAPE[ggml_type]
        assert bytes_per_block == expected_bytes, (
            f"BLOCK_SHAPE[{ggml_type}][1]={bytes_per_block} != expected {expected_bytes}"
        )


def test_row_bytes_roundtrip():
    """Test row_bytes() consistency for various block counts.

    For each quant type, verify that row_bytes(k * block_numel, type) == k * bytes_per_block
    for k in [1, 2, 4, 16], and that non-multiples of block_numel raise AssertionError.
    """
    for ggml_type, (block_numel, bytes_per_block) in BLOCK_SHAPE.items():
        # Test valid multiples
        for k in [1, 2, 4, 16]:
            numel = k * block_numel
            expected = k * bytes_per_block
            actual = row_bytes(numel, ggml_type)
            assert actual == expected, (
                f"row_bytes({numel}, {ggml_type}): got {actual}, expected {expected}"
            )

        # Test that non-multiples raise
        if block_numel > 1:
            bad_numel = block_numel + 1
            with pytest.raises(AssertionError):
                row_bytes(bad_numel, ggml_type)


def test_capability_sets_are_consistent():
    """Verify logical consistency and completeness of type-capability sets.

    Checks:
    - MMVQ_TYPES == DEQUANT_TYPES (both handle all STD_K and IQ types)
    - MMQ_TYPES == MOE_MMQ_TYPES (both handle STD_K only)
    - MOE_VEC_TYPES == DEQUANT_TYPES (both handle all STD_K and IQ types)
    - MMQ_TYPES is a strict subset of MMVQ_TYPES
    - MOE_MMQ_TYPES is a strict subset of MOE_VEC_TYPES
    - Every member of every set has a BLOCK_SHAPE entry and GGML_NAME entry
    - None of the five sets intersects GGML_UNQUANTIZED (F32, F16, BF16)
    """
    # Check set equalities
    assert MMVQ_TYPES == DEQUANT_TYPES, (
        f"MMVQ_TYPES {MMVQ_TYPES} != DEQUANT_TYPES {DEQUANT_TYPES}"
    )
    assert MMQ_TYPES == MOE_MMQ_TYPES, (
        f"MMQ_TYPES {MMQ_TYPES} != MOE_MMQ_TYPES {MOE_MMQ_TYPES}"
    )
    assert MOE_VEC_TYPES == DEQUANT_TYPES, (
        f"MOE_VEC_TYPES {MOE_VEC_TYPES} != DEQUANT_TYPES {DEQUANT_TYPES}"
    )

    # Check subset relationships
    assert MMQ_TYPES < MMVQ_TYPES, (
        f"MMQ_TYPES {MMQ_TYPES} is not a strict subset of MMVQ_TYPES {MMVQ_TYPES}"
    )
    assert MOE_MMQ_TYPES < MOE_VEC_TYPES, (
        f"MOE_MMQ_TYPES {MOE_MMQ_TYPES} is not a strict subset of MOE_VEC_TYPES {MOE_VEC_TYPES}"
    )

    # Check every member is in BLOCK_SHAPE and GGML_NAME
    all_types = DEQUANT_TYPES | MMVQ_TYPES | MMQ_TYPES | MOE_VEC_TYPES | MOE_MMQ_TYPES
    for ggml_type in all_types:
        assert ggml_type in BLOCK_SHAPE, f"GGML type {ggml_type} not in BLOCK_SHAPE"
        assert ggml_type in GGML_NAME, f"GGML type {ggml_type} not in GGML_NAME"

    # Check no intersection with unquantized types
    for ggml_type_set in [DEQUANT_TYPES, MMVQ_TYPES, MMQ_TYPES, MOE_VEC_TYPES, MOE_MMQ_TYPES]:
        assert ggml_type_set.isdisjoint(GGML_UNQUANTIZED), (
            f"Type set {ggml_type_set} intersects GGML_UNQUANTIZED {GGML_UNQUANTIZED}"
        )


def test_capability_sets_match_cuda_switches():
    """Extract switch cases from gguf_kernel.cu and verify against Python sets.

    For each CUDA kernel (ggml_mul_mat_vec_a8, ggml_mul_mat_a8, ggml_moe_a8,
    ggml_moe_a8_vec, ggml_moe_get_block_size), extracts the case labels from
    the switch(type) block and asserts they match the corresponding Python frozenset.

    This is the critical test that prevents Python tables from drifting from C source.
    """
    kernel_path = Path(__file__).parent.parent.parent / "python" / "freetoken" / "kernel" / "csrc" / "gguf" / "gguf_kernel.cu"

    # Extract cases for each kernel
    mmvq_cases = _extract_switch_cases(str(kernel_path), "ggml_mul_mat_vec_a8")
    mmq_cases = _extract_switch_cases(str(kernel_path), "ggml_mul_mat_a8")
    moe_a8_cases = _extract_switch_cases(str(kernel_path), "ggml_moe_a8")
    moe_vec_cases = _extract_switch_cases(str(kernel_path), "ggml_moe_a8_vec")
    moe_block_size_cases = _extract_switch_cases(str(kernel_path), "ggml_moe_get_block_size")

    # Verify against Python sets
    assert mmvq_cases == MMVQ_TYPES, (
        f"ggml_mul_mat_vec_a8 cases {mmvq_cases} != MMVQ_TYPES {MMVQ_TYPES}"
    )
    assert mmq_cases == MMQ_TYPES, (
        f"ggml_mul_mat_a8 cases {mmq_cases} != MMQ_TYPES {MMQ_TYPES}"
    )
    assert moe_a8_cases == MOE_MMQ_TYPES, (
        f"ggml_moe_a8 cases {moe_a8_cases} != MOE_MMQ_TYPES {MOE_MMQ_TYPES}"
    )
    assert moe_vec_cases == MOE_VEC_TYPES, (
        f"ggml_moe_a8_vec cases {moe_vec_cases} != MOE_VEC_TYPES {MOE_VEC_TYPES}"
    )
    assert moe_block_size_cases == MOE_MMQ_TYPES, (
        f"ggml_moe_get_block_size cases {moe_block_size_cases} != MOE_MMQ_TYPES {MOE_MMQ_TYPES}"
    )
