#pragma once

#include <cstdint>

// Layout consumed by the b10434 single-token ABI. ``qs`` is signed Q8_1 payload;
// ``d`` is the scale and ``s`` is the activation sum.
struct freetoken_block_q8_1 {
  uint16_t d;
  uint16_t s;
  int8_t qs[32];
};

static_assert(sizeof(freetoken_block_q8_1) == 36, "Q8_1 ABI changed");
