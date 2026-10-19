// Thread placement and spin helpers shared by the CPU MoE pool and the locked-layer fetch threads.
#pragma once

#include "common.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOE_HAS_AFFINITY 1
#else
#define CPU_MOE_HAS_AFFINITY 0
#endif

namespace freetoken::cpu_moe {

inline void pin_this_thread(int cpu) {
#if CPU_MOE_HAS_AFFINITY
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu;
#endif
}

inline void cpu_relax() {
#if CPU_MOE_X86
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

}  // namespace freetoken::cpu_moe
