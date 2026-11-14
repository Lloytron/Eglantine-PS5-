// SPDX-License-Identifier: GPL-3.0-or-later
// EDEN_UP_HOST_STACK_KIB=<256|512|1024> (off by default; any other value or unset: off).
//
// Idea from mihawk-99/PS5_Hookshot PR #1 by CHI-GETSU (12 Nov 2026: smaller host thread stacks;
// they measured a deepest use of 141 KiB, the shader compiler included). Our numbers and the
// reasoning: the stack and stream memory finding of 12 Nov 2026.
//
// Every host thread whose creator places no stack and asks for less than 2 MiB (std::thread, the
// pipeline builders, Mesa's queues, the wrapper's own threads, and the GPU worker's 1 MiB too)
// gets 2 MiB of direct memory from the RADV platform's thread wrap (mihawk-sdk-review
// platform/src/threads.c, linked through --wrap=pthread_create/join/detach). About 60 of them run
// in a game: ~125 MiB of the pool the heap grows from.
//
// tools/radv-link-eden.sh renames the platform's three wraps to ps5_platform_pthread_* in its copy
// of the archive, and these take their place. Switch off (or not decided yet): each call goes
// straight on to the platform's, as before. Switch on: a thread that would get the platform's
// 2 MiB, and asked for no more than the switch's size, runs on a stack of that size instead, cut
// from 2 MiB blocks taken at the top of the pool like the platform's (src/direct_memory.c), each
// stack with an unmapped 256 KiB guard below it. A thread that asks for more than that size (but
// less than 2 MiB), that places its own stack, or that finds no stack here, goes to the platform as
// before. Blocks are kept once taken; a stack goes back to its block when its thread is joined
// (pthread_join) or, for a detached thread, by a reaper thread (on a stack of its own here) that
// joins it once its start routine has returned or it called pthread_exit, as the platform does.
// The decision is made by main right after env.txt is applied (eden_host_stacks_init): threads
// made before that keep the platform's stacks (and their join/detach go on to the platform).
//
// Line "EDEN_UP_HOST_STACK" with the pool ledger (headless/performance.cpp): threads given a small
// stack, live and peak, blocks held, the MiB those threads would have held at 2 MiB each less what
// the blocks hold, and the threads sent on to the platform while on.
#define _GNU_SOURCE 1
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ps5_platform_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int ps5_platform_pthread_join(pthread_t, void **);
int ps5_platform_pthread_detach(pthread_t);
int __real_pthread_join(pthread_t, void **);
/* The platform's own placement: address space outside the Vulkan driver's GPU window. */
int ps5p_reserve_placed(size_t bytes, uintptr_t hint, size_t alignment, int no_place, void **address);
int sceKernelAllocateDirectMemory(int64_t, int64_t, size_t, size_t, int, int64_t *);
int sceKernelReleaseDirectMemory(int64_t, size_t);
int sceKernelMapDirectMemory(void **, size_t, int, int, int64_t, size_t);
int sceKernelMunmap(void *, size_t);
int64_t sceKernelGetDirectMemorySize(void);

#define HS_PLATFORM_STACK ((size_t)2 << 20) /* PS5_THREAD_STACK_BYTES */
#define HS_BLOCK ((size_t)2 << 20)
/* Unmapped below each stack. No stack probes in this build (no -fstack-clash-protection), so a frame
 * larger than the guard can step over it into the next slot's live stack, which sits right below:
 * T181's largest frames are 206 KiB (settings load), 141 KiB (radv_check_gpu_hangs), 120 KiB
 * (psbc tessellation compile), 96-103 KiB (RADV shader objects, Dynarmic decode), 70 KiB
 * (nir_opt_varyings). 256 KiB covers them all; it is address space only, no memory. */
#define HS_GUARD ((size_t)256 << 10)
#define HS_UNIT ((size_t)64 << 10)
#define HS_TYPE_CPU 12
#define HS_PROT (0x01 | 0x02)
#define HS_MAP_FIXED 0x10
#define HS_MAX_BLOCKS 64 /* 64 x 2 MiB: 128 threads at 1 MiB, 512 at 256 KiB */

struct hs_block {
  char *base;        /* reserved: per slot a guard, then the stack */
  int64_t direct;    /* the 2 MiB block */
  uint8_t used[8];   /* one bit per slot */
};

struct hs_record {
  struct hs_record *next;
  pthread_t thread;
  unsigned block, slot;
  void *(*start)(void *);
  void *argument;
  bool detached, finished;
};

static atomic_int hs_state;           /* 0 not decided (off), 1 on, 2 off */
static size_t hs_stack;               /* the switch's size in bytes, when on */
static unsigned hs_slots;             /* slots per block */
static pthread_mutex_t hs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t hs_reaper_wake = PTHREAD_COND_INITIALIZER;
static struct hs_block hs_blocks[HS_MAX_BLOCKS];
static unsigned hs_block_count;
static struct hs_record *hs_records;
static bool hs_reaper_started;
static pthread_key_t hs_finish_key;
static pthread_once_t hs_key_once = PTHREAD_ONCE_INIT;
static unsigned long long hs_made, hs_live, hs_peak, hs_platform, hs_no_slot;

static void hs_finished(void *opaque) {
  struct hs_record *const record = opaque;
  pthread_mutex_lock(&hs_lock);
  record->finished = true;
  if (record->detached) pthread_cond_signal(&hs_reaper_wake);
  pthread_mutex_unlock(&hs_lock);
}
static void hs_key_create(void) {
  if (pthread_key_create(&hs_finish_key, hs_finished) != 0) atomic_store(&hs_state, 2);
}

void eden_host_stacks_init(void) {
  const char *value = getenv("EDEN_UP_HOST_STACK_KIB");
  const unsigned long kib = value ? strtoul(value, NULL, 10) : 0;
  if (kib == 256 || kib == 512 || kib == 1024) {
    hs_stack = (size_t)kib << 10;
    hs_slots = (unsigned)(HS_BLOCK / hs_stack);
    pthread_once(&hs_key_once, hs_key_create);
    if (atomic_load(&hs_state) == 0) atomic_store(&hs_state, 1);
    fprintf(stderr, "EDEN_UP_HOST_STACK on kib=%lu slots_per_block=%u guard_kib=%zu\n", kib, hs_slots,
            HS_GUARD >> 10);
  } else {
    atomic_store(&hs_state, 2);
    if (value) fprintf(stderr, "EDEN_UP_HOST_STACK off: kib=%s is not 256, 512 or 1024\n", value);
  }
}

/* Under hs_lock. A new 2 MiB block, every slot mapped; false when the pool or the table has none. */
static bool hs_block_add(void) {
  if (hs_block_count == HS_MAX_BLOCKS) return false;
  const size_t span = (size_t)hs_slots * (HS_GUARD + hs_stack);
  void *base = NULL;
  if (ps5p_reserve_placed(span, 0, HS_UNIT, -1, &base) != 0) return false;
  int64_t direct = -1;
  /* Through the wrap: a 2 MiB block goes to the top of the pool, as the platform's stacks do. */
  if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), HS_BLOCK, HS_UNIT, HS_TYPE_CPU, &direct) != 0) {
    sceKernelMunmap(base, span);
    return false;
  }
  for (unsigned slot = 0; slot < hs_slots; ++slot) {
    void *const at = (char *)base + (size_t)slot * (HS_GUARD + hs_stack) + HS_GUARD;
    void *mapped = at;
    if (sceKernelMapDirectMemory(&mapped, hs_stack, HS_PROT, HS_MAP_FIXED, direct + (int64_t)(slot * hs_stack),
                                 HS_UNIT) != 0 || mapped != at) {
      sceKernelMunmap(base, span); /* the whole range: mapped slots and guards alike */
      sceKernelReleaseDirectMemory(direct, HS_BLOCK);
      return false;
    }
  }
  struct hs_block *const block = &hs_blocks[hs_block_count++];
  block->base = base;
  block->direct = direct;
  memset(block->used, 0, sizeof(block->used));
  return true;
}

/* Under hs_lock. */
static bool hs_take(unsigned *block_out, unsigned *slot_out) {
  for (int pass = 0; pass < 2; ++pass) {
    for (unsigned b = 0; b < hs_block_count; ++b) {
      for (unsigned s = 0; s < hs_slots; ++s) {
        if (!(hs_blocks[b].used[s / 8] & (1u << (s % 8)))) {
          hs_blocks[b].used[s / 8] |= (uint8_t)(1u << (s % 8));
          *block_out = b;
          *slot_out = s;
          if (++hs_live > hs_peak) hs_peak = hs_live;
          return true;
        }
      }
    }
    if (pass == 0 && !hs_block_add()) break;
  }
  return false;
}
/* Under hs_lock. */
static void hs_give(unsigned block, unsigned slot) {
  hs_blocks[block].used[slot / 8] &= (uint8_t)~(1u << (slot % 8));
  --hs_live;
}
static void *hs_slot_address(unsigned block, unsigned slot) {
  return hs_blocks[block].base + (size_t)slot * (HS_GUARD + hs_stack) + HS_GUARD;
}

static void *hs_reaper(void *unused) {
  (void)unused;
  pthread_mutex_lock(&hs_lock);
  for (;;) {
    struct hs_record *found = NULL;
    for (struct hs_record **at = &hs_records; *at; at = &(*at)->next) {
      if ((*at)->detached && (*at)->finished) {
        found = *at;
        *at = found->next;
        break;
      }
    }
    if (!found) {
      pthread_cond_wait(&hs_reaper_wake, &hs_lock);
      continue;
    }
    pthread_mutex_unlock(&hs_lock);
    __real_pthread_join(found->thread, NULL);
    pthread_mutex_lock(&hs_lock);
    hs_give(found->block, found->slot);
    free(found);
  }
  return NULL;
}

/* Under hs_lock: the reaper, on a small stack of its own (detached, never ends). */
static void hs_reaper_ensure(void) {
  if (hs_reaper_started) return;
  unsigned block, slot;
  pthread_attr_t attributes;
  if (pthread_attr_init(&attributes) != 0) return;
  pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  const bool own = hs_take(&block, &slot);
  if (own) pthread_attr_setstack(&attributes, hs_slot_address(block, slot), hs_stack);
  pthread_t thread;
  /* The platform: a placed stack is created as asked; without one, its own 2 MiB. */
  if (ps5_platform_pthread_create(&thread, &attributes, hs_reaper, NULL) == 0) hs_reaper_started = true;
  else if (own) hs_give(block, slot);
  pthread_attr_destroy(&attributes);
}

static void *hs_start(void *opaque) {
  struct hs_record *const record = opaque;
  pthread_setspecific(hs_finish_key, record);
  return record->start(record->argument);
}

/* A copy that keeps what the caller set, for a stack placed here; false when the platform
 * should make this thread as before (its own stack placed, or asking for more than the switch). */
static bool hs_attributes(const pthread_attr_t *attributes, pthread_attr_t *copy, bool *detached) {
  *detached = false;
  if (attributes) {
    void *address = NULL;
    size_t size = 0;
    if (pthread_attr_getstackaddr(attributes, &address) == 0 && address != NULL) return false;
    if (pthread_attr_getstacksize(attributes, &size) == 0 && size > hs_stack) return false;
  }
  if (pthread_attr_init(copy) != 0) return false;
  if (attributes) {
    int value;
    struct sched_param parameter;
    if (pthread_attr_getdetachstate(attributes, &value) == 0) *detached = value == PTHREAD_CREATE_DETACHED;
    if (pthread_attr_getinheritsched(attributes, &value) == 0) pthread_attr_setinheritsched(copy, value);
    if (pthread_attr_getschedpolicy(attributes, &value) == 0) pthread_attr_setschedpolicy(copy, value);
    if (pthread_attr_getschedparam(attributes, &parameter) == 0) pthread_attr_setschedparam(copy, &parameter);
    if (pthread_attr_getscope(attributes, &value) == 0) pthread_attr_setscope(copy, value);
  }
  pthread_attr_setdetachstate(copy, PTHREAD_CREATE_JOINABLE);
  return true;
}

int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attributes, void *(*start)(void *),
                          void *argument) {
  if (atomic_load_explicit(&hs_state, memory_order_acquire) != 1)
    return ps5_platform_pthread_create(thread, attributes, start, argument);
  pthread_attr_t copy;
  bool detached;
  if (!hs_attributes(attributes, &copy, &detached)) {
    pthread_mutex_lock(&hs_lock);
    ++hs_platform;
    pthread_mutex_unlock(&hs_lock);
    return ps5_platform_pthread_create(thread, attributes, start, argument);
  }
  struct hs_record *const record = calloc(1, sizeof(*record));
  if (!record) {
    pthread_attr_destroy(&copy);
    return EAGAIN;
  }
  record->start = start;
  record->argument = argument;
  record->detached = detached;
  pthread_mutex_lock(&hs_lock);
  const bool have = hs_take(&record->block, &record->slot);
  if (!have) {
    ++hs_no_slot;
    ++hs_platform;
    pthread_mutex_unlock(&hs_lock);
    free(record);
    pthread_attr_destroy(&copy);
    return ps5_platform_pthread_create(thread, attributes, start, argument);
  }
  pthread_attr_setstack(&copy, hs_slot_address(record->block, record->slot), hs_stack);
  if (detached) hs_reaper_ensure();
  /* Listed before the thread runs, so its end finds it. The platform makes a thread on a placed
   * stack as asked (and passes the creator's MXCSR on). */
  const int result = ps5_platform_pthread_create(&record->thread, &copy, hs_start, record);
  if (result == 0) {
    record->next = hs_records;
    hs_records = record;
    *thread = record->thread;
    ++hs_made;
  } else {
    hs_give(record->block, record->slot);
  }
  pthread_mutex_unlock(&hs_lock);
  pthread_attr_destroy(&copy);
  if (result != 0) free(record);
  return result;
}

int __wrap_pthread_join(pthread_t thread, void **value) {
  if (atomic_load_explicit(&hs_state, memory_order_acquire) != 1) return ps5_platform_pthread_join(thread, value);
  pthread_mutex_lock(&hs_lock);
  bool ours = false;
  for (struct hs_record *record = hs_records; record; record = record->next)
    ours |= pthread_equal(record->thread, thread) && !record->detached;
  pthread_mutex_unlock(&hs_lock);
  if (!ours) return ps5_platform_pthread_join(thread, value);
  const int result = __real_pthread_join(thread, value);
  if (result == 0) {
    pthread_mutex_lock(&hs_lock);
    for (struct hs_record **at = &hs_records; *at; at = &(*at)->next) {
      if (pthread_equal((*at)->thread, thread)) {
        struct hs_record *const record = *at;
        *at = record->next;
        hs_give(record->block, record->slot);
        free(record);
        break;
      }
    }
    pthread_mutex_unlock(&hs_lock);
  }
  return result;
}

int __wrap_pthread_detach(pthread_t thread) {
  if (atomic_load_explicit(&hs_state, memory_order_acquire) != 1) return ps5_platform_pthread_detach(thread);
  pthread_mutex_lock(&hs_lock);
  for (struct hs_record *record = hs_records; record; record = record->next) {
    if (pthread_equal(record->thread, thread)) {
      const int result = record->detached ? EINVAL : 0;
      record->detached = true;
      hs_reaper_ensure();
      if (record->finished) pthread_cond_signal(&hs_reaper_wake);
      pthread_mutex_unlock(&hs_lock);
      return result;
    }
  }
  pthread_mutex_unlock(&hs_lock);
  return ps5_platform_pthread_detach(thread);
}

/* 0: the switch is off. headless/performance.cpp prints it with the pool ledger. */
int eden_host_stacks_report(char *line, size_t size) {
  if (atomic_load_explicit(&hs_state, memory_order_acquire) != 1) return 0;
  pthread_mutex_lock(&hs_lock);
  const unsigned long long made = hs_made, live = hs_live, peak = hs_peak, blocks = hs_block_count,
                           platform = hs_platform, no_slot = hs_no_slot;
  pthread_mutex_unlock(&hs_lock);
  const long long saved = (long long)(live * (HS_PLATFORM_STACK >> 20)) - (long long)(blocks * (HS_BLOCK >> 20));
  snprintf(line, size,
           "EDEN_UP_HOST_STACK kib=%zu made=%llu live=%llu peak=%llu blocks=%llu held_mib=%llu saved_mib=%lld "
           "to_platform=%llu no_slot=%llu",
           hs_stack >> 10, made, live, peak, blocks, blocks * (HS_BLOCK >> 20), saved, platform, no_slot);
  return 1;
}
