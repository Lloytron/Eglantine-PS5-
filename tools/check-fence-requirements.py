#!/usr/bin/env python3
"""Compile actual T119 fence bodies with mocks; no PS5 SDK or driver needed."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source_root = Path(os.environ.get('FENCE_SOURCE_ROOT', root / 'Eglantine/src'))
source = (source_root / 'video_core/fence_manager.h').read_text()


def body(text, signature):
    start = text.index(signature)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


code = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>
using u64=unsigned long long;using u32=unsigned;using u8=unsigned char;
#define LOG_WARNING(...) ((void)0)
namespace Settings {
inline bool delayed{};
bool IsGPUFenceBehaviorDefault(){return false;}bool IsGPULevelHigh(){return false;}
bool IsGPUFenceBehaviorBalanced(){return delayed;}bool IsGPUFenceBehaviorAccurate(){return false;}
}
namespace VideoCore {
inline bool g_signaling_query_report_fence{};
namespace Readback {
inline bool g_signaling_held_release{};
inline std::atomic<u64> g_held_releases_pending{},g_downloads_in_flight{};
}
}
namespace VideoCommon {
inline bool no_alloc{};
bool ReportNoAllocEnabled(){return no_alloc;}
namespace Nx2Profile {enum class Zone{Fence,FenceSkipped};struct Scope{Scope(Zone){}};}
namespace Nx2FastDraw {enum class Reason{Fence};void Break(Reason){}}
namespace GpuSplit {
bool OnBackThread(){return true;}
struct Split { bool BackRunning(){return true;}void RequestIdleHook(){}; };
Split& Get(){static Split split;return split;}
}
struct Cache {
 std::mutex mutex;bool pending{},side_work{};unsigned checks{},commits{},pops{};
 bool HasUncommittedFlushes(){++checks;return pending;}
 bool HasAsyncFlushWork(){return pending||side_work;}
 void CommitAsyncFlushes(){++commits;pending=false;}
 void PopAsyncFlushes(){++pops;}
};
struct Raster {unsigned invalidates{},flushes{};void InvalidateGPUCache(){++invalidates;}void FlushCommands(){++flushes;}};
struct TestFence {bool stubbed;};
struct Manager {
 using TFence=std::shared_ptr<TestFence>;
 using OperationList=std::deque<std::function<void()>>;
 static constexpr size_t MaxSpareOperationLists=32;
 static constexpr bool can_async_check=false;
 static constexpr u8 TextureFlushes=1,BufferFlushes=2;
 static constexpr u32 NoSyncpoint=~u32{0};
 Cache texture_cache,buffer_cache,query_cache;Raster rasterizer;
 std::mutex guard,uncommitted_guard,released_guard,ring_guard;
 std::condition_variable cv,released_cv;
 unsigned skipped_fences{},queued{},readback_mode{},held_noted{};bool skip{};
 std::deque<std::function<void()>> uncommitted_operations;
 std::deque<std::deque<std::function<void()>>> pending_operations;
 std::vector<std::deque<std::function<void()>>> spare_operations;
 std::deque<u8> pending_flushes;std::queue<TFence> fences;
 struct Held {u32 syncpoint;u64 needs;std::chrono::steady_clock::time_point held_at;};
 std::deque<Held> held_increments;
 std::atomic<u64> downloads_committed{},downloads_landed{},held_front_needs{~u64{0}};
 struct Syncpoint {unsigned guest{},host{};void IncrementGuest(u32){++guest;}void IncrementHost(u32){++host;}} syncpoint_manager;
 struct Ring {unsigned pushed{};void Push(TFence&&){++pushed;}} delayed_destruction_ring;
 bool ShouldWait(){return false;}bool IsFenceSignaled(const TFence&){return true;}
 void WaitFence(const TFence&){}
 void PopAsyncFlushes(u8 mask){if(mask&1)texture_cache.PopAsyncFlushes();if(mask&2)buffer_cache.PopAsyncFlushes();query_cache.PopAsyncFlushes();}
 bool SkipEmptyFence(){return skip;}
 int ReadbackWait(){return readback_mode;}
 void ApplyReleasedIncrements(bool){}void PublishHeldPush(){}void NoteReadbackIncrement(bool){++held_noted;}
 TFence CreateFence(bool stubbed){return std::make_shared<TestFence>(TestFence{stubbed});}
 void QueueFence(const TFence&){++queued;}
 struct CacheFlushes{bool should_flush;u8 committed;};
'''
for signature in ('OperationList TakeUncommittedOperations()',
                  'void KeepSpareOperations(OperationList& list)',
                  'void TryReleasePendingFences()',
                  'CacheFlushes CommitCacheFlushes()',
                  'void SignalFenceImpl(std::function<void()>&& func, u32 syncpoint)',
                  'void SignalFence(std::function<void()>&& func)'):
    if signature.startswith('void TryRelease'):
        code += 'template<bool force_wait>\n'
    code += body(source, signature) + '\n'
code += r'''
};
}
int main(){
 using namespace VideoCommon;
 for(bool fast:{false,true}){
  no_alloc=fast;
  for(bool delayed:{false,true})for(unsigned mask=0;mask<8;++mask){
   Settings::delayed=delayed;Manager m;
   m.texture_cache.pending=mask&1;m.buffer_cache.pending=mask&2;m.query_cache.pending=mask&4;
   unsigned callbacks{};m.SignalFence([&]{++callbacks;});
   assert(m.queued==1&&m.fences.size()==1&&m.fences.front()->stubbed==(mask==0));
   assert(m.texture_cache.checks==1&&m.buffer_cache.checks==1&&m.query_cache.checks==1);
   assert(m.texture_cache.commits==bool(mask&1)&&m.buffer_cache.commits==bool(mask&2)&&m.query_cache.commits==1);
   assert(m.pending_flushes.back()==(mask&3)&&m.rasterizer.flushes==bool(mask));
   assert(callbacks==!delayed&&m.rasterizer.invalidates==1);
   m.TryReleasePendingFences<false>();
   assert(callbacks==1&&m.delayed_destruction_ring.pushed==1&&m.fences.empty());
   assert(m.query_cache.pops==1);
   assert(m.spare_operations.size()==(fast?1u:0u));
  }
  Settings::delayed=false;
  {Manager m;m.skip=true;unsigned calls{};m.SignalFence([&]{++calls;});
   assert(calls==1&&!m.queued&&m.query_cache.commits==0);}
  {Manager m;m.texture_cache.side_work=true;m.SignalFence([]{});
   assert(m.queued==1&&m.fences.front()->stubbed&&m.texture_cache.commits==1);
   assert(m.pending_flushes.back()==1&&!m.rasterizer.flushes);}
  // Actual fence drain preserves FIFO and lets operations re-enter the uncommitted lock.
  Settings::delayed=true;
  {Manager m;std::vector<unsigned> order;
   for(unsigned i=0;i<100;++i){
    m.SignalFence([&,i]{std::scoped_lock lock{m.uncommitted_guard};order.push_back(i);});
    m.TryReleasePendingFences<false>();
   }
   for(unsigned i=0;i<100;++i)assert(order[i]==i);
   if(fast)assert(m.spare_operations.size()<=2);else assert(m.spare_operations.empty());
  }
  // Current T119 held-readback operation must still run and decrement the in-flight count.
  {Manager m;m.readback_mode=1;m.buffer_cache.pending=true;
   m.SignalFenceImpl([]{},7);
   assert(m.downloads_committed==1&&VideoCore::Readback::g_downloads_in_flight==1);
   m.TryReleasePendingFences<false>();
   assert(m.downloads_landed==1&&VideoCore::Readback::g_downloads_in_flight==0);
   assert(m.held_increments.size()==1&&m.held_increments.front().needs==1);
  }
 }
}
'''
compiler = os.environ.get('CXX') or shutil.which('clang++-18') or shutil.which('clang++')
if not compiler:
    raise SystemExit('No local C++ compiler found')
# Override TMPDIR when invoking this from the offline task, keeping all artifacts in out/.
with tempfile.TemporaryDirectory(prefix='fence-requirements-') as temp:
    path = Path(temp)
    (path / 'check.cpp').write_text(code)
    subprocess.run([compiler, '-std=c++20', '-O1', '-g', '-pthread',
                    '-fsanitize=address,undefined', str(path / 'check.cpp'), '-o', str(path / 'check')], check=True)
    subprocess.run([str(path / 'check')], check=True, timeout=10)
print('Fence requirements PASS: off/on, all immediate/delayed masks, empty skip, side-work stub, FIFO/reentrant callbacks, list recycling (on only) and held-readback bookkeeping')
