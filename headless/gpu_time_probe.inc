namespace {
// Development GPU execution time (dev_vulkan.h gpu_time): timestamps at the start and end
// of each upload+graphics submission, summed once the GPU has run it and printed every
// 5 s as EDEN_GPU_TIME. Probe/counters are worker-thread-only. Origin is captured on
// the recording thread, then copied into the submission command.
using ::Eden::VulkanProbe::GpuSubmitOrigin;
using ::Eden::VulkanProbe::gpu_submit_origin;
using ::Eden::VulkanProbe::GpuSubmitOriginScope;
struct GpuTimeProbe {
    static constexpr u32 SLOTS = 512;
    vk::QueryPool pool;
    PFN_vkCmdWriteTimestamp write{};
    double period_ns{};
    u32 valid_bits{};
    u64 next{}, first_pending{};
    u32 current{};
    u64 busy_ns{}, max_ns{}, submissions{}, total_ns{}, total_submissions{};
    std::chrono::steady_clock::time_point report{};
    bool failed{};
    struct OriginCount { GpuSubmitOrigin origin; u64 count{}; };
    std::array<OriginCount, 64> origins{};
    u32 origin_slots{};
    u64 origin_overflow{};
} gpu_time;

#ifndef EDEN_GPU_TIME_PROBE_CHECK
// EDEN_UP_GPU_PASS_TIME=1 (gpu_pass_probe.h; it also turns EDEN_GPU_TIME on): timestamps at the
// boundaries of the scheduler's render passes and of the tagged work between them, on a second
// query pool, so the time inside one submission is split by what ran. Each stamp is written at
// BOTTOM_OF_PIPE (all earlier work of the queue done) and only outside render passes. The queries
// are reset from the host (vkResetQueryPool, hostQueryReset: RADV has it and Eden enables it),
// once at the start and again after each submission's results were read: no reset is recorded.
// That matters on RADV: after a timestamp, a recorded vkCmdResetQueryPool in the same command
// buffer adds PS/CS partial flushes, an L2 invalidate and CB/DB flushes (active_query_flush_bits,
// radv_query.c), which would cost far more than the stamps. Without host reset the probe stays
// off ("EDEN_GPU_PASS failed=no_host_query_reset"). A stamp costs RADV one EOP timestamp event
// and emits the barrier flushes already pending a little earlier than the next draw would.
// A segment's time is the difference of its two stamps, so the segments of a submission
// add up to its whole time (upload start to its last stamp); work that overlaps the next segment
// counts for the one that finishes later. Worker thread only, like gpu_time.
struct GpuPassLabel {
    u64 key{};   ///< 0: no big target (TargetSmall); else width, height, format, samples, depth
    u64 bytes{}; ///< the largest attachment's bytes
    const char* format{"?"};
};
struct GpuPassProbe {
    static constexpr u32 STAMPS = 1U << 16;
    static constexpr u32 BUCKETS = 40;
    static constexpr u8 NO_BUCKET = 0xFF;
    vk::QueryPool pool;
    bool ready{}, failed{};
    u64 next{}, first_pending{};
    std::array<u8, STAMPS> bucket_of{};
    struct Submission { u64 first{}; u32 count{}; u64 upload_end{}; };
    std::array<Submission, GpuTimeProbe::SLOTS> submissions{};
    Submission* open{};
    bool in_pass{};
    u8 pass_bucket{};
    u8 outside_kind{};
    struct Bucket {
        u64 key{}, bytes{};
        u64 ns{}, segments{}, draws{}, clears{};
        char name[48]{};
    };
    std::array<Bucket, BUCKETS> buckets{};
    u32 used{::Eden::GpuPass::FixedKinds};
    u64 presents_at_report{};
    u64 stamps{}, nonmonotonic{}, in_pass_marks{}, lost_ns{};
    std::vector<u64> scratch;
} gpu_pass;

bool GpuPassActive() {
    return ::Eden::GpuPass::on && gpu_pass.ready && !gpu_pass.failed && gpu_pass.open != nullptr;
}

void GpuPassFail(const char* why) {
    gpu_pass.failed = true;
    std::printf("EDEN_GPU_PASS failed=%s\n", why);
}

// Writes one stamp that closes the segment of `bucket`. Outside render passes only.
void GpuPassStamp(VkCommandBuffer cmdbuf, u8 bucket) {
    if (gpu_pass.next - gpu_pass.first_pending >= GpuPassProbe::STAMPS) {
        GpuPassFail("ring"); // results stopped arriving; never overwrite a pending stamp
        return;
    }
    const u32 index = static_cast<u32>(gpu_pass.next++ % GpuPassProbe::STAMPS);
    gpu_pass.bucket_of[index] = bucket;
    gpu_time.write(cmdbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *gpu_pass.pool, index);
    ++gpu_pass.open->count;
}

u8 GpuPassBucket(const GpuPassLabel& label) {
    using namespace ::Eden::GpuPass;
    if (label.key == 0) return TargetSmall;
    for (u32 i = FixedKinds; i < gpu_pass.used; ++i) {
        if (gpu_pass.buckets[i].key == label.key) return static_cast<u8>(i);
    }
    if (gpu_pass.used == GpuPassProbe::BUCKETS) return TargetOverflow;
    auto& bucket = gpu_pass.buckets[gpu_pass.used];
    bucket.key = label.key;
    bucket.bytes = label.bytes;
    std::snprintf(bucket.name, sizeof(bucket.name), "rt_%ux%u_%s_s%u%s",
                  static_cast<unsigned>(label.key >> 48), static_cast<unsigned>((label.key >> 32) & 0xFFFF),
                  label.format, static_cast<unsigned>((label.key >> 8) & 0xFF), (label.key & 1) ? "_depth" : "");
    return static_cast<u8>(gpu_pass.used++);
}

// gpu_pass_probe.h's mark: a tag scope's start (enter) or end, recorded on the recording thread.
void GpuPassMark(void* raw, std::uint32_t kind, ::Eden::GpuPass::Counts counts, bool enter) {
    if (!GpuPassActive() || kind >= ::Eden::GpuPass::FixedKinds) return;
    const auto cmdbuf = static_cast<VkCommandBuffer>(raw);
    if (gpu_pass.in_pass) {
        // Recorded before the scope's own RequestOutsideRenderPassOperationContext: the kind
        // applies once the pass has ended. Its draws belong to the pass.
        auto& pass = gpu_pass.buckets[gpu_pass.pass_bucket];
        pass.draws += counts.draws;
        pass.clears += counts.clears;
        ++gpu_pass.in_pass_marks;
    } else {
        auto& outside = gpu_pass.buckets[gpu_pass.outside_kind];
        outside.draws += counts.draws;
        outside.clears += counts.clears;
        GpuPassStamp(cmdbuf, gpu_pass.outside_kind);
    }
    if (enter) ++gpu_pass.buckets[kind].segments;
    gpu_pass.outside_kind = static_cast<u8>(kind);
}

// The scheduler's render pass begin and end (vulkan_scheduler.cpp, prepare-vulkan-port.py).
// `load_clears`: deferred clears this pass's begin realizes (its load ops), counted for the pass.
void GpuPassBegin(VkCommandBuffer cmdbuf, const GpuPassLabel& label, ::Eden::GpuPass::Counts counts,
                  u32 load_clears) {
    if (!GpuPassActive()) return;
    if (gpu_pass.in_pass) { // never expected: the scheduler ends a pass before it begins one
        ++gpu_pass.in_pass_marks;
        return;
    }
    auto& outside = gpu_pass.buckets[gpu_pass.outside_kind];
    outside.draws += counts.draws;
    outside.clears += counts.clears;
    GpuPassStamp(cmdbuf, gpu_pass.outside_kind);
    gpu_pass.pass_bucket = GpuPassBucket(label);
    ++gpu_pass.buckets[gpu_pass.pass_bucket].segments;
    gpu_pass.buckets[gpu_pass.pass_bucket].clears += load_clears;
    gpu_pass.in_pass = true;
}

void GpuPassEnd(VkCommandBuffer cmdbuf, ::Eden::GpuPass::Counts counts) {
    if (!GpuPassActive() || !gpu_pass.in_pass) return;
    auto& pass = gpu_pass.buckets[gpu_pass.pass_bucket];
    pass.draws += counts.draws;
    pass.clears += counts.clears;
    gpu_pass.in_pass = false;
    GpuPassStamp(cmdbuf, gpu_pass.pass_bucket);
}

// GpuTimeBegin: the submission's first stamp (top of its upload buffer) and the one reserved for
// the upload buffer's end.
void GpuPassSubmitBegin(const Device& device, vk::CommandBuffer upload_cmdbuf) {
    gpu_pass.open = nullptr;
    auto& submission = gpu_pass.submissions[gpu_time.current];
    submission = {};
    if (!::Eden::GpuPass::on || gpu_pass.failed) return;
    if (!gpu_pass.ready) {
        if (!device.IsHostQueryResetSupported()) { GpuPassFail("no_host_query_reset"); return; }
        gpu_pass.pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = GpuPassProbe::STAMPS,
            .pipelineStatistics = 0,
        });
        device.GetLogical().ResetQueryPool(*gpu_pass.pool, 0, GpuPassProbe::STAMPS);
        gpu_pass.scratch.resize(GpuPassProbe::STAMPS);
        gpu_pass.presents_at_report = ::Eden::GpuPass::presents.load(std::memory_order_relaxed);
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::Other].name, 48, "other");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::UploadBuffer].name, 48, "upload_buffer");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::UploadImage].name, 48, "upload_image");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::Astc].name, 48, "astc_decode");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::Copy].name, 48, "copy_blit");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::Download].name, 48, "download");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::Compute].name, 48, "compute");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::TargetSmall].name, 48, "rt_under_64mib");
        std::snprintf(gpu_pass.buckets[::Eden::GpuPass::TargetOverflow].name, 48, "rt_big_overflow");
        gpu_pass.ready = true;
        std::printf("EDEN_GPU_PASS_INIT stamps=%u buckets=%u big_mib=64\n", GpuPassProbe::STAMPS,
                    GpuPassProbe::BUCKETS);
    }
    if (gpu_pass.next + 2 - gpu_pass.first_pending > GpuPassProbe::STAMPS) {
        GpuPassFail("ring");
        return;
    }
    const VkCommandBuffer cmd = *upload_cmdbuf;
    submission.first = gpu_pass.next;
    submission.upload_end = gpu_pass.next + 1;
    submission.count = 2;
    const u32 start = static_cast<u32>(submission.first % GpuPassProbe::STAMPS);
    const u32 upload_end = static_cast<u32>(submission.upload_end % GpuPassProbe::STAMPS);
    gpu_pass.next += 2;
    gpu_pass.bucket_of[start] = GpuPassProbe::NO_BUCKET;
    gpu_pass.bucket_of[upload_end] = ::Eden::GpuPass::UploadBuffer;
    gpu_time.write(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *gpu_pass.pool, start);
    gpu_pass.in_pass = false;
    gpu_pass.outside_kind = ::Eden::GpuPass::Other;
    gpu_pass.open = &submission;
}

// Right before the upload buffer ends (SubmitExecution).
void GpuPassUploadEnd(vk::CommandBuffer upload_cmdbuf) {
    if (!GpuPassActive()) return;
    gpu_time.write(*upload_cmdbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *gpu_pass.pool,
                   static_cast<u32>(gpu_pass.open->upload_end % GpuPassProbe::STAMPS));
}

// GpuTimeEnd: the last stamp closes whatever ran since the last mark.
void GpuPassSubmitEnd(vk::CommandBuffer cmdbuf) {
    if (!GpuPassActive()) return;
    if (!gpu_pass.in_pass) GpuPassStamp(*cmdbuf, gpu_pass.outside_kind);
    gpu_pass.open = nullptr;
}

// The stamps of one submission whose GpuTimeProbe stamps are ready. False: not ready yet.
bool GpuPassCollect(const Device& device, u32 slot) {
    if (!gpu_pass.ready) return true;
    auto& submission = gpu_pass.submissions[slot];
    if (submission.count == 0 || gpu_pass.failed) {
        if (submission.count) gpu_pass.first_pending = submission.first + submission.count;
        submission.count = 0;
        return true;
    }
    const u32 count = submission.count;
    const u32 begin = static_cast<u32>(submission.first % GpuPassProbe::STAMPS);
    const u32 head = (std::min)(count, GpuPassProbe::STAMPS - begin);
    if (device.GetLogical().GetQueryResults(*gpu_pass.pool, begin, head, head * sizeof(u64),
                                            gpu_pass.scratch.data(), sizeof(u64),
                                            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return false;
    if (head < count && device.GetLogical().GetQueryResults(*gpu_pass.pool, 0, count - head,
                                                            (count - head) * sizeof(u64),
                                                            gpu_pass.scratch.data() + head, sizeof(u64),
                                                            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return false;
    // Read: the GPU is done with these queries; reset them for their next turn round the ring.
    device.GetLogical().ResetQueryPool(*gpu_pass.pool, begin, head);
    if (head < count) device.GetLogical().ResetQueryPool(*gpu_pass.pool, 0, count - head);
    const u64 mask = gpu_time.valid_bits == 64 ? ~u64{0} : (u64{1} << gpu_time.valid_bits) - 1;
    for (u32 i = 1; i < count; ++i) {
        const u8 bucket = gpu_pass.bucket_of[(submission.first + i) % GpuPassProbe::STAMPS];
        const u64 delta = (gpu_pass.scratch[i] - gpu_pass.scratch[i - 1]) & mask;
        const u64 ns = static_cast<u64>(double(delta) * gpu_time.period_ns);
        if (delta > (mask >> 1) || ns > 10'000'000'000ULL) { // out of order: an overlap or a wrap
            ++gpu_pass.nonmonotonic;
            continue;
        }
        if (bucket < GpuPassProbe::BUCKETS) gpu_pass.buckets[bucket].ns += ns;
        else gpu_pass.lost_ns += ns;
    }
    gpu_pass.stamps += count;
    gpu_pass.first_pending = submission.first + count;
    submission.count = 0;
    return true;
}

void GpuPassReport(double wall_ms, u64 busy_ns) {
    if (!gpu_pass.ready) return;
    const u64 presents = ::Eden::GpuPass::presents.load(std::memory_order_relaxed);
    const u64 frames = presents - gpu_pass.presents_at_report;
    gpu_pass.presents_at_report = presents;
    u64 measured = gpu_pass.lost_ns;
    std::array<u8, GpuPassProbe::BUCKETS> order{};
    for (u32 i = 0; i < gpu_pass.used; ++i) {
        order[i] = static_cast<u8>(i);
        measured += gpu_pass.buckets[i].ns;
    }
    std::sort(order.begin(), order.begin() + gpu_pass.used,
              [](u8 a, u8 b) { return gpu_pass.buckets[a].ns > gpu_pass.buckets[b].ns; });
    const double per = frames ? 1.0 / double(frames) : 0.0;
    std::printf("EDEN_GPU_PASS wall_ms=%.1f frames=%llu gpu_ms=%.1f busy_ms=%.1f stamps=%llu "
                "out_of_order=%llu marks_in_pass=%llu failed=%d\n",
                wall_ms, static_cast<unsigned long long>(frames), measured / 1e6, busy_ns / 1e6,
                static_cast<unsigned long long>(gpu_pass.stamps),
                static_cast<unsigned long long>(gpu_pass.nonmonotonic),
                static_cast<unsigned long long>(gpu_pass.in_pass_marks), int(gpu_pass.failed));
    for (u32 n = 0; n < gpu_pass.used; ++n) {
        auto& bucket = gpu_pass.buckets[order[n]];
        if (bucket.ns == 0 && bucket.segments == 0 && bucket.draws == 0 && bucket.clears == 0) continue;
        std::printf("EDEN_GPU_PASS_BUCKET %s mib=%llu gpu_ms=%.1f per_frame_ms=%.2f share=%.1f%% "
                    "passes=%llu per_frame=%.2f draws=%llu per_frame=%.1f clears=%llu per_frame=%.2f\n",
                    bucket.name, static_cast<unsigned long long>(bucket.bytes >> 20), bucket.ns / 1e6,
                    bucket.ns / 1e6 * per, measured ? 100.0 * double(bucket.ns) / double(measured) : 0.0,
                    static_cast<unsigned long long>(bucket.segments), double(bucket.segments) * per,
                    static_cast<unsigned long long>(bucket.draws), double(bucket.draws) * per,
                    static_cast<unsigned long long>(bucket.clears), double(bucket.clears) * per);
        bucket.ns = bucket.segments = bucket.draws = bucket.clears = 0;
    }
    gpu_pass.stamps = gpu_pass.nonmonotonic = gpu_pass.in_pass_marks = gpu_pass.lost_ns = 0;
}
#endif // EDEN_GPU_TIME_PROBE_CHECK

void GpuTimeCountOrigin(GpuSubmitOrigin origin) {
    if (!::Eden::DevVulkan::gpu_time || gpu_time.failed) return;
    for (u32 i = 0; i < gpu_time.origin_slots; ++i) {
        auto& row = gpu_time.origins[i];
        if (row.origin.caller == origin.caller && row.origin.finish == origin.finish) {
            ++row.count;
            return;
        }
    }
    if (gpu_time.origin_slots == gpu_time.origins.size()) { ++gpu_time.origin_overflow; return; }
    gpu_time.origins[gpu_time.origin_slots++] = {origin, 1};
}

void GpuTimeBegin(const Device& device, vk::CommandBuffer cmdbuf) {
    if (!::Eden::DevVulkan::gpu_time || gpu_time.failed) return;
    if (!gpu_time.write) {
        const auto& dld = device.GetDispatchLoader();
        gpu_time.write = reinterpret_cast<PFN_vkCmdWriteTimestamp>(
            dld.vkGetDeviceProcAddr(*device.GetLogical(), "vkCmdWriteTimestamp"));
        gpu_time.period_ns = device.GetPhysical().GetProperties().limits.timestampPeriod;
        const auto families = device.GetPhysical().GetQueueFamilyProperties();
        if (device.GetGraphicsFamily() < families.size())
            gpu_time.valid_bits = families[device.GetGraphicsFamily()].timestampValidBits;
        if (!gpu_time.write || gpu_time.period_ns <= 0 || !gpu_time.valid_bits || gpu_time.valid_bits > 64) { gpu_time.failed = true; return; }
        gpu_time.pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = GpuTimeProbe::SLOTS * 2,
            .pipelineStatistics = 0,
        });
        gpu_time.report = std::chrono::steady_clock::now();
        std::printf("EDEN_GPU_TIME_INIT period_ns=%.9f valid_bits=%u slots=%u\n",
                    gpu_time.period_ns, gpu_time.valid_bits, GpuTimeProbe::SLOTS);
        std::printf("EDEN_GPU_SUBMIT_ANCHOR address=%p coverage=upload+graphics\n",
                    reinterpret_cast<void*>(&GpuTimeCountOrigin));
    }
    if (gpu_time.next - gpu_time.first_pending >= GpuTimeProbe::SLOTS) {
        gpu_time.failed = true; // results stopped arriving; never overwrite a pending slot
        std::printf("EDEN_GPU_TIME failed=ring\n");
        return;
    }
    gpu_time.current = static_cast<u32>(gpu_time.next++ % GpuTimeProbe::SLOTS);
    cmdbuf.ResetQueryPool(*gpu_time.pool, gpu_time.current * 2, 2);
    gpu_time.write(*cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *gpu_time.pool, gpu_time.current * 2);
#ifndef EDEN_GPU_TIME_PROBE_CHECK
    GpuPassSubmitBegin(device, cmdbuf);
#endif
}

void GpuTimeEnd(vk::CommandBuffer cmdbuf) {
    if (!::Eden::DevVulkan::gpu_time || gpu_time.failed || !gpu_time.write) return;
#ifndef EDEN_GPU_TIME_PROBE_CHECK
    GpuPassSubmitEnd(cmdbuf);
#endif
    gpu_time.write(*cmdbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *gpu_time.pool, gpu_time.current * 2 + 1);
}

void GpuTimeCollect(const Device& device) {
    if (!::Eden::DevVulkan::gpu_time || gpu_time.failed || !gpu_time.write) return;
    while (gpu_time.first_pending + 1 < gpu_time.next) {
        const u32 slot = static_cast<u32>(gpu_time.first_pending % GpuTimeProbe::SLOTS);
        std::array<u64, 2> stamps{};
        if (device.GetLogical().GetQueryResults(*gpu_time.pool, slot * 2, 2, sizeof(stamps), stamps.data(),
                                               sizeof(u64), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) break;
#ifndef EDEN_GPU_TIME_PROBE_CHECK
        if (!GpuPassCollect(device, slot)) break; // both sets of one submission, or neither
#endif
        const u64 mask = gpu_time.valid_bits == 64 ? ~u64{0} : (u64{1} << gpu_time.valid_bits) - 1;
        const u64 ns = static_cast<u64>(double((stamps[1] - stamps[0]) & mask) * gpu_time.period_ns);
        gpu_time.busy_ns += ns;
        gpu_time.total_ns += ns;
        ++gpu_time.total_submissions;
        gpu_time.max_ns = (std::max)(gpu_time.max_ns, ns);
        ++gpu_time.submissions;
        ++gpu_time.first_pending;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(now - gpu_time.report).count();
    if (wall >= 5'000'000'000) {
        std::printf("EDEN_GPU_TIME wall_ms=%.1f busy_ms=%.1f submissions=%llu max_ms=%.2f cumulative_ns=%llu cumulative_submissions=%llu pending=%llu\n", wall / 1e6,
                    gpu_time.busy_ns / 1e6, static_cast<unsigned long long>(gpu_time.submissions),
                    gpu_time.max_ns / 1e6, static_cast<unsigned long long>(gpu_time.total_ns),
                    static_cast<unsigned long long>(gpu_time.total_submissions),
                    static_cast<unsigned long long>(gpu_time.next - gpu_time.first_pending));
#ifndef EDEN_GPU_TIME_PROBE_CHECK
        GpuPassReport(wall / 1e6, gpu_time.busy_ns);
#endif
        gpu_time.busy_ns = gpu_time.max_ns = gpu_time.submissions = 0;
        u64 origin_total = gpu_time.origin_overflow;
        for (u32 i = 0; i < gpu_time.origin_slots; ++i) {
            auto& row = gpu_time.origins[i];
            origin_total += row.count;
            if (row.count) std::printf("EDEN_GPU_SUBMIT_ORIGIN caller=%p finish=%d count=%llu\n",
                row.origin.caller, int(row.origin.finish), static_cast<unsigned long long>(row.count));
            row.count = 0;
        }
        std::printf("EDEN_GPU_SUBMIT_TOTAL count=%llu overflow=%llu\n",
                    static_cast<unsigned long long>(origin_total),
                    static_cast<unsigned long long>(gpu_time.origin_overflow));
        gpu_time.origin_overflow = 0;
        gpu_time.report = now;
    }
}
void GpuTimeReset(const Device& device) {
    if (gpu_time.pool) {
        const auto result = device.GetDispatchLoader().vkDeviceWaitIdle(*device.GetLogical());
        if (result != VK_SUCCESS) std::fprintf(stderr, "EDEN_GPU_TIME reset_idle_result=%d\n", int(result));
    }
    // Called only after the scheduler worker joined, while its device is alive.
    gpu_time = {};
#ifndef EDEN_GPU_TIME_PROBE_CHECK
    gpu_pass.pool = {};
    gpu_pass.ready = gpu_pass.failed = false;
    gpu_pass.open = nullptr;
    gpu_pass.next = gpu_pass.first_pending = 0;
#endif
}
} // namespace
