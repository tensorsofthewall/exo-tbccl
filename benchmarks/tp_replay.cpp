// Phase 67: exact-size collective latency and trace-driven TP2 cadence replay over the installed TBCCL (Host memory, process per rank, two hosts or loopback).
// A consumer built OUTSIDE the TBCCL tree against its public C++ header and static library; nothing in TBCCL is changed.
//
//   tp_replay --rank R --peers ip0:port0,ip1:port1 --mode micro  [--sizes 1024,2048,...] [--ops p2p,allgather,allreduce] [--dtype bf16|f32]
//                                                                [--gaps 0,200] [--gap-mode spin|sleep] [--iters N] [--warmup W] [--label NAME]
//   tp_replay --rank R --peers ... --mode replay --profile FILE [--passes N] [--gap-mode spin|sleep] [--label NAME]
//
// micro: per (op, size, gap) the latency of the operation itself (post -> wait returns), excluding the application gap that precedes each iteration (the gap is spent
//   by spinning, like a CUDA synchronize, or sleeping, like an idle thread; 0 = back to back). p2p is the ping-pong round trip (rank 0 send+wait then recv+wait), reported
//   as the round trip; allgather is the N=2 AllGather of `bytes` per rank; allreduce is the in-place N=2 SUM (bf16 by default: bytes/2 elements).
// replay: this rank's profile {"rank":R,"ops":[{"op":"allreduce","bytes":2048,"gap_us":120},{"op":"all_gather","bytes":8,"gap_us":300}...]} is executed in order per
//   pass (one pass = one decode token): spin/sleep the recorded gap (the shard compute), then issue the collective and wait. Each rank replays its own file, so each
//   rank's wait for the other rank's compute reproduces itself. Reports the pass wall time per token and the share not spent in gaps.

#include <tbccl/communicator.hpp>

#include <sys/resource.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace
{
double g_sleep_ratio = 1.0;
bool g_spin = true;

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

double cpu_seconds()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

double measure_sleep_ratio()
{
    double total = 0;
    for (int i = 0; i < 15; ++i)
    {
        const auto t0 = Clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(2000));
        total += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }
    const double ratio = total / 15 / 2000.0;
    return ratio > 1.15 ? ratio : 1.0;
}

void gap(long us)
{
    if (us <= 0) return;
    if (g_spin)
    {
        const auto deadline = Clock::now() + std::chrono::microseconds(us);
        while (Clock::now() < deadline) {}
        return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(static_cast<double>(us) / g_sleep_ratio)));
}

struct ProfileOp
{
    std::string op;
    std::size_t bytes = 0;
    long gap_us = 0;
};

std::vector<ProfileOp> read_profile(const std::string &path)
{
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<ProfileOp> ops;
    for (std::size_t at = text.find("{\"op\":\""); at != std::string::npos; at = text.find("{\"op\":\"", at + 1))
    {
        ProfileOp p;
        const std::size_t name = at + 7, end = text.find('"', name);
        p.op = text.substr(name, end - name);
        p.bytes = std::strtoull(text.c_str() + text.find("\"bytes\":", end) + 8, nullptr, 10);
        p.gap_us = std::atol(text.c_str() + text.find("\"gap_us\":", end) + 9);
        ops.push_back(p);
    }
    return ops;
}

void report(const std::string &label, int rank, const char *op, std::size_t bytes, long gap_us, std::vector<double> &us, double cores)
{
    std::sort(us.begin(), us.end());
    const auto at = [&](double q) { return us[std::min(us.size() - 1, static_cast<std::size_t>(q * us.size()))]; };
    std::printf("{\"label\":\"%s\",\"rank\":%d,\"op\":\"%s\",\"bytes\":%zu,\"gap_us\":%ld,\"gap_mode\":\"%s\",\"median_us\":%.2f,\"p25_us\":%.2f,\"p75_us\":%.2f,\"p95_us\":%.2f,\"n\":%zu,\"cpu_cores\":%.2f}\n",
                label.c_str(), rank, op, bytes, gap_us, g_spin ? "spin" : "sleep", at(0.5), at(0.25), at(0.75), at(0.95), us.size(), cores);
    std::fflush(stdout);
}
} // namespace

int main(int argc, char **argv)
{
    std::size_t rank = 0;
    std::string peers_arg, mode = "micro", sizes_arg = "2048", ops_arg = "p2p,allgather,allreduce", gaps_arg = "0", label = "tp", profile, dtype = "bf16";
    int iters = 100, warmup = 10, passes = 50;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--rank") rank = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--peers") peers_arg = next();
        else if (a == "--mode") mode = next();
        else if (a == "--sizes") sizes_arg = next();
        else if (a == "--ops") ops_arg = next();
        else if (a == "--gaps") gaps_arg = next();
        else if (a == "--gap-mode") g_spin = next() != "sleep";
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--passes") passes = std::atoi(next().c_str());
        else if (a == "--label") label = next();
        else if (a == "--profile") profile = next();
        else if (a == "--dtype") dtype = next();
    }
    tbccl::CommunicatorOptions o;
    o.rank = rank;
    for (const auto &p : split(peers_arg, ','))
    {
        const auto colon = p.rfind(':');
        o.peers.push_back({p.substr(0, colon), static_cast<std::uint16_t>(std::atoi(p.c_str() + colon + 1))});
    }
    if (o.peers.size() != 2)
    {
        std::fprintf(stderr, "need --peers ip0:port0,ip1:port1\n");
        return 2;
    }
    o.bootstrap_timeout = std::chrono::milliseconds(60000);
    if (!g_spin) g_sleep_ratio = measure_sleep_ratio();
    auto comm = tbccl::Communicator::create(o);
    const std::size_t peer = 1 - rank;
    const bool leader = rank == 0;
    const tbccl::DataType dt = dtype == "f32" ? tbccl::DataType::Float32 : tbccl::DataType::BFloat16;
    const std::size_t esize = dtype == "f32" ? 4 : 2;
    const auto view = [](std::vector<std::uint8_t> &v, std::size_t n) { return tbccl::BufferView{tbccl::MemoryKind::Host, v.data(), n, -1}; };

    if (mode == "replay")
    {
        const auto ops = read_profile(profile);
        if (ops.empty())
        {
            std::fprintf(stderr, "empty or unreadable profile %s\n", profile.c_str());
            return 2;
        }
        std::size_t max_bytes = 8, tokens = 0;
        long gap_total = 0;
        for (const auto &p : ops)
        {
            max_bytes = std::max(max_bytes, p.bytes);
            gap_total += p.gap_us;
            if (p.op == "all_gather") ++tokens;
        }
        if (tokens == 0) tokens = 1;
        std::vector<std::uint8_t> buf(max_bytes, 0x80), g0(max_bytes), g1(max_bytes);
        const auto play = [&](const ProfileOp &p) {
            gap(p.gap_us);
            if (p.op == "allreduce") comm->all_reduce(view(buf, p.bytes), view(buf, p.bytes), p.bytes / esize, dt, tbccl::ReduceOp::Sum).wait();
            else if (p.op == "all_gather") comm->all_gather(view(buf, p.bytes), {view(g0, p.bytes), view(g1, p.bytes)}).wait();
            else if (p.op == "barrier") comm->barrier().wait();
        };
        comm->barrier().wait();
        for (int w = 0; w < std::max(2, passes / 10); ++w)
            for (const auto &p : ops) play(p);
        comm->barrier().wait();
        std::vector<double> wall_us;
        const double cpu0 = cpu_seconds();
        const auto run0 = Clock::now();
        for (int i = 0; i < passes; ++i)
        {
            const auto t0 = Clock::now();
            for (const auto &p : ops) play(p);
            wall_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / static_cast<double>(tokens));
        }
        const double cores = (cpu_seconds() - cpu0) / std::max(std::chrono::duration<double>(Clock::now() - run0).count(), 1e-9);
        comm->barrier().wait();
        std::sort(wall_us.begin(), wall_us.end());
        const auto at = [&](double q) { return wall_us[std::min(wall_us.size() - 1, static_cast<std::size_t>(q * wall_us.size()))]; };
        std::printf("{\"label\":\"%s\",\"rank\":%zu,\"mode\":\"replay\",\"ops_per_token\":%zu,\"tokens_per_pass\":%zu,\"gap_total_us_per_token\":%.1f,\"gap_mode\":\"%s\","
                    "\"token_wall_median_us\":%.1f,\"p25_us\":%.1f,\"p75_us\":%.1f,\"p95_us\":%.1f,\"passes\":%d,\"cpu_cores\":%.2f}\n",
                    label.c_str(), rank, ops.size() / tokens, tokens, static_cast<double>(gap_total) / static_cast<double>(tokens), g_spin ? "spin" : "sleep", at(0.5), at(0.25), at(0.75),
                    at(0.95), passes, cores);
        return 0;
    }

    for (const auto &op : split(ops_arg, ','))
        for (const auto &size_text : split(sizes_arg, ','))
            for (const auto &gap_text : split(gaps_arg, ','))
            {
                const std::size_t bytes = std::strtoull(size_text.c_str(), nullptr, 10);
                const long gap_us = std::atol(gap_text.c_str());
                std::vector<std::uint8_t> a(bytes, 0x80), in(bytes), g0(bytes), g1(bytes);
                const auto one = [&]() {
                    if (op == "p2p")
                    {
                        if (leader)
                        {
                            comm->send(view(a, bytes), bytes, tbccl::DataType::UInt8, peer).wait();
                            comm->recv(view(in, bytes), bytes, tbccl::DataType::UInt8, peer).wait();
                        }
                        else
                        {
                            comm->recv(view(in, bytes), bytes, tbccl::DataType::UInt8, peer).wait();
                            comm->send(view(in, bytes), bytes, tbccl::DataType::UInt8, peer).wait();
                        }
                    }
                    else if (op == "allgather") comm->all_gather(view(a, bytes), {view(g0, bytes), view(g1, bytes)}).wait();
                    else if (op == "allreduce") comm->all_reduce(view(a, bytes), view(a, bytes), bytes / esize, dt, tbccl::ReduceOp::Sum).wait();
                };
                comm->barrier().wait();
                for (int i = 0; i < warmup; ++i) { gap(gap_us); one(); }
                comm->barrier().wait();
                std::vector<double> us;
                const double cpu0 = cpu_seconds();
                const auto wall0 = Clock::now();
                for (int i = 0; i < iters; ++i)
                {
                    gap(gap_us);
                    const auto t0 = Clock::now();
                    one();
                    us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
                }
                const double cores = (cpu_seconds() - cpu0) / std::max(std::chrono::duration<double>(Clock::now() - wall0).count(), 1e-9);
                comm->barrier().wait();
                report(label, static_cast<int>(rank), op.c_str(), bytes, gap_us, us, cores);
            }
    return 0;
}
