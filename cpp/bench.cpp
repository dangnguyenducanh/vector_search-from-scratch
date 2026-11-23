// Pure C++ benchmark of the HNSW kernels: no Python anywhere in the timed code.
//   python cpp_benchmark.py --export    # once: data/{docs,queries,levels,truth}.bin
//   make -C cpp bench && ./cpp/hnsw_bench data [reps]
// Builds the index with the same layer draws as HNSWIndex(seed=0), then times
// 500 single queries per ef; prints recall@10 and median/best µs per query.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hnsw.cpp"

template <class T>
static std::vector<T> load(const std::string& path, int& rows, int& cols) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(&rows, 4, 1, f) != 1 || std::fread(&cols, 4, 1, f) != 1) {
        std::fprintf(stderr, "can't read %s (run: python cpp_benchmark.py --export)\n", path.c_str());
        std::exit(1);
    }
    std::vector<T> v((size_t)rows * cols);
    if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) {
        std::fprintf(stderr, "%s: short file\n", path.c_str());
        std::exit(1);
    }
    std::fclose(f);
    return v;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const int M = 16, M0 = 32, EFC = 100, K = 10;
    int n, dim, nq, qdim, nl, one, nt, k;
    auto docs = load<float>(dir + "/docs.bin", n, dim);
    auto queries = load<float>(dir + "/queries.bin", nq, qdim);
    auto levels = load<i32>(dir + "/levels.bin", nl, one);
    auto truth = load<i32>(dir + "/truth.bin", nt, k);
    if (qdim != dim || nl != n || nt != nq || k != K) return std::fprintf(stderr, "shape mismatch\n"), 1;

    // graph arrays, laid out exactly as HNSWIndex lays them out
    int depth = std::max(1, *std::max_element(levels.begin(), levels.end()));
    std::vector<i32> slot(n, -1);
    int n_up = 0;
    for (int i = 0; i < n; ++i)
        if (levels[i] > 0) slot[i] = n_up++;
    std::vector<i32> l0((size_t)n * M0), c0(n), up((size_t)std::max(n_up, 1) * depth * M), cu((size_t)std::max(n_up, 1) * depth);
    std::vector<i64> visited(n, 0);
    std::vector<uint8_t> dead(n, 0);
    Graph g{docs.data(), l0.data(), c0.data(), up.data(), cu.data(), slot.data(), levels.data(),
            visited.data(), dead.data(), dim, M, M0, depth};

    i32 entry = -1;
    double t = now();
    i64 tag = hnsw_insert(&g, 0, &entry, 0, n, EFC);
    std::printf("%d docs x %d, M=%d efC=%d, %d queries, 1 thread\nbuild: %.1f s\n", n, dim, M, EFC, nq, now() - t);
    std::printf("%4s %7s %11s %9s\n", "ef", "recall", "median µs", "best µs");

    std::vector<i32> ids(K);
    std::vector<float> sc(K);
    for (int ef : {16, 32, 64, 128}) {
        std::vector<double> us;
        int hits = 0;
        for (int r = 0; r < reps; ++r) {
            hits = 0;
            t = now();
            for (int q = 0; q < nq; ++q) {
                int got = hnsw_query(&g, ++tag, entry, &queries[(size_t)q * dim], K, ef, nullptr, ids.data(), sc.data());
                const i32* tr = &truth[(size_t)q * K];
                for (int i = 0; i < got; ++i) hits += std::count(tr, tr + K, ids[i]);
            }
            us.push_back(1e6 * (now() - t) / nq);
        }
        std::sort(us.begin(), us.end());
        std::printf("%4d %7.3f %11.1f %9.1f\n", ef, hits / double(K * nq), us[us.size() / 2], us[0]);
    }
}
