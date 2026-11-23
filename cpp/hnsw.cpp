// HNSW graph kernels in C++: a port of the @njit functions in hnsw.py, same
// algorithm, same graph layout. They work in place on the NumPy arrays that
// HNSWIndex owns and are called through ctypes from hnsw.py (backend="cpp"). No bounds checks:
// HNSWIndex._check() validates loaded files, as for the Numba kernels.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

using i32 = int32_t;
using i64 = int64_t;

// Field order must match _Graph in hnsw.py.
struct Graph {
    float* vecs;              // [n, dim], L2-normalized
    i32 *l0, *c0;             // layer 0: l0[node, :c0[node]], row width M0
    i32 *up, *cu;             // upper layers: up[slot, level-1, :cu[slot, level-1]], row width M
    i32 *slot, *levels;
    i64* visited;             // visited[x] == tag  <=>  seen in this search
    uint8_t* dead;
    i64 dim, M, M0, depth;
};

struct Span {
    i32* p;
    int n;
};

static inline const float* vec(const Graph& g, i64 x) { return g.vecs + x * g.dim; }

static inline Span links(const Graph& g, i64 node, int level) {
    if (level == 0) return {g.l0 + node * g.M0, g.c0[node]};
    i64 r = (i64)g.slot[node] * g.depth + level - 1;
    return {g.up + r * g.M, g.cu[r]};
}

static inline void set_links(const Graph& g, i64 node, int level, const i32* ids, int n) {
    if (level == 0) {
        if (n) std::memmove(g.l0 + node * g.M0, ids, n * sizeof(i32));
        g.c0[node] = n;
    } else {
        i64 r = (i64)g.slot[node] * g.depth + level - 1;
        if (n) std::memmove(g.up + r * g.M, ids, n * sizeof(i32));
        g.cu[r] = n;
    }
}

static inline bool has(Span s, i32 x) {
    for (int i = 0; i < s.n; ++i)
        if (s.p[i] == x) return true;
    return false;
}

static inline float dot(const float* a, const float* b, i64 d) {
    i64 i = 0;
    float r = 0.0f;
#ifdef __AVX2__
    // 4 accumulators: one FMA chain is latency-bound at 384 dims
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 32 <= d; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= d; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    r = _mm_cvtss_f32(h);
#endif
    for (; i < d; ++i) r += a[i] * b[i];
    return r;
}

static inline void prefetch(const Graph& g, i32 x) {
    const char* p = reinterpret_cast<const char*>(vec(g, x));
    __builtin_prefetch(p);
    __builtin_prefetch(p + 64);  // the hardware streamer picks up the rest
}

// (score, id) pairs compared like Python tuples, so ties break exactly as heapq does.
using P = std::pair<float, i32>;
static const std::greater<P> MINHEAP{};

// --- search primitives -----------------------------------------------------
// Beam search on one layer -> up to ef (ids, scores), best first. allow: nodes
// that may be returned (nullptr = all); other nodes are still walked through.
static void search_layer(const Graph& g, const float* q, const i32* eps, int neps, int ef, int level,
                         i64 tag, const uint8_t* allow, std::vector<i32>& ids, std::vector<float>& scores) {
    static thread_local std::vector<P> cand, found;  // cand: min-heap on -score; found: min-heap on score
    cand.clear();
    found.clear();
    for (int i = 0; i < neps; ++i) {
        i32 e = eps[i];
        g.visited[e] = tag;
        float s = dot(vec(g, e), q, g.dim);
        cand.push_back({-s, e});
        std::push_heap(cand.begin(), cand.end(), MINHEAP);
        if (!allow || allow[e]) {
            found.push_back({s, e});
            std::push_heap(found.begin(), found.end(), MINHEAP);
            if ((int)found.size() > ef) {
                std::pop_heap(found.begin(), found.end(), MINHEAP);
                found.pop_back();
            }
        }
    }
    while (!cand.empty()) {
        std::pop_heap(cand.begin(), cand.end(), MINHEAP);
        P c = cand.back();
        cand.pop_back();
        if ((int)found.size() >= ef && -c.first < found.front().first) break;  // best unexplored is worse than worst kept
        Span L = links(g, c.second, level);
        for (int i = 0; i < L.n; ++i)
            if (g.visited[L.p[i]] != tag) prefetch(g, L.p[i]);
        for (int i = 0; i < L.n; ++i) {
            i32 nb = L.p[i];
            if (g.visited[nb] == tag) continue;
            g.visited[nb] = tag;
            float s = dot(vec(g, nb), q, g.dim);
            if ((int)found.size() < ef || s > found.front().first) {
                cand.push_back({-s, nb});
                std::push_heap(cand.begin(), cand.end(), MINHEAP);
                if (!allow || allow[nb]) {
                    found.push_back({s, nb});
                    std::push_heap(found.begin(), found.end(), MINHEAP);
                    if ((int)found.size() > ef) {
                        std::pop_heap(found.begin(), found.end(), MINHEAP);
                        found.pop_back();
                    }
                }
            }
        }
    }
    int n = (int)found.size();
    ids.resize(n);
    scores.resize(n);
    for (int i = n - 1; i >= 0; --i) {  // min-heap pops worst first
        std::pop_heap(found.begin(), found.end(), MINHEAP);
        scores[i] = found.back().first;
        ids[i] = found.back().second;
        found.pop_back();
    }
}

// Greedy walk from the top layer down to (but not into) to_level.
static i32 descend(const Graph& g, const float* q, i32 entry, int to_level) {
    i32 ep = entry;
    float best = dot(vec(g, entry), q, g.dim);
    for (int level = g.levels[entry]; level > to_level; --level) {
        bool moved = true;
        while (moved) {
            moved = false;
            Span L = links(g, ep, level);  // fixed for this pass, as in the Numba loop
            for (int i = 0; i < L.n; ++i) {
                float s = dot(vec(g, L.p[i]), q, g.dim);
                if (s > best) {
                    ep = L.p[i];
                    best = s;
                    moved = true;
                }
            }
        }
    }
    return ep;
}

// Diversity heuristic (paper Alg. 4); ids/scores best-first, scores = similarity
// to the base. Writes up to m ids to kept (must not alias ids), returns the count.
static int select_nbrs(const Graph& g, const i32* ids, const float* scores, int n, int m, i32* kept) {
    if (n <= m) {
        if (n) std::memcpy(kept, ids, n * sizeof(i32));
        return n;
    }
    int k = 0;
    for (int i = 0; i < n && k < m; ++i) {
        bool ok = true;
        for (int j = 0; j < k; ++j)
            if (dot(vec(g, ids[i]), vec(g, kept[j]), g.dim) > scores[i]) {
                ok = false;
                break;
            }
        if (ok) kept[k++] = ids[i];
    }
    return k;
}

// Sort cand by similarity to base (best first), then run the heuristic into out.
static void prune(const Graph& g, i32 base, const std::vector<i32>& cand, int cap, std::vector<i32>& out) {
    static thread_local std::vector<P> tmp;
    static thread_local std::vector<i32> ids;
    static thread_local std::vector<float> sc;
    int m = (int)cand.size();
    tmp.resize(m);
    for (int i = 0; i < m; ++i) tmp[i] = {-dot(vec(g, cand[i]), vec(g, base), g.dim), cand[i]};
    std::sort(tmp.begin(), tmp.end());
    ids.resize(m);
    sc.resize(m);
    for (int i = 0; i < m; ++i) {
        ids[i] = tmp[i].second;
        sc[i] = -tmp[i].first;
    }
    out.resize(std::max(m, 1));
    out.resize(select_nbrs(g, ids.data(), sc.data(), m, cap, out.data()));
}

// Link node to the selected candidates and link them back. A neighbor list over
// cap is shrunk with the same heuristic, or with shrink=false left as it is.
static void connect(const Graph& g, i32 node, int level, const std::vector<i32>& ids,
                    const std::vector<float>& scores, int M, int cap, bool shrink) {
    static thread_local std::vector<i32> nbrs, cand, out;
    nbrs.resize(std::max<size_t>(ids.size(), 1));
    int k = select_nbrs(g, ids.data(), scores.data(), (int)ids.size(), M, nbrs.data());
    set_links(g, node, level, nbrs.data(), k);
    for (int t = 0; t < k; ++t) {
        i32 nb = nbrs[t];
        Span cur = links(g, nb, level);
        if (has(cur, node)) continue;
        int m = cur.n;
        if (m < cap) {  // room in the row: append in place
            cur.p[m] = node;
            set_links(g, nb, level, cur.p, m + 1);
            continue;
        }
        if (!shrink) continue;
        cand.assign(cur.p, cur.p + m);
        cand.push_back(node);
        prune(g, nb, cand, cap, out);
        set_links(g, nb, level, out.data(), (int)out.size());
    }
}

// Re-insert u on one layer, searching from seed; u is pre-marked visited so the
// walk never enters or returns it. Returns tag.
static i64 relink(const Graph& g, i64 tag, i32 u, i32 seed, int level, int efc, const uint8_t* allow, bool shrink) {
    static thread_local std::vector<i32> ids;
    static thread_local std::vector<float> sc;
    tag += 1;
    g.visited[u] = tag;
    search_layer(g, vec(g, u), &seed, 1, efc, level, tag, allow, ids, sc);
    connect(g, u, level, ids, sc, (int)g.M, level == 0 ? (int)g.M0 : (int)g.M, shrink);
    return tag;
}

extern "C" {

// Insert rows start..stop-1 (paper Alg. 1). Updates *entry, returns tag.
i64 hnsw_insert(const Graph* gp, i64 tag, i32* entry_io, i32 start, i32 stop, i32 efc) {
    const Graph& g = *gp;
    i32 entry = *entry_io;
    std::vector<i32> cur(1), ids;
    std::vector<float> sc;
    for (i32 node = start; node < stop; ++node) {
        if (entry < 0) {
            entry = node;
            continue;
        }
        const float* q = vec(g, node);
        int level = g.levels[node], top = g.levels[entry];
        cur.assign(1, level < top ? descend(g, q, entry, level) : entry);
        for (int lc = std::min(level, top); lc >= 0; --lc) {
            tag += 1;
            search_layer(g, q, cur.data(), (int)cur.size(), efc, lc, tag, nullptr, ids, sc);
            connect(g, node, lc, ids, sc, (int)g.M, lc == 0 ? (int)g.M0 : (int)g.M, true);
            cur.swap(ids);
        }
        if (level > top) entry = node;
    }
    *entry_io = entry;
    return tag;
}

// Top-k from the layer-0 beam search. Writes up to k results, returns the count.
int hnsw_query(const Graph* gp, i64 tag, i32 entry, const float* q, int k, int ef, const uint8_t* allow,
               i32* out_ids, float* out_scores) {
    const Graph& g = *gp;
    static thread_local std::vector<i32> ids;
    static thread_local std::vector<float> sc;
    i32 ep = descend(g, q, entry, 0);
    search_layer(g, q, &ep, 1, std::max(ef, k), 0, tag, allow, ids, sc);
    int n = std::min<int>(k, (int)ids.size());
    if (n) {
        std::memcpy(out_ids, ids.data(), n * sizeof(i32));
        std::memcpy(out_scores, sc.data(), n * sizeof(float));
    }
    return n;
}

// Unlink tombstoned nodes and reconnect everything they cut off (see hnsw.py _repair).
i64 hnsw_repair(const Graph* gp, i64 tag, i32 entry, i32 n, i32 efc) {
    const Graph& g = *gp;
    const int M = (int)g.M, M0 = (int)g.M0;
    std::vector<i32> buf, out;
    // 1. local repair: a dead neighbor is replaced by its live neighbors, re-pruned
    for (i32 u = 0; u < n; ++u) {
        if (g.dead[u]) continue;
        for (int lv = 0; lv <= g.levels[u]; ++lv) {
            Span cur = links(g, u, lv);
            bool hit = false;
            for (int i = 0; i < cur.n && !hit; ++i) hit = g.dead[cur.p[i]];
            if (!hit) continue;
            tag += 1;
            g.visited[u] = tag;  // never link u to itself
            buf.clear();
            for (int i = 0; i < cur.n; ++i) {
                i32 x = cur.p[i];
                if (g.dead[x]) {
                    Span L = links(g, x, lv);
                    for (int j = 0; j < L.n; ++j) {
                        i32 y = L.p[j];
                        if (!g.dead[y] && g.visited[y] != tag) {
                            g.visited[y] = tag;
                            buf.push_back(y);
                        }
                    }
                } else if (g.visited[x] != tag) {
                    g.visited[x] = tag;
                    buf.push_back(x);
                }
            }
            prune(g, u, buf, lv == 0 ? M0 : M, out);
            set_links(g, u, lv, out.data(), (int)out.size());
        }
    }
    for (i32 d = 0; d < n; ++d)
        if (g.dead[d])
            for (int lv = 0; lv <= g.levels[d]; ++lv) set_links(g, d, lv, nullptr, 0);
    // 2. an entry with no links on a layer is a dead end there: link it from any live node on that layer
    for (int lv = g.levels[entry]; lv >= 0; --lv)
        if (links(g, entry, lv).n == 0)
            for (i32 s = 0; s < n; ++s)
                if (!g.dead[s] && s != entry && g.levels[s] >= lv) {
                    tag = relink(g, tag, entry, s, lv, efc, nullptr, true);
                    break;
                }
    // 3. the entry's layer-0 component: nodes it reaches that can also walk back to it
    std::vector<uint8_t> fwd(n, 0), bwd(n, 0);
    std::vector<i32> stack;
    stack.reserve(n);
    fwd[entry] = 1;
    stack.push_back(entry);
    while (!stack.empty()) {
        i32 x = stack.back();
        stack.pop_back();
        for (int i = 0; i < g.c0[x]; ++i) {
            i32 y = g.l0[(i64)x * M0 + i];
            if (!fwd[y]) fwd[y] = 1, stack.push_back(y);
        }
    }
    std::vector<i64> start(n + 1, 0);  // reverse edges, CSR
    for (i32 x = 0; x < n; ++x)
        for (int i = 0; i < g.c0[x]; ++i) start[g.l0[(i64)x * M0 + i] + 1] += 1;
    for (i32 i = 0; i < n; ++i) start[i + 1] += start[i];
    std::vector<i32> rev(start[n]);
    std::vector<i64> fill(start.begin(), start.end() - 1);
    for (i32 x = 0; x < n; ++x)
        for (int i = 0; i < g.c0[x]; ++i) rev[fill[g.l0[(i64)x * M0 + i]]++] = x;
    bwd[entry] = 1;
    stack.push_back(entry);
    while (!stack.empty()) {
        i32 x = stack.back();
        stack.pop_back();
        for (i64 i = start[x]; i < start[x + 1]; ++i)
            if (!bwd[rev[i]]) bwd[rev[i]] = 1, stack.push_back(rev[i]);
    }
    std::vector<uint8_t> member(n);
    for (i32 i = 0; i < n; ++i) member[i] = fwd[i] & bwd[i];
    // 4. re-insert every live non-member with links into the component
    for (i32 u = 0; u < n; ++u) {
        if (g.dead[u] || member[u]) continue;
        // out-links only into the component; back-links only where a member has room
        tag = relink(g, tag, u, entry, 0, efc, member.data(), false);
        Span mine = links(g, u, 0);
        bool joined = false;
        for (int i = 0; i < mine.n && !joined; ++i) joined = has(links(g, mine.p[i], 0), u);
        if (!joined && mine.n > 0) {
            // splice u into its nearest member's link nb -> b, making it nb -> u -> b
            Span cur = links(g, mine.p[0], 0);
            if (cur.n > 0) {
                int j = 0;
                for (int i = 1; i < cur.n; ++i)
                    if (dot(vec(g, cur.p[i]), vec(g, u), g.dim) > dot(vec(g, cur.p[j]), vec(g, u), g.dim)) j = i;
                i32 b = cur.p[j];
                cur.p[j] = u;
                if (!has(mine, b)) {  // len(mine) <= M < M0: room
                    mine.p[mine.n] = b;
                    set_links(g, u, 0, mine.p, mine.n + 1);
                }
            }
        }
        member[u] = 1;
    }
    return tag;
}

}  // extern "C"
