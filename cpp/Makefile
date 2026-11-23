# make -C cpp        -> cpp/libhnsw.so   (HNSWIndex(backend="cpp") loads it)
# make -C cpp bench  -> cpp/hnsw_bench   (standalone C++ benchmark, no Python)
# Float flags match Numba's fastmath={"reassoc", "contract"}: no inf/nan assumptions.
CXX ?= g++
FLAGS = -O3 -march=native -std=c++17 -Wall -Wextra \
        -fassociative-math -fno-signed-zeros -fno-trapping-math -ffp-contract=fast

libhnsw.so: hnsw.cpp
	$(CXX) $(FLAGS) -fPIC -shared -o $@ $<

bench: hnsw_bench
hnsw_bench: bench.cpp hnsw.cpp
	$(CXX) $(FLAGS) -o $@ bench.cpp

clean:
	rm -f libhnsw.so hnsw_bench

.PHONY: bench clean
