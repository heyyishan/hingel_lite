# Verified Environment Capability Matrix

Captured by direct probe on the build machine (`scripts/probe_environment.sh`).

| Component | Detected | Notes |
|---|---|---|
| OS | Debian GNU/Linux 13 (trixie), kernel 6.x | container, 2 vCPU |
| CPU | Intel Xeon @ 2.60 GHz, 2 vCPU | flags: sse4.2, avx2, fma, avx512f, avx512dq, avx512bw, avx512vl, avx512_vnni |
| RAM | 1.9 GiB total | no swap |
| Disk | 25 GiB (21 GiB free) | |
| gcc / g++ | 14.2.0 (Debian 14.2.0-19) | |
| `-std=c23` | **accepted** | `constexpr`, `typeof`, `nullptr`, `bool` keyword, `1'000` digit separators verified |
| `#embed` | **NOT supported** | C23 preprocessor feature absent in GCC 14 — documented limitation |
| `-std=c++26` | **accepted**, `__cplusplus == 202400` | GCC 14 implements the language core but libstdc++ 14 is partial |
| C++26 library | `std::print`, `std::println`, `std::expected`, `std::span`, `std::format`, ranges, concepts, `<bit>` | **working** |
| C++26 library | `<flat_map>`, `<mdspan>` | **absent** in libstdc++ 14 — compat shim / avoided |
| CMake / Ninja | 3.31.6 / 1.12.1 | |
| Eigen | 3.4.0 (system) | |
| GoogleTest | 1.16.0 | system static libs |
| Google Benchmark | 1.9.1 | system |
| PostgreSQL | 17.11 | running, db `attraction_engine` + `attraction_engine_test` |
| pgvector | 0.8.0 | `CREATE EXTENSION vector` verified |
| Redis | 8.0.2 | running on 127.0.0.1:6379, `PING`→PONG |
| Python | 3.13.16 | |
| PyTorch | 2.15.0+cpu | CPU-only build |
| OpenCLIP | 4.2.1 | `ViT-B-32` / `laion2b_s34b_b79k` weights downloaded & forward pass verified (512-d) |
| Sentence Transformers | 6.2.0 | `all-MiniLM-L6-v2` downloaded & encode verified (384-d) |
| Node / npm | 20.20.2 / 10.8.2 | |
| Docker | **not available** in this container | compose files shipped but unverified → documented |
