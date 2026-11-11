# Hinge Lite (Attraction Engine)

**A production-grade dating-app ranking & matching engine** — mathematical foundations, native performance cores, and privacy-first design.

> **Note on naming**: The repository was created as `hingel_lite` (typo). The intended name is **Hinge Lite**.

---

## Overview

**Hinge Lite** is a complete, rigorously engineered attraction / matching engine inspired by modern dating apps such as Hinge. It implements the full pipeline that real systems need:

- High-quality photo analysis & optimal ordering
- Multi-signal compatibility ranking (visual + text + attributes + interests)
- List-level diversity (MMR)
- Bayesian sequential experimentation & Thompson sampling
- Counterfactual simulation
- Conversation analysis with strong privacy guarantees
- Full REST API + frontend + PostgreSQL/pgvector + Redis

Every formula is documented, tested, and pinned to the code that actually runs it. The native layers (C23 + C++26) deliver AVX2/AVX-512 kernels, blocked GEMM, and exact/ANN retrieval so the system stays fast even on large candidate pools.

---

## Key Capabilities

| Module | What it does |
|--------|--------------|
| **Photo Intelligence** | OpenCLIP ViT-B-32 embeddings + 7-component quality score (exposure, contrast, sharpness, resolution, composition, face framing, background). Zero identity / attractiveness / demographic inference. |
| **Photograph Ordering** | Optimizes \(J(\pi)\) = quality + diversity + coherence. Exhaustive search (n ≤ 9), beam, simulated annealing, random restart + 2-swap polish. Sensitivity & ablation reports. |
| **Compatibility Ranking** | Weighted baseline → harmonic mean of bidirectional scores → optional pairwise MLP ranker (trained only on synthetic data) → MMR diversity re-ranking. |
| **Retrieval** | Exact scan + pgvector HNSW. Reports recall@k of ANN vs exact. |
| **Bayesian Laboratory** | Beta-Binomial conjugate updates, probability of superiority, expected loss stopping rule, Thompson allocation, hierarchical shrinkage, multiplicity correction. |
| **Counterfactual Simulator** | Deterministic Q-score with explicit honesty statement that it is *not* causal. |
| **Conversations** | Reciprocity & structure analysis. Raw text is never stored. Psychological inference is refused. |
| **Privacy & Ethics** | Explicit policy endpoint, data export, no sensitive attributes, synthetic data clearly labelled. |

All of the above is covered by a 62-check end-to-end suite that must pass.

---

## Architecture Highlights

- **Native cores**: C23 (vector kernels) + modern C++26 (Eigen blocked GEMM, stats, optimizers) with Python bindings.
- **ML stack**: OpenCLIP (`ViT-B-32`) + `all-MiniLM-L6-v2`.
- **Data plane**: PostgreSQL 17 + pgvector 0.8 + Redis.
- **API**: FastAPI-style service with auth, profiles, photos, intelligence, laboratory, experiments, counterfactuals, conversations, privacy.
- **Frontend**: Present (see `frontend/`).
- **Migrations**: Alembic.
- **Benchmarks**: Honest numbers vs NumPy baselines published under `benchmarks/results/`.

---

## Mathematical Foundations

Full derivations, implementation references, and tests live in [`docs/mathematics.md`](docs/mathematics.md). Highlights:

- Cosine similarity on L2-normalised embeddings (exact equivalence to squared Euclidean for unit vectors).
- Photo ordering objective \(J(\pi)\) with positional attention weights, kernel-weighted diversity, and coherence.
- Image quality as a transparent weighted sum of seven measurable components.
- Harmonic-mean ranking + MMR.
- Full Beta-Binomial sequential analysis (conjugacy, P(superiority), expected loss, Thompson sampling, hierarchical shrinkage).
- Ranking metrics (P@k, R@k, nDCG@k) computed only on synthetic data with known ground truth.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Native | C23, C++26, Eigen, AVX2/AVX-512 |
| Python | 3.13, FastAPI-style service, PyTorch (CPU), OpenCLIP, Sentence-Transformers |
| Data | PostgreSQL 17 + pgvector, Redis, Alembic |
| Frontend | Present in repo |
| Testing | pytest + extensive native tests + 62-point E2E suite |
| Build | CMake, Ninja |

Verified environment matrix: [`docs/environment.md`](docs/environment.md).

---

## Quick Start (high level)

1. Clone the repository.
2. Install system dependencies (PostgreSQL + pgvector, Redis, modern GCC, Eigen, etc.).
3. Build the native libraries (`c23/` and `cpp/`).
4. Set up the Python environment and install the package.
5. Run Alembic migrations.
6. Start the API service.
7. (Optional) Launch the frontend.

Detailed environment and build notes are in the docs. The E2E suite (`docs/e2e_output.txt`) demonstrates a full happy path including photo analysis, ordering, ranking, experiments, and privacy export.

---

## Project Structure (top level)

```
hingel_lite/
├── c23/                 # High-performance C23 kernels
├── cpp/                 # C++26 core (similarity, optimizers, stats)
├── python/
│   └── attraction_engine/   # API, services, ML, analytics, stats
├── frontend/
├── docs/                # mathematics.md, environment.md, E2E output, screenshots
├── benchmarks/
├── alembic/
├── scripts/
└── data/
```

---

## Design Principles

- **Honesty over hype** — every model limitation, synthetic-data boundary, and non-causal claim is stated in the API responses themselves.
- **Performance with correctness** — native kernels are measured against reference implementations; ANN recall is reported, never assumed.
- **Privacy by construction** — no demographic inference, no raw conversation storage, explicit export & deletion paths.
- **Auditability** — quality scores, specificity components, posterior parameters, and ranking features are all inspectable.

---

## Status

- Fully functional core engine with comprehensive test coverage.
- 62/62 E2E checks passing on the reference environment.
- Ready for local experimentation, research, or as a high-quality portfolio / systems piece.

---

## License & Attribution

See the repository for the license. This is an independent engineering project and is not affiliated with Hinge or Match Group.

---

**Built with mathematical rigor, systems craftsmanship, and a clear ethical boundary.**

Explore [`docs/mathematics.md`](docs/mathematics.md) first — it is the single best entry point into how the system actually thinks.
