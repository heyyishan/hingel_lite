# Mathematical foundations

Every formula below is the one the code actually implements; each section names the
module and test that pins it down. Where the product makes a modelling choice, that
choice is stated rather than buried.

---

## 1. Embeddings and similarity (Modules A, C)

Images are embedded with OpenCLIP `ViT-B-32` (512-d), text with
`all-MiniLM-L6-v2` (384-d). Both are L2-normalised, so inner product is cosine
similarity:

$$
s(a,b) \;=\; \frac{\langle a, b\rangle}{\lVert a\rVert_2\,\lVert b\rVert_2}
\;\equiv\; \langle \hat a, \hat b \rangle \quad\text{for } \hat a = a/\lVert a\rVert_2 .
$$

The C23 kernel `ae_cosine_unit` is a **raw inner product** and is only valid on unit
vectors; `ae_cosine_raw` performs the division. The binding exposes `cosine()`
(always correct) and `cosine_unit()` (documented contract, plain dot product). The
conversion between cosine similarity and squared Euclidean distance for unit
vectors is

$$
\lVert \hat a - \hat b\rVert_2^2 = 2 - 2\langle \hat a,\hat b\rangle
\;\Longrightarrow\;
s = 1 - \tfrac14 d^2 \quad (d = \lVert a-b\rVert_2 \le 2).
$$

*Implementation*: `c23/src/core.c:ae_cosine_*`, `cpp/src/similarity*.cpp`,
`ml/encoders.py`. *Tests*: `test_native_bindings.py::test_cosine_matches_definition`,
`Similarity.EigenBatchMatchesScalarPath` (C++).

### 1.1 Blocked GEMM path

For a $m\times d$ query block $Q$ and $n\times d$ candidate matrix $X$ the full
similarity matrix is $S = QX^{\top}$ with $S_{ij} = s(q_i, x_j)$. Computing it row by
row costs $m\cdot n$ independent dot products and is memory-bound; the benchmark
(§8) shows a 4–25× gap to BLAS at large $n$. The API therefore uses Eigen's blocked
product (`batch_cosine_eigen`) for matrix work and keeps the dispatched scalar
kernel for single pairs.

---

## 2. Photograph ordering: the objective J(π) (Module B)

We must choose a permutation $\pi = (\pi_1,\dots,\pi_n)$ of the $n$ photographs into
display slots $1..n$. Three terms compose the objective:

$$
J(\pi) \;=\; \underbrace{\sum_{k=1}^{n} w_k\, q_{\pi_k}}_{\text{quality}}
\;+\; \lambda\,\underbrace{D(\pi)}_{\text{diversity}}
\;+\; \mu\,\underbrace{C(\pi)}_{\text{coherence}} .
$$

**(a) Quality with positional weights.** $q_i\in[0,1]$ is the measured image quality
(§3). $w_k$ is a normalised attention weight for slot $k$; slot 1 is seen most, so

$$
w_k \;\propto\; \frac{1}{1 + \beta(k-1)}, \qquad \sum_k w_k = 1,\quad \beta = 0.35 .
$$

**(b) Diversity via a kernel-weighted redundancy penalty.** Let
$\kappa(\Delta) = 1/\Delta$ for slot distance $\Delta\ge 1$ and $\kappa(1)=1$. Then

$$
R(\pi) = \sum_{i<j} \kappa\!\left(|i-j|\right)\bigl(1 - s(\pi_i,\pi_j)\bigr),
\qquad
D(\pi) = 1 - \frac{R(\pi)}{\sum_{i<j}\kappa(|i-j|)} \in [0,1].
$$

So $D$ is the mean *dissimilarity* of all pairs, with nearer pairs weighted more —
adjacent photographs that look identical are penalised hardest. ($\kappa$ is
position-independent, so $\sum\kappa$ depends only on $n$; this is what makes
$D\in[0,1]$.)

**(c) Coherence.** Each slot has a target "energy" $t_k$ (slot 1 should be the
strongest). With $\sigma_i$ the normalised per-photo strength,

$$
C(\pi) = 1 - \frac{1}{n}\sum_{k=1}^n \left|\sigma_{\pi_k} - t_k\right| \in [0,1].
$$

**Degenerate case (oracle check).** $\lambda=\mu=0 \Rightarrow J$ is a linear
assignment with sorted $w$, so the optimum is $q_{\pi_1}\ge q_{\pi_2}\ge\cdots$ —
a plain descending sort by quality. The implementation reproduces this exactly;
`test_ml_pipeline.py::test_lambda_mu_zero_reproduces_the_quality_oracle` asserts it.

### 2.1 Search

$n!\;$ orderings. Exhaustive search is used up to $n\le 9$ (`exhaustive_limit`) and
proves optimality; above that the API runs greedy, beam (width 8), simulated
annealing (20 000 iter, 2 reheats) and random restart (16 restarts), then a
steepest-descent 2-swap polish. The response reports `proved_optimal` per strategy
and the gap to the best, so a heuristic result is never presented as an optimum.

Measured on this machine (`cpp/tests`, 40 random problems, $n=8$):
beam+polish matched the exhaustive optimum in **35/40** cases, mean relative gap
**4.4e-04**, max **1.01e-02**.

### 2.2 Sensitivity and ablation

*Sensitivity* re-optimises over a $(\lambda,\mu)$ grid and reports
$J^\star_{\lambda,\mu}$ against the score of the *fixed* incumbent order
$J_{\lambda,\mu}(\pi_0)$. The latter is the number that matters: it shows how much
the current ordering would lose if the user's weighting changed.

*Ablation* removes each photograph, re-optimises, and reports
$\Delta J_i = J^\star - J^\star_{\text{without } i}$ — the objective mass that
photograph carries.

---

## 3. Image quality (Module A)

Quality is a fixed weighted sum of seven measured components:

| component | weight | measurement |
|---|---|---|
| exposure | 0.18 | distance of mean luma from mid-grey |
| contrast | 0.12 | RMS contrast, saturating at a reference sigma |
| sharpness | 0.22 | variance of Laplacian, log-scaled |
| resolution | 0.10 | short-edge pixels vs. a 640 px reference |
| composition | 0.13 | rule-of-thirds / centre-bias of the saliency centroid |
| face | 0.15 | detected face area fraction (a *framing* signal) |
| background | 0.10 | background entropy away from the subject |

$q = \sum_k w_k c_k$ with $\sum w_k = 1$, so $q\in[0,1]$. Weights are returned with
every analysis and tested to reconstruct the score
(`test_ml_pipeline.py::test_image_analysis_reports_quality_without_sensitive_attributes`).

The `face` component measures **how much of the frame a face occupies**. No
identity, attractiveness, age, ethnicity, religion or health attribute is computed
anywhere in the pipeline; `docs/ethics.md` states this and the API's `/privacy/policy`
publishes it.

---

## 4. Text specificity (Modules A, E)

A prompt answer is scored on four saturating sub-signals:

$$
u = \tfrac14\Bigl[\min(1, E/4) + \min(1, K/5) + \mathbb{1}[\text{time/place anchor}] + \min(1, \ell/25)\Bigr],
$$

where $E$ = digits + capitalised entities, $K$ = distinct concrete-noun hits,
$\ell$ = token count. `specificity()` returns the score **and** every component, so
the number is auditable.

---

## 5. Compatibility and ranking (Module C)

### 5.1 Weighted baseline

For viewer $v$ and candidate $c$:

$$
S(v,c) = \frac{a V + b T + g A + d P}{a+b+g+d},
\quad (a,b,g,d) = (0.25, 0.30, 0.25, 0.20),
$$

$$
V = \frac{(s_{v,c}+1)}{2}\ \text{(visual cosine mapped to }[0,1]\text{)},\quad
T = \text{same for text},\quad
A = \mathbb{E}\left[1 - \min(|\text{attr}_v - \text{attr}_c|, 1)\right],\quad
P = \text{interest cosine}.
$$

Ranking uses the **harmonic mean** of the two directions so a one-sided match cannot
score highly:

$$
H(v,c) = \frac{2\,S(v,c)\,S(c,v)}{S(v,c)+S(c,v)} .
$$

### 5.2 List-level diversity (MMR)

Final selection maximises relevance minus redundancy against what is already chosen:

$$
\text{MMR} = \arg\max_{i \notin \mathcal{S}}\Bigl[\lambda\,\text{rel}(i) - (1-\lambda)\max_{j\in\mathcal{S}} s(i,j)\Bigr],
$$

with $\lambda=1$ recovering pure relevance.

### 5.3 Pairwise MLP ranker

A two-layer MLP over pair features (visual, textual, attribute, interest similarities
plus their product/difference) predicts which of two candidates a synthetic viewer
prefers, trained with binary cross-entropy on **synthetic** pairs from the
mixture-of-Gaussians generator.

Measured (this machine, `artifacts/pair_ranker.txt`):
**7 970 pairs** (5 977 train / 1 993 val), best epoch 128,
**val loss 0.50706, accuracy 0.7787, Brier 0.1648, ECE 0.0467**.
A constant 0.5 predictor has Brier 0.25, so the head carries real signal — on
synthetic data. It has never seen a real person's data and the UI says so.

### 5.4 Ranking metrics

$$
\text{P@}k = \frac{1}{k}\sum_{i\le k}\mathbb{1}[y_i \ge \tau],\quad
\text{R@}k = \frac{\#\{\text{relevant in top } k\}}{\#\{\text{relevant}\}},
$$

$$
\text{nDCG@}k = \frac{\sum_{i=1}^{k} y_i / \log_2(i+1)}{\sum_{i=1}^{k} y_{(i)} / \log_2(i+1)},
$$

with $\tau = 0.66$ (the generator's "strong affinity" threshold) and $y_{(i)}$ the
ideal ordering. This is computable **only because the pool is synthetic** and the
affinity is planted.

### 5.5 Retrieval

Exact scan vs. pgvector HNSW ($m=16$, `ef_construction=64`, cosine ops). The API
reports both times and `recall@k` of ANN against exact on the same vectors, so the
index's approximation is measured rather than assumed.

---

## 6. Beta-Binomial experimentation (Module D)

### 6.1 Conjugacy

With prior $\theta\sim\text{Beta}(\alpha_0,\beta_0)$ and $s$ successes in
$n=s+f$ Bernoulli trials,

$$
\theta\mid s,f \;\sim\; \text{Beta}(\alpha_0+s,\ \beta_0+f),
\qquad
\mathbb{E}[\theta]=\frac{\alpha}{\alpha+\beta},\quad
\text{Var}[\theta]=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)} .
$$

Sequential updating is identical to batch updating — the streaming endpoint relies
on this and `test_stats_bayesian.py::test_sequential_updates_equal_the_batch_update`
verifies it.

### 6.2 Probability of superiority

$$
P(\theta_A>\theta_B)=\int_0^1 f_A(x)\,F_B(x)\,dx,
$$

evaluated by composite Simpson's rule on 4 096 panels (`ae::prob_superiority`,
`cpp/src/stats.cpp`). The identity $P(\theta_A>\theta_B)+P(\theta_B>\theta_A)=1$ is
asserted to 1e-6, and symmetry to 1e-3 (`cpp/tests/test_stats.cpp`).

### 6.3 Expected loss and stopping

Choosing arm $A$ costs

$$
L(A) = \mathbb{E}\bigl[\max(0,\ \theta_B-\theta_A)\bigr]
     = \int_0^1 f_A(x)\int_x^1 (y-x) f_B(y)\,dy\,dx .
$$

The sequential rule stops when the leading arm's expected loss is below
`loss_tolerance` **and** both arms have at least `min_observations` observations —
the second condition is what prevents a two-observation "winner".

### 6.4 Thompson allocation

Draw $\tilde\theta_k \sim \text{Beta}(\alpha_k,\beta_k)$ and allocate the next batch
proportionally to wins over $N=2000$ simulated draws. Seedable and reproducible;
the API returns allocation, share and seed.

### 6.5 Multiplicity

With $k$ simultaneous comparisons the per-comparison level is

$$
\alpha_{\text{Bonf}} = \alpha/k,\qquad
\alpha_{\text{Šidák}} = 1-(1-\alpha)^{1/k}.
$$

### 6.6 Sample size (planning)

$$
n_{\text{per arm}} =
\frac{\left(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_2-p_1)^2},
$$

$\bar p=(p_1+p_2)/2$. For $(p_1,p_2)=(0.40,0.45)$, $\alpha=0.05$, power $0.8$:
**1 531 per arm (3 062 total)**. This is a fixed-horizon planning number and does
**not** apply to the sequential loss rule — the API says so in the payload.

### 6.7 Hierarchical shrinkage

Arm rates are partially pooled toward a population Beta fitted by moment matching,
with shrinkage factor $\kappa$:

$$
\tilde\theta_k = (1-\kappa)\,\hat\theta_k + \kappa\,\mu_{\text{pop}},
$$

so thin arms move further than well-sampled ones (tested explicitly).

---

## 7. Counterfactual simulator (Module E)

$$
Q(\text{config}) = 0.55\,\bar q_{\text{sel}} + 0.25\,D_{\text{sel}} + 0.20\,\bar u_{\text{text}},
\qquad
\Delta Q = Q(\text{modified}) - Q(\text{baseline}).
$$

Attribution by permutation importance: shuffle one input block $B$ ($N=200$ draws),
recompute $Q$, and report the mean drop $\overline{Q - Q_{\sigma(B)}}$.

**This is not a causal quantity.** $Q$ is a deterministic function of declared
features with hand-specified weights; no assignment was randomised, so $\Delta Q$
carries no sampling uncertainty (`uncertainty.value = null`) and the API returns an
`interpretation` string saying exactly this. To obtain causal evidence, run the two
configurations as an experiment (§6) with randomised assignment.

---

## 8. Benchmarks and complexity

| operation | time | where |
|---|---|---|
| dot product, $d$ | $O(d)$, 2/4/8 doubles per AVX2/AVX-512 instruction | `ae_vec_dot` |
| full similarity $m\times n$ | $O(mnd)$, blocked GEMM | `batch_cosine_eigen` |
| top-$k$ of $n$ (heap) | $O(n\log k)$ | `ae::top_k<heap_bounded>` |
| top-$k$ (full sort) | $O(n\log n)$ | reference strategy |
| MMR selection, $k$ of $n$ | $O(nk)$ similarity work | `ae::mmr_select` |
| ordering, exhaustive | $O(n!)$ — capped at $n\le 9$ | `ae::optimize` |
| ordering, beam | $O(b\,n^3)$ per width $b$ | `ae::optimize` |
| posterior moments | closed form, $O(1)$ | `ae::beta_posterior` |
| $P(\theta_A>\theta_B)$ | $O(N)$ Simpson, $N=4096$ | `ae::prob_superiority` |
| credible interval | $O(\log(1/\varepsilon))$ bisection on the CDF | `ae::credible_interval` → `beta_quantile` |

Measured numbers, with the NumPy baselines and the caveats that make them honest,
are generated into `benchmarks/results/benchmarks_table.md` by
`scripts/run_benchmarks.sh` and summarized in `docs/benchmarks.md`.

---

## References

1. K. Pearson (1895). *Contributions to the mathematical theory of evolution II:
   Skew variation in homogeneous material.* Phil. Trans. R. Soc. A 186, 343–414 —
   the Beta family used for conjugate inference.
2. G. E. P. Box & G. C. Tiao (1973). *Bayesian Inference in Statistical Analysis.*
   Addison-Wesley — conjugate updating and expected-loss decisions.
3. W. R. Thompson (1933). *On the likelihood that one unknown probability exceeds
   another in view of the evidence of two samples.* Biometrika 25(3–4), 285–294 —
   probability of superiority and Thompson sampling.
4. C. R. Harris et al. (2020). *Array programming with NumPy.* Nature 585, 357–362 —
   the baseline implementation for the benchmark comparisons.
5. M. Cherti et al. (2023). *Reproducible scaling laws for contrastive
   language-image learning.* CVPR — OpenCLIP, source of `ViT-B-32 laion2b`.
6. N. Reimers & I. Gurevych (2019). *Sentence-BERT: Sentence embeddings using Siamese
   BERT-networks.* EMNLP — `all-MiniLM-L6-v2`.
7. J. Johnson, M. Douze & H. Jégou (2019). *Billion-scale similarity search with
   GPUs.* IEEE Trans. Big Data — HNSW/FAISS-style ANN, the design pgvector follows.
8. C. Malkov & D. A. Yashunin (2020). *Efficient and robust approximate nearest
   neighbor search using HNSW graphs.* IEEE TPAMI 42(4) — the index used by pgvector.
9. J. Carbonell & J. Goldstein (1998). *The use of MMR, diversity-based reranking for
   reordering documents and producing summaries.* SIGIR — MMR.
10. M. Narasimhan & J. Kulkarni (2011). *A linear combinatorial optimization problem
    with applications* — assignment-style objectives; see also D. Bertsekas,
    *Nonlinear Programming*, for the assignment polytope behind J(π).
11. T. Fawcett (2006). *An introduction to ROC analysis.* Pattern Recognition
    Letters 27(8) — the basis for Brier/ECE-style probability calibration reporting.
12. W. Kahan (1965) / N. J. Higham (2002). *Accuracy and Stability of Numerical
    Algorithms* — compensated summation used in the C23 kernels.
