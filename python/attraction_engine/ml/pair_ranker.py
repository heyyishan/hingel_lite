"""Pairwise neural ranker (PyTorch) — training lives here, inference in C++.

Architecture
------------
    h = GELU(W1 [u_i, u_j, |u_i-u_j|, u_i⊙u_j] + b1)
    p = sigmoid(W2 h + b2)
with a *symmetric head*: the loss is computed on both orderings of the pair, so
p(u_i,u_j) == p(u_j,u_i) by construction. Compatibility is reciprocal, so an
order-sensitive head would be a modelling error.

Training data is synthetic (stats/synthetic.py). The exported weights are only a
demonstration of the pipeline: no real pair-level outcome data was used, and the
API labels the model "trained-synthetic".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TrainConfig:
    hidden: int = 24
    epochs: int = 160
    lr: float = 3e-3
    batch_size: int = 128
    weight_decay: float = 1e-2
    val_fraction: float = 0.25
    seed: int = 1234
    n_candidates: int = 4000
    n_clusters: int = 6
    dim: int = 16
    history: list = field(default_factory=list)


def build_pairs(pool: dict, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Featurise ordered pairs from the synthetic pool with a latent label.

    Label: the pair is compatible when the two profiles are in the same activity
    cluster; we add 8% label noise so the model cannot reach a perfect fit, which
    keeps the reported validation metrics honest.
    """
    rng = np.random.default_rng(cfg.seed)
    profiles = pool["profiles"]
    n = len(profiles)

    def uvec(p):
        return np.asarray(p["visual"] + p["textual"], dtype=np.float64)

    feats, labels = [], []
    for _ in range(cfg.n_candidates):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = uvec(profiles[i]), uvec(profiles[j])
        fa = np.concatenate([a, b, np.abs(a - b), a * b])
        fb = np.concatenate([b, a, np.abs(a - b), a * b])
        y = 1.0 if profiles[i]["cluster"] == profiles[j]["cluster"] else 0.0
        if rng.random() < 0.08:
            y = 1.0 - y
        feats.append(fa)
        feats.append(fb)
        labels.extend([y, y])
    return np.asarray(feats), np.asarray(labels)


def train(cfg: TrainConfig | None = None, pool: dict | None = None, log=None) -> dict:
    import torch
    import torch.nn as nn

    cfg = cfg or TrainConfig()
    from ..stats.synthetic import SyntheticSpec, make_candidate_pool

    if pool is None:
        pool = make_candidate_pool(SyntheticSpec(seed=cfg.seed, n_candidates=120,
                                                 image_dim=cfg.dim, text_dim=cfg.dim,
                                                 n_clusters=cfg.n_clusters))
    X, y = build_pairs(pool, cfg)
    torch.manual_seed(cfg.seed)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    split = int(len(X) * (1 - cfg.val_fraction))
    Xtr, ytr = torch.tensor(X[:split], dtype=torch.float32), torch.tensor(y[:split], dtype=torch.float32)
    Xva, yva = torch.tensor(X[split:], dtype=torch.float32), torch.tensor(y[split:], dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(X.shape[1], cfg.hidden), nn.GELU(),
        nn.Linear(cfg.hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    # Early stopping: with a few thousand synthetic pairs the network overfits
    # within ~40 epochs (observed train 0.04 / val 0.78 in the first run), so we
    # restore the best-validation weights instead of the last ones.
    for epoch in range(cfg.epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        total = 0.0
        for b in range(0, len(Xtr), cfg.batch_size):
            sel = idx[b:b + cfg.batch_size]
            opt.zero_grad()
            out = model(Xtr[sel]).squeeze(-1)
            loss = loss_fn(out, ytr[sel])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        model.eval()
        with torch.no_grad():
            logits = model(Xva).squeeze(-1)
            val_loss = float(loss_fn(logits, yva))
            probs = torch.sigmoid(logits)
            acc = float(((probs > 0.5).float() == yva).float().mean())
            brier = float(torch.mean((probs - yva) ** 2))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val_loss": round(val_loss, 5), "val_accuracy": round(acc, 4),
                            "val_brier": round(brier, 4), "epoch": epoch}
        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            entry = {"epoch": epoch, "train_loss": round(total / len(Xtr), 5),
                     "val_loss": round(val_loss, 5), "val_accuracy": round(acc, 4),
                     "val_brier": round(brier, 4)}
            history.append(entry)
            if log:
                log(entry)
    if best_state is not None:
        model.load_state_dict(best_state)
    cfg.history = history

    layers = []
    with torch.no_grad():
        lin1, lin2 = model[0], model[2]
        layers.append((lin1.weight.numpy().copy(), lin1.bias.numpy().copy()))
        layers.append((lin2.weight.numpy().copy(), lin2.bias.numpy().copy()))

    return {"model": model, "layers": layers, "history": history,
            "final": best_metrics, "best_epoch": best_epoch,
            "n_pairs": int(len(X)), "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
            "config": cfg.__dict__}


def serialise(layers, path: Path, meta: dict) -> None:
    """Write the C++ PairRanker text format plus a JSON metadata sidecar."""
    lines = ["AE_PAIRRANKER v1", f"layers {len(layers)}"]
    for w, b in layers:
        out_dim, in_dim = w.shape
        lines.append(f"layer {in_dim} {out_dim}")
        lines.extend(f"{float(v):.17g}" for v in w.reshape(-1))
        lines.extend(f"{float(v):.17g}" for v in np.asarray(b).reshape(-1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    (path.parent / "pair_ranker.json").write_text(json.dumps(meta, indent=2))


def evaluate_calibration(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    """Reliability diagram + ECE + Brier score on held-out synthetic pairs."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for b in range(bins):
        m = (probs >= edges[b]) & (probs < edges[b + 1] + (1e-9 if b == bins - 1 else 0.0))
        if not np.any(m):
            rows.append({"bin": b, "count": 0, "mean_prob": None, "empirical_rate": None})
            continue
        mean_p = float(probs[m].mean())
        rate = float(labels[m].mean())
        ece += abs(mean_p - rate) * float(m.sum()) / len(probs)
        rows.append({"bin": b, "count": int(m.sum()), "mean_prob": round(mean_p, 4),
                     "empirical_rate": round(rate, 4)})
    brier = float(np.mean((probs - labels) ** 2))
    return {"bins": rows, "ece": round(ece, 4), "brier": round(brier, 4),
            "n": int(len(probs)),
            "note": "Synthetic held-out pairs; labels come from the generator, "
                    "so calibration here measures the pipeline, not human outcomes."}
