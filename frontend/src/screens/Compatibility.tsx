import { useState } from "react";
import { api, RankResponse } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useProfiles } from "../lib/store";
import { Card, ErrorBox, Loading, Stat, Pill, Caveat } from "../components/ui";
import { NoProfile, ProfileBar } from "../components/ProfileBar";
import { num, pct } from "../lib/format";

export function Compatibility() {
  const { selected } = useProfiles();
  const [topK, setTopK] = useState(10);
  const [lambdaDiv, setLambdaDiv] = useState(0);
  const [seed, setSeed] = useState(11);

  const rank = useApi<RankResponse>(
    () =>
      selected
        ? api.post<RankResponse>("/api/compatibility/rank", {
            profile_id: selected.id, top_k: topK, seed, lambda_diversity: lambdaDiv, pool_size: 300,
          })
        : Promise.resolve(null as never),
    [selected?.id, topK, lambdaDiv, seed],
  );

  return (
    <>
      <ProfileBar
        title="Compatibility Explorer"
        description="Modules C. Weighted multi-feature baseline plus a pairwise MLP head trained on synthetic pairs; MMR reranking for list-level diversity."
      />
      {!selected && <NoProfile />}
      {selected && (
        <div className="space-y-6">
          <Card title="Ranking parameters">
            <div className="grid gap-4 sm:grid-cols-4">
              <label className="block">
                <span className="label">Top k</span>
                <input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                       className="input mt-1" />
              </label>
              <label className="block">
                <span className="label">MMR λ (diversity)</span>
                <input type="range" min={0} max={0.9} step={0.05} value={lambdaDiv}
                       onChange={(e) => setLambdaDiv(Number(e.target.value))} className="mt-3 w-full accent-[#7c5cff]" />
                <span className="text-xs text-slate-400">{lambdaDiv.toFixed(2)}</span>
              </label>
              <label className="block">
                <span className="label">Pool seed</span>
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="input mt-1" />
              </label>
              <div className="flex items-end">
                <button className="btn-ghost w-full" onClick={rank.reload} disabled={rank.loading}>
                  {rank.loading ? "Ranking…" : "Re-rank"}
                </button>
              </div>
            </div>
          </Card>

          {rank.loading && <Loading what="the candidate pool" />}
          {Boolean(rank.error) && <ErrorBox error={rank.error} onRetry={rank.reload} />}

          {rank.data && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="nDCG@k" value={num(rank.data.metrics.ndcg_at_k, 4)} hint={rank.data.metrics.relevant_definition} tone="ok" />
                <Stat label="Precision@k" value={num(rank.data.metrics.precision_at_k, 3)} hint={`${rank.data.metrics.n_relevant} relevant in pool`} />
                <Stat label="Recall@k" value={num(rank.data.metrics.recall_at_k, 3)} hint={`k = ${rank.data.metrics.k}`} />
                <Stat
                  label="Neural head"
                  value={rank.data.neural_head_applied ? "applied" : "baseline only"}
                  tone={rank.data.neural_head_applied ? "ok" : "warn"}
                  hint={rank.data.model_status}
                />
              </div>

              <Card title={`Top ${rank.data.ranking.length} candidates`} subtitle="Scores are the symmetric (harmonic) blend of forward and backward feature similarity.">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="text-slate-500">
                      <tr>
                        <th className="text-left font-normal">#</th>
                        <th className="text-left font-normal">score</th>
                        <th className="text-right font-normal">visual</th>
                        <th className="text-right font-normal">text</th>
                        <th className="text-right font-normal">attrs</th>
                        <th className="text-right font-normal">interests</th>
                        <th className="text-right font-normal">confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rank.data.ranking.map((r, i) => (
                        <tr key={r.index} className="border-t border-ink-700/60">
                          <td className="py-1.5 text-slate-500">{i + 1}</td>
                          <td className="font-medium tabular-nums text-slate-100">{num(r.harmonic, 4)}</td>
                          <td className="text-right tabular-nums text-slate-400">{num(r.forward.visual, 3)}</td>
                          <td className="text-right tabular-nums text-slate-400">{num(r.forward.textual, 3)}</td>
                          <td className="text-right tabular-nums text-slate-400">{num(r.forward.attributes, 3)}</td>
                          <td className="text-right tabular-nums text-slate-400">{num(r.forward.interests, 3)}</td>
                          <td className="text-right tabular-nums text-slate-400">
                            {pct(r.confidence, 0)} {r.cold_start && <Pill tone="warn">cold start</Pill>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 font-mono text-[11px] text-slate-500">{rank.data.formula}</p>
              </Card>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Declared weights" subtitle="Every ranking exposes the exact linear blend it used.">
                  <ul className="space-y-1 text-xs">
                    {Object.entries(rank.data.weights).map(([k, v]) => (
                      <li key={k} className="flex justify-between border-b border-ink-700/60 py-1">
                        <span className="text-slate-300">{k}</span>
                        <span className="tabular-nums">{num(v, 3)}</span>
                      </li>
                    ))}
                  </ul>
                  <Caveat>
                    Ablating each weight and re-scoring shows what the metric owes to it — a weight that
                    barely moves nDCG is not doing work.
                  </Caveat>
                  <ul className="mt-2 space-y-1 text-xs">
                    {Object.entries(rank.data.ablations).map(([name, a]) => {
                      const delta = a.ndcg_at_k - rank.data!.metrics.ndcg_at_k;
                      return (
                        <li key={name} className="flex justify-between">
                          <span className="text-slate-400">{name.replace(/_/g, " ")}</span>
                          <span className="tabular-nums text-slate-200">
                            nDCG {num(a.ndcg_at_k, 4)} ({delta >= 0 ? "+" : ""}{num(delta, 4)})
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </Card>

                <Card title="What these numbers mean" subtitle="Synthetic-evaluation metrics, not real-world match rates.">
                  <ul className="space-y-2 text-xs leading-relaxed text-slate-400">
                    <li>· The candidate pool is generated by a mixture-of-Gaussians with planted affinity — that is why a ground-truth nDCG can be computed at all.</li>
                    <li>· The MLP ranker was trained on 7,970 synthetic pairs from that generator (val accuracy 0.779, Brier 0.165); it has never seen a real person's data.</li>
                    <li>· High nDCG here means the scorer recovers the generator's affinity structure. It says nothing about the real world.</li>
                  </ul>
                  <Caveat>{rank.data.caveat}</Caveat>
                </Card>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
