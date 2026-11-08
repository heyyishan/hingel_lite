import { useState } from "react";
import { api, AblationRow, LabPhotos, OptimizeResponse, Photo, SensitivityResponse } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useProfiles } from "../lib/store";
import { Card, ErrorBox, Loading, Stat, Pill, Caveat } from "../components/ui";
import { NoProfile, ProfileBar } from "../components/ProfileBar";
import { num, pct } from "../lib/format";

export function PhotoLaboratory() {
  const { selected } = useProfiles();
  const [lambda, setLambda] = useState(0.35);
  const [mu, setMu] = useState(0.2);
  const [applied, setApplied] = useState<string | null>(null);

  // Signed URLs for the thumbnails (an <img> cannot send the bearer token).
  const urls = useApi<{ photos: Photo[] }>(
    () =>
      selected
        ? api.get<{ photos: Photo[] }>(`/api/profiles/${selected.id}/photos`)
        : Promise.resolve({ photos: [] }),
    [selected?.id],
  );
  const urlFor = (photoId: string) => urls.data?.photos.find((p) => p.id === photoId)?.url ?? "";

  const lab = useApi<LabPhotos>(
    () => (selected ? api.get<LabPhotos>(`/api/laboratory/${selected.id}/photos`) : Promise.resolve(null as never)),
    [selected?.id],
  );
  const opt = useApi<OptimizeResponse>(
    () =>
      selected
        ? api.post<OptimizeResponse>(`/api/laboratory/${selected.id}/optimize`, { profile_id: selected.id, lambda, mu })
        : Promise.resolve(null as never),
    [selected?.id, lambda, mu],
  );
  const sens = useApi<SensitivityResponse>(
    () =>
      selected
        ? api.post<SensitivityResponse>(`/api/laboratory/${selected.id}/sensitivity`, { profile_id: selected.id })
        : Promise.resolve(null as never),
    [selected?.id],
  );
  const abl = useApi<{ rows: AblationRow[]; caveat?: string }>(
    () => (selected ? api.post(`/api/laboratory/${selected.id}/ablation`, { profile_id: selected.id }) : Promise.resolve(null as never)),
    [selected?.id],
  );

  const apply = async (perm: number[]) => {
    if (!selected) return;
    try {
      await api.post(`/api/laboratory/${selected.id}/apply`, { profile_id: selected.id, permutation: perm });
      setApplied(`Stored order: ${perm.map((i) => `#${i + 1}`).join(" → ")}`);
      lab.reload();
    } catch (e) {
      setApplied(e instanceof Error ? e.message : "Failed to apply");
    }
  };

  const noisy = opt.data?.n !== undefined && opt.data.n > 8;

  return (
    <>
      <ProfileBar
        title="Photo Laboratory"
        description="Module B. Maximise J(π) = Σ wₖ q_{π(k)} + λ·D(π) + μ·C(π) over orderings, then prove the result against exhaustive search where feasible."
      />
      {!selected && <NoProfile />}
      {selected && (
        <div className="space-y-6">
          <Card title="Objective weights" subtitle="λ rewards visual diversity, μ rewards flow between adjacent photographs. Both at zero reproduces the pure quality sort.">
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                { label: "λ · diversity", value: lambda, set: setLambda, hint: "0 = ignore redundancy between photos" },
                { label: "μ · coherence", set: setMu, value: mu, hint: "0 = ignore whether the sequence flows" },
              ].map((s) => (
                <label key={s.label} className="block">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>{s.label}</span>
                    <span className="tabular-nums text-slate-200">{s.value.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0} max={0.8} step={0.05} value={s.value}
                         onChange={(e) => s.set(Number(e.target.value))}
                         className="mt-2 w-full accent-[#7c5cff]" />
                  <p className="mt-1 text-[11px] text-slate-500">{s.hint}</p>
                </label>
              ))}
            </div>
          </Card>

          {opt.loading && <Loading what="the optimiser" />}
          {Boolean(opt.error) && <ErrorBox error={opt.error} onRetry={opt.reload} />}
          {opt.data && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Best objective" value={num(opt.data.best.objective, 4)}
                      hint={`+${pct(opt.data.best.improvement_vs_upload_order)} vs upload order`} tone="ok" />
                <Stat label="Strategy" value={opt.data.best.strategy}
                      hint={opt.data.best.proved_optimal ? "proved optimal by exhaustive search" : "heuristic — exhaustive infeasible"} />
                <Stat label="Evaluations" value={opt.data.best.evaluations.toLocaleString()} hint={`${opt.data.best.elapsed_ms.toFixed(2)} ms`} />
                <Stat label="Photographs" value={opt.data.n} hint={`n! = ${factorial(opt.data.n).toLocaleString()} permutations`} />
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Proposed order" subtitle="Click to persist as the profile's photo order.">
                  <div className="grid grid-cols-3 gap-2">
                    {opt.data.best.permutation.map((idx, slot) => (
                      <figure key={idx} className="overflow-hidden rounded-lg border border-ink-700">
                        <img src={urlFor(opt.data!.photo_ids[idx])}
                             alt={`Slot ${slot + 1}`} className="aspect-square w-full object-cover" loading="lazy" />
                        <figcaption className="flex justify-between p-1.5 text-[11px]">
                          <span className="text-accent">#{slot + 1}</span>
                          <span className="text-slate-400">{opt.data!.photo_labels[idx]}</span>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                  <button className="btn-primary mt-3 w-full" onClick={() => apply(opt.data!.best.permutation)}>
                    Apply this order
                  </button>
                  {applied && <p className="mt-2 text-xs text-ok">{applied}</p>}
                </Card>

                <Card title="Strategy comparison" subtitle="Every strategy is run on the same problem; disagreement is visible, not hidden.">
                  <table className="w-full text-xs">
                    <thead className="text-slate-500">
                      <tr><th className="text-left font-normal">strategy</th>
                          <th className="text-right font-normal">objective</th>
                          <th className="text-right font-normal">gap</th>
                          <th className="text-right font-normal">ms</th></tr>
                    </thead>
                    <tbody>
                      {opt.data.results.map((r) => (
                        <tr key={r.strategy} className="border-t border-ink-700/60">
                          <td className="py-1.5">
                            {r.strategy} {r.proved_optimal && <Pill tone="ok">optimal</Pill>}
                          </td>
                          <td className="text-right tabular-nums">{num(r.objective, 4)}</td>
                          <td className="text-right tabular-nums text-slate-400">
                            {num(opt.data!.best.objective - r.objective, 5)}
                          </td>
                          <td className="text-right tabular-nums text-slate-400">{r.elapsed_ms.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {noisy && (
                    <Caveat>
                      n = {opt.data.n}: exhaustive search is infeasible, so the “best” row is the best
                      of the heuristics run, not a proven optimum.
                    </Caveat>
                  )}
                </Card>
              </div>

              <Card title="Similarity matrix" subtitle="Cosine similarity between photographs at the chosen embeddings — the raw input to the diversity term.">
                <div className="overflow-x-auto">
                  <table className="text-[11px]">
                    <tbody>
                      {lab.data?.similarity_matrix.map((row, i) => (
                        <tr key={i}>
                          <th className="pr-2 text-right font-normal text-slate-500">#{i + 1}</th>
                          {row.map((v, j) => (
                            <td key={j} className="p-1 text-center tabular-nums"
                                style={{ background: `rgba(124,92,255,${Math.max(0, v) * 0.55})` }}>
                              {v.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Sensitivity" subtitle="Same photographs, different (λ, μ): does the optimum move?">
                  {sens.loading && <Loading what="the sensitivity grid" />}
                  {Boolean(sens.error) && <ErrorBox error={sens.error} onRetry={sens.reload} />}
                  {sens.data && (
                    <table className="w-full text-xs">
                      <thead className="text-slate-500">
                        <tr><th className="text-left font-normal">λ</th><th className="text-left font-normal">μ</th>
                            <th className="text-right font-normal">optimised J</th>
                            <th className="text-right font-normal">order</th></tr>
                      </thead>
                      <tbody>
                        {sens.data.grid.map((g, i) => (
                          <tr key={i} className="border-t border-ink-700/60">
                            <td>{g.lambda.toFixed(2)}</td>
                            <td>{g.mu.toFixed(2)}</td>
                            <td className="text-right tabular-nums">{num(g.optimized_total, 4)}</td>
                            <td className="text-right font-mono text-slate-400">
                              {g.optimized_permutation.map((x) => x + 1).join("")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </Card>

                <Card title="Leverage ablation" subtitle="Drop each photograph and re-optimise: how much objective does it actually carry?">
                  {abl.loading && <Loading what="the ablation" />}
                  {Boolean(abl.error) && <ErrorBox error={abl.error} onRetry={abl.reload} />}
                  {abl.data && (
                    <ul className="space-y-1.5 text-xs">
                      {abl.data.rows.map((r) => (
                        <li key={r.removed_index} className="flex justify-between border-b border-ink-700/60 py-1">
                          <span className="text-slate-300">− {r.photo_id ? `#${(r.position ?? 0) + 1}` : r.removed_index}</span>
                          <span className="tabular-nums text-slate-200">loss {num(r.loss, 4)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                  <Caveat>
                    A large loss means the set depended on that photograph. It is a sensitivity measure on
                    the declared objective, not evidence about how other people respond.
                  </Caveat>
                </Card>
              </div>
              <Caveat>{opt.data.caveat}</Caveat>
            </>
          )}
        </div>
      )}
    </>
  );
}

function factorial(n: number): number {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}
