import { useMemo, useState } from "react";
import { api, CounterfactualResponse, Photo } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useProfiles } from "../lib/store";
import { Card, Caveat, ErrorBox, Loading, Stat, Bar } from "../components/ui";
import { NoProfile, ProfileBar } from "../components/ProfileBar";
import { num, pct } from "../lib/format";

export function Counterfactual() {
  const { selected } = useProfiles();
  const photos = useApi<{ photos: Photo[] }>(
    () => (selected ? api.get<{ photos: Photo[] }>(`/api/profiles/${selected.id}/photos`) : Promise.resolve({ photos: [] })),
    [selected?.id],
  );
  const [keep, setKeep] = useState<Set<string>>(new Set());
  const [text, setText] = useState("");
  const [result, setResult] = useState<CounterfactualResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const available = photos.data?.photos ?? [];
  // Selection is compared against itself: only the *order* and the text change.
  // Q averages over the selection, so a different-sized "modified" set would
  // compare different quantities (the API rejects that).
  const baselineIds = available.map((p) => p.id);

  const modifiedIds = useMemo(() => {
    const kept = baselineIds.filter((id) => keep.has(id));
    return kept.length ? kept : baselineIds;
  }, [baselineIds, keep]);

  const run = async () => {
    if (!selected || baselineIds.length < 2) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<CounterfactualResponse>("/api/counterfactual/simulate", {
        profile_id: selected.id,
        baseline_photo_ids: baselineIds,
        modified_photo_ids: modifiedIds,
        modified_prompts: text.trim() ? [text, null, null] : null,
        seed: 42,
      });
      setResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <ProfileBar
        title="Counterfactual Simulator"
        description="Module E. Evaluate the same declared objective on two configurations and attribute the difference by permutation importance. This is a prediction under a stated model, not a causal effect."
      />
      {!selected && <NoProfile />}
      {selected && (
        <div className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Baseline" subtitle="Your profile exactly as stored, in upload order.">
              <div className="grid grid-cols-3 gap-2">
                {available.map((p) => (
                  <figure key={p.id} className="overflow-hidden rounded-lg border border-ink-700">
                    <img src={p.url} alt={`#${p.position + 1}`}
                         className="aspect-square w-full object-cover" loading="lazy" />
                    <figcaption className="p-1 text-[10px] text-slate-400">#{p.position + 1} · q {pct(p.quality, 0)}</figcaption>
                  </figure>
                ))}
              </div>
            </Card>

            <Card title="Modified" subtitle="Toggle photographs to drop, and rewrite one prompt answer.">
              <div className="grid grid-cols-3 gap-2">
                {available.map((p) => {
                  const dropped = !keep.has(p.id);
                  return (
                    <button key={p.id} onClick={() => {
                      const next = new Set(keep.size === 0 ? baselineIds : keep);
                      if (next.has(p.id)) next.delete(p.id); else next.add(p.id);
                      setKeep(next);
                    }} className={`overflow-hidden rounded-lg border text-left transition ${
                      dropped ? "border-bad/60 opacity-40" : "border-ok/60"
                    }`}>
                      <img src={p.url} alt={`#${p.position + 1}`}
                           className="aspect-square w-full object-cover" loading="lazy" />
                      <div className="p-1 text-[10px] text-slate-400">{dropped ? "dropped" : "kept"}</div>
                    </button>
                  );
                })}
              </div>
              <textarea
                className="input mt-3 min-h-[80px] text-xs"
                placeholder="Rewrite a prompt answer (optional) — concrete detail raises specificity."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <p className="mt-1 text-[11px] text-slate-500">
                Dropping every photograph is not allowed: {modifiedIds.length} of {baselineIds.length} selected.
              </p>
              <button className="btn-primary mt-3 w-full" onClick={run} disabled={busy || baselineIds.length < 2}>
                {busy ? "Simulating…" : "Simulate"}
              </button>
            </Card>
          </div>

          {busy && <Loading what="the counterfactual" />}
          {Boolean(error) && <ErrorBox error={error} onRetry={run} />}

          {result && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Q baseline" value={num(result.q_baseline, 4)} />
                <Stat label="Q modified" value={num(result.q_modified, 4)} />
                <Stat label="ΔQ" value={`${result.delta_q >= 0 ? "+" : ""}${num(result.delta_q, 4)}`}
                      tone={result.delta_q >= 0 ? "ok" : "bad"}
                      hint={result.relative_change !== null ? `${(result.relative_change * 100).toFixed(1)}% relative` : ""} />
                <Stat label="Attribution method" value="permutation" hint={result.attribution.method} />
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Component comparison" subtitle="Q = 0.55·photo quality + 0.25·diversity + 0.20·text specificity.">
                  {Object.keys(result.components_modified.weights ?? {}).map((k) => {
                    const w = result.components_modified.weights![k];
                    const b = Number(result.components_baseline[k] ?? 0);
                    const m = Number(result.components_modified[k] ?? 0);
                    return (
                      <div key={k} className="mb-3">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300">{k} <span className="text-slate-500">(w={w})</span></span>
                          <span className="tabular-nums">{num(b, 3)} → {num(m, 3)}</span>
                        </div>
                        <Bar value={Math.max(b, m)} />
                      </div>
                    );
                  })}
                </Card>

                <Card title="Permutation importance" subtitle="Drop in ΔQ when each input block is shuffled.">
                  <ul className="space-y-2 text-xs">
                    {Object.entries(result.attribution.importance).map(([k, v]) => (
                      <li key={k}>
                        <div className="flex justify-between">
                          <span className="text-slate-300">{k.replace(/_/g, " ")}</span>
                          <span className="tabular-nums">{num(v, 4)}</span>
                        </div>
                        <Bar value={Math.abs(v)} max={Math.max(...Object.values(result.attribution.importance).map(Math.abs), 1e-9)} />
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[11px] text-slate-500">
                    {result.attribution.n_permutations} shuffles per block, seed fixed for reproducibility.
                  </p>
                </Card>
              </div>

              <Card title="Sensitivity" subtitle="One-step changes around the modified configuration.">
                <ul className="space-y-1 text-xs">
                  {result.sensitivity.map((s) => (
                    <li key={s.change} className="flex flex-wrap justify-between gap-2 border-b border-ink-700/60 py-1">
                      <span className="text-slate-300">{s.change}</span>
                      <span className="tabular-nums text-slate-200">ΔQ {num(s.delta_q, 4)}</span>
                      {s.note && <span className="w-full text-[11px] text-slate-500">{s.note}</span>}
                    </li>
                  ))}
                </ul>
              </Card>

              <Card title="Uncertainty and interpretation">
                <p className="text-xs leading-relaxed text-slate-300">{result.interpretation}</p>
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  <strong className="text-slate-400">Uncertainty:</strong> {result.uncertainty.explanation}
                </p>
                <p className="mt-1 font-mono text-[11px] text-slate-600">{result.model_version}</p>
                <Caveat>
                  To turn this prediction into evidence you would have to run the two configurations as
                  an experiment with randomised assignment — that is what the Experimentation Lab is for.
                </Caveat>
              </Card>
            </>
          )}
        </div>
      )}
    </>
  );
}
