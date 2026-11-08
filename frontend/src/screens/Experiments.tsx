import { useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Analysis, api, Experiment } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, Caveat, ErrorBox, Empty, Loading, Pill, Stat } from "../components/ui";
import { num, pct } from "../lib/format";

export function Experiments() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const list = useApi<{ experiments: Experiment[] }>(
    () => api.get<{ experiments: Experiment[] }>("/api/experiments"),
    [],
  );
  const current = selectedId ?? list.data?.experiments[0]?.id ?? null;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-white">Experimentation Lab</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">
          Module D. Beta-Binomial posteriors, P(θ_A &gt; θ_B) by numerical integration,
          expected-loss stopping and Thompson allocation. Missing impressions are never counted as failures.
        </p>
      </header>

      {list.loading && <Loading what="experiments" />}
      {Boolean(list.error) && <ErrorBox error={list.error} onRetry={list.reload} />}
      {list.data && list.data.experiments.length === 0 && (
        <Empty title="No experiments yet">
          Create one below. The prior, the minimum observation count and the loss tolerance are all
          recorded with the experiment so the decision rule is reproducible.
        </Empty>
      )}

      <div className="grid gap-6 lg:grid-cols-4">
        <div className="lg:col-span-1 space-y-2">
          {list.data?.experiments.map((e) => (
            <button
              key={e.id}
              onClick={() => setSelectedId(e.id)}
              className={`w-full rounded-lg border p-3 text-left transition ${
                current === e.id ? "border-accent bg-accent/10" : "border-ink-700 hover:border-ink-600"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-100">{e.name}</span>
                <Pill tone={e.status === "concluded" ? "ok" : "default"}>{e.status}</Pill>
              </div>
              <div className="mt-1 text-[11px] text-slate-500">
                {e.variants.length} variants · {e.metric} · prior {e.prior}
              </div>
            </button>
          ))}
          <CreateExperiment onCreated={list.reload} />
        </div>

        <div className="lg:col-span-3">
          {current ? <ExperimentDetail id={current} onChanged={list.reload} /> : (
            <Empty title="Select or create an experiment" />
          )}
        </div>
      </div>
    </div>
  );
}

function ExperimentDetail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const analysis = useApi<Analysis>(() => api.get<Analysis>(`/api/experiments/${id}/analysis`), [id]);
  const [counts, setCounts] = useState<Record<string, { successes: number; failures: number; missing: number }>>({});
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const record = async (variantId: string) => {
    const c = counts[variantId] ?? { successes: 0, failures: 0, missing: 0 };
    setBusy(true);
    setMsg(null);
    try {
      await api.post(`/api/experiments/${id}/observations`, { variant_id: variantId, ...c });
      setCounts((prev) => ({ ...prev, [variantId]: { successes: 0, failures: 0, missing: 0 } }));
      analysis.reload();
      onChanged();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Failed to record");
    } finally {
      setBusy(false);
    }
  };

  if (analysis.loading) return <Loading what="the posterior" />;
  if (analysis.error) return <ErrorBox error={analysis.error} onRetry={analysis.reload} />;
  if (!analysis.data) return null;
  const a = analysis.data;

  // Densities come from the backend (native beta_pdf evaluated server-side), so the
  // chart plots the same numbers the analysis reports rather than a second,
  // unverified implementation in the browser.
  const labelOf = (variantId: string) => a.variants.find((v) => v.id === variantId)?.label ?? variantId;
  const series = Object.entries(a.density).map(([variantId, d]) => ({
    variantId,
    label: labelOf(variantId),
    x: d.x,
    pdf: d.pdf,
  }));
  const densityData = series.length
    ? series[0].x.map((x, i) => {
        const row: Record<string, number> = { x };
        series.forEach((s) => {
          row[s.label] = s.pdf[i];
        });
        return row;
      })
    : [];
  const palette = ["#7c5cff", "#34d399", "#fbbf24", "#f87171", "#60a5fa", "#f472b6"];

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        <Stat
          label="Decision"
          value={a.decision.stop ? "stop" : "continue"}
          tone={a.decision.stop ? "ok" : "warn"}
          hint={a.decision.reason}
        />
        <Stat
          label="Leader"
          value={a.decision.leader}
          hint={`P(${a.decision.leader} is best) = ${num(a.decision.leader_probability, 4)} · P(${a.decision.a} > ${a.decision.b}) = ${num(a.decision.prob_a_better, 4)}`}
          tone={a.decision.leader_probability > 0.95 ? "ok" : "warn"}
        />
        <Stat
          label="Expected loss"
          value={num(a.decision.expected_loss_choose_leader, 5)}
          hint={`if we ship the leader; tolerance ${a.experiment.loss_tolerance}`}
        />
        <Stat label="Family α" value={`${a.multiplicity.bonferroni_alpha.toFixed(4)}`}
              hint={`Bonferroni over ${a.multiplicity.variants} variants (Šidák ${a.multiplicity.sidak_alpha.toFixed(4)})`} />
      </div>

      <Card title="Posterior densities" subtitle={`Prior: ${a.prior.name} — Beta(${a.prior.alpha}, ${a.prior.beta}). ${a.prior.explanation}`}>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={densityData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#25303f" strokeDasharray="3 3" />
              <XAxis dataKey="x" stroke="#64748b" fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip
                contentStyle={{ background: "#111823", border: "1px solid #25303f", fontSize: 12 }}
                labelFormatter={(v) => `conversion ${(Number(v) * 100).toFixed(1)}%`}
              />
              {series.map((sv, i) => (
                <Line key={sv.variantId} type="monotone" dataKey={sv.label}
                      stroke={palette[i % palette.length]} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Variants" subtitle="Posterior is Beta(prior_α + s, prior_β + f). Missing observations are excluded from the denominator.">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="text-left font-normal">variant</th>
                <th className="text-right font-normal">s</th>
                <th className="text-right font-normal">f</th>
                <th className="text-right font-normal">missing</th>
                <th className="text-right font-normal">rate</th>
                <th className="text-right font-normal">posterior mean</th>
                <th className="text-right font-normal">95% CI</th>
                <th className="text-right font-normal">Thompson share</th>
              </tr>
            </thead>
            <tbody>
              {a.variants.map((v, i) => (
                <tr key={v.id} className="border-t border-ink-700/60">
                  <td className="py-2">{v.label} <span className="text-slate-500">({v.maturity})</span></td>
                  <td className="text-right tabular-nums">{v.successes}</td>
                  <td className="text-right tabular-nums">{v.failures}</td>
                  <td className="text-right tabular-nums text-slate-500">{v.missing}</td>
                  <td className="text-right tabular-nums">{pct(v.observed_rate)}</td>
                  <td className="text-right tabular-nums text-slate-100">{num(v.posterior.mean, 4)}</td>
                  <td className="text-right tabular-nums text-slate-400">
                    [{num(v.credible_interval.lo, 3)}, {num(v.credible_interval.hi, 3)}]
                  </td>
                  <td className="text-right tabular-nums">{pct(a.thompson.share[i], 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Observation timeline" subtitle="Raw recorded batches, in the order they were added.">
        <ul className="space-y-1 text-xs">
          {a.timeline.length === 0 && <li className="text-slate-500">No observations recorded yet.</li>}
          {a.timeline.map((t, i) => (
            <li key={i} className="flex flex-wrap justify-between gap-2 border-b border-ink-700/60 py-1">
              <span className="text-slate-300">{t.variant}</span>
              <span className="tabular-nums text-slate-400">
                +{t.context.successes} successes · +{t.context.failures} failures
                {t.context.missing ? ` · ${t.context.missing} missing` : ""}
              </span>
              <span className="text-slate-500">{new Date(t.observed_at).toLocaleString()}</span>
            </li>
          ))}
        </ul>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Record observations" subtitle="Per-arm entry — each variant accumulates its own counts.">
          <div className="space-y-3">
            {a.variants.map((v) => {
              const c = counts[v.id] ?? { successes: 0, failures: 0, missing: 0 };
              const set = (patch: Partial<typeof c>) =>
                setCounts((prev) => ({ ...prev, [v.id]: { ...c, ...patch } }));
              return (
                <div key={v.id} className="rounded-lg border border-ink-700 p-2">
                  <div className="mb-1 text-xs text-slate-300">{v.label}</div>
                  <div className="flex gap-2">
                    {(["successes", "failures", "missing"] as const).map((field) => (
                      <label key={field} className="flex-1">
                        <span className="text-[10px] uppercase tracking-wide text-slate-500">{field.slice(0, 4)}</span>
                        <input type="number" min={0} value={c[field]} className="input mt-0.5 py-1 text-xs"
                               onChange={(e) => set({ [field]: Number(e.target.value) } as Partial<typeof c>)} />
                      </label>
                    ))}
                    <button className="btn-ghost mt-4 px-2 py-1 text-xs" disabled={busy} onClick={() => record(v.id)}>
                      add
                    </button>
                  </div>
                </div>
              );
            })}
            {msg && <p className="text-xs text-bad">{msg}</p>}
          </div>
        </Card>

        <Card title="Model checks">
          <ul className="space-y-2 text-xs leading-relaxed text-slate-400">
            <li>
              · Hierarchical fit: population α {num(a.hierarchical.population.alpha, 2)}, β{" "}
              {num(a.hierarchical.population.beta, 2)} (mean {pct(a.hierarchical.population.mean)});
              shrinkage {num(a.hierarchical.shrinkage, 2)} pulls thin arms toward the pool,
              overdispersion {num(a.hierarchical.overdispersion, 4)}.
            </li>
            <li>
              · Capacity: {a.capacity.observed_total} observations against a planning target of{" "}
              {a.capacity.planning.per_arm} per arm ({pct(a.capacity.planning.power, 0)} power,
              α {a.capacity.planning.alpha}).
            </li>
            <li>· Multiplicity: {a.multiplicity.note}</li>
            <li>· {a.missing_data_policy}</li>
          </ul>
          <Caveat>
            A stopping rule evaluated on the same data that produced the posterior is optimistic.
            The expected-loss rule here is a decision aid, not a frequentist guarantee.
          </Caveat>
        </Card>
      </div>
    </div>
  );
}

function CreateExperiment({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [labels, setLabels] = useState(["control", "treatment"]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await api.post<Experiment>("/api/experiments", {
        name,
        hypothesis: hypothesis || null,
        variants: labels.filter(Boolean).map((label) => ({ label })),
      });
      setName("");
      setHypothesis("");
      setOpen(false);
      onCreated();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button className="btn-ghost w-full" onClick={() => setOpen(true)}>
        + New experiment
      </button>
    );
  }
  return (
    <div className="card space-y-2">
      <input className="input" placeholder="Experiment name" value={name} onChange={(e) => setName(e.target.value)} />
      <textarea className="input min-h-[60px]" placeholder="Hypothesis (optional)"
                value={hypothesis} onChange={(e) => setHypothesis(e.target.value)} />
      <div className="space-y-1">
        {labels.map((l, i) => (
          <input key={i} className="input" value={l}
                 onChange={(e) => setLabels(labels.map((x, j) => (j === i ? e.target.value : x)))} />
        ))}
      </div>
      <div className="flex gap-2">
        <button className="btn-ghost flex-1 text-xs" onClick={() => setLabels([...labels, `variant ${labels.length + 1}`])}>
          + variant
        </button>
      </div>
      {Boolean(err) && <ErrorBox error={err} />}
      <div className="flex gap-2">
        <button className="btn-primary flex-1" disabled={busy || !name} onClick={submit}>
          {busy ? "Creating…" : "Create"}
        </button>
        <button className="btn-ghost" onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </div>
  );
}
