import { useState } from "react";
import {
  Bar as RBar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, BenchmarkResponse, Capabilities } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, Caveat, ErrorBox, Loading, Stat } from "../components/ui";
import { ms } from "../lib/format";

export function Benchmarks() {
  const caps = useApi<Capabilities>(() => api.get<Capabilities>("/api/system/capabilities"), []);
  const history = useApi<{ runs: { id: string; suite: string; machine: Record<string, unknown>; created_at: string; n_results: number }[] }>(
    () => api.get("/api/benchmarks/history"),
    [],
  );
  const [run, setRun] = useState<BenchmarkResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const execute = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<BenchmarkResponse>("/api/benchmarks/run", {
        sizes: [4096, 32768], k: 10, dim: 128, repeats: 3, threads: [1, 2],
      });
      setRun(res);
      history.reload();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const topk = run?.results.filter((r) => r.benchmark === "top_k") ?? [];
  const cosine = run?.results.filter((r) => r.benchmark === "batch_cosine") ?? [];
  const chartData = run
    ? Object.entries(
        cosine.reduce<Record<string, Record<string, number | string>>>((acc, r) => {
          const key = `n=${r.n} t=${r.threads}`;
          acc[key] = acc[key] ?? { name: key };
          acc[key][r.kernel ?? "kernel"] = r.median_ms;
          return acc;
        }, {}),
      ).map(([, v]) => v)
    : [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-white">Systems Benchmarking</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">
          Live measurements from this machine, executed now through the native core. Stored runs are
          retrievable; the offline suite (Google Benchmark + NumPy baselines) writes
          <code className="mx-1 rounded bg-ink-700 px-1">benchmarks/results/</code>.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="CPU target" value={caps.data?.native.target ?? "…"} hint={`dispatch mask ${caps.data?.native.mask ?? "—"}`} tone="ok" />
        <Stat label="Compiler" value={caps.data?.native.compiler ?? "…"}
              hint={`${caps.data?.native.compiler_version ?? ""} · C++${caps.data?.native.cxx_dialect ?? ""}`} />
        <Stat label="OpenMP" value={caps.data?.native.has_openmp ? "on" : "off"} />
        <Stat label="Stored runs" value={history.data?.runs.length ?? 0} hint="from /api/benchmarks/history" />
      </div>

      <Card title="Run the suite" subtitle="Top-k strategies, in-house SIMD cosine, Eigen GEMM and a NumPy BLAS baseline on identical arrays.">
        <button className="btn-primary" onClick={execute} disabled={busy}>
          {busy ? "Measuring…" : "Run benchmarks now"}
        </button>
        {busy && <Loading what="measurements" />}
        {Boolean(error) && <div className="mt-3"><ErrorBox error={error} onRetry={execute} /></div>}
        {run && (
          <p className="mt-3 text-xs text-slate-400">
            run <span className="font-mono text-slate-300">{run.run_id}</span> · {run.note}
          </p>
        )}
        {run?.clamped.map((c) => (
          <p key={c.requested} className="mt-2 rounded-lg border border-warn/40 bg-warn/10 p-2 text-xs text-amber-200">
            n={c.requested} was reduced to {c.used}: {c.reason}.
          </p>
        ))}
      </Card>

      {run && (
        <>
          <Card title="Cosine kernels vs NumPy BLAS" subtitle="Lower is better. The in-house scalar kernel is memory-bound; Eigen closes the gap and wins at 64×4096 in the offline suite.">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke="#25303f" strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} tickFormatter={(v) => `${Number(v).toFixed(1)}`} />
                  <Tooltip contentStyle={{ background: "#111823", border: "1px solid #25303f", fontSize: 12 }}
                           formatter={(v) => `${Number(v).toFixed(3)} ms`} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {["eigen", "inhouse", "numpy_blas"].map((k, i) => (
                    <RBar key={k} dataKey={k} fill={["#7c5cff", "#fbbf24", "#34d399"][i]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title="Top-k strategies" subtitle="Measured per (n, strategy); the heap wins as n grows because it never materialises a full sort.">
            <table className="w-full text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="text-left font-normal">n</th>
                  <th className="text-left font-normal">strategy</th>
                  <th className="text-right font-normal">median</th>
                  <th className="text-right font-normal">min</th>
                  <th className="text-right font-normal">p95</th>
                </tr>
              </thead>
              <tbody>
                {topk.map((r) => (
                  <tr key={`${r.n}-${r.strategy}`} className="border-t border-ink-700/60">
                    <td className="py-1.5 tabular-nums">{r.n.toLocaleString()}</td>
                    <td>{r.strategy}</td>
                    <td className="text-right tabular-nums">{ms(r.median_ms)}</td>
                    <td className="text-right tabular-nums text-slate-400">{ms(r.min_ms)}</td>
                    <td className="text-right tabular-nums text-slate-400">{ms(r.p95_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Caveat>
            These numbers include Python call overhead and were taken on a shared 2-vCPU container.
            Absolute values do not transfer between machines; the ratios between strategies do.
          </Caveat>
        </>
      )}

      {history.data && history.data.runs.length > 0 && (
        <Card title="Stored runs" subtitle="Every run is persisted with its machine description.">
          <ul className="space-y-1 text-xs">
            {history.data.runs.slice(0, 8).map((r) => (
              <li key={r.id} className="flex flex-wrap justify-between gap-2 border-b border-ink-700/60 py-1">
                <span className="font-mono text-slate-400">{r.id.slice(0, 8)}</span>
                <span className="text-slate-300">{r.suite}</span>
                <span className="text-slate-500">{r.n_results} measurements</span>
                <span className="text-slate-500">{new Date(r.created_at).toLocaleString()}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
