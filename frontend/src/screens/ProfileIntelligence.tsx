import { api, Photo, Report } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useProfiles } from "../lib/store";
import { Card, ErrorBox, Loading, Stat, Pill, Caveat, Bar } from "../components/ui";
import { NoProfile, ProfileBar } from "../components/ProfileBar";
import { pct, num } from "../lib/format";

const SEVERITY_TONE: Record<string, "bad" | "warn" | "ok"> = { high: "bad", medium: "warn", low: "ok" };

export function ProfileIntelligence() {
  const { selected } = useProfiles();
  const report = useApi<Report>(
    () => (selected ? api.post<Report>(`/api/intelligence/${selected.id}/report`, {}) : Promise.resolve(null as never)),
    [selected?.id],
  );
  const thumbs = useApi<{ photos: Photo[] }>(
    () =>
      selected
        ? api.get<{ photos: Photo[] }>(`/api/profiles/${selected.id}/photos`)
        : Promise.resolve({ photos: [] }),
    [selected?.id],
  );
  const urlFor = (id: string) => thumbs.data?.photos.find((p) => p.id === id)?.url ?? "";

  return (
    <>
      <ProfileBar
        title="Profile Intelligence"
        description="Module A. Quality, composition, visual diversity, text specificity and their coherence — measured from your own uploads."
      />
      {!selected && <NoProfile />}
      {selected && report.loading && <Loading what="the intelligence report" />}
      {Boolean(report.error) && <ErrorBox error={report.error} onRetry={report.reload} />}

      {report.data && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Visual diversity"
              value={num(report.data.visual_diversity.diversity)}
              hint={`1 − mean pairwise cosine · ${report.data.visual_diversity.redundant_pairs.length} pair(s) above ${report.data.visual_diversity.threshold}`}
              tone={report.data.visual_diversity.diversity < 0.15 ? "warn" : "ok"}
            />
            <Stat
              label="Evidence coverage"
              value={pct(report.data.confidence.evidence_coverage, 0)}
              hint={report.data.confidence.interpretation}
            />
            <Stat
              label="Photographs"
              value={report.data.photos.length}
              hint={`mean quality ${pct(
                report.data.photos.reduce((a, p) => a + p.quality_estimate, 0) /
                  Math.max(1, report.data.photos.length),
              )}`}
            />
            <Stat
              label="Recommendations"
              value={report.data.recommendations.length}
              hint={`${report.data.recommendations.filter((r) => r.severity === "high").length} high severity`}
              tone={report.data.recommendations.some((r) => r.severity === "high") ? "warn" : "ok"}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-5">
            <Card className="lg:col-span-3" title="Recommendations" subtitle="Ordered by expected impact on perceived quality. Each carries the reason and the concrete action.">
              <ul className="space-y-3">
                {report.data.recommendations.map((r) => (
                  <li key={r.code} className="rounded-lg border border-ink-700 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-medium text-slate-100">{r.title}</h3>
                      <Pill tone={SEVERITY_TONE[r.severity] ?? "default"}>{r.severity}</Pill>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-slate-400">{r.why}</p>
                    <p className="mt-2 text-xs leading-relaxed text-slate-300">
                      <span className="text-accent">→ </span>{r.action}
                    </p>
                  </li>
                ))}
              </ul>
            </Card>

            <div className="lg:col-span-2 space-y-6">
              <Card title="Quality components" subtitle="Weighted sum reported per photograph; weights are fixed and shown.">
                {report.data.photos[0]?.quality_components && (
                  <div className="space-y-2">
                    {Object.entries(report.data.photos[0].quality_components).map(([k, v]) => (
                      <div key={k}>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">{k}</span>
                          <span className="tabular-nums text-slate-200">{num(v as number, 3)}</span>
                        </div>
                        <Bar value={v as number} />
                      </div>
                    ))}
                  </div>
                )}
                <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
                  Components for photograph #1. Sharpness is a variance-of-Laplacian proxy; “face” only
                  reports the detected area fraction, never identity, attractiveness or demographics.
                </p>
              </Card>

              <Card title="Text specificity" subtitle="Concrete nouns, numbers, place/time anchors — a proxy for how much a prompt actually says.">
                <ul className="space-y-2">
                  {report.data.text_blocks.map((b) => (
                    <li key={b.index}>
                      <div className="flex justify-between text-xs">
                        <span className="truncate pr-2 text-slate-300">{b.text.slice(0, 48)}…</span>
                        <span className="tabular-nums text-slate-200">{num(b.specificity.score)}</span>
                      </div>
                      <Bar value={b.specificity.score} tone={b.specificity.score < 0.25 ? "warn" : "ok"} />
                    </li>
                  ))}
                </ul>
              </Card>
            </div>
          </div>

          <Card title="Photographs" subtitle="Stored bytes are served through a short-lived signature, so thumbnails work without embedding the bearer token in the page.">
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
              {report.data.photos.map((p, i) => (
                <figure key={p.index} className="overflow-hidden rounded-lg border border-ink-700">
                  {urlFor(report.data!.photo_ids[i]) ? (
                    <img src={urlFor(report.data!.photo_ids[i])} alt={`Photograph ${p.index + 1}`}
                         className="aspect-square w-full object-cover" loading="lazy" />
                  ) : (
                    <div className="aspect-square w-full bg-ink-700" />
                  )}
                  <figcaption className="p-1 text-[10px] text-slate-400">
                    #{p.index + 1} · q {pct(p.quality_estimate, 0)}
                  </figcaption>
                </figure>
              ))}
            </div>
          </Card>

          <Card title="Redundant pairs" subtitle="Cosine similarity above the threshold means two photographs are saying the same thing.">
            {report.data.visual_diversity.redundant_pairs.length === 0 ? (
              <p className="text-xs text-slate-400">No pair exceeds the threshold — the set is visually distinct.</p>
            ) : (
              <ul className="space-y-1 text-xs text-slate-300">
                {report.data.visual_diversity.redundant_pairs.map((p) => (
                  <li key={`${p.a}-${p.b}`} className="flex justify-between border-b border-ink-700/60 py-1">
                    <span>#{p.a + 1} ↔ #{p.b + 1}</span>
                    <span className="tabular-nums">{num(p.similarity, 4)}</span>
                  </li>
                ))}
              </ul>
            )}
            <Caveat>
              Similarity is CLIP image-embedding cosine. It measures visual redundancy, not whether a
              photograph is a good one.
            </Caveat>
          </Card>

          <Card title="Limitations" subtitle="Reported by the backend with every analysis.">
            <ul className="space-y-1 text-xs leading-relaxed text-slate-400">
              {report.data.limitations.map((l) => (
                <li key={l}>· {l}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </>
  );
}
