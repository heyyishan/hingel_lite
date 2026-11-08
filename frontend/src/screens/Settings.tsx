import { useState } from "react";
import { api, Capabilities } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useProfiles } from "../lib/store";
import { Card, Caveat, ErrorBox, Loading, Stat, Pill } from "../components/ui";

interface Policy {
  [k: string]: unknown;
}

export function Settings() {
  const caps = useApi<Capabilities>(() => api.get<Capabilities>("/api/system/capabilities"), []);
  const policy = useApi<Policy>(() => api.get<Policy>("/api/privacy/policy"), []);
  const audit = useApi<{ events: { kind: string; at: string | null; payload?: unknown }[] }>(
    () => api.get("/api/privacy/audit"),
    [],
  );
  const { selected, refresh } = useProfiles();
  const [exported, setExported] = useState<string | null>(null);
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [note, setNote] = useState<string | null>(null);

  const doExport = async () => {
    setBusy(true);
    setErr(null);
    try {
      const data = await api.get<unknown>("/api/privacy/export");
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      setExported(URL.createObjectURL(blob));
      setNote("Export ready — download it below. It contains every row this account owns.");
      audit.reload();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const deleteProfile = async () => {
    if (!selected) return;
    setBusy(true);
    setErr(null);
    try {
      await api.del(`/api/profiles/${selected.id}`);
      setNote(`Profile “${selected.name}” and its photographs were deleted. Files removed from disk.`);
      refresh();
      audit.reload();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const sweep = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await api.post<{ retention_days: number; cutoff: string; photographs_removed: number }>(
        "/api/privacy/retention/sweep",
        {},
      );
      setNote(
        `Retention sweep (${res.retention_days} days, cutoff ${new Date(res.cutoff).toLocaleString()}) ` +
          `removed ${res.photographs_removed} photographs.`,
      );
      audit.reload();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-white">Settings &amp; Privacy</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">
          What is stored, where it lives, how to take it out and how to make it disappear.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Native core" value={caps.data?.native.native_available ? "yes" : "no"}
              tone={caps.data?.native.native_available ? "ok" : "bad"}
              hint={`${caps.data?.native.compiler ?? ""} ${caps.data?.native.compiler_version ?? ""}`} />
        <Stat label="Database" value={caps.data?.database.dialect ?? "…"}
              hint={caps.data?.database.pgvector ? `pgvector ${caps.data.database.pgvector}` : "no vector index"} />
        <Stat label="Cache" value={caps.data?.cache.enabled ? "redis" : "off"} hint="optional by design" />
        <Stat label="Photo storage" value="local disk" hint={caps.data?.storage_dir} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Data export" subtitle="Everything this account owns, as JSON: profiles, prompts, photo metadata, embeddings, experiments, conversations and analytics.">
          <button className="btn-primary" onClick={doExport} disabled={busy}>Build export</button>
          {exported && (
            <a className="btn-ghost ml-2 inline-block" href={exported} download="attraction-engine-export.json">
              Download
            </a>
          )}
          <Caveat>
            Photographs are referenced by hash and path, not embedded, so an export cannot be used to
            re-upload images. The files themselves are yours on disk.
          </Caveat>
        </Card>

        <Card title="Delete a profile" subtitle="Removes the profile, its prompts, photographs, embeddings and stored image files.">
          <select className="input" value={selected?.id ?? ""} disabled
                  onChange={() => undefined} aria-label="Profile to delete">
            {selected && <option>{selected.name}</option>}
          </select>
          <input className="input mt-2" placeholder="Type the profile name to confirm"
                 value={confirm} onChange={(e) => setConfirm(e.target.value)} />
          <button className="btn-ghost mt-2 w-full border-bad/60 text-bad hover:border-bad"
                  disabled={busy || !selected || confirm !== selected.name} onClick={deleteProfile}>
            Delete profile permanently
          </button>
        </Card>

        <Card title="Retention sweep" subtitle="Applies the configured retention window to everything not marked to keep.">
          <button className="btn-ghost" onClick={sweep} disabled={busy}>Run sweep now</button>
          <div className="mt-3 text-xs leading-relaxed text-slate-400">
            {(policy.data?.deletion as string) ?? "Deletion is irreversible and removes stored files."}
          </div>
        </Card>

        <Card title="Audit log" subtitle="Every privileged action on this account is recorded.">
          {audit.loading && <Loading what="audit events" />}
          {Boolean(audit.error) && <ErrorBox error={audit.error} onRetry={audit.reload} />}
          {audit.data?.events.length === 0 && (
            <p className="text-xs text-slate-500">No events recorded yet.</p>
          )}
          <ul className="max-h-56 space-y-1 overflow-auto text-xs">
            {audit.data?.events.map((e, i) => (
              <li key={`${e.kind}-${i}`} className="flex justify-between border-b border-ink-700/60 py-1">
                <span className="text-slate-300">{e.kind}</span>
                <span className="text-slate-500">{e.at ? new Date(e.at).toLocaleString() : "—"}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      {note && <p className="rounded-lg border border-ok/40 bg-ok/10 p-3 text-xs text-emerald-200">{note}</p>}
      {Boolean(err) && <ErrorBox error={err} />}

      <Card title="Published privacy policy" subtitle="Returned by GET /api/privacy/policy — the same text the backend enforces.">
        {policy.loading && <Loading what="the policy" />}
        {policy.data && (
          <ul className="space-y-2 text-xs leading-relaxed text-slate-400">
            {Object.entries(policy.data).map(([k, v]) => (
              <li key={k}>
                <span className="text-slate-300">{k.replace(/_/g, " ")}: </span>
                {typeof v === "string" ? v : JSON.stringify(v)}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Environment">
        <div className="grid gap-2 text-xs sm:grid-cols-2">
          {caps.data && Object.entries(caps.data.machine).map(([k, v]) => (
            <div key={k} className="kv"><span className="text-slate-400">{k}</span><span className="truncate text-right">{String(v)}</span></div>
          ))}
        </div>
        <p className="mt-3 flex flex-wrap gap-2">
          <Pill>{caps.data?.encoders.image_encoder ?? "…"}</Pill>
          <Pill>{caps.data?.encoders.text_encoder ?? "…"}</Pill>
          <Pill>image dim {caps.data?.encoders.image_dim ?? "—"}</Pill>
          <Pill>text dim {caps.data?.encoders.text_dim ?? "—"}</Pill>
        </p>
        <Caveat>{caps.data?.encoders.notes}</Caveat>
      </Card>
    </div>
  );
}
