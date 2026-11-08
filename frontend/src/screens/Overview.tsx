import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import { api, Capabilities, Photo, Profile } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useProfiles } from "../lib/store";
import { Card, ErrorBox, Loading, Stat, Pill, Caveat, Empty } from "../components/ui";
import { pct, when, bytes } from "../lib/format";

export function Overview() {
  const { profiles, selected, refresh, loading: profilesLoading, error: profilesError } = useProfiles();
  const caps = useApi<Capabilities>(() => api.get<Capabilities>("/api/system/capabilities"), []);
  const photos = useApi<{ photos: Photo[] }>(
    () => (selected ? api.get<{ photos: Photo[] }>(`/api/profiles/${selected.id}/photos`) : Promise.resolve({ photos: [] })),
    [selected?.id],
  );

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-white">Overview</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">
          One profile, six analytical modules. Everything below is computed locally from your own
          uploads and from explicitly labelled synthetic data — never from scraped profiles.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Profiles" value={profiles.length} hint="single-account installation" />
        <Stat label="Photographs" value={selected?.photo_count ?? 0} hint={selected ? selected.name : "no profile"} />
        <Stat
          label="Native core"
          value={caps.data ? caps.data.native.target : "…"}
          tone={caps.data?.native.native_available ? "ok" : "bad"}
          hint={caps.data ? `gcc ${caps.data.native.compiler_version} · openmp ${caps.data.native.has_openmp ? "on" : "off"}` : ""}
        />
        <Stat
          label="Cache"
          value={caps.data?.cache.enabled ? "redis" : "off"}
          tone={caps.data?.cache.enabled ? "ok" : "warn"}
          hint="optional: the app is fully useful without it"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <Card title="Your profiles" subtitle="Create a profile, then upload photographs to unlock the laboratory.">
            {profilesLoading && <Loading what="profiles" />}
            {Boolean(profilesError) && <ErrorBox error={profilesError} onRetry={refresh} />}
            {!profilesLoading && profiles.length === 0 && (
              <Empty title="No profiles yet">Use the form on the right to create the first one.</Empty>
            )}
            <ul className="space-y-2">
              {profiles.map((p) => (
                <li key={p.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-ink-700 p-3">
                  <div>
                    <div className="text-sm font-medium text-slate-100">
                      {p.name} {p.is_primary && <Pill tone="ok">primary</Pill>}
                    </div>
                    <div className="text-xs text-slate-500">
                      {p.photo_count} photographs · {p.prompt_count} prompts · created {when(p.created_at)}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Link to="/profile" className="btn-ghost">Intelligence</Link>
                    <Link to="/laboratory" className="btn-ghost">Laboratory</Link>
                  </div>
                </li>
              ))}
            </ul>
          </Card>

          <Card
            title="Photographs"
            subtitle="Uploaded images are analysed in-process: quality components, composition, zero-shot scene labels and a CLIP embedding stored in pgvector."
          >
            {photos.loading && <Loading what="photographs" />}
            {Boolean(photos.error) && <ErrorBox error={photos.error} onRetry={photos.reload} />}
            {photos.data && photos.data.photos.length === 0 && (
              <Empty title="No photographs">Upload 2–6 images to enable ordering optimisation and diversity analysis.</Empty>
            )}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {photos.data?.photos.map((ph) => (
                <figure key={ph.id} className="overflow-hidden rounded-lg border border-ink-700">
                  <img
                    src={ph.url}
                    alt={`Photograph ${ph.position + 1}`}
                    className="aspect-square w-full object-cover"
                    loading="lazy"
                  />
                  <figcaption className="space-y-1 p-2 text-[11px] text-slate-400">
                    <div className="flex justify-between">
                      <span className="text-slate-300">#{ph.position + 1}</span>
                      <span>{ph.width}×{ph.height}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>quality</span>
                      <span className="tabular-nums text-slate-200">{pct(ph.quality)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>scene</span>
                      <span className="truncate pl-2 text-slate-200">{ph.analysis.dominant_scene ?? "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>size</span>
                      <span>{bytes(ph.bytes)}</span>
                    </div>
                  </figcaption>
                </figure>
              ))}
            </div>
            {selected && <ProfileUpload profile={selected} onDone={() => { photos.reload(); refresh(); }} />}
          </Card>
        </div>

        <div className="space-y-6">
          <CreateProfile onCreated={refresh} />
          <Card title="What this is not">
            <ul className="space-y-2 text-xs leading-relaxed text-slate-400">
              <li>· Not a dating app, and it sends no messages on any platform.</li>
              <li>· No scraping, no automated swiping, no persuasion features.</li>
              <li>· No inference of ethnicity, religion, orientation or health from images or text.</li>
              <li>· The compatibility pool is synthetic and labelled as such, because we hold no real dating data.</li>
              <li>· Benchmarks in this app are measured on this machine, not copied from a paper.</li>
            </ul>
            <Caveat>
              {caps.data?.disclaimer ??
                "Scores rank candidates under declared features and weights. They do not predict whether two people will form a relationship."}
            </Caveat>
          </Card>
        </div>
      </div>
    </div>
  );
}

function ProfileUpload({ profile, onDone }: { profile: Profile; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true);
    setErr(null);
    setMsg(null);
    const done: string[] = [];
    try {
      for (const file of Array.from(files)) {
        const res = await api.upload<{
          id: string;
          quality: number;
          deduplicated: boolean;
          analysis: { dominant_scene: string | null; quality_estimate: number };
        }>(`/api/profiles/${profile.id}/photos`, file);
        done.push(
          `${file.name} → ${res.analysis.dominant_scene ?? "unlabelled"} (q=${pct(res.quality)}` +
            `${res.deduplicated ? ", duplicate of an existing photo" : ""})`,
        );
      }
      setMsg(done.join(" · "));
      onDone();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-4 border-t border-ink-700 pt-4">
      <label className="label">Add photographs (JPEG/PNG/WebP, ≤ 20 MB each)</label>
      <input
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        disabled={busy}
        onChange={(e) => upload(e.target.files)}
        className="mt-2 block w-full cursor-pointer rounded-lg border border-dashed border-ink-600 bg-ink-900 p-3 text-xs text-slate-400 file:mr-3 file:rounded file:border-0 file:bg-accent file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-white"
      />
      {busy && <p className="mt-2 text-xs text-accent">Analysing… the first upload loads the CLIP weights and takes ~10 s.</p>}
      {msg && <p className="mt-2 break-words text-xs text-ok">{msg}</p>}
      {Boolean(err) && <div className="mt-2"><ErrorBox error={err} /></div>}
    </div>
  );
}

function CreateProfile({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [interests, setInterests] = useState("");
  const [q1, setQ1] = useState("My ideal Sunday");
  const [a1, setA1] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await api.post<Profile>("/api/profiles", {
        name,
        description: description || null,
        interests: interests.split(",").map((s) => s.trim()).filter(Boolean),
        prompts: a1.trim() ? [{ kind: "prompt", question: q1, answer: a1, position: 0 }] : [],
      });
      setName("");
      setDescription("");
      setInterests("");
      setA1("");
      onCreated();
    } catch (e2) {
      setErr(e2);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title="New profile" subtitle="Prompt answers are scored for specificity only — never for personality.">
      <form onSubmit={submit} className="space-y-3">
        <input className="input" placeholder="Display name" value={name} onChange={(e) => setName(e.target.value)} required />
        <textarea className="input min-h-[70px]" placeholder="Short bio (optional)" value={description}
                  onChange={(e) => setDescription(e.target.value)} />
        <input className="input" placeholder="Interests, comma separated" value={interests}
               onChange={(e) => setInterests(e.target.value)} />
        <input className="input" value={q1} onChange={(e) => setQ1(e.target.value)} aria-label="Prompt question" />
        <textarea className="input min-h-[70px]" placeholder="Your answer — concrete detail scores higher than vague enthusiasm"
                  value={a1} onChange={(e) => setA1(e.target.value)} />
        {Boolean(err) && <ErrorBox error={err} />}
        <button className="btn-primary w-full" disabled={busy}>{busy ? "Creating…" : "Create profile"}</button>
      </form>
    </Card>
  );
}
