import { useState } from "react";
import { api, ConversationAnalytics, ConversationSummary } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Card, Caveat, Empty, ErrorBox, Loading, Stat } from "../components/ui";
import { num, pct } from "../lib/format";

export function Conversations() {
  const list = useApi<{ conversations: ConversationSummary[] }>(
    () => api.get<{ conversations: ConversationSummary[] }>("/api/conversations"),
    [],
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  const detail = useApi<{ id: string; label: string; analytics: ConversationAnalytics }>(
    () =>
      selectedId
        ? api.get<{ id: string; label: string; analytics: ConversationAnalytics }>(`/api/conversations/${selectedId}`)
        : Promise.resolve(null as never),
    [selectedId],
  );

  const importRaw = async (file: File) => {
    setBusy(true);
    setErr(null);
    try {
      const parsed = JSON.parse(await file.text()) as { label?: string; me?: string; messages: unknown[] };
      const res = await api.post<{ id: string }>("/api/conversations/import", {
        label: parsed.label ?? file.name,
        me: parsed.me ?? "me",
        messages: parsed.messages,
      });
      list.reload();
      setSelectedId(res.id);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const loadSynthetic = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await api.post<{ id: string }>("/api/conversations/synthetic", { messages: 80, seed: 3 });
      list.reload();
      setSelectedId(res.id);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const a = detail.data?.analytics;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-white">Conversation Analytics</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">
          Module F. Behavioural aggregates from timestamps and senders. Message text is analysed for
          length and turnover, then discarded — it is never stored or shown back.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-4">
        <div className="space-y-3">
          <Card title="Import">
            <input type="file" accept="application/json" disabled={busy}
                   onChange={(e) => e.target.files?.[0] && importRaw(e.target.files[0])}
                   className="block w-full cursor-pointer rounded-lg border border-dashed border-ink-600 bg-ink-900 p-2 text-[11px] text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-accent file:px-2 file:py-1 file:text-[11px] file:text-white" />
            <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
              JSON: <code className="text-slate-400">{`{ "label": "...", "me": "me", "messages": [{ "sender", "timestamp", "text" }] }`}</code>
            </p>
            <button className="btn-ghost mt-3 w-full text-xs" onClick={loadSynthetic} disabled={busy}>
              Load a synthetic conversation
            </button>
            {Boolean(err) && <div className="mt-2"><ErrorBox error={err} /></div>}
          </Card>

          {list.loading && <Loading what="conversations" />}
          {list.data?.conversations.map((c) => (
            <button key={c.id} onClick={() => setSelectedId(c.id)}
                    className={`w-full rounded-lg border p-3 text-left transition ${
                      selectedId === c.id ? "border-accent bg-accent/10" : "border-ink-700 hover:border-ink-600"}`}>
              <div className="text-sm text-slate-100">{c.label}</div>
              <div className="text-[11px] text-slate-500">{c.message_count} messages</div>
            </button>
          ))}
        </div>

        <div className="lg:col-span-3 space-y-4">
          {!selectedId && <Empty title="Select or import a conversation" />}
          {detail.loading && <Loading what="the analytics" />}
          {Boolean(detail.error) && <ErrorBox error={detail.error} onRetry={detail.reload} />}
          {a && (
            <>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Messages" value={a.messages} hint={`${a.sessions} sessions over ${num(a.duration_hours, 1)} h`} />
                <Stat label="Reciprocity" value={num(a.reciprocity.index, 3)}
                      hint={`message ${num(a.reciprocity.message_balance, 2)} · word ${num(a.reciprocity.word_balance, 2)}`} />
                <Stat label="Median reply (me)" value={`${Math.round(a.reply_latency_s.me?.median ?? 0)} s`}
                      hint={`them: ${Math.round(a.reply_latency_s.them?.median ?? 0)} s`} />
                <Stat label="Double texts" value={a.engagement.double_texts}
                      hint="consecutive messages before a reply" />
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Message share" subtitle="Who writes how much. Balance is symmetric: far ahead or far behind both lower the index.">
                  {Object.entries(a.message_share).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <div className="flex justify-between text-xs"><span className="text-slate-300">{k}</span><span className="tabular-nums">{pct(v)}</span></div>
                      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-600">
                        <div className="h-full bg-accent" style={{ width: `${v * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </Card>

                <Card title="Engagement" subtitle="Descriptive only.">
                  <div className="kv"><span>Messages / session</span><span className="tabular-nums">{num(a.engagement.messages_per_session, 1)}</span></div>
                  <div className="kv"><span>Longest session</span><span className="tabular-nums">{a.engagement.longest_session_messages}</span></div>
                  <div className="kv"><span>Busiest hours (UTC)</span><span className="tabular-nums">{a.engagement.busiest_hours_utc.join(", ")}</span></div>
                  <div className="kv"><span>Continuity</span><span className="tabular-nums">{num(a.continuity, 2)}</span></div>
                </Card>
              </div>

              <Card title="What is deliberately not computed" subtitle="The module refuses psychological inference.">
                <ul className="space-y-1 text-xs leading-relaxed text-slate-400">
                  {a.limitations.map((l) => <li key={l}>· {l}</li>)}
                </ul>
                <Caveat>{a.sequence_model.reason}</Caveat>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
