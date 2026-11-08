import { ReactNode } from "react";

export function Card({ title, subtitle, right, children, className = "" }: {
  title?: string; subtitle?: string; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-100">{title}</h2>}
            {subtitle && <p className="mt-1 max-w-2xl text-xs leading-relaxed text-slate-400">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone = "default" }: {
  label: string; value: ReactNode; hint?: string; tone?: "default" | "ok" | "warn" | "bad";
}) {
  const colour = { default: "text-slate-100", ok: "text-ok", warn: "text-warn", bad: "text-bad" }[tone];
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className={`mt-2 text-2xl font-semibold tabular-nums ${colour}`}>{value}</div>
      {hint && <div className="mt-1 text-xs leading-relaxed text-slate-400">{hint}</div>}
    </div>
  );
}

export function Loading({ what = "data" }: { what?: string }) {
  return (
    <div className="flex items-center gap-3 py-8 text-sm text-slate-400" role="status" aria-live="polite">
      <span className="h-3 w-3 animate-ping rounded-full bg-accent" />
      Loading {what}…
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-xl border border-bad/40 bg-bad/10 p-4 text-sm text-red-200" role="alert">
      <div className="font-semibold text-bad">Something failed</div>
      <p className="mt-1 break-words font-mono text-xs text-red-200/90">{message}</p>
      {onRetry && (
        <button className="btn-ghost mt-3" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-ink-600 p-8 text-center">
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-lg text-xs leading-relaxed text-slate-400">{children}</div>}
    </div>
  );
}

export function Pill({ children, tone = "default" }: { children: ReactNode; tone?: "default" | "ok" | "warn" | "bad" }) {
  const cls = {
    default: "border-ink-600 text-slate-300",
    ok: "border-ok/50 text-ok",
    warn: "border-warn/50 text-warn",
    bad: "border-bad/50 text-bad",
  }[tone];
  return <span className={`rounded-full border px-2 py-0.5 text-xs ${cls}`}>{children}</span>;
}

export function Bar({ value, max = 1, tone = "accent" }: { value: number; max?: number; tone?: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const colour = tone === "accent" ? "bg-accent" : tone === "ok" ? "bg-ok" : tone === "warn" ? "bg-warn" : "bg-bad";
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-600">
      <div className={`h-full rounded-full ${colour}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

/** Honest disclosure strip: every screen that shows modelled numbers carries one. */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 rounded-lg border border-ink-600 bg-ink-900/60 p-3 text-xs leading-relaxed text-slate-400">
      {children}
    </p>
  );
}
