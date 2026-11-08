import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setToken } from "../lib/api";

export function Login() {
  const [handle, setHandle] = useState("owner");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<{ token: string }>("/api/auth/login", { handle, password });
      setToken(res.token);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <div className="w-full max-w-sm">
        <h1 className="text-2xl font-semibold tracking-tight text-white">
          attraction<span className="text-accent">·</span>engine
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          A local, single-account research instrument. Everything you upload stays on this machine;
          no third-party service is called at request time.
        </p>
        <form onSubmit={submit} className="mt-6 space-y-3">
          <div>
            <label className="label" htmlFor="handle">Handle</label>
            <input id="handle" className="input mt-1" value={handle} onChange={(e) => setHandle(e.target.value)}
                   autoComplete="username" required />
          </div>
          <div>
            <label className="label" htmlFor="password">Password</label>
            <input id="password" type="password" className="input mt-1" value={password}
                   onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
          </div>
          {error && <p className="rounded-lg border border-bad/40 bg-bad/10 p-2 text-xs text-red-200">{error}</p>}
          <button className="btn-primary w-full" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
          The installation account is created on first run. In development the default password is
          <code className="mx-1 rounded bg-ink-700 px-1">attraction-engine</code>
          (see <code className="rounded bg-ink-700 px-1">README.md</code> — change it before exposing
          this to a network).
        </p>
      </div>
    </div>
  );
}
