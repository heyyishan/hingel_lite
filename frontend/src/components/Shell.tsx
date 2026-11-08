import { ReactNode, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { api, setToken, Capabilities } from "../lib/api";
import { useApi } from "../lib/useApi";
import { Pill } from "./ui";

export const NAV = [
  { to: "/", label: "Overview", icon: "◎" },
  { to: "/profile", label: "Profile Intelligence", icon: "◧" },
  { to: "/laboratory", label: "Photo Laboratory", icon: "⌗" },
  { to: "/compatibility", label: "Compatibility Explorer", icon: "⇄" },
  { to: "/experiments", label: "Experimentation Lab", icon: "⚖" },
  { to: "/counterfactual", label: "Counterfactual Simulator", icon: "⟳" },
  { to: "/conversations", label: "Conversation Analytics", icon: "✎" },
  { to: "/benchmarks", label: "Systems Benchmarking", icon: "⏱" },
  { to: "/settings", label: "Settings & Privacy", icon: "⚙" },
];

export function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const caps = useApi<Capabilities>(() => api.get<Capabilities>("/api/system/capabilities"), []);

  const signOut = async () => {
    try {
      await api.post("/api/auth/logout");
    } catch {
      /* the token is being discarded anyway */
    }
    setToken(null);
    navigate("/login", { replace: true });
  };

  const nativeOk = caps.data?.native.native_available;
  const cacheOk = caps.data?.cache.enabled;

  return (
    <div className="flex min-h-full">
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-64 shrink-0 border-r border-ink-700 bg-ink-800/95 p-4 backdrop-blur transition-transform lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="mb-6 px-2">
          <div className="text-lg font-semibold tracking-tight text-white">
            attraction<span className="text-accent">·</span>engine
          </div>
          <p className="mt-1 text-[11px] leading-tight text-slate-500">
            multimodal dating intelligence · synthetic-data research instrument
          </p>
        </div>
        <nav className="space-y-0.5">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? "bg-accent/15 text-white" : "text-slate-400 hover:bg-ink-700 hover:text-slate-100"
                }`
              }
            >
              <span className="w-4 text-center text-accent">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-6 space-y-2 border-t border-ink-700 pt-4 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">native core</span>
            <Pill tone={nativeOk ? "ok" : "bad"}>{caps.data?.native.target ?? "…"}</Pill>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">cache</span>
            <Pill tone={cacheOk ? "ok" : "warn"}>{cacheOk ? "redis" : "off"}</Pill>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">encoder</span>
            <span className="truncate text-slate-300">{caps.data?.encoders.image_encoder ?? "…"}</span>
          </div>
        </div>
        <button onClick={signOut} className="btn-ghost mt-6 w-full text-xs">
          Sign out
        </button>
      </aside>

      {open && <div className="fixed inset-0 z-20 bg-black/50 lg:hidden" onClick={() => setOpen(false)} />}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-ink-700 bg-ink-900/85 px-4 py-3 backdrop-blur lg:hidden">
          <button className="btn-ghost px-2 py-1" onClick={() => setOpen((o) => !o)} aria-label="Menu">
            ☰
          </button>
          <span className="font-semibold">attraction·engine</span>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8">{children}</main>
        <footer className="border-t border-ink-700/60 px-4 py-4 text-center text-[11px] leading-relaxed text-slate-500">
          {caps.data?.disclaimer ??
            "Scores rank candidates under declared features and weights. They do not predict whether two people will form a relationship."}
        </footer>
      </div>
    </div>
  );
}
