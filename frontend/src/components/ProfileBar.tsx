import { Link } from "react-router-dom";
import { useProfiles } from "../lib/store";
import { Empty } from "./ui";

/** Header with the profile selector; every screen renders it above its content. */
export function ProfileBar({ title, description }: { title: string; description: string }) {
  const { profiles, selected, select } = useProfiles();
  return (
    <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-white">{title}</h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-slate-400">{description}</p>
      </div>
      {profiles.length > 0 && (
        <label className="shrink-0 text-xs text-slate-400">
          <span className="label mr-2">Profile</span>
          <select
            className="input mt-1 w-full sm:w-56"
            value={selected?.id ?? ""}
            onChange={(e) => select(e.target.value)}
          >
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} · {p.photo_count} photos
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}

export function NoProfile() {
  return (
    <Empty title="No profile yet">
      Create a profile on <Link className="text-accent underline" to="/">Overview</Link> and upload at
      least two photographs. Every screen here operates on a profile you own.
    </Empty>
  );
}
