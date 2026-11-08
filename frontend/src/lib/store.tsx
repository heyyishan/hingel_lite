/**
 * Profile selection shared across screens.
 *
 * Screens are keyed on the selected profile, so it must survive navigation but
 * not reloads: sessionStorage is enough and avoids a global state library.
 */
import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { api, Profile } from "./api";

interface Ctx {
  profiles: Profile[];
  selected: Profile | null;
  select: (id: string) => void;
  refresh: () => void;
  loading: boolean;
  error: unknown;
}

const ProfileCtx = createContext<Ctx | null>(null);
const KEY = "ae.profile";

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(sessionStorage.getItem(KEY));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .get<Profile[]>("/api/profiles")
      .then((rows) => {
        if (cancelled) return;
        setProfiles(rows);
        setError(null);
        const stillThere = rows.some((r) => r.id === selectedId);
        if (!stillThere) {
          const next = rows.find((r) => r.is_primary) ?? rows[0] ?? null;
          setSelectedId(next?.id ?? null);
          if (next) sessionStorage.setItem(KEY, next.id);
          else sessionStorage.removeItem(KEY);
        }
      })
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  const value = useMemo<Ctx>(
    () => ({
      profiles,
      selected: profiles.find((p) => p.id === selectedId) ?? null,
      select: (id: string) => {
        setSelectedId(id);
        sessionStorage.setItem(KEY, id);
      },
      refresh: () => setTick((t) => t + 1),
      loading,
      error,
    }),
    [profiles, selectedId, loading, error],
  );

  return <ProfileCtx.Provider value={value}>{children}</ProfileCtx.Provider>;
}

export function useProfiles(): Ctx {
  const ctx = useContext(ProfileCtx);
  if (!ctx) throw new Error("useProfiles must be used inside ProfileProvider");
  return ctx;
}
