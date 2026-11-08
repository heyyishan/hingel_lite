import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { hasToken } from "./lib/api";
import { ProfileProvider } from "./lib/store";
import { Shell } from "./components/Shell";
import { Login } from "./screens/Login";
import { Overview } from "./screens/Overview";
import { ProfileIntelligence } from "./screens/ProfileIntelligence";
import { PhotoLaboratory } from "./screens/PhotoLaboratory";
import { Compatibility } from "./screens/Compatibility";
import { Experiments } from "./screens/Experiments";
import { Counterfactual } from "./screens/Counterfactual";
import { Conversations } from "./screens/Conversations";
import { Benchmarks } from "./screens/Benchmarks";
import { Settings } from "./screens/Settings";

function RequireAuth({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  if (!hasToken()) return <Navigate to="/login" replace state={{ from: location }} />;
  return <>{children}</>;
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/*"
        element={
          <RequireAuth>
            <ProfileProvider>
              <Shell>
                <Routes>
                  <Route path="/" element={<Overview />} />
                  <Route path="/profile" element={<ProfileIntelligence />} />
                  <Route path="/laboratory" element={<PhotoLaboratory />} />
                  <Route path="/compatibility" element={<Compatibility />} />
                  <Route path="/experiments" element={<Experiments />} />
                  <Route path="/counterfactual" element={<Counterfactual />} />
                  <Route path="/conversations" element={<Conversations />} />
                  <Route path="/benchmarks" element={<Benchmarks />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Shell>
            </ProfileProvider>
          </RequireAuth>
        }
      />
    </Routes>
  );
}
