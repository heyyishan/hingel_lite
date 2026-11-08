/**
 * Typed API client.
 *
 * Design rules that matter for correctness:
 *  - relative URLs only, so the app works behind any proxy (no baked-in host);
 *  - the bearer token lives in memory + sessionStorage, never localStorage;
 *  - a single place maps HTTP failures onto typed errors so every screen can
 *    render a real error state instead of an empty one.
 */

const TOKEN_KEY = "ae.token";
let token: string | null = sessionStorage.getItem(TOKEN_KEY);

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? (typeof detail === "string" ? detail : `HTTP ${status}`));
    this.status = status;
    this.detail = detail;
  }
}

export function setToken(next: string | null) {
  token = next;
  if (next) sessionStorage.setItem(TOKEN_KEY, next);
  else sessionStorage.removeItem(TOKEN_KEY);
}

export function hasToken() {
  return Boolean(token);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch (err) {
    throw new ApiError(0, String(err), "Cannot reach the API. Is the backend running?");
  }
  if (res.status === 401) {
    setToken(null);
    throw new ApiError(401, "unauthorised", "Session expired — sign in again.");
  }
  if (!res.ok) {
    let detail: unknown = await res.text();
    try {
      detail = JSON.parse(detail as string);
    } catch {
      /* keep the raw text */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  get: <T>(p: string) => request<T>(p),
  post: <T>(p: string, body?: unknown) =>
    request<T>(p, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  patch: <T>(p: string, body: unknown) =>
    request<T>(p, { method: "PATCH", body: JSON.stringify(body) }),
  del: <T>(p: string) => request<T>(p, { method: "DELETE" }),
  upload: <T>(p: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<T>(p, { method: "POST", body: form });
  },
};

// ------------------------------------------------------------------ types
export interface Profile {
  id: string;
  name: string;
  description: string | null;
  is_primary: boolean;
  structured_prefs: Record<string, unknown>;
  interests: string[];
  created_at: string | null;
  photo_count: number;
  prompt_count: number;
}

export interface Photo {
  id: string;
  position: number;
  width: number;
  height: number;
  bytes: number;
  sha256: string;
  mime: string;
  quality: number;
  /** Server-issued signed URL: <img> cannot carry the bearer header. */
  url: string;
  analysis: {
    dominant_scene: string | null;
    quality_estimate?: number;
    quality_components?: Record<string, number>;
    faces: { count: number; largest_fraction: number | null };
    lighting?: { mean_luma: number; contrast_rms: number; entropy_bits: number };
    flags: string[];
    scene_source?: string;
    embedding_encoder?: string;
    embedding_dim?: number;
  };
}

export interface OptimizeResult {
  strategy: string;
  permutation: number[];
  objective: number;
  quality_term: number;
  diversity_term: number;
  coherence_term: number;
  redundancy_term: number;
  improvement_vs_upload_order: number;
  proved_optimal: boolean;
  evaluations: number;
  elapsed_ms: number;
}

export interface OptimizeResponse {
  n: number;
  parameters: Record<string, number>;
  position_weights: number[];
  slot_affinity: number[];
  results: OptimizeResult[];
  best: OptimizeResult;
  baseline: { objective: number; permutation: number[] };
  photo_ids: string[];
  photo_labels: string[];
  caveat: string;
}

export interface ReportPhoto {
  index: number;
  width: number;
  height: number;
  quality_estimate: number;
  quality_components: Record<string, number>;
  quality_weights: Record<string, number>;
  dominant_scene: string | null;
  scene_distribution?: Record<string, number>;
  scene_source?: string;
  faces: { count: number; largest_fraction: number | null };
  lighting?: { mean_luma: number; contrast_rms: number; entropy_bits: number };
  flags: string[];
  embedding_dim?: number;
  embedding_encoder?: string;
}

export interface Report {
  profile: { id: string; name: string };
  photos: ReportPhoto[];
  photo_ids: string[];
  cached: boolean;
  synthetic_inputs: unknown;
  visual_diversity: {
    mean_pairwise_similarity: number;
    diversity: number;
    redundant_pairs: { a: number; b: number; similarity: number }[];
    nearest_neighbour: { index: number; nearest: number; similarity: number }[];
    threshold: number;
  };
  text_blocks: { index: number; text: string; specificity: { score: number; components: Record<string, unknown> } }[];
  coherence: Record<string, unknown>;
  text_image_consistency: Record<string, unknown>;
  recommendations: { code: string; severity: string; title: string; why: string; action: string }[];
  strongest_photos: unknown[];
  weakest_photos: unknown[];
  ordering_suggestion: unknown;
  confidence: { evidence_coverage: number; inputs: Record<string, unknown>; interpretation: string };
  limitations: string[];
}

export interface RankItem {
  index: number;
  forward: { visual: number; textual: number; attributes: number; interests: number; score: number };
  backward: number;
  harmonic: number;
  confidence: number;
  cold_start: boolean;
  neural_probability?: number | null;
}

export interface RankResponse {
  ranking: RankItem[];
  weights: Record<string, number>;
  formula: string;
  caveat: string;
  neural_head_applied: boolean;
  model_status: string;
  lambda_diversity: number;
  metrics: {
    k: number;
    relevant_definition: string;
    precision_at_k: number;
    recall_at_k: number;
    ndcg_at_k: number;
    n_relevant: number;
  };
  /** One entry per ablated feature set, keyed by name (e.g. "visual_only"). */
  ablations: Record<string, {
    k: number;
    relevant_definition: string;
    precision_at_k: number;
    recall_at_k: number;
    ndcg_at_k: number;
    n_relevant: number;
  }>;
  pool_label?: string;
  viewer?: unknown;
}

export interface Variant {
  id: string;
  label: string;
  successes: number;
  failures: number;
  missing: number;
  configuration: Record<string, unknown>;
}

export interface Experiment {
  id: string;
  name: string;
  hypothesis: string | null;
  metric: string;
  prior: string;
  min_observations: number;
  loss_tolerance: number;
  family_level: number;
  seed: number;
  status: string;
  created_at: string | null;
  variants: Variant[];
}

export interface AnalysisVariant {
  id: string;
  label: string;
  successes: number;
  failures: number;
  observed: number;
  missing: number;
  observed_rate: number | null;
  posterior: { alpha: number; beta: number; mean: number; variance: number; mode: number | null };
  credible_interval: { level: number; lo: number; hi: number; width: number };
  maturity: string;
}

export interface Analysis {
  experiment: Experiment;
  prior: { name: string; alpha: number; beta: number; explanation: string };
  variants: AnalysisVariant[];
  pairwise: { a: string; b: string; prob_a_better: number; expected_loss_if_a: number; expected_loss_if_b: number }[];
  decision: {
    stop: boolean;
    reason: string;
    /** P(θ_first > θ_second); named explicitly because "a better" is misleading. */
    prob_a_better: number;
    a: string;
    b: string;
    leader: string;
    leader_probability: number;
    expected_loss_choose_a: number;
    expected_loss_choose_b: number;
    expected_loss_choose_leader: number;
  };
  thompson: { draws: number; allocation: number[]; share: number[]; seed: number };
  multiplicity: { variants: number; bonferroni_alpha: number; sidak_alpha: number; note: string };
  /** keyed by variant id: { x: [...], pdf: [...] } */
  density: Record<string, { x: number[]; pdf: number[] }>;
  timeline: { variant: string; observed_at: string; context: { successes: number; failures: number; missing: number } }[];
  hierarchical: {
    population: { alpha: number; beta: number; mean: number };
    overdispersion: number;
    shrinkage: number;
    posteriors: { id: string; label: string; alpha: number; beta: number; mean: number }[];
  };
  capacity: {
    observed_total: number;
    min_observations: number;
    planning: { per_arm: number; total: number; alpha: number; power: number; assumptions: string[] };
  };
  missing_data_policy: string;
}

export interface CounterfactualResponse {
  q_baseline: number;
  q_modified: number;
  delta_q: number;
  relative_change: number | null;
  components_baseline: Record<string, number> & { weights?: Record<string, number> };
  components_modified: Record<string, number> & { weights?: Record<string, number> };
  attribution: { method: string; importance: Record<string, number>; n_permutations: number };
  sensitivity: { change: string; delta_q: number; note?: string }[];
  uncertainty: { kind: string; value: null; explanation: string };
  model_version: string;
  interpretation: string;
}

export interface ConversationSummary {
  id: string;
  label: string;
  message_count: number;
  created_at: string | null;
  summary: Record<string, unknown>;
}

export interface ConversationAnalytics {
  messages: number;
  duration_hours: number;
  sessions: number;
  initiation_balance: number;
  message_share: Record<string, number>;
  word_counts: Record<string, number>;
  reply_latency_s: Record<string, { n: number; median: number; p90: number; mean: number }>;
  reciprocity: { index: number; message_balance: number; word_balance: number; latency_balance: number; note: string };
  engagement: { messages_per_session: number; longest_session_messages: number; busiest_hours_utc: number[]; double_texts: number };
  continuity: number;
  limitations: string[];
  sequence_model: { available: boolean; reason: string };
}

export interface Capabilities {
  native: {
    native_available: boolean;
    compiler: string;
    compiler_version: string;
    cxx_dialect?: number;
    target: string;
    mask: number;
    has_openmp: boolean;
  };
  encoders: {
    image_encoder: string;
    text_encoder: string;
    image_dim: number;
    text_dim: number;
    l2_normalized: boolean;
    notes: string;
  };
  database: { dialect: string; pgvector: string | null };
  cache: { enabled: boolean; url?: string };
  machine: Record<string, unknown>;
  storage_dir: string;
  disclaimer: string;
}

export interface LabPhotos {
  photos: { id: string; position: number; quality: number; embedding_dim: number; encoder: string; dominant_scene: string | null }[];
  similarity_matrix: number[][];
  dim: number;
}

export interface SensitivityResponse {
  grid: { lambda: number; mu: number; optimized_total: number; fixed_perm_total: number; optimized_permutation: number[] }[];
  note?: string;
}

export interface AblationRow {
  removed_index: number;
  loss: number;
  photo_id?: string;
  position?: number;
  weight_mass?: number;
}

export interface BenchmarkResult {
  benchmark: string;
  kernel?: string;
  n: number;
  k?: number;
  strategy?: string;
  threads?: number;
  median_ms: number;
  min_ms: number;
  p95_ms?: number;
  pairs_per_ms?: number;
}

export interface BenchmarkResponse {
  run_id: string;
  machine: Record<string, unknown>;
  results: BenchmarkResult[];
  /** Sizes the server reduced to stay inside its memory budget. */
  clamped: { requested: number; used: number; reason: string }[];
  note: string;
}
