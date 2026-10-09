// Mirrors src/cgrag/schemas.py (QueryResponse and the types it contains).
export type Verdict = "GENUINE" | "EXPLAINED" | "NOT_COMPARABLE";

export interface ConditionCheck {
  condition: string;
  requested: string;
  covered: boolean;
  observed: string[];
  chunk_ids: string[];
}

export interface ApplicabilityResult {
  coverage: number;
  checks: ConditionCheck[];
  missing: string[];
  re_retrieved: boolean;
  warning: string | null;
  reasoning?: string;
  profile_guided?: boolean;           // passages added because their condition profiles record a missing condition
  joint_covered?: boolean | null;     // model + dataset + language recorded TOGETHER by one result (null = not checked)
  escalated?: boolean;                // the larger model gave a second opinion before a scope warning
}

export interface ContradictionPair {
  verdict: Verdict;
  chunk_a: string;
  chunk_b: string;
  paper_a: string;
  paper_b: string;
  metric: string | null;
  value_a: number | null;
  value_b: number | null;
  differing: string[];
  reason: string;
  nli_contradiction: number | null;
}

export interface ClaimCheck {
  sentence: string;
  supported: boolean;
  chunk_id: string | null;
  entailment: number;
}

export interface SourceRef {
  chunk_id: string;
  paper_id: string;
  paper_title: string;
  page: number;
  section: string;
  score: number;
  preview: string;
}

/** The conditions Stage 1 read from the question (QueryConditions): single values plus lists of further models / datasets / languages. */
export type QueryConditions = Record<string, string | string[] | null>;

export interface QueryAnalysis {
  intent: string;
  complexity: string;
  conditions: QueryConditions;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: SourceRef[];
  analysis: QueryAnalysis | null;
  applicability: ApplicabilityResult | null;
  contradictions: ContradictionPair[];
  claim_checks: ClaimCheck[];
  regenerated: boolean;
  retrieval_weak?: boolean;           // Stage 5's own verdict (no passage above the threshold), before Stage 6 may withdraw it
  trace: string[];
  timings_ms: Record<string, number>;
}

export interface Paper {
  paper_id: string;
  title: string;
  chunks: number;
  profiles: number;
}

export interface Health {
  status: "starting" | "ok" | string;   // "starting" while the server loads its models (about 30 s)
  llm?: string;
  chunks?: number;
  profiles?: number;
  papers?: number;
  warm?: { status: string; seconds: number | null; error: string | null };
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface UploadJob {
  status: string;                      // queued | running | done | failed
  error?: string;
  chunks?: number;
  profiles?: number;
  title?: string;
  paper_id?: string;
}

// ---- POST /query/stream (text/event-stream): the live "chain of execution" ----

export type StepStatus = "running" | "done" | "skipped";

export interface StepEvent {
  type: "step";
  id: string;                          // "4_retrieve"
  stage: number;
  component: string;                   // "Hybrid retrieval"
  blurb?: string;                      // one line on what the component does
  status: StepStatus;
  ms?: number;                         // only when done
  reason?: string;                     // only when skipped
  t_ms: number;                        // ms since the run started
}

export interface NoteEvent {
  type: "note";
  stage: number;
  text: string;
  t_ms: number;
}

export type StreamEvent =
  | { type: "queued" }
  | StepEvent
  | NoteEvent
  | { type: "result"; response: QueryResponse }
  | { type: "error"; message: string };

// ---- UI state (kept in the browser; chats are saved to localStorage) ----

export interface StepState {
  id: string;
  stage: number;
  component: string;
  blurb: string;
  status: StepStatus;
  start: number;                       // t_ms when the step started (or was skipped)
  seenAt?: number;                     // Date.now() when the browser learnt that the step started (for its live timer)
  ms?: number;
  reason?: string;
  notes: string[];
}

export interface RunState {
  mode: "stream" | "replay";          // replay = the stream was unavailable; the steps were rebuilt from trace + timings_ms
  steps: StepState[];
  orphanNotes: string[];               // notes whose stage had no step (shown at the end)
  queued: boolean;
  startedAt: number;                   // Date.now() when the question was sent
  endedAt?: number;
  totalMs?: number;
  outcome?: "done" | "stopped" | "error";
}

export interface Exchange {
  id: string;
  question: string;
  response?: QueryResponse;
  error?: string;
  stopped?: boolean;
  run?: RunState;
}

export interface Chat {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  exchanges: Exchange[];
}
