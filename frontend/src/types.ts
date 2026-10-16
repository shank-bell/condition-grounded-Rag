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
  profile_guided?: boolean;
  joint_covered?: boolean | null;
  escalated?: boolean;
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
  retrieval_weak?: boolean;
  faithfulness?: number | null;
  answer_relevancy?: number | null;
  faithfulness_judge?: string | null;
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
  status: "starting" | "ok" | string;
  llm?: string;
  chunks?: number;
  profiles?: number;
  papers?: number;
  warm?: { status: string; seconds: number | null; error: string | null };
  auth_required?: boolean;
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface UploadJob {
  status: string;
  error?: string;
  chunks?: number;
  profiles?: number;
  title?: string;
  paper_id?: string;
}


export type StepStatus = "running" | "done" | "skipped";

export interface StepEvent {
  type: "step";
  id: string;
  stage: number;
  component: string;
  blurb?: string;
  status: StepStatus;
  ms?: number;
  reason?: string;
  t_ms: number;
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


export interface StepState {
  id: string;
  stage: number;
  component: string;
  blurb: string;
  status: StepStatus;
  start: number;
  seenAt?: number;
  ms?: number;
  reason?: string;
  notes: string[];
}

export interface RunState {
  mode: "stream" | "replay";
  steps: StepState[];
  orphanNotes: string[];
  queued: boolean;
  startedAt: number;
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
