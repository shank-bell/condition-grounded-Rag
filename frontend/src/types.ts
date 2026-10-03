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

export interface QueryAnalysis {
  intent: string;
  complexity: string;
  conditions: Record<string, string | null>;
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
