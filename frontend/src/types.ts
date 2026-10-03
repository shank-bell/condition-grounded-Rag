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
  status: "starting" | "ok" | string;
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
