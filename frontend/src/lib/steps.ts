// The live "chain of execution": the pipeline's timed steps, built from the stream's events or, when the stream is unavailable,
// replayed from the finished response (trace + timings_ms).
import type { QueryResponse, RunState, StepEvent, StepState, StreamEvent } from "../types";

/** Mirrors STEP_INFO in src/cgrag/pipeline/run.py: id -> (stage, component, what it does in one line). */
export const STEP_INFO: Record<string, [number, string, string]> = {
  "1_understand": [1, "Query understanding", "SciBERT reads the intent and complexity; the language model extracts the conditions (model, dataset, language ...)"],
  "2_plan": [2, "Orchestrator agent", "Decides whether to refine or split the question and which checks to run"],
  "3_refine": [3, "Query refinement", "Rewrites the question into search queries"],
  "4_retrieve": [4, "Hybrid retrieval", "BGE-M3 vectors + BM25 keywords + result cards"],
  "5_rerank": [5, "Reranker", "A cross-encoder keeps the passages that really answer the question"],
  "2b_replan": [2, "Orchestrator review", "Looks at what came back and may re-plan once"],
  "6_applicability": [6, "Applicability agent", "Checks that the passages cover every condition of the question; searches again or warns"],
  "7_contradiction": [7, "Contradiction resolver", "Separates real conflicts between papers from differences caused by different conditions"],
  "8_generate": [8, "Answer generation", "The language model writes the answer from the passages and the recorded conditions"],
  "9_critic": [9, "Claim critic", "An NLI model checks every sentence against its source and regenerates once if needed"],
};
export const STEP_ORDER = Object.keys(STEP_INFO);

export function newRun(mode: RunState["mode"] = "stream"): RunState {
  return { mode, steps: [], orphanNotes: [], queued: false, startedAt: Date.now() };
}

function stepFromEvent(e: StepEvent, prev?: StepState): StepState {
  const info = STEP_INFO[e.id];
  return {
    id: e.id,
    stage: e.stage ?? info?.[0] ?? 0,
    component: e.component || prev?.component || info?.[1] || e.id,
    blurb: e.blurb || prev?.blurb || info?.[2] || "",
    status: e.status,
    start: prev?.start ?? (e.status === "done" && e.ms !== undefined ? Math.max(0, e.t_ms - e.ms) : e.t_ms),
    seenAt: prev?.seenAt ?? Date.now(),
    ms: e.ms ?? prev?.ms,
    reason: e.reason ?? prev?.reason,
    notes: prev?.notes ?? [],
  };
}

/** One stream event applied to the run (pure: returns a new object). A note goes to the latest step of its stage. */
export function applyEvent(run: RunState, e: StreamEvent): RunState {
  switch (e.type) {
    case "queued":
      return { ...run, queued: true };
    case "step": {
      const i = run.steps.findIndex((s) => s.id === e.id);
      const steps = run.steps.slice();
      if (i >= 0) steps[i] = stepFromEvent(e, steps[i]);
      else steps.push(stepFromEvent(e));
      return { ...run, queued: false, steps };
    }
    case "note": {
      let i = -1;
      for (let j = run.steps.length - 1; j >= 0; j--) {
        if (run.steps[j].stage === e.stage) {
          i = j;
          break;
        }
      }
      if (i < 0) return { ...run, orphanNotes: [...run.orphanNotes, e.text] };
      const steps = run.steps.slice();
      steps[i] = { ...steps[i], notes: [...steps[i].notes, e.text] };
      return { ...run, steps };
    }
    default:
      return run;
  }
}

/** Closes a run: running steps become done (or stay unfinished when the run was stopped). */
export function finishRun(run: RunState, outcome: NonNullable<RunState["outcome"]>, totalMs?: number): RunState {
  const endedAt = Date.now();
  return {
    ...run,
    queued: false,
    outcome,
    endedAt,
    totalMs: totalMs ?? endedAt - run.startedAt,
  };
}

/** Which step a trace line belongs to ("6 coverage 0.67 ..." -> 6_applicability; "2 re-plan ..." -> 2b_replan). */
function traceStep(line: string, have: Set<string>): string | null {
  const m = line.match(/^(\d+)\s/);
  if (!m) return null;
  const stage = Number(m[1]);
  if (stage === 2 && /^2 re-plan/.test(line) && have.has("2b_replan")) return "2b_replan";
  for (const id of STEP_ORDER) if (STEP_INFO[id][0] === stage && have.has(id)) return id;
  return null;
}

/** The steps rebuilt from a finished response (the stream was unavailable). Steps that never ran are shown as skipped. */
export function stepsFromResponse(r: QueryResponse): { steps: StepState[]; orphanNotes: string[] } {
  const t = r.timings_ms || {};
  const have = new Set(STEP_ORDER.filter((id) => t[id] !== undefined));
  const steps: StepState[] = [];
  let clock = 0;
  for (const id of STEP_ORDER) {
    const [stage, component, blurb] = STEP_INFO[id];
    if (have.has(id)) {
      steps.push({ id, stage, component, blurb, status: "done", start: clock, ms: Math.round(t[id]), notes: [] });
      clock += t[id];
    } else if (id !== "2b_replan") {
      steps.push({ id, stage, component, blurb, status: "skipped", start: clock, reason: "not needed for this question", notes: [] });
    }
  }
  const orphanNotes: string[] = [];
  const byId = (id: string) => steps.find((s) => s.id === id && s.status === "done");
  for (const line of r.trace || []) {
    const text = line.replace(/^\d+\s+/, "");
    // "4 retrieved 20 -> 5 kept 10 (weak evidence)" holds the results of two steps, as the live stream reports them
    const split = text.match(/^retrieved (\d+) -> 5 kept (\d+)(.*)$/);
    if (split && byId("4_retrieve") && byId("5_rerank")) {
      byId("4_retrieve")!.notes.push(`${split[1]} candidate passages`);
      byId("5_rerank")!.notes.push(`kept ${split[2]} of ${split[1]} passages${split[3]}`);
      continue;
    }
    const id = traceStep(line, have);
    const step = id ? steps.find((s) => s.id === id) : undefined;
    if (step) step.notes.push(text);
    else orphanNotes.push(text);
  }
  return { steps, orphanNotes };
}

export function fmtMs(ms: number | undefined): string {
  if (ms === undefined || Number.isNaN(ms)) return "";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function fmtSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}
