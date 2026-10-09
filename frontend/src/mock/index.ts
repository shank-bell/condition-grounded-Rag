// DEV ONLY: a fake backend for designing the page without the GPU. Enabled with the URL parameter ?mock=1 (see MOCK in api.ts).
// It replays the event sequence of POST /query/stream with realistic delays and real answers captured from the backend.
// Health and the paper list come from the real API when it is up, otherwise from canned values.
import { StreamUnavailable, MOCK } from "../api";
import { STEP_INFO } from "../lib/steps";
import type { Health, Paper, QueryResponse, StreamEvent, UploadJob } from "../types";
import { SAMPLE_BERT_SQUAD, SAMPLE_DISTILBERT_GLUE, SAMPLE_KANNADA_NLI, SAMPLE_XLMR_KANNADA } from "./samples";

const params = (() => {
  try {
    return new URLSearchParams(window.location.search);
  } catch {
    return new URLSearchParams();
  }
})();
const SPEED = Math.max(0.25, Number(params.get("speed")) || 1);

/** A canned answer with a markdown table and inline code (to check how they render); its run skips Stage 3. */
const { "3_refine": _skipped, ...timingsWithoutRefine } = SAMPLE_DISTILBERT_GLUE.timings_ms;
const SAMPLE_TABLE: QueryResponse = {
  ...SAMPLE_DISTILBERT_GLUE,
  timings_ms: timingsWithoutRefine,
  trace: SAMPLE_DISTILBERT_GLUE.trace
    .filter((t) => !t.startsWith("3 "))
    .map((t) => (t.startsWith("2 refinement") ? "2 refinement: none (the question is already specific)" : t)),
  question: "Show DistilBERT and BERT-base on GLUE as a table",
  answer:
    "On the GLUE dev sets, DistilBERT keeps most of BERT-base's accuracy with 40 % fewer parameters [1]. The comparison uses the `dev` split and the medians of 5 runs; see the `Table 1` caption of the DistilBERT paper.\n\n" +
    "| Task | BERT-base | DistilBERT | Difference |\n|---|---:|---:|---:|\n| CoLA | 56.3 | 51.3 | -5.0 |\n| MNLI | 86.7 | 82.2 | -4.5 |\n| MRPC | 88.6 | 87.5 | -1.1 |\n| QNLI | 91.8 | 89.2 | -2.6 |\n| QQP | 89.6 | 88.5 | -1.1 |\n| RTE | 69.3 | 59.9 | -9.4 |\n\n" +
    "### Notes\n1. The largest gap is on **RTE** (9.4 points), a small data set [1].\n2. Inference is about 60 % faster: 410 s vs 668 s on STS-B [3].\n\n" +
    "> The parameter counts (66M vs 110M) come from the same paper [3].\n\nA long identifier that must wrap: `bert-base-multilingual-cased-finetuned-on-xnli-translate-train-all-languages-v2`.",
};

/** The SQuAD answer with its first conflict turned into a GENUINE one (to check the red badge). Ask a question containing "genuine". */
const SAMPLE_GENUINE: QueryResponse = {
  ...SAMPLE_BERT_SQUAD,
  contradictions: SAMPLE_BERT_SQUAD.contradictions.map((c, i) =>
    i === 0
      ? {
          ...c,
          verdict: "GENUINE" as const,
          differing: [],
          nli_contradiction: 0.93,
          reason: "Same model, same dataset version, same split and same setting, yet the numbers differ by more than 2 points: no recorded condition explains it.",
        }
      : c,
  ),
};

function pick(question: string): QueryResponse {
  const q = question.toLowerCase();
  let r: QueryResponse;
  if (q.includes("genuine")) r = SAMPLE_GENUINE;
  else if (q.includes("table") || q.includes("markdown")) r = SAMPLE_TABLE;
  else if (q.includes("xlm") && q.includes("kannada")) r = SAMPLE_XLMR_KANNADA;
  else if (q.includes("squad")) r = SAMPLE_BERT_SQUAD;
  else if (q.includes("distil") || q.includes("glue")) r = SAMPLE_DISTILBERT_GLUE;
  else r = SAMPLE_KANNADA_NLI;
  return { ...r, question };
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const t = window.setTimeout(resolve, ms / SPEED);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

/** The notes the real stream sends, rebuilt from the response's trace (stage number = first word of the line). */
function notesFor(r: QueryResponse, stage: number): string[] {
  const out: string[] = [];
  for (const line of r.trace) {
    const m = line.match(/^(\d+)\s+(.*)$/);
    if (!m || Number(m[1]) !== stage) continue;
    let text = m[2];
    if (stage === 4 && /^retrieved/.test(text)) {
      const k = text.match(/retrieved (\d+) -> 5 kept (\d+)(.*)$/);
      if (k) {
        out.push(`${k[1]} candidate passages`);
        continue;
      }
    }
    if (stage === 9 && r.regenerated) text = text.replace(", regenerated once", "") + ", regenerated once";
    out.push(text);
  }
  return out;
}

export async function mockStream(question: string, onEvent: (e: StreamEvent) => void, signal: AbortSignal): Promise<void> {
  if (MOCK === "plain") throw new StreamUnavailable("404 Not Found (mock)");
  const r = pick(question);
  const t0 = performance.now();
  const now = () => Math.round((performance.now() - t0) * SPEED);
  const emit = (e: StreamEvent) => {
    if (!signal.aborted) onEvent(e);
  };
  if (params.get("queued")) {
    emit({ type: "queued" });
    await sleep(2500, signal);
  }
  const step = async (id: string, after?: () => void) => {
    const [stage, component, blurb] = STEP_INFO[id];
    const ms = r.timings_ms[id];
    if (ms === undefined) {
      const reason = id === "3_refine" ? "simple question: searched with the question as asked" : id === "7_contradiction" ? "fewer than two passages kept" : "not needed for this question";
      emit({ type: "step", id, stage, component, blurb, status: "skipped", reason, t_ms: now() });
      return;
    }
    emit({ type: "step", id, stage, component, blurb, status: "running", t_ms: now() });
    if (id === "9_critic" && r.regenerated) {
      await sleep(ms * 0.3, signal);
      emit({ type: "note", stage: 9, text: "1 sentence(s) not backed by the sources: writing the answer once more", t_ms: now() });
      await sleep(ms * 0.7, signal);
    } else {
      await sleep(ms, signal);
    }
    if (MOCK === "error" && id === "8_generate") throw new Error("mock: the language model did not answer (Ollama timeout)");
    emit({ type: "step", id, stage, component, status: "done", ms: Math.round(ms), t_ms: now() });
    after?.();
  };
  const notes = (stage: number) => () => notesFor(r, stage).forEach((text) => emit({ type: "note", stage, text, t_ms: now() }));
  try {
    await step("1_understand");
    await step("2_plan", () => {
      notes(1)();
      notesFor(r, 2).filter((t) => !t.startsWith("re-plan")).forEach((text) => emit({ type: "note", stage: 2, text, t_ms: now() }));
    });
    await step("3_refine", notes(3));
    await step("4_retrieve", () => {
      const first = notesFor(r, 4)[0];
      if (first) emit({ type: "note", stage: 4, text: first, t_ms: now() });
    });
    await step("5_rerank", () => {
      const k = r.trace.join("\n").match(/retrieved (\d+) -> 5 kept (\d+)( \(weak evidence\))?/);
      if (k) emit({ type: "note", stage: 5, text: `kept ${k[2]} of ${k[1]} passages${k[3] ?? ""}`, t_ms: now() });
    });
    await step("2b_replan");
    await step("6_applicability", () => {
      notes(6)();
      notesFor(r, 4).slice(1).forEach((text) => emit({ type: "note", stage: 4, text, t_ms: now() }));
    });
    await step("7_contradiction", notes(7));
    await step("8_generate", () => emit({ type: "note", stage: 8, text: `wrote ${r.answer.length} characters from ${r.sources.length} passages`, t_ms: now() }));
    await step("9_critic", notes(9));
  } catch (err) {
    if (signal.aborted) throw err;
    emit({ type: "error", message: err instanceof Error ? err.message : String(err) });
    return;
  }
  emit({ type: "result", response: { ...r, timings_ms: { ...r.timings_ms, total: now() } } });
}

export async function mockQuery(question: string, signal?: AbortSignal): Promise<QueryResponse> {
  const r = pick(question);
  await sleep(r.timings_ms.total ?? 9000, signal);
  return r;
}

const MOCK_HEALTH: Health = {
  status: "ok", llm: "gemma4:12b (mock)", chunks: 1385, profiles: 10275, papers: 6,
  warm: { status: "ready", seconds: 29, error: null },
};

export async function mockHealth(real: () => Promise<Health>): Promise<Health> {
  if (MOCK === "down") throw new TypeError("Failed to fetch (mock)");
  if (MOCK === "warming") return { status: "starting", warm: { status: "starting", seconds: null, error: null } };
  try {
    return await real();
  } catch {
    return MOCK_HEALTH;
  }
}

const MOCK_PAPERS: Paper[] = [
  { paper_id: "1606.05250", title: "SQuAD: 100,000+ Questions for Machine Comprehension of Text", chunks: 26, profiles: 53 },
  { paper_id: "1806.03822", title: "Know What You Don't Know: Unanswerable Questions for SQuAD", chunks: 20, profiles: 61 },
  { paper_id: "1810.04805", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", chunks: 43, profiles: 180 },
  { paper_id: "1910.01108", title: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", chunks: 21, profiles: 96 },
  { paper_id: "1911.02116", title: "Unsupervised Cross-lingual Representation Learning at Scale", chunks: 38, profiles: 412 },
  { paper_id: "2212.05409", title: "Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages", chunks: 64, profiles: 1203 },
];

export async function mockPapers(real: () => Promise<Paper[]>): Promise<Paper[]> {
  try {
    return await real();
  } catch {
    return MOCK_PAPERS;
  }
}

const jobs = new Map<string, { name: string; polls: number }>();

export async function mockUpload(file: File): Promise<{ job_id: string; paper_id: string }> {
  await sleep(600);
  if (!file.name.toLowerCase().endsWith(".pdf")) throw new Error("400 Bad Request: {\"detail\":\"Upload a PDF file.\"}");
  const id = Math.random().toString(16).slice(2, 10);
  jobs.set(id, { name: file.name, polls: 0 });
  return { job_id: id, paper_id: file.name.replace(/\.pdf$/i, "") };
}

export async function mockJob(id: string): Promise<UploadJob> {
  const job = jobs.get(id);
  if (!job) throw new Error("404 Not Found: Unknown job.");
  job.polls += 1;
  if (job.polls < 2) return { status: "queued" };
  if (job.polls < 4) return { status: "running" };
  return { status: "done", chunks: 31, profiles: 142, title: job.name };
}
