import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { AnswerCard } from "./components/AnswerCard";
import type { Health, Paper, QueryResponse, Turn } from "./types";

interface Exchange {
  id: string;
  question: string;
  response?: QueryResponse;
  error?: string;
}

const EXAMPLES = [
  "Does BERT work well for Kannada question answering?",
  "What F1 does BERT-large get on SQuAD v2.0, and how does it compare with v1.1?",
  "Do papers agree on BERT-large accuracy on MNLI?",
];

export default function App() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [uploadNote, setUploadNote] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [h, p] = await Promise.all([api.health(), api.papers()]);
      setHealth(h);
      setPapers(p);
      setApiDown(false);
    } catch {
      setApiDown(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [exchanges]);

  async function ask(question: string) {
    const q = question.trim();
    if (q.length < 3 || busy) return;
    const history: Turn[] = exchanges.flatMap((e) =>
      e.response ? [{ role: "user" as const, content: e.question }, { role: "assistant" as const, content: e.response.answer }] : [],
    ).slice(-6);
    const id = `q${exchanges.length + 1}`;
    setExchanges((xs) => [...xs, { id, question: q }]);
    setInput("");
    setBusy(true);
    try {
      const response = await api.query(q, history);
      setExchanges((xs) => xs.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (err) {
      setExchanges((xs) => xs.map((x) => (x.id === id ? { ...x, error: String(err) } : x)));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(input);
  }

  async function onUpload(file: File | undefined) {
    if (!file) return;
    setUploadNote(`Uploading ${file.name}…`);
    try {
      const { job_id } = await api.upload(file);
      setUploadNote(`Reading ${file.name}: extracting conditions (this takes a few minutes)…`);
      for (;;) {
        await new Promise((r) => setTimeout(r, 4000));
        const job = await api.job(job_id);
        if (job.status === "done") {
          setUploadNote(`Added ${file.name}: ${job.chunks} passages, ${job.profiles} condition profiles.`);
          break;
        }
        if (job.status === "failed") {
          setUploadNote(`Could not add ${file.name}: ${job.error}`);
          break;
        }
      }
      void refresh();
    } catch (err) {
      setUploadNote(`Upload failed: ${String(err)}`);
    }
  }

  return (
    <div className="layout">
      <header className="top">
        <h1>Condition-Grounded Scientific RAG</h1>
        <p className="muted">
          Answers state the conditions each finding holds under, warn when the evidence does not cover your question, and
          separate real conflicts between papers from differences caused by different experiments.
        </p>
      </header>

      <main className="chat">
        {apiDown && (
          <p className="warning" role="alert">
            The API is not reachable. Start it with <code>uvicorn cgrag.api.main:app --port 8000</code>.
          </p>
        )}
        {exchanges.length === 0 && (
          <section className="examples">
            <h2>Try a question</h2>
            {EXAMPLES.map((q) => (
              <button key={q} type="button" className="example" onClick={() => void ask(q)} disabled={busy || apiDown}>
                {q}
              </button>
            ))}
          </section>
        )}
        {exchanges.map((x) => (
          <div key={x.id} className="exchange">
            <p className="question">{x.question}</p>
            {x.response && <AnswerCard r={x.response} id={x.id} />}
            {x.error && <p className="warning" role="alert">{x.error}</p>}
            {!x.response && !x.error && <p className="muted pending">Working through the pipeline…</p>}
          </div>
        ))}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the indexed papers…"
          aria-label="Question"
          maxLength={2000}
          disabled={apiDown}
        />
        <button type="submit" disabled={busy || apiDown || input.trim().length < 3}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>

      <aside className="side">
        <h2>Indexed papers</h2>
        {health && (
          <p className="muted small">
            {health.papers} papers · {health.chunks} passages · {health.profiles} condition profiles · {health.llm}
          </p>
        )}
        <label className="upload">
          Add a paper (PDF)
          <input type="file" accept="application/pdf" onChange={(e) => void onUpload(e.target.files?.[0])} />
        </label>
        {uploadNote && <p className="small" role="status">{uploadNote}</p>}
        <ul className="papers">
          {papers.map((p) => (
            <li key={p.paper_id}>
              <span>{p.title || p.paper_id}</span>
              <span className="muted small"> {p.profiles} profiles</span>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
