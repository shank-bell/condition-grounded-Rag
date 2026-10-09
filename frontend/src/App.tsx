import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, MOCK, StreamUnavailable } from "./api";
import { AssistantTurn } from "./components/AnswerCard";
import { Composer } from "./components/Composer";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ArrowDown, Cross, Menu, Plus, Spark, Warning } from "./components/Icons";
import { Sidebar, type UploadState } from "./components/Sidebar";
import { useMediaQuery, usePrefersReducedMotion } from "./lib/hooks";
import { applyEvent, finishRun, newRun, stepsFromResponse } from "./lib/steps";
import { loadChats, loadFlag, saveChats, saveFlag } from "./lib/storage";
import type { Chat, Exchange, Health, Paper, StreamEvent, Turn } from "./types";

const APP_NAME = "Condition-Grounded Scientific RAG";

const EXAMPLES = [
  "How well do models perform on Kannada NLI?",
  "What accuracy does XLM-R get on XNLI for Kannada?",
  "What F1 does BERT-large get on SQuAD v2.0?",
  "Compare DistilBERT and BERT-base on GLUE",
];

const uid = () => Math.random().toString(36).slice(2, 9) + Date.now().toString(36).slice(-4);
const titleFrom = (q: string) => (q.length > 64 ? `${q.slice(0, 61).trimEnd()}…` : q);
const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

/** The last six turns (three questions and their answers) go with a follow-up question. */
function historyOf(exchanges: Exchange[]): Turn[] {
  return exchanges
    .flatMap((e) =>
      e.response ? [{ role: "user" as const, content: e.question }, { role: "assistant" as const, content: e.response.answer }] : [],
    )
    .slice(-6);
}

function isNetworkError(err: unknown): boolean {
  return err instanceof TypeError || (err instanceof Error && /Failed to fetch|NetworkError|network/i.test(err.message));
}

function describe(err: unknown): string {
  if (isNetworkError(err)) return "The API is not reachable (network error). Is the server running on port 8000?";
  return err instanceof Error ? err.message : String(err);
}

function chatFromHash(): string | null {
  try {
    const m = window.location.hash.match(/chat=([\w-]+)/);
    return m ? m[1] : null;
  } catch {
    return null;
  }
}

function StatusPill({ apiDown, health }: { apiDown: boolean; health: Health | null }) {
  let tone = "wait";
  let text = "Connecting…";
  if (apiDown) {
    tone = "down";
    text = "API not reachable";
  } else if (health?.status === "starting") {
    tone = "warm";
    text = "Warming up the models";
  } else if (health?.status === "ok") {
    tone = health.warm?.status === "failed" ? "warm" : "ok";
    text = health.warm?.status === "failed" ? "Ready · warm-up failed" : "Ready";
  }
  return (
    <span className={`status-pill is-${tone}`} role="status" title={health?.llm ? `Answer model: ${health.llm}` : undefined}>
      <span className="status-dot" aria-hidden="true" />
      {text}
    </span>
  );
}

export default function App() {
  const [chats, setChats] = useState<Chat[]>(loadChats);
  const [activeId, setActiveId] = useState<string | null>(chatFromHash);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState<{ chatId: string; exId: string } | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [collapsed, setCollapsed] = useState(() => loadFlag("sidebarCollapsed"));
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [fresh, setFresh] = useState<Record<string, true>>({});
  const [atBottom, setAtBottom] = useState(true);
  const [undo, setUndo] = useState<Chat | null>(null);

  const mobile = useMediaQuery("(max-width: 900px)");
  const reduced = usePrefersReducedMotion();
  const reducedRef = useRef(reduced);
  reducedRef.current = reduced;
  const abortRef = useRef<AbortController | null>(null);
  const runningRef = useRef(running);
  runningRef.current = running;
  const chatsRef = useRef(chats);
  chatsRef.current = chats;
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const undoTimer = useRef(0);

  const active = chats.find((c) => c.id === activeId) ?? null;
  const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);
  const empty = !active || active.exchanges.length === 0;

  // ---- server state: /health every 2.5 s until ready, then every 20 s (not while a question runs: one GPU) ----
  const refreshPapers = useCallback(async () => {
    try {
      setPapers(await api.papers());
    } catch {
      /* the health poll reports a dead API */
    }
  }, []);

  const pollNow = useRef<() => void>(() => {});
  useEffect(() => {
    let stop = false;
    let timer = 0;
    let hadOk = false;
    const tick = async () => {
      window.clearTimeout(timer);
      let ok = false;
      if (runningRef.current) ok = true;
      else {
        try {
          const h = await api.health();
          if (stop) return;
          setHealth(h);
          setApiDown(false);
          ok = h.status === "ok";
          if (ok && !hadOk) void refreshPapers();
          hadOk = ok;
        } catch {
          if (stop) return;
          setApiDown(true);
          hadOk = false;
        }
      }
      if (!stop) timer = window.setTimeout(tick, ok ? 20000 : 2500);
    };
    pollNow.current = () => void tick();
    void tick();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [refreshPapers]);

  const warming = !apiDown && health?.status === "starting";
  const blocked: string | null = apiDown
    ? "The API is not reachable"
    : health?.status !== "ok"
      ? warming
        ? "Warming up the models…"
        : "Connecting to the API…"
      : running && running.chatId !== active?.id
        ? "Another chat is still answering; wait for it to finish"
        : null;

  // ---- persistence: chats in localStorage, the open chat in the URL (#chat=...) ----
  useEffect(() => {
    const t = window.setTimeout(() => saveChats(chats), 350);
    return () => window.clearTimeout(t);
  }, [chats]);
  useEffect(() => {
    const save = () => saveChats(chatsRef.current);
    window.addEventListener("pagehide", save);
    return () => window.removeEventListener("pagehide", save);
  }, []);
  useEffect(() => {
    try {
      const url = new URL(window.location.href);
      url.hash = active ? `chat=${active.id}` : "";
      window.history.replaceState(null, "", url.toString());
    } catch {
      /* ignore */
    }
    document.title = active ? `${active.title} · ${APP_NAME}` : APP_NAME;
  }, [active?.id, active?.title]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => saveFlag("sidebarCollapsed", collapsed), [collapsed]);

  // ---- scrolling ----
  // While the pipeline works the view follows the newest step. Once the answer arrives it follows the answer only until the
  // start of the answer would leave the top of the view ("cap"), so the reader starts at the beginning. Scrolling up by hand
  // stops the following; the round button (or scrolling back down) resumes it.
  const capRef = useRef<string | null>(null);
  const lastTop = useRef(0);
  const follow = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (stickRef.current) {
      let target = el.scrollHeight - el.clientHeight;
      const cap = capRef.current;
      // the question's own bubble stays whole at the top of the view, with the start of the answer right below it
      const turn = cap ? el.querySelector<HTMLElement>(`[data-ex="${cap}"]`) : null;
      if (turn) {
        const top = turn.getBoundingClientRect().top - el.getBoundingClientRect().top + el.scrollTop;
        target = Math.min(target, Math.max(0, top - 14));
      }
      if (Math.abs(el.scrollTop - target) > 1) el.scrollTop = target;
    }
    lastTop.current = el.scrollTop;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 56);
  }, []);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const moved = el.scrollTop - lastTop.current;
    if (Math.abs(moved) > 2) {
      // a scroll the page did not make itself: the reader's
      if (moved < 0) stickRef.current = false;
      else capRef.current = null;
      if (distance < 56) stickRef.current = true;
    }
    lastTop.current = el.scrollTop;
    setAtBottom(distance < 56);
  };

  useLayoutEffect(() => {
    const el = scrollRef.current;
    const inner = threadRef.current;
    if (!el || !inner) return;
    stickRef.current = true;
    el.scrollTop = el.scrollHeight;
    lastTop.current = el.scrollTop;
    setAtBottom(true);
    const ro = new ResizeObserver(follow);
    ro.observe(inner);
    return () => ro.disconnect();
  }, [active?.id, empty, follow]);

  const toBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    capRef.current = null;
    stickRef.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior: reduced ? "auto" : "smooth" });
  };

  // ---- keyboard: Escape closes the drawer ----
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);
  useEffect(() => {
    if (!mobile) setDrawerOpen(false);
  }, [mobile]);

  // ---- asking ----
  const patch = useCallback((chatId: string, exId: string, fn: (x: Exchange) => Exchange) => {
    setChats((cs) =>
      cs.map((c) => (c.id !== chatId ? c : { ...c, updatedAt: Date.now(), exchanges: c.exchanges.map((x) => (x.id === exId ? fn(x) : x)) })),
    );
  }, []);

  async function execute(chatId: string, exId: string, q: string, history: Turn[]) {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning({ chatId, exId });
    const upd = (fn: (x: Exchange) => Exchange) => patch(chatId, exId, fn);
    let finished = false;
    const fail = (err: unknown) => {
      finished = true;
      upd((x) => ({ ...x, error: describe(err), run: x.run && finishRun(x.run, "error") }));
      if (isNetworkError(err)) {
        setApiDown(true);
        pollNow.current();
      }
    };
    const stopped = () => upd((x) => ({ ...x, stopped: true, run: x.run && finishRun(x.run, "stopped") }));
    const onEvent = (e: StreamEvent) => {
      if (finished || ctrl.signal.aborted) return;
      if (e.type === "result") {
        finished = true;
        capRef.current = exId;
        upd((x) => ({ ...x, response: e.response, run: x.run && finishRun(x.run, "done", e.response.timings_ms?.total) }));
        setFresh((f) => ({ ...f, [exId]: true }));
      } else if (e.type === "error") {
        finished = true;
        upd((x) => ({ ...x, error: e.message || "The pipeline failed.", run: x.run && finishRun(x.run, "error") }));
      } else {
        upd((x) => (x.run ? { ...x, run: applyEvent(x.run, e) } : x));
      }
    };
    try {
      await api.queryStream(q, history, onEvent, ctrl.signal);
      if (!finished && !ctrl.signal.aborted) fail(new Error("The answer stream ended before the answer arrived."));
      else if (ctrl.signal.aborted && !finished) stopped();
    } catch (err) {
      if (ctrl.signal.aborted) {
        if (!finished) stopped();
      } else if (err instanceof StreamUnavailable && !finished) {
        // No live stream: ask the plain endpoint, then replay the steps from the response's trace and timings.
        upd((x) => (x.run ? { ...x, run: { ...x.run, mode: "replay" } } : x));
        try {
          const r = await api.query(q, history, ctrl.signal);
          const { steps, orphanNotes } = stepsFromResponse(r);
          if (!reducedRef.current) {
            for (let k = 1; k <= steps.length && !ctrl.signal.aborted; k++) {
              upd((x) => (x.run ? { ...x, run: { ...x.run, steps: steps.slice(0, k) } } : x));
              await sleep(90);
            }
          }
          finished = true;
          capRef.current = exId;
          upd((x) => ({ ...x, response: r, run: x.run && finishRun({ ...x.run, steps, orphanNotes }, "done", r.timings_ms?.total) }));
          setFresh((f) => ({ ...f, [exId]: true }));
        } catch (err2) {
          if (ctrl.signal.aborted) stopped();
          else fail(err2);
        }
      } else if (!finished) {
        fail(err);
      }
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setRunning(null);
    }
  }

  function ask(question: string) {
    const q = question.trim();
    if (q.length < 3 || running || blocked) return;
    const exId = uid();
    const ex: Exchange = { id: exId, question: q, run: newRun() };
    let chatId = active?.id ?? null;
    const history = historyOf(active?.exchanges ?? []);
    if (!chatId) {
      chatId = uid();
      const now = Date.now();
      setChats((cs) => [{ id: chatId!, title: titleFrom(q), createdAt: now, updatedAt: now, exchanges: [ex] }, ...cs]);
      setActiveId(chatId);
    } else {
      const id = chatId;
      setChats((cs) => cs.map((c) => (c.id === id ? { ...c, updatedAt: Date.now(), exchanges: [...c.exchanges, ex] } : c)));
    }
    setInput("");
    capRef.current = null;
    stickRef.current = true;
    void execute(chatId, exId, q, history);
  }

  function retry(chatId: string, exId: string) {
    if (running || blocked) return;
    const chat = chats.find((c) => c.id === chatId);
    const idx = chat?.exchanges.findIndex((x) => x.id === exId) ?? -1;
    if (!chat || idx < 0) return;
    const q = chat.exchanges[idx].question;
    patch(chatId, exId, (x) => ({ id: x.id, question: x.question, run: newRun() }));
    setFresh(({ [exId]: _drop, ...rest }) => rest);
    capRef.current = null;
    stickRef.current = true;
    void execute(chatId, exId, q, historyOf(chat.exchanges.slice(0, idx)));
  }

  const stop = () => abortRef.current?.abort();

  // ---- chats ----
  function newChat() {
    setActiveId(null);
    setInput("");
    setDrawerOpen(false);
    window.setTimeout(() => inputRef.current?.focus(), 30);
  }

  function selectChat(id: string) {
    setActiveId(id);
    setDrawerOpen(false);
  }

  function deleteChat(id: string) {
    const chat = chats.find((c) => c.id === id);
    if (!chat || running?.chatId === id) return;
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
    setUndo(chat);
    window.clearTimeout(undoTimer.current);
    undoTimer.current = window.setTimeout(() => setUndo(null), 6000);
  }

  function undoDelete() {
    if (!undo) return;
    const chat = undo;
    setChats((cs) => (cs.some((c) => c.id === chat.id) ? cs : [chat, ...cs]));
    setUndo(null);
  }

  // ---- paper upload ----
  async function onFile(file: File | undefined) {
    if (!file) return;
    setUpload({ state: "busy", text: `Uploading ${file.name}…` });
    try {
      const { job_id } = await api.upload(file);
      setUpload({ state: "busy", text: `Reading ${file.name}: extracting conditions (this takes a few minutes)…` });
      for (;;) {
        await sleep(4000);
        const job = await api.job(job_id);
        if (job.status === "done") {
          setUpload({ state: "done", text: `Added ${file.name}: ${job.chunks} passages, ${job.profiles} condition profiles.` });
          break;
        }
        if (job.status === "failed") {
          setUpload({ state: "error", text: `Could not add ${file.name}: ${job.error}` });
          break;
        }
        setUpload({
          state: "busy",
          text: job.status === "queued" ? `${file.name} is queued: another paper is being read…` : `Reading ${file.name}: extracting conditions (this takes a few minutes)…`,
        });
      }
      void refreshPapers();
      pollNow.current();
    } catch (err) {
      setUpload({ state: "error", text: `Upload failed: ${describe(err)}` });
    }
  }

  const titleOf = (pid: string) => papers.find((p) => p.paper_id === pid)?.title || pid;
  const sidebarVisible = mobile ? drawerOpen : !collapsed;
  const modelLabel = health?.llm;

  const notices = (
    <>
      {apiDown && (
        <div className="notice is-error" role="alert">
          <Warning size={16} />
          <span>
            The API is not reachable. Start it with <code>uvicorn cgrag.api.main:app --port 8000</code>.
          </span>
        </div>
      )}
      {warming && (
        <div className="notice is-warm" role="status">
          <span className="spinner" aria-hidden="true" />
          <span>Warming up the language models (about 30 seconds). You can ask as soon as this message disappears.</span>
        </div>
      )}
      {!apiDown && health?.warm?.status === "failed" && (
        <div className="notice is-error" role="alert">
          <Warning size={16} />
          <span>Warm-up failed ({health.warm.error}); the first question will be slow while the models load.</span>
        </div>
      )}
      {upload && !sidebarVisible && (
        <div className={`notice is-upload is-${upload.state}`} role="status">
          {upload.state === "busy" ? <span className="spinner" aria-hidden="true" /> : upload.state === "error" ? <Warning size={16} /> : <Spark size={15} />}
          <span>{upload.text}</span>
          {upload.state !== "busy" && (
            <button type="button" className="icon-btn notice-close" onClick={() => setUpload(null)} aria-label="Dismiss">
              <Cross size={13} />
            </button>
          )}
        </div>
      )}
    </>
  );

  const composer = (variant: "welcome" | "dock") => (
    <Composer
      variant={variant}
      value={input}
      onChange={setInput}
      onSend={() => ask(input)}
      onStop={stop}
      onAttach={() => fileRef.current?.click()}
      running={!!running && running.chatId === active?.id}
      blocked={blocked}
      placeholder={
        apiDown ? "The API is not reachable" : warming ? "Warming up the models…" : variant === "welcome" ? "Ask about the indexed papers…" : "Ask a follow-up…"
      }
      modelLabel={modelLabel}
      inputRef={inputRef}
    />
  );

  return (
    <div className={`app${!mobile && collapsed ? " is-collapsed" : ""}${mobile ? " is-mobile" : ""}`}>
      <Sidebar
        collapsed={collapsed}
        mobile={mobile}
        drawerOpen={drawerOpen}
        onToggle={() => (mobile ? setDrawerOpen(false) : setCollapsed((v) => !v))}
        chats={sorted}
        activeId={active?.id ?? null}
        runningChatId={running?.chatId ?? null}
        onSelect={selectChat}
        onNew={newChat}
        onDelete={deleteChat}
        health={health}
        papers={papers}
        upload={upload}
        onUpload={() => fileRef.current?.click()}
      />
      {mobile && (
        <div className={`scrim${drawerOpen ? " is-open" : ""}`} onClick={() => setDrawerOpen(false)} aria-hidden="true" />
      )}

      <main className="main" inert={(mobile && drawerOpen) || undefined}>
        <header className="topbar">
          {mobile && (
            <button type="button" className="icon-btn" onClick={() => setDrawerOpen(true)} aria-label="Open sidebar" title="Open sidebar">
              <Menu size={20} />
            </button>
          )}
          <h1 className="topbar-title" title={active?.title}>
            {active ? active.title : "New chat"}
          </h1>
          <div className="topbar-right">
            {MOCK && (
              <span className="mock-tag" title="Dev mock: canned answers, no GPU (remove ?mock from the URL for the real API)">
                mock
              </span>
            )}
            <StatusPill apiDown={apiDown} health={health} />
            {mobile && (
              <button type="button" className="icon-btn" onClick={newChat} aria-label="New chat" title="New chat">
                <Plus size={19} />
              </button>
            )}
          </div>
        </header>

        {empty ? (
          <div className="welcome">
            <div className="welcome-inner">
              <div className="greeting">
                <Spark size={40} className="greeting-spark" />
                <h2>What would you like to know about the papers?</h2>
              </div>
              <p className="welcome-sub">
                {health?.status === "ok"
                  ? `${health.papers ?? papers.length} NLP papers · every answer says which conditions it holds under, and warns when the papers do not cover your question.`
                  : "Answers say which conditions each finding holds under, and warn when the papers do not cover your question."}
              </p>
              <div className="welcome-box">
                {notices}
                {composer("welcome")}
                <div className="pills" role="group" aria-label="Example questions">
                  {EXAMPLES.map((q) => (
                    <button key={q} type="button" className="pill" onClick={() => ask(q)} disabled={!!blocked || !!running}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="scroll" ref={scrollRef} onScroll={onScroll}>
              <div className="thread" ref={threadRef}>
                {active!.exchanges.map((x) => {
                  const live = running?.exId === x.id && !x.response && !x.error && !x.stopped;
                  return (
                    <section key={x.id} className="turn" data-ex={x.id} aria-label={`Question: ${x.question}`}>
                      <div className="turn-user">
                        <div className="bubble">{x.question}</div>
                      </div>
                      <ErrorBoundary scope="turn" onRetry={() => retry(active!.id, x.id)}>
                        <AssistantTurn
                          x={x}
                          live={live}
                          animate={!!fresh[x.id]}
                          canRetry={!running && !blocked}
                          onRetry={() => retry(active!.id, x.id)}
                          onRevealed={() => setFresh(({ [x.id]: _drop, ...rest }) => rest)}
                          titleOf={titleOf}
                        />
                      </ErrorBoundary>
                    </section>
                  );
                })}
              </div>
            </div>
            <div className="dock">
              <button
                type="button"
                className={`to-bottom${atBottom ? "" : " is-visible"}`}
                onClick={toBottom}
                aria-label="Scroll to the newest message"
                title="Scroll to bottom"
                tabIndex={atBottom ? -1 : 0}
              >
                <ArrowDown size={17} />
              </button>
              <div className="dock-inner">
                {notices}
                {composer("dock")}
                <p className="dock-hint">Answers are checked against the papers but can still be wrong · Shift + Enter for a new line</p>
              </div>
            </div>
          </>
        )}

        {undo && (
          <div className="toast" role="status">
            Chat deleted.
            <button type="button" className="text-btn" onClick={undoDelete}>
              Undo
            </button>
          </div>
        )}
      </main>

      <input
        ref={fileRef}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={(e) => {
          void onFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
