// The live "chain of execution" above an answer: which pipeline component runs right now, like a "Thinking" block.
// Collapses by itself into one line when the answer is complete; a click opens it again with every note and time.
import { useEffect, useId, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../lib/hooks";
import { prettyNote } from "../lib/notes";
import { fmtMs } from "../lib/steps";
import type { QueryResponse, RunState, StepState } from "../types";
import { Check, ChevronDown, Cross, Dash, Spark } from "./Icons";

/** The current time, refreshed every `every` ms while `active` (for the live timers). */
function useNow(active: boolean, every = 200): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), every);
    return () => window.clearInterval(t);
  }, [active, every]);
  return now;
}

/** "21.6 s": the same precision as the time line under the answer (the live header counts whole seconds). */
function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

/** A note line can carry several parts ("coverage 0.67, ... | agent: ..."): one line each (Python-style text made readable first). */
function noteLines(text: string): string[] {
  return prettyNote(text).split(/\s+\|\s+/).filter(Boolean);
}

function StepRow({ s, live, now, total }: { s: StepState; live: boolean; now: number; total: number }) {
  const running = s.status === "running";
  const current = live && running;
  const unfinished = !live && running;                     // the run stopped while this step was running
  let time = "";
  if (s.status === "done") time = fmtMs(s.ms);
  else if (s.status === "skipped") time = "skipped";
  else if (current && s.seenAt) time = seconds(Math.max(0, now - s.seenAt));
  else if (unfinished) time = "stopped";
  const showBar = !live && total > 0 && s.status === "done" && s.ms !== undefined;
  return (
    <li className={`step step-${unfinished ? "stopped" : s.status}${current ? " is-current" : ""}`} aria-current={current ? "step" : undefined}>
      <span className="step-icon" aria-hidden="true">
        {current ? <span className="spinner" /> : s.status === "done" ? <Check size={13} /> : unfinished ? <Cross size={12} /> : <Dash size={13} />}
      </span>
      <div className="step-main">
        <div className="step-head">
          <span className="step-name">{s.component}</span>
          <span className="stage-tag">Stage {s.stage}</span>
          <span className="step-time">
            {showBar && (
              <span className="step-bar" aria-hidden="true">
                <span
                  style={{
                    left: `${Math.min(100, (s.start / total) * 100)}%`,
                    width: `${Math.max(2.5, ((s.ms ?? 0) / total) * 100)}%`,
                  }}
                />
              </span>
            )}
            <span className="step-time-text">{time}</span>
          </span>
        </div>
        {s.blurb && <p className="step-blurb">{s.blurb}</p>}
        {s.status === "skipped" && s.reason && <p className="step-note">Skipped: {s.reason}</p>}
        {s.notes.flatMap(noteLines).map((n, i) => (
          <p key={i} className="step-note">
            {n}
          </p>
        ))}
      </div>
    </li>
  );
}

export function Thinking({ run, live, response }: { run: RunState; live: boolean; response?: QueryResponse }) {
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const [autoOpen, setAutoOpen] = useState(live);
  const [stuck, setStuck] = useState(false);
  const bodyId = useId();
  const sentinel = useRef<HTMLDivElement>(null);
  const now = useNow(live);
  const reduced = usePrefersReducedMotion();

  // The header is pinned while the steps scroll under it; a soft fade below it shows that (only while it is really pinned).
  useEffect(() => {
    const el = sentinel.current;
    const scroller = el?.closest(".scroll");
    if (!live || !el || !scroller || typeof IntersectionObserver === "undefined") {
      setStuck(false);
      return;
    }
    const io = new IntersectionObserver(
      ([e]) => setStuck(!e.isIntersecting && e.boundingClientRect.top < (e.rootBounds?.top ?? 0)),
      { root: scroller },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [live]);

  // Open while the pipeline works, close by itself a moment after the answer arrives (unless the user chose).
  useEffect(() => {
    if (live) {
      setAutoOpen(true);
      return;
    }
    const t = window.setTimeout(() => setAutoOpen(false), reduced ? 0 : 380);
    return () => window.clearTimeout(t);
  }, [live, reduced]);

  const open = userOpen ?? autoOpen;
  const done = run.steps.filter((s) => s.status === "done").length;
  const current = live ? [...run.steps].reverse().find((s) => s.status === "running") : undefined;
  const serverTotal = response?.timings_ms?.total;
  const elapsed = live ? Math.max(0, now - run.startedAt) : serverTotal ?? run.totalMs ?? 0;
  const span = Math.max(1, ...run.steps.map((s) => s.start + (s.ms ?? 0)));
  const total = Math.max(span, serverTotal ?? 0);

  let label: string;
  if (live && run.queued) label = "Waiting for the previous question to finish…";
  else if (live) label = "Working through the pipeline…";
  else if (run.outcome === "stopped") label = `Stopped after ${done} step${done === 1 ? "" : "s"}`;
  else if (run.outcome === "error") label = `Stopped by an error after ${done} step${done === 1 ? "" : "s"}`;
  else label = `Worked through ${done} step${done === 1 ? "" : "s"}`;

  return (
    <div className={`thinking${live ? " is-live" : ""}${open ? " is-open" : ""}${live && stuck ? " is-stuck" : ""}`}>
      <div className="thinking-sentinel" ref={sentinel} aria-hidden="true" />
      {/* while the pipeline works this bar stays pinned at the top of the conversation, so the live timer is always in view */}
      <div className="thinking-bar">
        <button
          type="button"
          className="thinking-head"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setUserOpen(!open)}
        >
          <Spark size={16} animated={live} className="thinking-spark" />
          <span className={`thinking-label${live ? " shimmer" : ""}`}>{label}</span>
          {live && !open && current && <span className="thinking-current">{current.component}</span>}
          <span className="thinking-time">{live ? `${Math.floor(elapsed / 1000)} s` : `· ${seconds(elapsed)}`}</span>
          <ChevronDown size={15} className="thinking-chevron" />
        </button>
        <span className="sr-only" aria-live="polite">
          {current ? `Running: ${current.component}` : ""}
        </span>
      </div>
      {open && (
        <div className="thinking-body" id={bodyId}>
          {run.steps.length === 0 && live && (
            <p className="thinking-wait">
              <span className="spinner" aria-hidden="true" />
              {run.queued
                ? "The server answers one question at a time; yours starts as soon as the previous one is done."
                : run.mode === "replay"
                  ? "Live steps are not available from this server; the steps are shown when the answer arrives."
                  : "Connecting to the pipeline…"}
            </p>
          )}
          <ol className="steps">
            {run.steps.map((s) => (
              <StepRow key={s.id} s={s} live={live} now={now} total={total} />
            ))}
          </ol>
          {run.orphanNotes.length > 0 && (
            <div className="thinking-extra">
              {run.orphanNotes.map((n, i) => (
                <p key={i} className="step-note">
                  {n}
                </p>
              ))}
            </div>
          )}
          {!live && run.mode === "replay" && (
            <p className="thinking-foot">Replayed from the response: the server did not stream its steps.</p>
          )}
          {!live && response && (
            <details className="raw-trace">
              <summary>Raw trace and timings</summary>
              <pre>
                {response.trace.join("\n")}
                {"\n\n"}
                {Object.entries(response.timings_ms)
                  .map(([k, v]) => `${k.padEnd(18)} ${(v / 1000).toFixed(2)} s`)
                  .join("\n")}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
