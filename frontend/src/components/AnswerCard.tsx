// One assistant turn: the live pipeline steps, the scope warning, the answer (revealed smoothly), the evidence
// (coverage, conflicts between papers, claim check, sources) and a small action row (copy, retry).
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { usePrefersReducedMotion } from "../lib/hooks";
import type { ApplicabilityResult, ClaimCheck, ContradictionPair, Exchange, QueryResponse, SourceRef, Verdict } from "../types";
import { Book, Check, ChevronDown, Compare, Copy, Cross, Retry, Shield, Spark, Target, Warning } from "./Icons";
import { Inline, Markdown, tidyPartial, type CiteProps } from "./Markdown";
import { Thinking } from "./Thinking";

const VERDICT: Record<Verdict, { label: string; tone: string }> = {
  GENUINE: { label: "Genuine conflict", tone: "genuine" },
  EXPLAINED: { label: "Explained", tone: "explained" },
  NOT_COMPARABLE: { label: "Not comparable", tone: "neutral" },
};

const BULLET = /^\s*([*-]|\d+[.)])\s+/;
const nice = (s: string) => s.replace(/_/g, " ");

/** "BERT: Pre-training of Deep ..." -> "BERT" (the part before a colon, when there is a short one). */
export function shortTitle(title: string): string {
  const t = title.trim();
  const i = t.indexOf(":");
  if (i > 1 && i <= 42) return t.slice(0, i).trim();
  return t;
}

function fmtNum(v: number): string {
  return Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100);
}

/** Types the answer out in about a second (word by word), after `delay` ms (the thinking block folds away first);
 *  instant when the user prefers reduced motion. */
function useReveal(text: string, animate: boolean, delay = 0): { shown: string; done: boolean; finish: () => void } {
  const reduced = usePrefersReducedMotion();
  const play = animate && !reduced;
  const [n, setN] = useState(play ? 0 : text.length);
  useEffect(() => {
    if (!play) {
      setN(text.length);
      return;
    }
    setN(0);
    let raf = 0;
    let t0 = 0;
    const dur = Math.min(2400, Math.max(700, text.length * 2.4));
    const tick = (t: number) => {
      if (!t0) t0 = t;
      const p = Math.min(1, (t - t0) / dur);
      let k = Math.floor(text.length * (1 - Math.pow(1 - p, 1.6)));
      while (k < text.length && /\S/.test(text[k])) k++;          // end on a word boundary
      setN(Math.max(1, k));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    const wait = window.setTimeout(() => {
      raf = requestAnimationFrame(tick);
    }, delay);
    return () => {
      window.clearTimeout(wait);
      cancelAnimationFrame(raf);
    };
  }, [text, play, delay]);
  const done = n >= text.length;
  return { shown: done ? text : tidyPartial(text.slice(0, n)), done, finish: () => setN(text.length) };
}

function Section({
  icon,
  title,
  summary,
  open,
  onToggle,
  children,
  tone,
  short,
}: {
  icon: ReactNode;
  title: string;
  summary?: ReactNode;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
  tone?: string;
  /** A short summary (a count) stays beside the title on narrow screens instead of dropping to its own line. */
  short?: boolean;
}) {
  return (
    <div className={`ev-section${open ? " is-open" : ""}${tone ? ` tone-${tone}` : ""}`}>
      <button type="button" className={`ev-head${short ? " is-short" : ""}`} aria-expanded={open} onClick={onToggle}>
        <span className="ev-icon" aria-hidden="true">
          {icon}
        </span>
        <span className="ev-title">{title}</span>
        {summary !== undefined && <span className="ev-summary">{summary}</span>}
        <ChevronDown size={15} className="ev-chevron" />
      </button>
      {open && <div className="ev-body">{children}</div>}
    </div>
  );
}

function Coverage({ a, weakWithdrawn, retrievalWeak }: { a: ApplicabilityResult; weakWithdrawn: boolean; retrievalWeak: boolean }) {
  const [open, setOpen] = useState(true);
  const covered = a.checks.filter((c) => c.covered).length;
  const pct = Math.round(a.coverage * 100);
  const tone = a.checks.length === 0 ? "neutral" : a.coverage >= 1 ? "ok" : a.coverage > 0 ? "part" : "none";
  const missing = a.checks.filter((c) => !c.covered);
  return (
    <Section
      icon={<Target size={16} />}
      title="Evidence"
      open={open}
      onToggle={() => setOpen((v) => !v)}
      summary={
        a.checks.length === 0 ? (
          <span className="muted">no conditions named</span>
        ) : (
          <>
            <span className="cov-bar" role="meter" aria-label="Evidence coverage" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
              <span className={`cov-fill tone-${tone}`} style={{ width: `${Math.max(pct, 4)}%` }} />
            </span>
            <span className={`cov-text tone-${tone}`}>
              {covered} of {a.checks.length} conditions
            </span>
          </>
        )
      }
    >
      {a.checks.length === 0 ? (
        <p className="ev-note">The question names no conditions (model, dataset, language …), so there was nothing to check.</p>
      ) : (
        <ul className="cond-chips" aria-label="Conditions of the question">
          {a.checks.map((c) => (
            <li
              key={`${c.condition}-${c.requested}`}
              className={`cond-chip ${c.covered ? "is-covered" : "is-missing"}`}
              title={
                c.covered
                  ? `Found in ${c.chunk_ids.length} passage${c.chunk_ids.length === 1 ? "" : "s"}`
                  : c.observed.length
                    ? `Sources record: ${c.observed.join(", ")}`
                    : "No source states this"
              }
            >
              <span className="cond-mark" aria-label={c.covered ? "covered" : "not covered"}>
                {c.covered ? <Check size={12} /> : <Cross size={11} />}
              </span>
              <span className="cond-key">{nice(c.condition)}</span>
              <span className="cond-val">{c.requested}</span>
            </li>
          ))}
        </ul>
      )}
      {missing.map((c) => (
        <p key={c.condition} className="ev-note">
          <strong>{nice(c.condition)} = {c.requested}</strong>:{" "}
          {c.observed.length ? `the sources record ${c.observed.join(", ")}` : "no source states it"}.
        </p>
      ))}
      {a.joint_covered === true && <p className="ev-note">One result records these conditions together.</p>}
      {a.joint_covered === false && (
        <p className="ev-note">No single result records these conditions together: each appears in the sources, but in different combinations.</p>
      )}
      {a.re_retrieved && <p className="ev-note">The applicability agent searched again for the missing condition.</p>}
      {a.profile_guided && <p className="ev-note">Passages were added through the condition profiles to look for the missing condition.</p>}
      {a.escalated && <p className="ev-note">The larger model gave a second opinion before this warning was shown.</p>}
      {retrievalWeak && (
        <p className="ev-note">
          The reranker scored no passage above its relevance threshold
          {weakWithdrawn ? "; the flag was withdrawn because every named condition is recorded in the kept passages." : "."}
        </p>
      )}
      {a.reasoning && (
        <p className="ev-note ev-agent">
          <span className="ev-agent-label">Agent notes</span> {a.reasoning}
        </p>
      )}
    </Section>
  );
}

function Conflicts({ items, titleOf }: { items: ContradictionPair[]; titleOf: (paperId: string) => string }) {
  const [open, setOpen] = useState(true);
  const counts = items.reduce<Record<string, number>>((acc, c) => ({ ...acc, [c.verdict]: (acc[c.verdict] ?? 0) + 1 }), {});
  const summary = (Object.keys(counts) as Verdict[]).map((v) => `${counts[v]} ${VERDICT[v].label.toLowerCase()}`).join(", ");
  return (
    <Section icon={<Compare size={16} />} title="Conflicts between papers" summary={summary} open={open} onToggle={() => setOpen((v) => !v)}>
      <ul className="conflicts">
        {items.map((c, i) => {
          const v = VERDICT[c.verdict] ?? VERDICT.NOT_COMPARABLE;
          return (
            <li key={i} className="conflict">
              <div className="conflict-line">
                <span className={`badge badge-${v.tone}`}>{v.label}</span>
                {c.value_a !== null && c.value_b !== null ? (
                  <span className="conflict-values">
                    {fmtNum(c.value_a)} <span className="vs">vs</span> {fmtNum(c.value_b)}
                    {c.metric ? ` ${c.metric}` : ""}
                  </span>
                ) : (
                  <span className="conflict-values muted">no comparable numbers</span>
                )}
                {c.differing.length > 0 && <span className="conflict-diff">differs in {c.differing.map(nice).join(", ")}</span>}
              </div>
              <p className="conflict-papers">
                <span title={titleOf(c.paper_a)}>
                  {shortTitle(titleOf(c.paper_a))} <span className="paper-id">{c.paper_a}</span>
                </span>
                <span className="vs"> vs </span>
                <span title={titleOf(c.paper_b)}>
                  {shortTitle(titleOf(c.paper_b))} <span className="paper-id">{c.paper_b}</span>
                </span>
              </p>
              {c.reason && <p className="conflict-reason">{c.reason}</p>}
            </li>
          );
        })}
      </ul>
    </Section>
  );
}

function Claims({ checks, regenerated, cite }: { checks: ClaimCheck[]; regenerated: boolean; cite: CiteProps }) {
  const [open, setOpen] = useState(false);
  const ok = checks.filter((c) => c.supported).length;
  const tone = ok === checks.length ? "ok" : ok === 0 ? "none" : "part";
  return (
    <Section
      icon={<Shield size={16} />}
      title="Claim check"
      open={open}
      onToggle={() => setOpen((v) => !v)}
      summary={
        <span className={`tone-${tone}`}>
          {ok} of {checks.length} statement{checks.length === 1 ? "" : "s"} supported{regenerated ? " · regenerated once" : ""}
        </span>
      }
    >
      <ul className="claims">
        {checks.map((c, i) => (
          <li key={i} className={c.supported ? "is-ok" : "is-bad"}>
            <span className="claim-mark" aria-label={c.supported ? "supported" : "not supported"}>
              {c.supported ? <Check size={12} /> : <Cross size={11} />}
            </span>
            <span className="claim-text">
              <Inline text={c.sentence.replace(BULLET, "")} cite={cite} />
            </span>
            <span className="claim-score" title="Entailment probability (NLI)">
              {c.entailment.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </Section>
  );
}

function Sources({
  sources,
  open,
  setOpen,
  active,
  setActive,
  flash,
  register,
}: {
  sources: SourceRef[];
  open: boolean;
  setOpen: (v: boolean) => void;
  active: number | null;
  setActive: (n: number | null) => void;
  flash: number | null;
  register: (n: number, el: HTMLElement | null) => void;
}) {
  return (
    <Section icon={<Book size={16} />} title="Sources" summary={`${sources.length}`} short open={open} onToggle={() => setOpen(!open)}>
      <ol className="src-list">
        {sources.map((s, i) => {
          const n = i + 1;
          const isOpen = active === n;
          return (
            <li key={s.chunk_id} ref={(el) => register(n, el)} className={`src${isOpen ? " is-open" : ""}${flash === n ? " is-flash" : ""}`}>
              <button
                type="button"
                className="src-chip"
                aria-expanded={isOpen}
                onClick={() => setActive(isOpen ? null : n)}
                title={`${s.paper_title || s.paper_id}, page ${s.page}`}
              >
                <span className="src-num">{n}</span>
                <span className="src-title">{shortTitle(s.paper_title || s.paper_id)}</span>
                <span className="src-page">p. {s.page}</span>
              </button>
              {isOpen && (
                <div className="src-card">
                  <p className="src-full">{s.paper_title || s.paper_id}</p>
                  <p className="src-meta">
                    arXiv {s.paper_id} · page {s.page} · {nice(s.section)} · reranker score {s.score.toFixed(2)}
                  </p>
                  <p className="src-preview">{s.preview}</p>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </Section>
  );
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

function ActionRow({ r, onRetry, canRetry }: { r: QueryResponse; onRetry: () => void; canRetry: boolean }) {
  const [copied, setCopied] = useState(false);
  const total = r.timings_ms?.total;
  return (
    <div className="actions">
      <button
        type="button"
        className="act-btn"
        aria-label={copied ? "Copied" : "Copy answer"}
        data-tip={copied ? "Copied" : "Copy"}
        onClick={async () => {
          if (await copyText(r.answer)) {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1600);
          }
        }}
      >
        {copied ? <Check size={15} /> : <Copy size={15} />}
      </button>
      <button type="button" className="act-btn" aria-label="Retry: ask this question again" data-tip="Retry" onClick={onRetry} disabled={!canRetry}>
        <Retry size={15} />
      </button>
      <span className="act-meta">
        {r.analysis && (
          <>
            {r.analysis.intent} · {r.analysis.complexity}
          </>
        )}
        {total !== undefined && <> · {(total / 1000).toFixed(1)} s</>}
      </span>
      <span className="sr-only" aria-live="polite">
        {copied ? "Answer copied" : ""}
      </span>
    </div>
  );
}

export function AssistantTurn({
  x,
  live,
  animate,
  canRetry,
  onRetry,
  onRevealed,
  titleOf,
}: {
  x: Exchange;
  live: boolean;
  animate: boolean;
  canRetry: boolean;
  onRetry: () => void;
  onRevealed: () => void;
  titleOf: (paperId: string) => string;
}) {
  const r = x.response;
  const { shown, done, finish } = useReveal(r?.answer ?? "", animate && !!r, x.run ? 420 : 0);

  // Once the answer is fully shown (and the evidence has faded in) it must not animate again, e.g. after switching chats.
  const revealedRef = useRef(onRevealed);
  revealedRef.current = onRevealed;
  useEffect(() => {
    if (!animate || !done || !r) return;
    const t = window.setTimeout(() => revealedRef.current(), 900);
    return () => window.clearTimeout(t);
  }, [animate, done, r]);
  const [sourcesOpen, setSourcesOpen] = useState(true);
  const [activeSrc, setActiveSrc] = useState<number | null>(null);
  const [flash, setFlash] = useState<number | null>(null);
  const srcEls = useRef(new Map<number, HTMLElement>());

  const register = useCallback((n: number, el: HTMLElement | null) => {
    if (el) srcEls.current.set(n, el);
    else srcEls.current.delete(n);
  }, []);

  const onCite = useCallback(
    (n: number) => {
      finish();
      setSourcesOpen(true);
      setActiveSrc(n);
      setFlash(n);
      window.setTimeout(() => setFlash(null), 1400);
      window.setTimeout(() => srcEls.current.get(n)?.scrollIntoView({ block: "center", behavior: "smooth" }), 60);
    },
    [finish],
  );

  const sourceTitle = (n: number) => {
    const s = r?.sources[n - 1];
    return s ? `${s.paper_title || s.paper_id}, page ${s.page}` : undefined;
  };
  const cite: CiteProps = { maxCite: r?.sources.length ?? 0, onCite, citeTitle: sourceTitle };
  const titleFromSources = (pid: string) => r?.sources.find((s) => s.paper_id === pid)?.paper_title || titleOf(pid);
  const weakWithdrawn = !!r?.trace.some((t) => t.includes("weak-evidence flag withdrawn"));
  const weakShown = !!r?.retrieval_weak && !weakWithdrawn;

  return (
    <div className={`turn-assistant${live ? " is-live" : ""}`}>
      <div className="assistant-gutter" aria-hidden="true">
        <Spark size={20} animated={live} />
      </div>
      <div className="assistant-body">
        {x.run && <Thinking key={x.run.startedAt} run={x.run} live={live} response={r} />}
        <div className="answer-anchor" aria-hidden="true" />
        {r?.applicability?.warning && (
          <div className="callout callout-warn" role="note">
            <Warning size={17} className="callout-icon" />
            <div>
              <p className="callout-title">Outside what the papers cover</p>
              <p>{r.applicability.warning}</p>
            </div>
          </div>
        )}
        {weakShown && (
          <div className="callout callout-warn" role="note">
            <Warning size={17} className="callout-icon" />
            <p>No retrieved passage is clearly relevant to the question, so the sources below may not answer it.</p>
          </div>
        )}
        {r && <Markdown text={shown} className={`answer${done ? "" : " is-typing"}`} {...cite} />}
        {r && done && (
          <div className={`evidence${animate ? " fade-in" : ""}`}>
            {r.applicability && <Coverage a={r.applicability} weakWithdrawn={weakWithdrawn} retrievalWeak={!!r.retrieval_weak} />}
            {r.contradictions.length > 0 && <Conflicts items={r.contradictions} titleOf={titleFromSources} />}
            {r.claim_checks.length > 0 && <Claims checks={r.claim_checks} regenerated={r.regenerated} cite={cite} />}
            {r.sources.length > 0 && (
              <Sources
                sources={r.sources}
                open={sourcesOpen}
                setOpen={setSourcesOpen}
                active={activeSrc}
                setActive={setActiveSrc}
                flash={flash}
                register={register}
              />
            )}
          </div>
        )}
        {x.error && (
          <div className="callout callout-error" role="alert">
            <Warning size={17} className="callout-icon" />
            <div>
              <p className="callout-title">The question could not be answered</p>
              <p className="callout-detail">{x.error}</p>
              <button type="button" className="text-btn" onClick={onRetry} disabled={!canRetry}>
                <Retry size={14} /> Try again
              </button>
            </div>
          </div>
        )}
        {x.stopped && !x.error && (
          <p className="stopped-line">
            You stopped this answer.{" "}
            <button type="button" className="text-btn" onClick={onRetry} disabled={!canRetry}>
              <Retry size={14} /> Ask again
            </button>
          </p>
        )}
        {r && done && <ActionRow r={r} onRetry={onRetry} canRetry={canRetry} />}
      </div>
    </div>
  );
}
