import { useState } from "react";
import type { ApplicabilityResult, ClaimCheck, ContradictionPair, QueryResponse, SourceRef, Verdict } from "../types";

const VERDICT_LABEL: Record<Verdict, string> = {
  GENUINE: "Genuine conflict",
  EXPLAINED: "Explained difference",
  NOT_COMPARABLE: "Not comparable",
};

/** One line of the answer: **bold** text, and [1] / [2, 3] markers as links to the source list. */
function Inline({ text, id }: { text: string; id: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[\d+(?:\s*,\s*\d+)*\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const bold = part.match(/^\*\*([^*]+)\*\*$/);
        if (bold) return <strong key={i}>{bold[1]}</strong>;
        const m = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
        if (!m) return <span key={i}>{part}</span>;
        return (
          <sup key={i} className="cite">
            {m[1].split(/\s*,\s*/).map((n) => (
              <a key={n} href={`#${id}-src-${n}`}>
                {n}
              </a>
            ))}
          </sup>
        );
      })}
    </>
  );
}

const BULLET = /^\s*[*-]\s+/;

/** The answer as paragraphs and bullet lists (the model writes light markdown; no library needed for that). */
function AnswerText({ text, id }: { text: string; id: string }) {
  const blocks: { bullets: boolean; lines: string[] }[] = [];
  for (const raw of text.split("\n")) {
    if (!raw.trim()) {
      blocks.push({ bullets: false, lines: [] });                       // a blank line ends the current block
      continue;
    }
    const bullet = BULLET.test(raw);
    const last = blocks[blocks.length - 1];
    if (last && last.lines.length > 0 && last.bullets === bullet) last.lines.push(raw.replace(BULLET, ""));
    else blocks.push({ bullets: bullet, lines: [raw.replace(BULLET, "")] });
  }
  return (
    <div className="answer-text">
      {blocks.filter((b) => b.lines.length > 0).map((b, i) =>
        b.bullets ? (
          <ul key={i} className="answer-list">
            {b.lines.map((l, j) => (
              <li key={j}>
                <Inline text={l} id={id} />
              </li>
            ))}
          </ul>
        ) : (
          <p key={i}>
            {b.lines.map((l, j) => (
              <span key={j}>
                {j > 0 && <br />}
                <Inline text={l} id={id} />
              </span>
            ))}
          </p>
        ),
      )}
    </div>
  );
}

function CoverageBar({ a }: { a: ApplicabilityResult }) {
  const pct = Math.round(a.coverage * 100);
  const tone = a.coverage >= 1 ? "ok" : a.coverage > 0 ? "part" : "none";
  if (a.checks.length === 0) {
    return <p className="muted small">The question names no conditions to check.</p>;
  }
  const covered = a.checks.filter((c) => c.covered).length;
  return (
    <section className="panel" aria-label="Applicability">
      <div className="panel-head">
        <h3>Evidence coverage</h3>
        <span className={`tone tone-${tone}`}>
          {covered} of {a.checks.length} conditions{a.re_retrieved ? " · searched again" : ""}
        </span>
      </div>
      <div className="bar" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={`bar-fill tone-${tone}`} style={{ width: `${Math.max(pct, 3)}%` }} />
      </div>
      <ul className="conditions">
        {a.checks.map((c) => (
          <li key={`${c.condition}-${c.requested}`} className={c.covered ? "covered" : "missing"}>
            <span className="mark" aria-hidden>{c.covered ? "✓" : "✗"}</span>
            <span className="cond-name">{c.condition.replace("_", " ")}</span>
            <strong>{c.requested}</strong>
            <span className="muted small">
              {c.covered
                ? `found in ${c.chunk_ids.length} passage${c.chunk_ids.length === 1 ? "" : "s"}`
                : c.observed.length
                  ? `sources record: ${c.observed.join(", ")}`
                  : "no source states this"}
            </span>
          </li>
        ))}
      </ul>
      {a.joint_covered === true && (
        <p className="muted small">✓ One result records these conditions together.</p>
      )}
      {a.joint_covered === false && (
        <p className="muted small">✗ No single result records these conditions together: each appears in the sources, but in different combinations.</p>
      )}
      {a.profile_guided && <p className="muted small">Passages were added through the condition profiles to look for the missing condition.</p>}
      {a.escalated && <p className="muted small">The larger model gave a second opinion before this warning was shown.</p>}
      {a.reasoning && <p className="muted small">Agent notes: {a.reasoning}</p>}
      {a.warning && <p className="warning" role="alert">{a.warning}</p>}
    </section>
  );
}

function Conflicts({ items }: { items: ContradictionPair[] }) {
  if (items.length === 0) return null;
  return (
    <section className="panel" aria-label="Conflicts between papers">
      <div className="panel-head">
        <h3>Conflicts between papers</h3>
      </div>
      <ul className="conflicts">
        {items.map((c, i) => (
          <li key={i}>
            <span className={`badge badge-${c.verdict}`}>{VERDICT_LABEL[c.verdict]}</span>
            {c.value_a !== null && c.value_b !== null && (
              <span className="values">
                {c.value_a} vs {c.value_b} {c.metric ?? ""}
              </span>
            )}
            <span className="muted small"> {c.paper_a} vs {c.paper_b}</span>
            <p className="small">{c.reason}</p>
            {c.differing.length > 0 && (
              <p className="small">
                Differs in: {c.differing.map((d) => <code key={d}>{d.replace("_", " ")}</code>)}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function Claims({ checks, regenerated }: { checks: ClaimCheck[]; regenerated: boolean }) {
  if (checks.length === 0) return null;
  const bad = checks.filter((c) => !c.supported).length;
  return (
    <details className="panel">
      <summary>
        Claim check: {checks.length - bad} of {checks.length} sentences supported{regenerated ? " · regenerated once" : ""}
      </summary>
      <ul className="claims">
        {checks.map((c, i) => (
          <li key={i} className={c.supported ? "" : "unsupported"}>
            <span className="mark" aria-hidden>{c.supported ? "✓" : "!"}</span> {c.sentence.replace(BULLET, "").replace(/\*\*/g, "")}
            <span className="muted small"> ({c.entailment.toFixed(2)})</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function Sources({ sources, id }: { sources: SourceRef[]; id: string }) {
  if (sources.length === 0) return null;
  return (
    <details className="panel" open>
      <summary>Sources ({sources.length})</summary>
      <ol className="sources">
        {sources.map((s, i) => (
          <li key={s.chunk_id} id={`${id}-src-${i + 1}`}>
            <strong>{s.paper_title || s.paper_id}</strong>
            <span className="muted small"> · page {s.page} · {s.section.replace("_", " ")}</span>
            <p className="small preview">{s.preview}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}

export function AnswerCard({ r, id }: { r: QueryResponse; id: string }) {
  const [showTrace, setShowTrace] = useState(false);
  const total = r.timings_ms.total;
  return (
    <article className="card answer">
      <AnswerText text={r.answer} id={id} />
      {r.applicability && <CoverageBar a={r.applicability} />}
      <Conflicts items={r.contradictions} />
      <Claims checks={r.claim_checks} regenerated={r.regenerated} />
      <Sources sources={r.sources} id={id} />
      <div className="foot">
        {r.analysis && (
          <span className="muted small">
            {r.analysis.intent} · {r.analysis.complexity}
          </span>
        )}
        {total !== undefined && <span className="muted small">{(total / 1000).toFixed(1)} s</span>}
        <button type="button" className="link" onClick={() => setShowTrace((v) => !v)}>
          {showTrace ? "Hide" : "Show"} pipeline trace
        </button>
      </div>
      {showTrace && (
        <pre className="trace">
          {r.trace.join("\n")}
          {"\n\n"}
          {Object.entries(r.timings_ms).map(([k, v]) => `${k.padEnd(18)} ${(v / 1000).toFixed(2)} s`).join("\n")}
        </pre>
      )}
    </article>
  );
}
