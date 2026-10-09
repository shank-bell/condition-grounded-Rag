// A small markdown renderer for the answers (no library): paragraphs, headings, bullet / numbered lists (nested), tables,
// block quotes, fenced code, inline code, bold, italic, links, and citation markers [1] / [1, 2] / [1-3] as pills.
import { Fragment, memo, useMemo, type ReactNode } from "react";

export interface CiteProps {
  /** Number of sources: markers above it are left as plain text. */
  maxCite?: number;
  onCite?: (n: number) => void;
  citeTitle?: (n: number) => string | undefined;
}

type Align = "left" | "right" | "center" | null;
interface ListItem {
  text: string;
  children: Block[];
}
type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; level: number; text: string }
  | { kind: "ul" | "ol"; start: number; items: ListItem[] }
  | { kind: "code"; lang: string; text: string }
  | { kind: "table"; head: string[]; align: Align[]; rows: string[][] }
  | { kind: "quote"; blocks: Block[] }
  | { kind: "hr" };

const LIST_RE = /^(\s*)([-*+•]|\d{1,3}[.)])\s+(.*)$/;
const FENCE_RE = /^\s{0,3}(```+|~~~+)\s*([\w+-]*)\s*$/;
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR_RE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const TABLE_SEP_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
const QUOTE_RE = /^\s{0,3}>\s?(.*)$/;

function splitRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

function startsBlock(line: string, next: string | undefined): boolean {
  return (
    FENCE_RE.test(line) || HEADING_RE.test(line) || HR_RE.test(line) || QUOTE_RE.test(line) || LIST_RE.test(line) ||
    (line.includes("|") && next !== undefined && TABLE_SEP_RE.test(next))
  );
}

function parseList(lines: string[], start: number): [Block, number] {
  const first = lines[start].match(LIST_RE)!;
  const base = first[1].length;
  const ordered = /\d/.test(first[2]);
  const items: ListItem[] = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j++;
      const m = j < lines.length ? lines[j].match(LIST_RE) : null;
      if (m && m[1].length >= base && (m[1].length >= base + 2 || /\d/.test(m[2]) === ordered)) {
        i = j;
        continue;
      }
      // an indented paragraph after a blank line still belongs to the last item
      if (j < lines.length && items.length && /^\s{2,}\S/.test(lines[j]) && !LIST_RE.test(lines[j])) {
        items[items.length - 1].text += "\n" + lines[j].trim();
        i = j + 1;
        continue;
      }
      break;
    }
    const m = line.match(LIST_RE);
    if (m) {
      const indent = m[1].length;
      if (indent < base) break;
      if (indent >= base + 2 && items.length) {
        const [child, next] = parseList(lines, i);
        items[items.length - 1].children.push(child);
        i = next;
        continue;
      }
      if (/\d/.test(m[2]) !== ordered) break;
      items.push({ text: m[3], children: [] });
      i++;
      continue;
    }
    if (items.length && /^\s+\S/.test(line)) {
      items[items.length - 1].text += "\n" + line.trim();          // indented continuation line
      i++;
      continue;
    }
    break;
  }
  const n = ordered ? parseInt(first[2], 10) : 1;
  return [{ kind: ordered ? "ol" : "ul", start: Number.isFinite(n) ? n : 1, items }, i];
}

export function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const fence = line.match(FENCE_RE);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++]);
      i++;                                                            // closing fence (or end of text)
      out.push({ kind: "code", lang: fence[2] || "", text: body.join("\n") });
      continue;
    }
    const h = line.match(HEADING_RE);
    if (h) {
      out.push({ kind: "h", level: h[1].length, text: h[2] });
      i++;
      continue;
    }
    if (HR_RE.test(line)) {
      out.push({ kind: "hr" });
      i++;
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1])) {
      const head = splitRow(line);
      const align: Align[] = splitRow(lines[i + 1]).map((c) =>
        c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : null,
      );
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) rows.push(splitRow(lines[i++]));
      out.push({ kind: "table", head, align, rows });
      continue;
    }
    if (QUOTE_RE.test(line)) {
      const inner: string[] = [];
      while (i < lines.length && lines[i].trim() && QUOTE_RE.test(lines[i])) inner.push(lines[i++].match(QUOTE_RE)![1]);
      out.push({ kind: "quote", blocks: parseBlocks(inner.join("\n")) });
      continue;
    }
    if (LIST_RE.test(line)) {
      const [list, next] = parseList(lines, i);
      out.push(list);
      i = next;
      continue;
    }
    const para: string[] = [line.trim()];
    i++;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i], lines[i + 1])) para.push(lines[i++].trim());
    out.push({ kind: "p", text: para.join("\n") });
  }
  return out;
}

// ---- inline ----

const INLINE = new RegExp(
  [
    "(`+)([^`][\\s\\S]*?)\\1(?!`)",                                   // 1,2 code
    "\\*\\*(?=\\S)([\\s\\S]*?\\S)\\*\\*",                             // 3 bold
    "__(?=\\S)([\\s\\S]*?\\S)__",                                     // 4 bold
    "\\[(\\d{1,3}(?:\\s*[,;\\u2013-]\\s*\\d{1,3})*)\\]",              // 5 citation
    "\\[([^\\]\\n]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)",                 // 6,7 link
    "(?<![\\w*])\\*(?=[^\\s*])([^*\\n]*?[^\\s*])\\*(?![\\w*])",        // 8 italic
    "(?<![\\w_])_(?=[^\\s_])([^_\\n]*?[^\\s_])_(?![\\w_])",            // 9 italic
    "~~(?=\\S)([^~\\n]*?\\S)~~",                                      // 10 strike
  ].join("|"),
  "g",
);

function citeNumbers(spec: string): number[] {
  const out: number[] = [];
  for (const part of spec.split(/\s*[,;]\s*/)) {
    const range = part.match(/^(\d+)\s*[–-]\s*(\d+)$/);
    if (range) {
      const a = Number(range[1]);
      const b = Number(range[2]);
      if (b >= a && b - a < 20) for (let k = a; k <= b; k++) out.push(k);
      else out.push(a, b);
    } else if (part) out.push(Number(part));
  }
  return out;
}

function withBreaks(text: string, key: string): ReactNode {
  const parts = text.split("\n");
  if (parts.length === 1) return text;
  return parts.map((p, i) => (
    <Fragment key={`${key}-${i}`}>
      {i > 0 && <br />}
      {p}
    </Fragment>
  ));
}

export function Inline({ text, cite }: { text: string; cite: CiteProps }): ReactNode {
  const nodes: ReactNode[] = [];
  let last = 0;
  let k = 0;
  const re = new RegExp(INLINE.source, "g");                         // own instance: Inline recurses into bold / italic
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) nodes.push(withBreaks(text.slice(last, m.index), `t${k++}`));
    const key = `i${k++}`;
    if (m[2] !== undefined) nodes.push(<code key={key}>{m[2]}</code>);
    else if (m[3] !== undefined || m[4] !== undefined) nodes.push(<strong key={key}><Inline text={(m[3] ?? m[4])!} cite={cite} /></strong>);
    else if (m[5] !== undefined) {
      const nums = citeNumbers(m[5]);
      const max = cite.maxCite ?? 0;
      if (nums.length && nums.every((n) => n >= 1 && n <= max)) {
        nodes.push(
          <span key={key} className="cites">
            {nums.map((n, j) => (
              <button
                key={`${n}-${j}`}
                type="button"
                className="cite"
                title={cite.citeTitle?.(n) ?? `Source ${n}`}
                aria-label={`Source ${n}${cite.citeTitle?.(n) ? `: ${cite.citeTitle(n)}` : ""}`}
                onClick={() => cite.onCite?.(n)}
              >
                {n}
              </button>
            ))}
          </span>,
        );
      } else nodes.push(m[0]);
    } else if (m[6] !== undefined) {
      nodes.push(
        <a key={key} href={m[7]} target="_blank" rel="noreferrer noopener">
          {m[6]}
        </a>,
      );
    } else if (m[8] !== undefined || m[9] !== undefined) nodes.push(<em key={key}><Inline text={(m[8] ?? m[9])!} cite={cite} /></em>);
    else if (m[10] !== undefined) nodes.push(<del key={key}>{m[10]}</del>);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(withBreaks(text.slice(last), `t${k++}`));
  return <>{nodes}</>;
}

function renderBlocks(blocks: Block[], cite: CiteProps, prefix: string): ReactNode[] {
  return blocks.map((b, i) => {
    const key = `${prefix}${i}`;
    switch (b.kind) {
      case "p":
        return (
          <p key={key}>
            <Inline text={b.text} cite={cite} />
          </p>
        );
      case "h": {
        const level = Math.min(6, Math.max(3, b.level + 2));
        const Tag = `h${level}` as "h3" | "h4" | "h5" | "h6";
        return (
          <Tag key={key} className="md-h">
            <Inline text={b.text} cite={cite} />
          </Tag>
        );
      }
      case "ul":
      case "ol": {
        const items = b.items.map((it, j) => (
          <li key={`${key}-${j}`}>
            <Inline text={it.text} cite={cite} />
            {it.children.length > 0 && renderBlocks(it.children, cite, `${key}-${j}-`)}
          </li>
        ));
        return b.kind === "ol" ? (
          <ol key={key} start={b.start !== 1 ? b.start : undefined}>
            {items}
          </ol>
        ) : (
          <ul key={key}>{items}</ul>
        );
      }
      case "code":
        return (
          <pre key={key} className="md-code" data-lang={b.lang || undefined}>
            <code>{b.text}</code>
          </pre>
        );
      case "table":
        return (
          <div key={key} className="md-table-wrap" role="region" aria-label="Table" tabIndex={0}>
            <table className="md-table">
              <thead>
                <tr>
                  {b.head.map((c, j) => (
                    <th key={j} style={b.align[j] ? { textAlign: b.align[j]! } : undefined}>
                      <Inline text={c} cite={cite} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {b.rows.map((row, r) => (
                  <tr key={r}>
                    {b.head.map((_, j) => (
                      <td key={j} style={b.align[j] ? { textAlign: b.align[j]! } : undefined}>
                        <Inline text={row[j] ?? ""} cite={cite} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      case "quote":
        return <blockquote key={key}>{renderBlocks(b.blocks, cite, `${key}-`)}</blockquote>;
      case "hr":
        return <hr key={key} />;
      default:
        return null;
    }
  });
}

/** While the answer is being revealed: close an open **bold** / `code`, drop a half-typed citation marker. */
export function tidyPartial(text: string): string {
  let t = text.replace(/\[[\d,\s–-]*$/, "");                    // half-typed citation marker
  t = t.replace(/\*+$/, "");                                         // half-typed bold marker
  if ((t.match(/\*\*/g) || []).length % 2 === 1) t += "**";
  if ((t.match(/`/g) || []).length % 2 === 1) t += "`";
  return t;
}

export const Markdown = memo(function Markdown({ text, className, ...cite }: { text: string; className?: string } & CiteProps) {
  const blocks = useMemo(() => parseBlocks(text), [text]);
  return <div className={`md${className ? ` ${className}` : ""}`}>{renderBlocks(blocks, cite, "b")}</div>;
});
