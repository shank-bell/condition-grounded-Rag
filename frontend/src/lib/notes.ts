
type Cursor = { s: string; i: number };

function ws(c: Cursor): void {
  while (c.i < c.s.length && /\s/.test(c.s[c.i])) c.i++;
}

function readString(c: Cursor): string | null {
  const q = c.s[c.i];
  if (q !== "'" && q !== '"') return null;
  let j = c.i + 1;
  let out = "";
  while (j < c.s.length && c.s[j] !== q) {
    if (c.s[j] === "\\" && j + 1 < c.s.length) {
      out += c.s[j + 1];
      j += 2;
    } else {
      out += c.s[j++];
    }
  }
  if (j >= c.s.length) return null;
  c.i = j + 1;
  return out;
}

function readList(c: Cursor): string[] | null {
  if (c.s[c.i] !== "[") return null;
  c.i++;
  const items: string[] = [];
  ws(c);
  while (c.i < c.s.length && c.s[c.i] !== "]") {
    const item = readString(c);
    if (item === null) return null;
    items.push(item);
    ws(c);
    if (c.s[c.i] === ",") {
      c.i++;
      ws(c);
    } else if (c.s[c.i] !== "]") return null;
  }
  if (c.s[c.i] !== "]") return null;
  c.i++;
  return items;
}

function parseDictBody(body: string): Array<[string, string]> | null {
  const c: Cursor = { s: body, i: 0 };
  const out: Array<[string, string]> = [];
  ws(c);
  while (c.i < body.length) {
    const key = readString(c);
    if (key === null) return null;
    ws(c);
    if (body[c.i] !== ":") return null;
    c.i++;
    ws(c);
    let value: string | null;
    if (body[c.i] === "'" || body[c.i] === '"') value = readString(c);
    else if (body[c.i] === "[") {
      const items = readList(c);
      value = items === null ? null : items.join(", ");
      if (items === null) return null;
    } else {
      const m = /^(None|True|False|-?\d+(?:\.\d+)?)/.exec(body.slice(c.i));
      if (!m) return null;
      value = m[1] === "None" ? null : m[1];
      c.i += m[1].length;
    }
    if (value !== null && value !== "") out.push([key, value]);
    ws(c);
    if (c.i < body.length) {
      if (body[c.i] !== ",") return null;
      c.i++;
      ws(c);
    }
  }
  return out;
}

const label = (k: string) => k.replace(/_/g, " ");

function condition(item: string): string {
  const eq = item.indexOf("=");
  return eq < 0 ? item : `${label(item.slice(0, eq).trim())} = ${item.slice(eq + 1).trim()}`;
}

export function prettyNote(line: string): string {
  let s = line;

  s = s.replace(/\bconditions=\{([^{}]*)\}/, (whole, body: string) => {
    const pairs = parseDictBody(body);
    if (!pairs) return whole;
    return pairs.length ? `conditions: ${pairs.map(([k, v]) => `${label(k)} = ${v}`).join(", ")}` : "conditions: none named";
  });

  s = s.replace(/^queries: (\[.*\])$/, (whole, list: string) => {
    const c: Cursor = { s: list, i: 0 };
    const items = readList(c);
    if (!items || c.i !== list.length || items.length === 0) return whole;
    return items.length === 1 ? `search query: ${items[0]}` : items.map((q, k) => `search query ${k + 1}: ${q}`).join(" | ");
  });

  s = s.replace(/\bmissing (\[[^\]]*\])/, (whole, list: string) => {
    const c: Cursor = { s: list, i: 0 };
    const items = readList(c);
    if (!items || c.i !== list.length) return whole;
    return items.length ? `missing ${items.map(condition).join(", ")}` : "missing none";
  });

  return s;
}
