import type { Health, Paper, QueryResponse, StreamEvent, Turn, UploadJob } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/** Dev switch: `?mock=1` replays canned answers through a fake event stream (no GPU needed). Other values:
 *  `plain` (no stream: the page falls back to the plain endpoint and replays the steps), `error`, `warming`, `down`.
 *  Extra parameters: `speed=3` (faster), `queued=1` (first wait behind another question). */
export const MOCK: string | null = (() => {
  try {
    return new URLSearchParams(window.location.search).get("mock");
  } catch {
    return null;
  }
})();

const mock = () => import("./mock");

/** The stream endpoint is missing or not a stream: use the plain endpoint instead. */
export class StreamUnavailable extends Error {}

/** The error text of a failed response: FastAPI's {"detail": "..."} when there is one. */
async function failure(res: Response): Promise<Error> {
  const text = await res.text().catch(() => "");
  let detail = text;
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    if (typeof j.detail === "string") detail = j.detail;
    else if (j.detail !== undefined) detail = JSON.stringify(j.detail);
  } catch {
    /* not JSON */
  }
  return new Error(`${res.status} ${res.statusText}${detail ? `: ${detail.slice(0, 240)}` : ""}`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

/** Splits a text/event-stream body into events (`event: <type>` + `data: <json>`, separated by a blank line). */
export async function readEventStream(body: ReadableStream<Uint8Array>, onEvent: (e: StreamEvent) => void): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const flush = (block: string) => {
    let type = "";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;                 // comment / keep-alive ping
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") type = value;
      else if (field === "data") data.push(value);
    }
    if (data.length === 0) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(data.join("\n"));
    } catch {
      return;                                                       // not JSON: ignore rather than break the answer
    }
    const evt = parsed as StreamEvent;
    if (evt && typeof evt === "object" && !("type" in evt) && type) (evt as { type: string }).type = type;
    onEvent(evt);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    buf = buf.replace(/\r\n?/g, "\n");
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      flush(buf.slice(0, i));
      buf = buf.slice(i + 2);
    }
  }
  buf += decoder.decode();
  if (buf.trim()) flush(buf.replace(/\r\n?/g, "\n"));
}

async function queryStream(question: string, history: Turn[], onEvent: (e: StreamEvent) => void, signal: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch(BASE + "/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question, history }),
      signal,
    });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new StreamUnavailable(String(err));
  }
  if (res.status === 404 || res.status === 405) throw new StreamUnavailable(`${res.status} ${res.statusText}`);
  if (!res.ok) throw await failure(res);
  if (!res.body || !(res.headers.get("content-type") || "").includes("text/event-stream")) {
    throw new StreamUnavailable("the server did not answer with an event stream");
  }
  await readEventStream(res.body, onEvent);
}

export const api = {
  health: async (): Promise<Health> => (MOCK ? (await mock()).mockHealth(() => request<Health>("/health")) : request<Health>("/health")),
  papers: async (): Promise<Paper[]> => (MOCK ? (await mock()).mockPapers(() => request<Paper[]>("/papers")) : request<Paper[]>("/papers")),
  /** POST /query: the whole answer at once (used when the stream is unavailable). */
  query: async (question: string, history: Turn[], signal?: AbortSignal): Promise<QueryResponse> =>
    MOCK
      ? (await mock()).mockQuery(question, signal)
      : request<QueryResponse>("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, history }),
          signal,
        }),
  /** POST /query/stream: the live steps, then the answer. Throws StreamUnavailable when the server has no stream. */
  queryStream: async (question: string, history: Turn[], onEvent: (e: StreamEvent) => void, signal: AbortSignal): Promise<void> =>
    MOCK ? (await mock()).mockStream(question, onEvent, signal) : queryStream(question, history, onEvent, signal),
  upload: async (file: File): Promise<{ job_id: string; paper_id: string }> => {
    if (MOCK) return (await mock()).mockUpload(file);
    const form = new FormData();
    form.append("file", file);
    return request<{ job_id: string; paper_id: string }>("/upload", { method: "POST", body: form });
  },
  job: async (id: string): Promise<UploadJob> => (MOCK ? (await mock()).mockJob(id) : request<UploadJob>(`/jobs/${id}`)),
};
