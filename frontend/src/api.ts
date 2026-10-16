import { readString, writeString } from "./lib/storage";
import type { Health, Paper, QueryResponse, StreamEvent, Turn, UploadJob } from "./types";

const ADDRESS_KEY = "cgrag.api";
const CODE_KEY = "cgrag.code";
const BUILD_ADDRESS: string | undefined = import.meta.env.VITE_API_URL;

const isLocalHost = (host: string) => host === "localhost" || host === "127.0.0.1";

export const hostedElsewhere = (): boolean => !isLocalHost(window.location.hostname);

export function cleanAddress(raw: string): string | null {
  try {
    const u = new URL(raw.trim());
    if (u.protocol !== "https:" && !(u.protocol === "http:" && isLocalHost(u.hostname))) return null;
    return u.origin;
  } catch {
    return null;
  }
}

export const apiAddress = (): string =>
  hostedElsewhere() ? (readString(ADDRESS_KEY) ?? BUILD_ADDRESS ?? "") : (BUILD_ADDRESS ?? "http://localhost:8000");

export const needsAddress = (): boolean => hostedElsewhere() && apiAddress() === "";

export function readCode(): string {
  try {
    return window.sessionStorage.getItem(CODE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveConnection(address: string | null, code: string): void {
  if (address !== null) writeString(ADDRESS_KEY, address);
  try {
    if (code) window.sessionStorage.setItem(CODE_KEY, code);
    else window.sessionStorage.removeItem(CODE_KEY);
  } catch {
  }
}

export function adoptAddressFromUrl(): void {
  try {
    const url = new URL(window.location.href);
    const given = url.searchParams.get("api");
    if (given === null) return;
    const clean = cleanAddress(given);
    if (clean && hostedElsewhere()) writeString(ADDRESS_KEY, clean);
    url.searchParams.delete("api");
    window.history.replaceState(null, "", url.toString());
  } catch {
  }
}

export class Unauthorized extends Error {
  constructor() {
    super("The access code is missing or wrong.");
  }
}

function withCode(init?: RequestInit): RequestInit {
  const code = readCode();
  if (!code) return init ?? {};
  const headers = new Headers(init?.headers);
  headers.set("X-Access-Code", code);
  return { ...init, headers };
}

export const MOCK: string | null = (() => {
  try {
    return new URLSearchParams(window.location.search).get("mock");
  } catch {
    return null;
  }
})();

const mock = () => import("./mock");

export class StreamUnavailable extends Error {}

async function failure(res: Response): Promise<Error> {
  const text = await res.text().catch(() => "");
  let detail = text;
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    if (typeof j.detail === "string") detail = j.detail;
    else if (j.detail !== undefined) detail = JSON.stringify(j.detail);
  } catch {
  }
  return new Error(`${res.status} ${res.statusText}${detail ? `: ${detail.slice(0, 240)}` : ""}`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiAddress() + path, withCode(init));
  if (res.status === 401) throw new Unauthorized();
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

export async function readEventStream(body: ReadableStream<Uint8Array>, onEvent: (e: StreamEvent) => void): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const flush = (block: string) => {
    let type = "";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;
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
      return;
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
    res = await fetch(
      apiAddress() + "/query/stream",
      withCode({
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ question, history }),
        signal,
      }),
    );
  } catch (err) {
    if (signal.aborted) throw err;
    throw new StreamUnavailable(String(err));
  }
  if (res.status === 401) throw new Unauthorized();
  if (res.status === 404 || res.status === 405) throw new StreamUnavailable(`${res.status} ${res.statusText}`);
  if (!res.ok) throw await failure(res);
  if (!res.body || !(res.headers.get("content-type") || "").includes("text/event-stream")) {
    throw new StreamUnavailable("the server did not answer with an event stream");
  }
  await readEventStream(res.body, onEvent);
}

export const api = {
  health: async (): Promise<Health> => (MOCK ? (await mock()).mockHealth(() => request<Health>("/health")) : request<Health>("/health")),
  check: async (): Promise<{ ok: boolean }> => (MOCK ? { ok: true } : request<{ ok: boolean }>("/auth/check")),
  papers: async (): Promise<Paper[]> => (MOCK ? (await mock()).mockPapers(() => request<Paper[]>("/papers")) : request<Paper[]>("/papers")),
  query: async (question: string, history: Turn[], signal?: AbortSignal): Promise<QueryResponse> =>
    MOCK
      ? (await mock()).mockQuery(question, signal)
      : request<QueryResponse>("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, history }),
          signal,
        }),
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
