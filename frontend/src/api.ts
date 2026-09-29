import type { Health, Paper, QueryResponse, Turn } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail.slice(0, 200)}` : ""}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  papers: () => request<Paper[]>("/papers"),
  query: (question: string, history: Turn[]) =>
    request<QueryResponse>("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ job_id: string; paper_id: string }>("/upload", { method: "POST", body: form });
  },
  job: (id: string) => request<{ status: string; error?: string; chunks?: number; profiles?: number }>(`/jobs/${id}`),
};
