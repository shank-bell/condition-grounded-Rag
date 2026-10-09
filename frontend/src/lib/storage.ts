// Chats live in the browser (localStorage). Every access is wrapped: in a private window, with blocked site data or a full
// quota the page keeps working, it just does not remember.
import type { Chat, Exchange } from "../types";

const CHATS_KEY = "cgrag.chats.v1";
const MAX_CHATS = 40;

function store(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readString(key: string): string | null {
  try {
    return store()?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

export function writeString(key: string, value: string | null): boolean {
  try {
    const s = store();
    if (!s) return false;
    if (value === null) s.removeItem(key);
    else s.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

/** An exchange that was still running when the page closed cannot resume: show it as interrupted. */
function settle(x: Exchange): Exchange {
  if (x.response || x.error || x.stopped) return x;
  return {
    ...x,
    error: "Interrupted: the page was closed or reloaded before the answer arrived.",
    run: x.run ? { ...x.run, outcome: "stopped", queued: false, endedAt: x.run.endedAt ?? x.run.startedAt } : undefined,
  };
}

export function loadChats(): Chat[] {
  const raw = readString(CHATS_KEY);
  if (!raw) return [];
  try {
    const data = JSON.parse(raw) as Chat[];
    if (!Array.isArray(data)) return [];
    return data
      .filter((c) => c && typeof c.id === "string" && Array.isArray(c.exchanges))
      .map((c) => ({ ...c, exchanges: c.exchanges.map(settle) }));
  } catch {
    return [];
  }
}

/** Saves the newest chats; when the quota is full, older chats are dropped until it fits. */
export function saveChats(chats: Chat[]): void {
  if (!store()) return;
  let keep = [...chats].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CHATS);
  while (keep.length > 0) {
    if (writeString(CHATS_KEY, JSON.stringify(keep))) return;
    keep = keep.slice(0, Math.floor(keep.length * 0.7));
  }
  writeString(CHATS_KEY, null);
}

export function loadFlag(key: string, fallback = false): boolean {
  const v = readString(`cgrag.${key}`);
  return v === null ? fallback : v === "1";
}

export function saveFlag(key: string, value: boolean): void {
  writeString(`cgrag.${key}`, value ? "1" : "0");
}
