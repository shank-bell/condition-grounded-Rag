// Left sidebar: app name + collapse toggle, New chat, Recents (saved chats), and the Library (indexed papers, upload).
import { useState } from "react";
import type { Chat, Health, Paper } from "../types";
import { Book, Chat as ChatIcon, ChevronDown, FileText, Library, PanelLeft, Plus, Spark, Trash, Upload } from "./Icons";

export interface UploadState {
  text: string;
  state: "busy" | "done" | "error";
}

interface Props {
  collapsed: boolean;                 // desktop: a slim rail
  mobile: boolean;                    // <= 900 px: an off-canvas drawer
  drawerOpen: boolean;
  onToggle: () => void;               // collapse / expand (desktop) or close (mobile)
  chats: Chat[];
  activeId: string | null;
  runningChatId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  health: Health | null;
  papers: Paper[];
  upload: UploadState | null;
  onUpload: () => void;
}

const fmt = (n: number | undefined) => (n === undefined ? "–" : n.toLocaleString("en-US"));

export function Sidebar(p: Props) {
  const [papersOpen, setPapersOpen] = useState(false);
  const rail = p.collapsed && !p.mobile;
  const hidden = p.mobile && !p.drawerOpen;
  const ready = p.health?.status === "ok";

  if (rail) {
    return (
      <aside className="sidebar is-rail" aria-label="Sidebar">
        <button type="button" className="icon-btn" onClick={p.onToggle} aria-label="Open sidebar" title="Open sidebar">
          <PanelLeft size={19} />
        </button>
        <button type="button" className="rail-new" onClick={p.onNew} aria-label="New chat" title="New chat">
          <Plus size={16} />
        </button>
        <button type="button" className="icon-btn" onClick={p.onToggle} aria-label="Show recent chats" title="Recents">
          <ChatIcon size={18} />
        </button>
        <span className="rail-spacer" />
        <button type="button" className="icon-btn" onClick={p.onToggle} aria-label="Show the library" title="Library">
          <Library size={19} />
        </button>
      </aside>
    );
  }

  return (
    <aside className={`sidebar${p.mobile ? " is-drawer" : ""}${p.drawerOpen ? " is-open" : ""}`} aria-label="Sidebar" inert={hidden || undefined}>
      <div className="sb-top">
        <div className="brand">
          <Spark size={22} className="brand-spark" />
          <span className="brand-name">
            Condition-Grounded
            <span>Scientific RAG</span>
          </span>
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={p.onToggle}
          aria-label={p.mobile ? "Close sidebar" : "Collapse sidebar"}
          title={p.mobile ? "Close sidebar" : "Collapse sidebar"}
        >
          <PanelLeft size={19} />
        </button>
      </div>

      <button type="button" className="new-chat" onClick={p.onNew}>
        <span className="new-chat-plus" aria-hidden="true">
          <Plus size={15} />
        </span>
        New chat
      </button>

      <nav className="recents" aria-label="Recent chats">
        <h2 className="sb-label">Recents</h2>
        {p.chats.length === 0 ? (
          <p className="sb-empty">Your questions will appear here.</p>
        ) : (
          <ul>
            {p.chats.map((c) => (
              <li key={c.id} className={`chat-item${c.id === p.activeId ? " is-active" : ""}`}>
                <button
                  type="button"
                  className="chat-link"
                  onClick={() => p.onSelect(c.id)}
                  aria-current={c.id === p.activeId ? "page" : undefined}
                  title={c.title}
                >
                  {c.id === p.runningChatId && <span className="chat-running" aria-label="answering" />}
                  <span className="chat-title-text">{c.title}</span>
                </button>
                <button
                  type="button"
                  className="chat-del"
                  onClick={() => p.onDelete(c.id)}
                  aria-label={`Delete chat: ${c.title}`}
                  title="Delete chat"
                  disabled={c.id === p.runningChatId}
                >
                  <Trash size={15} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </nav>

      <section className="library" aria-label="Library">
        <h2 className="sb-label">
          <Book size={14} /> Library
        </h2>
        {ready ? (
          <dl className="lib-stats">
            <div>
              <dt>papers</dt>
              <dd>{fmt(p.health?.papers)}</dd>
            </div>
            <div>
              <dt>passages</dt>
              <dd>{fmt(p.health?.chunks)}</dd>
            </div>
            <div>
              <dt>profiles</dt>
              <dd>{fmt(p.health?.profiles)}</dd>
            </div>
          </dl>
        ) : (
          <p className="lib-stats-wait">Counts appear when the API is ready.</p>
        )}
        <button type="button" className="lib-upload" onClick={p.onUpload} disabled={p.upload?.state === "busy" || !ready}>
          <Upload size={16} /> Add a paper (PDF)
        </button>
        {p.upload && (
          <p className={`lib-note is-${p.upload.state}`} role="status">
            {p.upload.state === "busy" && <span className="spinner" aria-hidden="true" />}
            {p.upload.text}
          </p>
        )}
        <div className={`lib-papers${papersOpen ? " is-open" : ""}`}>
          <button type="button" className="lib-papers-head" aria-expanded={papersOpen} onClick={() => setPapersOpen((v) => !v)}>
            <FileText size={15} />
            <span>Indexed papers ({p.papers.length})</span>
            <ChevronDown size={15} className="lib-chevron" />
          </button>
          {papersOpen && (
            <ul className="lib-list">
              {p.papers.map((paper) => (
                <li key={paper.paper_id} title={`${paper.title || paper.paper_id} (${paper.paper_id})`}>
                  <span className="lib-title">{paper.title || paper.paper_id}</span>
                  <span className="lib-meta">
                    {paper.paper_id} · {paper.profiles.toLocaleString("en-US")} profiles
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </aside>
  );
}
