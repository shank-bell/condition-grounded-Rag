import { useLayoutEffect, type FormEvent, type KeyboardEvent, type RefObject } from "react";
import { ArrowUp, Plus, StopSquare } from "./Icons";

const MAX_ROWS = 8;
const LINE = 24;            // px, matches .composer textarea line-height

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  onAttach: () => void;
  /** A question of this chat is running: the send button becomes a stop button. */
  running: boolean;
  /** Why sending is not possible right now (warming up, API down, another chat is answering), or null. */
  blocked: string | null;
  placeholder: string;
  modelLabel?: string;
  variant: "welcome" | "dock";
  inputRef: RefObject<HTMLTextAreaElement | null>;
}

export function Composer({ value, onChange, onSend, onStop, onAttach, running, blocked, placeholder, modelLabel, variant, inputRef }: Props) {
  const canSend = !running && !blocked && value.trim().length >= 3;

  // Grow with the text from one to eight rows, then scroll inside.
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    const max = LINE * MAX_ROWS;
    const h = Math.min(el.scrollHeight, max);
    el.style.height = `${Math.max(h, LINE)}px`;
    el.style.overflowY = el.scrollHeight > max ? "auto" : "hidden";
  }, [value, inputRef]);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    if (canSend) onSend();
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <form className={`composer composer-${variant}`} onSubmit={submit} aria-label="Ask a question">
      <textarea
        ref={inputRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        aria-label="Question"
        rows={1}
        maxLength={2000}
        spellCheck
        enterKeyHint="send"
      />
      <div className="composer-bar">
        <button type="button" className="icon-btn composer-attach" onClick={onAttach} aria-label="Add a paper (PDF)" title="Add a paper (PDF)">
          <Plus size={18} />
        </button>
        <div className="composer-right">
          {modelLabel && <span className="composer-model" title="Answer model">{modelLabel}</span>}
          {running ? (
            <button type="button" className="send-btn is-stop" onClick={onStop} aria-label="Stop answering" title="Stop answering">
              <StopSquare size={12} />
            </button>
          ) : (
            <button
              type="submit"
              className="send-btn"
              disabled={!canSend}
              aria-label="Send question"
              title={blocked ?? (value.trim().length < 3 ? "Type a question first" : "Send (Enter)")}
            >
              <ArrowUp size={17} />
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
