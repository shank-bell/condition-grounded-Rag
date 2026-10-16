import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  MOCK,
  Unauthorized,
  adoptAddressFromUrl,
  api,
  apiAddress,
  cleanAddress,
  hostedElsewhere,
  needsAddress,
  readCode,
  saveConnection,
} from "../api";

type Phase = { name: "checking" } | { name: "open" } | { name: "form"; message: string | null };

async function probe(): Promise<Phase> {
  if (MOCK) return { name: "open" };
  if (needsAddress()) return { name: "form", message: null };
  let authRequired = false;
  try {
    authRequired = Boolean((await api.health()).auth_required);
  } catch {
    if (!hostedElsewhere()) return { name: "open" };
    return {
      name: "form",
      message: `Could not reach the demo server at ${apiAddress()}. It may be switched off, or its address may have changed.`,
    };
  }
  if (!authRequired) return { name: "open" };
  if (!readCode()) return { name: "form", message: "This demo needs an access code." };
  try {
    await api.check();
    return { name: "open" };
  } catch (err) {
    if (err instanceof Unauthorized) {
      saveConnection(null, "");
      return { name: "form", message: "That access code was not accepted." };
    }
    return { name: "form", message: String(err) };
  }
}

export function Connect({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>({ name: "checking" });
  const [address, setAddress] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    adoptAddressFromUrl();
    setAddress(apiAddress());
    let stop = false;
    void probe().then((next) => {
      if (!stop) setPhase(next);
    });
    return () => {
      stop = true;
    };
  }, []);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const clean = hostedElsewhere() ? cleanAddress(address) : null;
    if (hostedElsewhere() && !clean) {
      setPhase({ name: "form", message: "Enter the server address, starting with https://" });
      return;
    }
    setBusy(true);
    saveConnection(clean, code.trim());
    setPhase(await probe());
    setCode("");
    setBusy(false);
  }

  if (phase.name === "open") return <>{children}</>;
  if (phase.name === "checking") {
    return (
      <main className="connect" aria-busy="true">
        <p className="connect-note">Connecting…</p>
      </main>
    );
  }
  return (
    <main className="connect">
      <form className="connect-card" onSubmit={submit}>
        <h1>Condition-Grounded Scientific RAG</h1>
        <p className="connect-lead">
          This page is only the front end. The system itself runs on the authors' machine. Enter the access code you were given.
        </p>
        {hostedElsewhere() && (
          <label>
            Server address
            <input
              type="url"
              inputMode="url"
              autoComplete="off"
              spellCheck={false}
              placeholder="https://…"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
            />
          </label>
        )}
        <label>
          Access code
          <input
            type="password"
            autoComplete="off"
            autoFocus={!needsAddress()}
            value={code}
            onChange={(e) => setCode(e.target.value)}
          />
        </label>
        {phase.message && (
          <p className="connect-note" role="alert">
            {phase.message}
          </p>
        )}
        <button type="submit" className="connect-btn" disabled={busy}>
          {busy ? "Connecting…" : "Connect"}
        </button>
      </form>
    </main>
  );
}
