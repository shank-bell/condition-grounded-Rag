// A safety net: if drawing something throws, show a readable message instead of a blank page.
// One boundary wraps the whole app; one wraps each assistant turn so a single odd response cannot take the conversation down.
import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** "page" = full-screen message with a reload button; "turn" = an inline callout inside the conversation. */
  scope?: "page" | "turn";
  onRetry?: () => void;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string | null }) {
    // Visible in the browser console for whoever debugs it; the page itself stays usable.
    console.error("Rendering failed:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    if (this.props.scope === "turn") {
      return (
        <div className="callout callout-error" role="alert">
          <div>
            <p className="callout-title">This answer could not be displayed</p>
            <p className="callout-detail">{error.message}</p>
            {this.props.onRetry && (
              <button
                type="button"
                className="text-btn"
                onClick={() => {
                  this.setState({ error: null });
                  this.props.onRetry?.();
                }}
              >
                Ask again
              </button>
            )}
          </div>
        </div>
      );
    }
    return (
      <div className="crash" role="alert">
        <h1>Something went wrong while drawing the page</h1>
        <p>{error.message}</p>
        <button type="button" className="text-btn" onClick={() => window.location.reload()}>
          Reload the page
        </button>
      </div>
    );
  }
}
