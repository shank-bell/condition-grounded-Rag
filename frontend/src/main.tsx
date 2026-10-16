import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { Connect } from "./components/Connect";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary scope="page">
      <Connect>
        <App />
      </Connect>
    </ErrorBoundary>
  </StrictMode>,
);
