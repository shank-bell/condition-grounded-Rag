// Small inline SVG icons (drawn for this app; 24 x 24 grid, stroke icons inherit the text colour).
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 18, children, ...rest }: P) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

/** The app's own mark: an eight-point compass spark (four long rays, four short ones). Filled with the accent colour. */
export function Spark({ size = 20, className, animated = false }: { size?: number; className?: string; animated?: boolean }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      className={`spark${animated ? " spark-animated" : ""}${className ? ` ${className}` : ""}`}
    >
      <path
        d="M12 1.6 L13 10 L16.4 7.6 L14 11 L22.4 12 L14 13 L16.4 16.4 L13 14 L12 22.4 L11 14 L7.6 16.4 L10 13 L1.6 12 L10 11 L7.6 7.6 L11 10 Z"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth={1.1}
        strokeLinejoin="round"
      />
    </svg>
  );
}

export const Plus = (p: P) => (
  <Svg {...p}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
);

export const ArrowUp = (p: P) => (
  <Svg {...p} strokeWidth={2.1}>
    <path d="M12 19V5M5.5 11.5 12 5l6.5 6.5" />
  </Svg>
);

export const ArrowDown = (p: P) => (
  <Svg {...p} strokeWidth={2}>
    <path d="M12 5v14M5.5 12.5 12 19l6.5-6.5" />
  </Svg>
);

export const StopSquare = ({ size = 14 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true" focusable="false">
    <rect x="2" y="2" width="10" height="10" rx="2" fill="currentColor" />
  </svg>
);

export const Copy = (p: P) => (
  <Svg {...p}>
    <rect x="8.5" y="8.5" width="11" height="11" rx="2.5" />
    <path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2" />
  </Svg>
);

export const Check = (p: P) => (
  <Svg {...p} strokeWidth={2.2}>
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </Svg>
);

export const Cross = (p: P) => (
  <Svg {...p} strokeWidth={2.2}>
    <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />
  </Svg>
);

export const Dash = (p: P) => (
  <Svg {...p} strokeWidth={2.2}>
    <path d="M7 12h10" />
  </Svg>
);

export const Retry = (p: P) => (
  <Svg {...p}>
    <path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" />
    <path d="M19.5 4.5v4.2h-4.2" />
  </Svg>
);

export const ChevronDown = (p: P) => (
  <Svg {...p} strokeWidth={2}>
    <path d="m6.5 9.5 5.5 5.5 5.5-5.5" />
  </Svg>
);

export const ChevronRight = (p: P) => (
  <Svg {...p} strokeWidth={2}>
    <path d="m9.5 6.5 5.5 5.5-5.5 5.5" />
  </Svg>
);

export const Menu = (p: P) => (
  <Svg {...p}>
    <path d="M4 7h16M4 12h16M4 17h10" />
  </Svg>
);

export const PanelLeft = (p: P) => (
  <Svg {...p}>
    <rect x="3.5" y="4.5" width="17" height="15" rx="3" />
    <path d="M9.5 4.5v15" />
  </Svg>
);

export const Trash = (p: P) => (
  <Svg {...p}>
    <path d="M4.5 7h15M9.5 7V5.2c0-.7.5-1.2 1.2-1.2h2.6c.7 0 1.2.5 1.2 1.2V7" />
    <path d="M6.5 7l.8 11.2c.1 1 .9 1.8 1.9 1.8h5.6c1 0 1.8-.8 1.9-1.8L17.5 7" />
  </Svg>
);

export const FileText = (p: P) => (
  <Svg {...p}>
    <path d="M14 3.5H7.5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8z" />
    <path d="M14 3.5V8h4.5M9 12.5h6M9 16h4" />
  </Svg>
);

export const Upload = (p: P) => (
  <Svg {...p}>
    <path d="M12 15V4.5M7.5 9 12 4.5 16.5 9" />
    <path d="M4.5 15v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V15" />
  </Svg>
);

export const Warning = (p: P) => (
  <Svg {...p}>
    <path d="M10.3 4.3 2.9 17.2A2 2 0 0 0 4.6 20h14.8a2 2 0 0 0 1.7-2.8L13.7 4.3a2 2 0 0 0-3.4 0z" />
    <path d="M12 9.5v4.2M12 16.8h.01" strokeWidth={2.2} />
  </Svg>
);

export const Chat = (p: P) => (
  <Svg {...p}>
    <path d="M20 12.5a7.5 7.5 0 0 1-11 6.6L4.5 20l1-4A7.5 7.5 0 1 1 20 12.5z" />
  </Svg>
);

export const Library = (p: P) => (
  <Svg {...p}>
    <path d="M5 4.5v15M9.5 4.5v15" />
    <path d="m14 5.2 3.6-.9 3 14.6-3.6.8z" />
  </Svg>
);

export const Target = (p: P) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8" />
    <circle cx="12" cy="12" r="3.5" />
  </Svg>
);

export const Compare = (p: P) => (
  <Svg {...p}>
    <path d="M7 4.5 3.5 8 7 11.5M3.5 8h13" />
    <path d="M17 12.5l3.5 3.5-3.5 3.5M20.5 16h-13" />
  </Svg>
);

export const Shield = (p: P) => (
  <Svg {...p}>
    <path d="M12 3.5 5 6.2v5.3c0 4.3 2.9 7.6 7 9 4.1-1.4 7-4.7 7-9V6.2z" />
    <path d="m9 12 2.2 2.2L15.3 10" />
  </Svg>
);

export const Book = (p: P) => (
  <Svg {...p}>
    <path d="M4.5 5.5c2.5-1.3 5-1.3 7.5 0v14c-2.5-1.3-5-1.3-7.5 0zM12 5.5c2.5-1.3 5-1.3 7.5 0v14c-2.5-1.3-5-1.3-7.5 0z" />
  </Svg>
);

export const ExternalJump = (p: P) => (
  <Svg {...p}>
    <path d="M9 6.5h-2a2 2 0 0 0-2 2v8.5a2 2 0 0 0 2 2h8.5a2 2 0 0 0 2-2v-2M13 5h6v6M19 5l-8.5 8.5" />
  </Svg>
);

export const Info = (p: P) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 11v5.2M12 7.8h.01" strokeWidth={2.2} />
  </Svg>
);
