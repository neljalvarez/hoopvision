import type { SVGProps } from 'react';

type P = SVGProps<SVGSVGElement> & { size?: number };

const base = ({ size = 20, ...p }: P) => ({
  width: size,
  height: size,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
  ...p,
});

export const IconHome = (p: P) => (
  <svg {...base(p)}><path d="M3 11l9-7 9 7" /><path d="M5 10v10h14V10" /></svg>
);
export const IconBall = (p: P) => (
  <svg {...base(p)}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3v18M5.6 5.6c3.5 3.5 3.5 9.3 0 12.8M18.4 5.6c-3.5 3.5-3.5 9.3 0 12.8" /></svg>
);
export const IconTarget = (p: P) => (
  <svg {...base(p)}><ellipse cx="12" cy="7" rx="8" ry="2.5" /><path d="M5 8l2 11h10l2-11M8.5 9.5l1 9M15.5 9.5l-1 9M12 9.5v9" /></svg>
);
export const IconCamera = (p: P) => (
  <svg {...base(p)}><path d="M4 7h3l2-2h6l2 2h3v12H4z" /><circle cx="12" cy="13" r="3.5" /></svg>
);
export const IconHistory = (p: P) => (
  <svg {...base(p)}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5M12 7v5l3 2" /></svg>
);
export const IconChart = (p: P) => (
  <svg {...base(p)}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
);
export const IconSettings = (p: P) => (
  <svg {...base(p)}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></svg>
);
export const IconLab = (p: P) => (
  <svg {...base(p)}><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3" /><path d="M7.5 15h9" /></svg>
);
export const IconPlay = (p: P) => (
  <svg {...base(p)}><path d="M7 4l13 8-13 8z" fill="currentColor" /></svg>
);
export const IconPause = (p: P) => (
  <svg {...base(p)}><path d="M7 4h3v16H7zM14 4h3v16h-3z" fill="currentColor" /></svg>
);
export const IconStop = (p: P) => (
  <svg {...base(p)}><rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor" /></svg>
);
export const IconUndo = (p: P) => (
  <svg {...base(p)}><path d="M9 14L4 9l5-5" /><path d="M4 9h11a5 5 0 0 1 0 10h-3" /></svg>
);
export const IconPlus = (p: P) => (
  <svg {...base(p)}><path d="M12 5v14M5 12h14" /></svg>
);
export const IconCheck = (p: P) => (
  <svg {...base(p)}><path d="M5 12l5 5L20 7" /></svg>
);
export const IconX = (p: P) => (
  <svg {...base(p)}><path d="M6 6l12 12M18 6L6 18" /></svg>
);
export const IconVolume = (p: P) => (
  <svg {...base(p)}><path d="M4 9h4l5-4v14l-5-4H4z" /><path d="M16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12" /></svg>
);
export const IconMute = (p: P) => (
  <svg {...base(p)}><path d="M4 9h4l5-4v14l-5-4H4z" /><path d="M17 9l5 6M22 9l-5 6" /></svg>
);
export const IconExpand = (p: P) => (
  <svg {...base(p)}><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></svg>
);
export const IconFlip = (p: P) => (
  <svg {...base(p)}><path d="M4 7h13l-3-3M20 17H7l3 3" /></svg>
);
export const IconBug = (p: P) => (
  <svg {...base(p)}><rect x="7" y="7" width="10" height="13" rx="5" /><path d="M12 7V4M9 4l1 3M15 4l-1 3M3 12h4M17 12h4M4 18l3-2M20 18l-3-2M4 6l3 2M20 6l-3 2" /></svg>
);
export const IconDownload = (p: P) => (
  <svg {...base(p)}><path d="M12 4v12M7 11l5 5 5-5M4 20h16" /></svg>
);
export const IconUpload = (p: P) => (
  <svg {...base(p)}><path d="M12 20V8M7 13l5-5 5 5M4 4h16" /></svg>
);
export const IconAlert = (p: P) => (
  <svg {...base(p)}><path d="M12 3l10 18H2z" /><path d="M12 10v5M12 18h.01" /></svg>
);
export const IconChevron = (p: P) => (
  <svg {...base(p)}><path d="M9 6l6 6-6 6" /></svg>
);
export const IconTrash = (p: P) => (
  <svg {...base(p)}><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" /></svg>
);
export const IconRecord = (p: P) => (
  <svg {...base(p)}><circle cx="12" cy="12" r="7" fill="currentColor" /></svg>
);
