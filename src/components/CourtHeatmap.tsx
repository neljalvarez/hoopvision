import type { SessionStats, ShootingZone } from '../session/types';
import { ZONES } from '../session/types';

/** Half-court positions (SVG units, basket at the top). */
const POS: Record<ShootingZone, { x: number; y: number }> = {
  rim: { x: 250, y: 70 },
  left_short_corner: { x: 120, y: 95 },
  right_short_corner: { x: 380, y: 95 },
  left_wing: { x: 105, y: 235 },
  right_wing: { x: 395, y: 235 },
  top_key: { x: 250, y: 330 },
  free_throw: { x: 250, y: 210 },
  left_corner_three: { x: 38, y: 60 },
  right_corner_three: { x: 462, y: 60 },
};

/** Sequential scale from cold (charcoal-blue) to hot (electric orange) — single hue family per end. */
function heat(p: number | null) {
  if (p === null) return '#2a2e35';
  const stops = [
    [0, [46, 58, 84]],
    [35, [96, 74, 70]],
    [50, [179, 82, 30]],
    [65, [255, 107, 0]],
    [100, [255, 178, 122]],
  ] as const;
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i];
    const [p0, c0] = stops[i - 1];
    if (p <= p1) {
      const f = (p - p0) / (p1 - p0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c.join(',')})`;
    }
  }
  return 'rgb(255,178,122)';
}

export function CourtHeatmap({
  perZone,
  selected,
  onSelect,
  compact,
}: {
  perZone?: SessionStats['perZone'];
  selected?: ShootingZone | null;
  onSelect?: (z: ShootingZone) => void;
  compact?: boolean;
}) {
  return (
    <svg viewBox="0 0 500 400" className="w-full" role="img" aria-label="Shooting efficiency by court location">
      <rect x="0" y="0" width="500" height="400" rx="16" fill="#131417" />
      <g fill="none" stroke="#3a3f48" strokeWidth="3">
        <rect x="8" y="8" width="484" height="384" rx="10" />
        <rect x="170" y="8" width="160" height="200" />
        <circle cx="250" cy="208" r="60" />
        <path d="M26 8 V100 A230 230 0 0 0 474 100 V8" />
        <path d="M215 52 h70" stroke="#faf8f3" strokeWidth="4" />
      </g>
      <circle cx="250" cy="66" r="11" fill="none" stroke="#ff6b00" strokeWidth="4" />
      {ZONES.map((z) => {
        const p = POS[z.id];
        const s = perZone?.[z.id];
        const n = s ? s.makes + s.misses : 0;
        const isSel = selected === z.id;
        return (
          <g
            key={z.id}
            transform={`translate(${p.x} ${p.y})`}
            onClick={onSelect ? () => onSelect(z.id) : undefined}
            className={onSelect ? 'cursor-pointer' : undefined}
            role={onSelect ? 'button' : undefined}
            aria-label={`${z.label}${s ? `: ${s.pct?.toFixed(0)}% on ${n}` : ''}`}
          >
            <circle r={compact ? 30 : 34} fill={heat(s && n ? s.pct : null)} stroke={isSel ? '#faf8f3' : '#0d0e10'} strokeWidth={isSel ? 4 : 2} />
            <text textAnchor="middle" y={n ? -2 : 4} className="font-display" fontSize="19" fontWeight="700" fill="#faf8f3">
              {n ? `${s!.pct!.toFixed(0)}%` : z.short}
            </text>
            {n > 0 && (
              <text textAnchor="middle" y="15" fontSize="11" fill="#faf8f3" opacity="0.85">
                {s!.makes}/{n}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
