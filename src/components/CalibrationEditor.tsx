import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import { type Calibration, defaultZones, type RimCalibration } from '../cv/calibration';
import { ZONE_COLORS } from '../cv/pipeline/overlay';
import type { Polygon, Vec2 } from '../cv/types';

export type EditStep = 'rim' | 'zones' | 'review';
type ZoneKey = 'approachZone' | 'scoringZone' | 'observationZone';

type Drag =
  | { kind: 'rimEnd'; end: -1 | 1 }
  | { kind: 'rimHeight' }
  | { kind: 'rimMove'; start: Vec2; rim: RimCalibration; zones: Pick<Calibration, ZoneKey> }
  | { kind: 'vertex'; zone: ZoneKey; index: number }
  | { kind: 'zoneMove'; zone: ZoneKey; start: Vec2; poly: Polygon };

export interface DraftState {
  rim: RimCalibration | null;
  approachZone: Polygon;
  scoringZone: Polygon;
  observationZone: Polygon;
  /** zones were hand-edited; stop regenerating them from the rim */
  zonesEdited: boolean;
}

export function draftFrom(cal: Calibration | null): DraftState {
  if (!cal) return { rim: null, approachZone: [], scoringZone: [], observationZone: [], zonesEdited: false };
  return { rim: cal.rim, approachZone: cal.approachZone, scoringZone: cal.scoringZone, observationZone: cal.observationZone, zonesEdited: true };
}

const shift = (poly: Polygon, dx: number, dy: number) => poly.map((p) => ({ x: p.x + dx, y: p.y + dy }));

/**
 * SVG editor laid over the video. Uses the video's intrinsic pixel space as viewBox with
 * `xMidYMid meet`, which matches `object-fit: contain` on the <video>, so editor coordinates
 * are exactly the coordinates the vision pipeline uses.
 */
export function CalibrationEditor({
  frameW,
  frameH,
  draft,
  onChange,
  step,
  activeZone,
  zoom = 1,
}: {
  frameW: number;
  frameH: number;
  draft: DraftState;
  onChange: (d: DraftState) => void;
  step: EditStep;
  activeZone?: ZoneKey | null;
  /** CSS zoom applied by the parent (used to re-measure handle sizes) */
  zoom?: number;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<Drag | null>(null);
  const [pendingEdge, setPendingEdge] = useState<Vec2 | null>(null);
  // handle sizes are defined in screen pixels so they stay small on a tiny rim but easy to
  // grab: the bounding rect already includes any CSS zoom applied by the parent
  const [cssScale, setCssScale] = useState(1);
  useLayoutEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const measure = () => {
      const r = svg.getBoundingClientRect();
      if (r.width && frameW) setCssScale(Math.min(r.width / frameW, r.height / frameH));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(svg);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [frameW, frameH, zoom]);
  const hr = 7 / cssScale;

  const toLocal = useCallback((e: { clientX: number; clientY: number }): Vec2 => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = svg.getScreenCTM();
    const p = m ? pt.matrixTransform(m.inverse()) : pt;
    return { x: Math.max(0, Math.min(frameW, p.x)), y: Math.max(0, Math.min(frameH, p.y)) };
  }, [frameW, frameH]);

  const withRim = (rim: RimCalibration, base: DraftState = draft): DraftState => {
    if (base.zonesEdited) return { ...base, rim };
    const z = defaultZones(rim, frameW, frameH);
    return { ...base, rim, approachZone: z.approachZone, scoringZone: z.scoringZone, observationZone: base.observationZone.length ? base.observationZone : z.observationZone };
  };

  const onBackgroundDown = (e: React.PointerEvent) => {
    if (step !== 'rim') return;
    const p = toLocal(e);
    if (!draft.rim) {
      if (!pendingEdge) {
        setPendingEdge(p);
        return;
      }
      const a = pendingEdge.x <= p.x ? pendingEdge : p;
      const b = pendingEdge.x <= p.x ? p : pendingEdge;
      const width = Math.hypot(b.x - a.x, b.y - a.y);
      if (width < 6) return;
      const rim: RimCalibration = { cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, width, height: width * 0.18, angle: Math.atan2(b.y - a.y, b.x - a.x) };
      setPendingEdge(null);
      onChange(withRim(rim, { ...draft, zonesEdited: false }));
    }
  };

  const startDrag = (d: Drag) => (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    drag.current = d;
  };

  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = toLocal(e);
    const rim = draft.rim;
    if (d.kind === 'rimEnd' && rim) {
      const c = Math.cos(rim.angle);
      const s = Math.sin(rim.angle);
      const other = { x: rim.cx - d.end * (rim.width / 2) * c, y: rim.cy - d.end * (rim.width / 2) * s };
      const a = d.end === 1 ? other : p;
      const b = d.end === 1 ? p : other;
      const width = Math.max(6, Math.hypot(b.x - a.x, b.y - a.y));
      onChange(withRim({ ...rim, cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, width, angle: Math.atan2(b.y - a.y, b.x - a.x), height: Math.min(rim.height, width) }));
    } else if (d.kind === 'rimHeight' && rim) {
      const nx = -Math.sin(rim.angle);
      const ny = Math.cos(rim.angle);
      const h = Math.abs((p.x - rim.cx) * nx + (p.y - rim.cy) * ny) * 2;
      onChange(withRim({ ...rim, height: Math.max(1, Math.min(rim.width, h)) }));
    } else if (d.kind === 'rimMove') {
      const dx = p.x - d.start.x;
      const dy = p.y - d.start.y;
      const moved = { ...d.rim, cx: d.rim.cx + dx, cy: d.rim.cy + dy };
      if (draft.zonesEdited) {
        onChange({ ...draft, rim: moved, approachZone: shift(d.zones.approachZone, dx, dy), scoringZone: shift(d.zones.scoringZone, dx, dy) });
      } else onChange(withRim(moved));
    } else if (d.kind === 'vertex') {
      const poly = draft[d.zone].map((v, i) => (i === d.index ? p : v));
      onChange({ ...draft, [d.zone]: poly, zonesEdited: true });
    } else if (d.kind === 'zoneMove') {
      onChange({ ...draft, [d.zone]: shift(d.poly, p.x - d.start.x, p.y - d.start.y), zonesEdited: true });
    }
  };

  const end = () => {
    drag.current = null;
  };

  const rim = draft.rim;
  const zoneDefs: { key: ZoneKey; color: string; fill: string; label: string }[] = [
    { key: 'observationZone', color: ZONE_COLORS.observation, fill: 'rgba(255,107,0,0.05)', label: 'Shot observation area' },
    { key: 'approachZone', color: ZONE_COLORS.approach, fill: 'rgba(59,130,246,0.16)', label: 'Approach zone' },
    { key: 'scoringZone', color: ZONE_COLORS.scoring, fill: 'rgba(168,85,247,0.22)', label: 'Scoring confirmation zone' },
  ];

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${frameW} ${frameH}`}
      preserveAspectRatio="xMidYMid meet"
      className="absolute inset-0 h-full w-full touch-none select-none"
      onPointerDown={onBackgroundDown}
      onPointerMove={onMove}
      onPointerUp={end}
      onPointerCancel={end}
    >
      {zoneDefs.map((z) => {
        const poly = draft[z.key];
        if (poly.length < 3) return null;
        const editable = step === 'zones' && (!activeZone || activeZone === z.key);
        const pts = poly.map((p) => `${p.x},${p.y}`).join(' ');
        return (
          <g key={z.key} opacity={step === 'rim' ? 0.2 : step === 'zones' && activeZone && activeZone !== z.key ? 0.35 : 1}>
            <polygon
              points={pts}
              fill={z.fill}
              stroke={z.color}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
              strokeDasharray={z.key === 'observationZone' ? '8 6' : undefined}
              onPointerDown={
                editable && z.key !== 'observationZone'
                  ? (e) => {
                      e.stopPropagation();
                      (e.target as Element).setPointerCapture?.(e.pointerId);
                      drag.current = { kind: 'zoneMove', zone: z.key, start: toLocal(e), poly };
                    }
                  : undefined
              }
              style={{ cursor: editable && z.key !== 'observationZone' ? 'move' : undefined }}
            />
            {editable &&
              poly.map((p, i) => (
                <g key={i} onPointerDown={startDrag({ kind: 'vertex', zone: z.key, index: i })} style={{ cursor: 'grab' }}>
                  <circle cx={p.x} cy={p.y} r={hr * 3} fill="transparent" />
                  <circle cx={p.x} cy={p.y} r={hr * 0.75} fill={z.color} stroke="#0d0e10" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                </g>
              ))}
          </g>
        );
      })}

      {pendingEdge && (
        <g>
          <circle cx={pendingEdge.x} cy={pendingEdge.y} r={hr} fill="#22c55e" stroke="#0d0e10" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        </g>
      )}

      {rim && (
        <g>
          <ellipse
            cx={rim.cx}
            cy={rim.cy}
            rx={rim.width / 2}
            ry={Math.max(1, rim.height / 2)}
            transform={`rotate(${(rim.angle * 180) / Math.PI} ${rim.cx} ${rim.cy})`}
            fill="rgba(34,197,94,0.12)"
            stroke={ZONE_COLORS.rim}
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            onPointerDown={
              step === 'rim'
                ? (e) => {
                    e.stopPropagation();
                    (e.target as Element).setPointerCapture?.(e.pointerId);
                    drag.current = { kind: 'rimMove', start: toLocal(e), rim, zones: { approachZone: draft.approachZone, scoringZone: draft.scoringZone, observationZone: draft.observationZone } };
                  }
                : undefined
            }
            style={{ cursor: step === 'rim' ? 'move' : undefined }}
          />
          {step === 'rim' &&
            ([-1, 1] as const).map((end) => {
              const x = rim.cx + end * (rim.width / 2) * Math.cos(rim.angle);
              const y = rim.cy + end * (rim.width / 2) * Math.sin(rim.angle);
              return (
                <g key={end} onPointerDown={startDrag({ kind: 'rimEnd', end })} style={{ cursor: 'ew-resize' }}>
                  <circle cx={x} cy={y} r={hr * 3} fill="transparent" />
                  <circle cx={x} cy={y} r={hr} fill="#22c55e" stroke="#0d0e10" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                </g>
              );
            })}
          {step === 'rim' && (
            <g onPointerDown={startDrag({ kind: 'rimHeight' })} style={{ cursor: 'ns-resize' }}>
              <circle cx={rim.cx - Math.sin(rim.angle) * (rim.height / 2 + hr * 1.6)} cy={rim.cy + Math.cos(rim.angle) * (rim.height / 2 + hr * 1.6)} r={hr * 2.6} fill="transparent" />
              <rect
                x={rim.cx - Math.sin(rim.angle) * (rim.height / 2 + hr * 1.6) - hr * 0.8}
                y={rim.cy + Math.cos(rim.angle) * (rim.height / 2 + hr * 1.6) - hr * 0.8}
                width={hr * 1.6}
                height={hr * 1.6}
                rx={hr * 0.3}
                fill="#faf8f3"
                stroke="#0d0e10"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )}
        </g>
      )}
    </svg>
  );
}
