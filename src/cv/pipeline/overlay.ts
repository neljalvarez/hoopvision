import type { Calibration } from '../calibration';
import type { ShotState } from '../shot/types';
import type { TrackSnapshot } from '../tracking/tracker';
import type { CropRegion, Detection, Polygon } from '../types';

export const ZONE_COLORS = {
  rim: '#22c55e',
  approach: '#3b82f6',
  scoring: '#a855f7',
  observation: '#ff6b00',
};

export interface OverlayOptions {
  calibration: Calibration | null;
  tracks: TrackSnapshot[];
  detections?: Detection[];
  crop?: CropRegion | null;
  showZones: boolean;
  showBoxes: boolean;
  showTrail: boolean;
  showDebug: boolean;
  state?: ShotState;
  /** extrapolate track positions to this time (s) for smooth display between detections */
  now?: number;
  ballClassIds?: number[];
}

function poly(ctx: CanvasRenderingContext2D, p: Polygon) {
  if (p.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(p[0].x, p[0].y);
  for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
  ctx.closePath();
}

const STATE_COLOR: Partial<Record<ShotState, string>> = {
  potential_shot: '#fbbf24',
  confirmed_shot: '#ff8f3d',
  approaching_rim: '#ff6b00',
  awaiting_outcome: '#a855f7',
  made: '#34d399',
  missed: '#f87171',
  uncertain: '#fbbf24',
};

/** Draws calibration zones, detections, tracks and trails in source-frame pixel space. */
export function drawOverlay(ctx: CanvasRenderingContext2D, o: OverlayOptions) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  const lw = Math.max(2, width / 480);
  const cal = o.calibration;

  if (cal && o.showZones) {
    ctx.lineWidth = lw;
    ctx.setLineDash([lw * 4, lw * 3]);
    ctx.strokeStyle = ZONE_COLORS.observation;
    poly(ctx, cal.observationZone);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(59,130,246,0.14)';
    ctx.strokeStyle = ZONE_COLORS.approach;
    poly(ctx, cal.approachZone);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = 'rgba(168,85,247,0.18)';
    ctx.strokeStyle = ZONE_COLORS.scoring;
    poly(ctx, cal.scoringZone);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = ZONE_COLORS.rim;
    ctx.lineWidth = lw * 1.5;
    ctx.beginPath();
    ctx.ellipse(cal.rim.cx, cal.rim.cy, cal.rim.width / 2, Math.max(1, cal.rim.height / 2), cal.rim.angle, 0, Math.PI * 2);
    ctx.stroke();
  }

  if (o.showDebug && o.crop) {
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(o.crop.x, o.crop.y, o.crop.w, o.crop.h);
    ctx.setLineDash([]);
  }

  if (o.showDebug && o.detections) {
    const balls = new Set(o.ballClassIds ?? [32]);
    for (const d of o.detections) {
      ctx.strokeStyle = balls.has(d.classId) ? 'rgba(250,250,250,0.8)' : 'rgba(34,197,94,0.8)';
      ctx.lineWidth = 1;
      ctx.strokeRect(d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
      ctx.fillStyle = '#fafafa';
      ctx.font = `${Math.max(10, width / 90)}px Inter, sans-serif`;
      ctx.fillText(d.score.toFixed(2), d.bbox.x, d.bbox.y - 3);
    }
  }

  const accent = (o.state && STATE_COLOR[o.state]) || '#ff6b00';
  for (const tr of o.tracks) {
    if (!tr.confirmed) continue;
    if (o.showTrail && tr.history.length > 1) {
      const pts = tr.history.filter((h) => h.observed);
      for (let i = 1; i < pts.length; i++) {
        const a = i / pts.length;
        ctx.strokeStyle = `rgba(255,107,0,${0.15 + 0.75 * a})`;
        ctx.lineWidth = lw * (0.6 + 1.4 * a);
        ctx.beginPath();
        ctx.moveTo(pts[i - 1].x, pts[i - 1].y);
        ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      }
    }
    let x = tr.x;
    let y = tr.y;
    if (o.now !== undefined && tr.lastSeen) {
      const dt = Math.min(0.12, Math.max(0, o.now - tr.lastSeen));
      x += tr.vx * dt;
      y += tr.vy * dt;
    }
    if (o.showBoxes) {
      ctx.strokeStyle = tr.observed ? accent : 'rgba(255,255,255,0.5)';
      ctx.lineWidth = lw * 1.2;
      ctx.beginPath();
      ctx.arc(x, y, tr.r * 1.15, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (o.showDebug) {
      ctx.fillStyle = '#faf8f3';
      ctx.font = `600 ${Math.max(11, width / 80)}px Inter, sans-serif`;
      ctx.fillText(`#${tr.id} ${(tr.score * 100).toFixed(0)}%`, x + tr.r * 1.3, y - tr.r);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + tr.vx * 0.1, y + tr.vy * 0.1);
      ctx.stroke();
    }
  }
}
