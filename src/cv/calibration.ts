import { pointInPolygon, rectPolygon } from './geometry';
import type { Polygon, Vec2 } from './types';

/** Regulation rim inner diameter (18 in). Used to convert pixels to metres at the rim's depth. */
export const RIM_DIAMETER_M = 0.457;
/** Regulation basketball diameter (size 7 ≈ 24 cm). */
export const BALL_DIAMETER_M = 0.24;
export const GRAVITY_MPS2 = 9.81;

/**
 * The rim as the camera sees it: an ellipse (a circle seen in perspective).
 * `width` is the projected rim diameter along its major axis, `height` the
 * projected minor axis (small when the camera is close to rim height),
 * `angle` the rotation of the major axis in radians (0 = horizontal).
 */
export interface RimCalibration {
  cx: number;
  cy: number;
  width: number;
  height: number;
  angle: number;
}

export interface MotionReference {
  width: number;
  height: number;
  /** Grayscale thumbnail, row-major, values 0..255. */
  pixels: number[];
}

export interface Calibration {
  id: string;
  createdAt: number;
  name: string;
  /** Size of the frame (in px) that all coordinates below refer to. */
  frameWidth: number;
  frameHeight: number;
  rim: RimCalibration;
  /** Blue: region above the rim the ball travels through on its way in. */
  approachZone: Polygon;
  /** Purple: region immediately under the rim (the net) that confirms a make. */
  scoringZone: Polygon;
  /** Orange: everything the camera can use to observe shots. Detections outside are ignored. */
  observationZone: Polygon;
  /** Thumbnail captured at calibration time, used to detect camera movement. */
  motionReference?: MotionReference;
}

/** Rim-local coordinates: u along the rim's major axis, v perpendicular (positive = below the rim). */
export interface RimLocal {
  u: number;
  v: number;
}

export function toRimLocal(rim: RimCalibration, p: Vec2): RimLocal {
  const dx = p.x - rim.cx;
  const dy = p.y - rim.cy;
  const c = Math.cos(rim.angle);
  const s = Math.sin(rim.angle);
  return { u: dx * c + dy * s, v: -dx * s + dy * c };
}

export function fromRimLocal(rim: RimCalibration, l: RimLocal): Vec2 {
  const c = Math.cos(rim.angle);
  const s = Math.sin(rim.angle);
  return { x: rim.cx + l.u * c - l.v * s, y: rim.cy + l.u * s + l.v * c };
}

/** Velocity vector expressed in rim-local axes. */
export function velocityToRimLocal(rim: RimCalibration, vx: number, vy: number): RimLocal {
  const c = Math.cos(rim.angle);
  const s = Math.sin(rim.angle);
  return { u: vx * c + vy * s, v: -vx * s + vy * c };
}

/** Pixels per metre at the rim's depth. */
export function pxPerMeter(rim: RimCalibration): number {
  return rim.width / RIM_DIAMETER_M;
}

/** Polygon that approximates a rotated rectangle in rim-local coordinates. */
export function rimLocalRect(rim: RimCalibration, u0: number, v0: number, u1: number, v1: number): Polygon {
  return [
    fromRimLocal(rim, { u: u0, v: v0 }),
    fromRimLocal(rim, { u: u1, v: v0 }),
    fromRimLocal(rim, { u: u1, v: v1 }),
    fromRimLocal(rim, { u: u0, v: v1 }),
  ];
}

/** Sensible starting zones derived from the rim, which the user can then drag. */
export function defaultZones(rim: RimCalibration, frameWidth: number, frameHeight: number) {
  const w = rim.width;
  const approachZone = rimLocalRect(rim, -1.6 * w, -2.6 * w, 1.6 * w, 0);
  const scoringZone = rimLocalRect(rim, -0.7 * w, 0, 0.7 * w, 1.3 * w);
  const mx = frameWidth * 0.01;
  const my = frameHeight * 0.01;
  const observationZone = rectPolygon(mx, my, frameWidth - 2 * mx, frameHeight - 2 * my);
  return { approachZone, scoringZone, observationZone };
}

export function createCalibration(
  rim: RimCalibration,
  frameWidth: number,
  frameHeight: number,
  name = 'My hoop',
): Calibration {
  return {
    id: `cal-${Date.now().toString(36)}`,
    createdAt: Date.now(),
    name,
    frameWidth,
    frameHeight,
    rim,
    ...defaultZones(rim, frameWidth, frameHeight),
  };
}

/** Re-express a calibration in a frame of a different size (e.g. a camera resolution change). */
export function scaleCalibration(cal: Calibration, frameWidth: number, frameHeight: number): Calibration {
  if (cal.frameWidth === frameWidth && cal.frameHeight === frameHeight) return cal;
  const sx = frameWidth / cal.frameWidth;
  const sy = frameHeight / cal.frameHeight;
  const sp = (p: Vec2) => ({ x: p.x * sx, y: p.y * sy });
  const s = Math.sqrt(sx * sy);
  return {
    ...cal,
    frameWidth,
    frameHeight,
    rim: { ...cal.rim, cx: cal.rim.cx * sx, cy: cal.rim.cy * sy, width: cal.rim.width * s, height: cal.rim.height * s },
    approachZone: cal.approachZone.map(sp),
    scoringZone: cal.scoringZone.map(sp),
    observationZone: cal.observationZone.map(sp),
  };
}

/** Aspect ratio mismatch means the field of view changed; the calibration cannot be reused safely. */
export function calibrationMatchesAspect(cal: Calibration, frameWidth: number, frameHeight: number): boolean {
  const a = cal.frameWidth / cal.frameHeight;
  const b = frameWidth / frameHeight;
  return Math.abs(a - b) / a < 0.02;
}

export function inZone(poly: Polygon, p: Vec2): boolean {
  return poly.length >= 3 && pointInPolygon(p, poly);
}

/** Validates a calibration and returns human-readable problems (empty = OK). */
export function validateCalibration(cal: Calibration): string[] {
  const problems: string[] = [];
  const { rim, frameWidth } = cal;
  if (rim.width < frameWidth * 0.015) problems.push('The rim is very small in the frame. Move the camera closer or zoom in so the rim is at least ~2% of the frame width.');
  if (rim.width > frameWidth * 0.5) problems.push('The rim is very large in the frame. Move the camera back so the shooter and the ball flight are visible.');
  if (rim.height > rim.width) problems.push('Rim height cannot exceed its width.');
  if (!inZone(cal.observationZone, { x: rim.cx, y: rim.cy })) problems.push('The observation area must contain the hoop.');
  if (cal.scoringZone.length < 3 || cal.approachZone.length < 3) problems.push('Approach and scoring zones are required.');
  return problems;
}
