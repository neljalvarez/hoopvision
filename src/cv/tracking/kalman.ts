/**
 * Constant-velocity Kalman filter for a ball in image space, with gravity as a known
 * control input. State: [x, y, vx, vy] (px, px/s). Measurement: [x, y].
 *
 * Matrices are tiny (4×4), so the algebra is written out explicitly for speed and to
 * avoid a linear-algebra dependency.
 */
export class BallKalman {
  /** state */
  x: number[];
  /** covariance, row-major 4×4 */
  P: number[];

  constructor(
    x0: number,
    y0: number,
    initialPosStd: number,
    initialVelStd: number,
    vx0 = 0,
    vy0 = 0,
  ) {
    this.x = [x0, y0, vx0, vy0];
    const p = initialPosStd ** 2;
    const v = initialVelStd ** 2;
    this.P = [p, 0, 0, 0, 0, p, 0, 0, 0, 0, v, 0, 0, 0, 0, v];
  }

  /**
   * Predict forward by dt seconds.
   * @param accelStd process noise (white acceleration), px/s²
   * @param gravity downward acceleration applied as control input, px/s²
   */
  predict(dt: number, accelStd: number, gravity: number): void {
    if (dt <= 0) return;
    const [x, y, vx, vy] = this.x;
    this.x = [x + vx * dt, y + vy * dt + 0.5 * gravity * dt * dt, vx, vy + gravity * dt];

    // P = F P F^T + Q with F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    const P = this.P;
    const n = new Array<number>(16);
    // FP
    const FP = new Array<number>(16);
    for (let c = 0; c < 4; c++) {
      FP[0 * 4 + c] = P[0 * 4 + c] + dt * P[2 * 4 + c];
      FP[1 * 4 + c] = P[1 * 4 + c] + dt * P[3 * 4 + c];
      FP[2 * 4 + c] = P[2 * 4 + c];
      FP[3 * 4 + c] = P[3 * 4 + c];
    }
    // (FP) F^T
    for (let r = 0; r < 4; r++) {
      n[r * 4 + 0] = FP[r * 4 + 0] + dt * FP[r * 4 + 2];
      n[r * 4 + 1] = FP[r * 4 + 1] + dt * FP[r * 4 + 3];
      n[r * 4 + 2] = FP[r * 4 + 2];
      n[r * 4 + 3] = FP[r * 4 + 3];
    }
    const q = accelStd * accelStd;
    const dt2 = dt * dt;
    const q11 = (q * dt2 * dt2) / 4;
    const q13 = (q * dt2 * dt) / 2;
    const q33 = q * dt2;
    n[0] += q11;
    n[2] += q13;
    n[8] += q13;
    n[10] += q33;
    n[5] += q11;
    n[7] += q13;
    n[13] += q13;
    n[15] += q33;
    this.P = n;
  }

  /** Innovation covariance S = H P H^T + R (2×2), returned as [s00, s01, s11]. */
  innovationCov(measStd: number): [number, number, number] {
    const r = measStd * measStd;
    return [this.P[0] + r, this.P[1], this.P[5] + r];
  }

  /** Squared Mahalanobis distance of a measurement from the predicted position. */
  mahalanobis2(mx: number, my: number, measStd: number): number {
    const [a, b, d] = this.innovationCov(measStd);
    const det = a * d - b * b;
    if (det <= 1e-9) return Infinity;
    const ex = mx - this.x[0];
    const ey = my - this.x[1];
    return (d * ex * ex - 2 * b * ex * ey + a * ey * ey) / det;
  }

  update(mx: number, my: number, measStd: number): void {
    const [a, b, d] = this.innovationCov(measStd);
    const det = a * d - b * b;
    if (det <= 1e-9) return;
    // S^-1
    const i00 = d / det;
    const i01 = -b / det;
    const i11 = a / det;
    const P = this.P;
    // K = P H^T S^-1 ; P H^T = columns 0 and 1 of P (4×2)
    const K: number[] = new Array(8);
    for (let r = 0; r < 4; r++) {
      const p0 = P[r * 4 + 0];
      const p1 = P[r * 4 + 1];
      K[r * 2 + 0] = p0 * i00 + p1 * i01;
      K[r * 2 + 1] = p0 * i01 + p1 * i11;
    }
    const ex = mx - this.x[0];
    const ey = my - this.x[1];
    for (let r = 0; r < 4; r++) this.x[r] += K[r * 2] * ex + K[r * 2 + 1] * ey;
    // P = (I - K H) P ; (K H) has nonzeros in columns 0,1
    const nP = new Array<number>(16);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        nP[r * 4 + c] = P[r * 4 + c] - K[r * 2] * P[0 * 4 + c] - K[r * 2 + 1] * P[1 * 4 + c];
      }
    }
    // keep symmetric
    for (let r = 0; r < 4; r++) {
      for (let c = r + 1; c < 4; c++) {
        const m = (nP[r * 4 + c] + nP[c * 4 + r]) / 2;
        nP[r * 4 + c] = m;
        nP[c * 4 + r] = m;
      }
    }
    this.P = nP;
  }

  get position() {
    return { x: this.x[0], y: this.x[1] };
  }

  get velocity() {
    return { x: this.x[2], y: this.x[3] };
  }

  get positionStd() {
    return Math.sqrt(Math.max(this.P[0], this.P[5]));
  }
}
