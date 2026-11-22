/** Minimal PNG decoder for tests (8-bit, non-interlaced, gray/gray+alpha/RGB/RGBA). */
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';

export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export function readPng(path: string): RgbaImage {
  const buf = readFileSync(path);
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      if (body[8] !== 8 || body[12] !== 0) throw new Error('unsupported PNG');
      colorType = body[9];
    } else if (type === 'IDAT') idat.push(body);
    pos += 12 + len;
  }
  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType as 0 | 2 | 4 | 6];
  if (!channels) throw new Error('unsupported color type');
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[y * stride + x - channels] : 0;
      const b = y > 0 ? out[(y - 1) * stride + x] : 0;
      const c = x >= channels && y > 0 ? out[(y - 1) * stride + x - channels] : 0;
      let v = line[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[y * stride + x] = v & 255;
    }
  }
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const s = out.subarray(i * channels, i * channels + channels);
    const [r, g, b] = channels >= 3 ? [s[0], s[1], s[2]] : [s[0], s[0], s[0]];
    data.set([r, g, b, channels === 4 ? s[3] : channels === 2 ? s[1] : 255], i * 4);
  }
  return { width, height, data };
}

/** Bilinear crop+resize into a grey (114) letterboxed square, like the browser canvas path. */
export function letterboxRgba(img: RgbaImage, crop: { x: number; y: number; w: number; h: number }, size: number): Uint8ClampedArray {
  const scale = Math.min(size / crop.w, size / crop.h);
  const out = new Uint8ClampedArray(size * size * 4).fill(114);
  const dw = Math.round(crop.w * scale);
  const dh = Math.round(crop.h * scale);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(img.width - 1.001, Math.max(0, crop.x + (x + 0.5) / scale - 0.5));
      const sy = Math.min(img.height - 1.001, Math.max(0, crop.y + (y + 0.5) / scale - 0.5));
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      for (let c = 0; c < 3; c++) {
        const p = (yy: number, xx: number) => img.data[(yy * img.width + xx) * 4 + c];
        const v = p(y0, x0) * (1 - fx) * (1 - fy) + p(y0, x0 + 1) * fx * (1 - fy) + p(y0 + 1, x0) * (1 - fx) * fy + p(y0 + 1, x0 + 1) * fx * fy;
        out[(y * size + x) * 4 + c] = v;
      }
      out[(y * size + x) * 4 + 3] = 255;
    }
  }
  return out;
}
