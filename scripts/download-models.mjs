// Downloads (or verifies) the YOLOX ONNX models into public/models.
// The models are committed to the repository, so this is only needed if they were removed
// or you want to verify their integrity.  Usage: npm run models:download [-- --force]
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dir = join(root, 'public', 'models');
const BASE = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0';
const MODELS = [
  { file: 'yolox_nano.onnx', sha256: 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d' },
  { file: 'yolox_tiny.onnx', sha256: '427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7' },
];
const force = process.argv.includes('--force');
const sha = (buf) => createHash('sha256').update(buf).digest('hex');

mkdirSync(dir, { recursive: true });
let failed = false;
for (const m of MODELS) {
  const path = join(dir, m.file);
  if (existsSync(path) && !force) {
    const ok = sha(readFileSync(path)) === m.sha256;
    console.log(`${ok ? '✓' : '✗'} ${m.file} ${ok ? 'present and verified' : 'present but checksum mismatch (re-run with --force)'}`);
    failed ||= !ok;
    continue;
  }
  process.stdout.write(`↓ ${m.file} … `);
  const res = await fetch(`${BASE}/${m.file}`);
  if (!res.ok) {
    console.log(`HTTP ${res.status}`);
    failed = true;
    continue;
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (sha(buf) !== m.sha256) {
    console.log('checksum mismatch, not saved');
    failed = true;
    continue;
  }
  writeFileSync(path, buf);
  console.log(`${(buf.length / 1e6).toFixed(1)} MB verified`);
}
process.exit(failed ? 1 : 0);
