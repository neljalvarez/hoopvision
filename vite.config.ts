import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Cross-origin isolation enables SharedArrayBuffer, which lets ONNX Runtime
// use multi-threaded WebAssembly. Everything (models, wasm, fonts) is served
// from our own origin so `require-corp` does not block anything.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react()],
  server: { headers: isolationHeaders, host: true },
  preview: { headers: isolationHeaders },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
} as never);
