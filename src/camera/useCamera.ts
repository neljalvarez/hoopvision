import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraStatus = 'idle' | 'requesting' | 'live' | 'denied' | 'unavailable' | 'error' | 'disconnected';

export interface CameraRequest {
  deviceId?: string | null;
  facingMode?: 'environment' | 'user';
  width: number;
  height: number;
  frameRate?: number;
}

export interface CameraInfo {
  label: string;
  width: number;
  height: number;
  frameRate: number | null;
  facingMode: string | null;
  deviceId: string | null;
}

/**
 * Camera access via MediaDevices.getUserMedia. Footage never leaves the device:
 * frames are only drawn into local canvases for on-device inference.
 */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [info, setInfo] = useState<CameraInfo | null>(null);
  const lastReq = useRef<CameraRequest | null>(null);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const all = await navigator.mediaDevices.enumerateDevices();
    setDevices(all.filter((d) => d.kind === 'videoinput'));
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => {
      t.onended = null;
      t.stop();
    });
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus('idle');
  }, []);

  const start = useCallback(
    async (req: CameraRequest) => {
      lastReq.current = req;
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('unavailable');
        setError(window.isSecureContext ? 'This browser does not support camera access.' : 'Camera access requires HTTPS (or localhost).');
        return;
      }
      streamRef.current?.getTracks().forEach((t) => {
        t.onended = null;
        t.stop();
      });
      setStatus('requesting');
      setError(null);
      const video: MediaTrackConstraints = {
        width: { ideal: req.width },
        height: { ideal: req.height },
        frameRate: { ideal: req.frameRate ?? 30, max: 60 },
        ...(req.deviceId ? { deviceId: { exact: req.deviceId } } : { facingMode: { ideal: req.facingMode ?? 'environment' } }),
      };
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
        streamRef.current = stream;
        const track = stream.getVideoTracks()[0];
        track.onended = () => setStatus('disconnected');
        const s = track.getSettings();
        setInfo({
          label: track.label,
          width: s.width ?? 0,
          height: s.height ?? 0,
          frameRate: s.frameRate ?? null,
          facingMode: (s.facingMode as string) ?? null,
          deviceId: s.deviceId ?? null,
        });
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          v.muted = true;
          v.playsInline = true;
          await v.play().catch(() => undefined);
        }
        setStatus('live');
        void refreshDevices();
      } catch (e) {
        const err = e as DOMException;
        if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
          setStatus('denied');
          setError('Camera permission was denied. Allow camera access for this site in your browser settings, then retry.');
        } else if (err.name === 'NotFoundError' || err.name === 'OverconstrainedError') {
          if (req.deviceId) return start({ ...req, deviceId: null });
          setStatus('unavailable');
          setError('No camera matching the requested settings was found.');
        } else if (err.name === 'NotReadableError') {
          setStatus('error');
          setError('The camera is in use by another app or tab.');
        } else {
          setStatus('error');
          setError(err.message || String(e));
        }
      }
    },
    [refreshDevices],
  );

  const retry = useCallback(() => lastReq.current && start(lastReq.current), [start]);

  // device plug/unplug + track health
  useEffect(() => {
    const onChange = () => {
      void refreshDevices();
      const track = streamRef.current?.getVideoTracks()[0];
      if (track && track.readyState === 'ended') setStatus('disconnected');
    };
    navigator.mediaDevices?.addEventListener?.('devicechange', onChange);
    const iv = setInterval(() => {
      const track = streamRef.current?.getVideoTracks()[0];
      if (track && (track.readyState === 'ended' || track.muted) && status === 'live') setStatus('disconnected');
    }, 1500);
    return () => {
      navigator.mediaDevices?.removeEventListener?.('devicechange', onChange);
      clearInterval(iv);
    };
  }, [refreshDevices, status]);

  // keep dimensions up to date when the phone rotates
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onResize = () => {
      const track = streamRef.current?.getVideoTracks()[0];
      const s = track?.getSettings();
      setInfo((i) => (i ? { ...i, width: v.videoWidth || s?.width || i.width, height: v.videoHeight || s?.height || i.height } : i));
    };
    v.addEventListener('resize', onResize);
    return () => v.removeEventListener('resize', onResize);
  });

  useEffect(() => () => stop(), [stop]);

  return { videoRef, streamRef, status, error, devices, info, start, stop, retry, refreshDevices };
}

/** Screen wake lock so the phone does not sleep during training. */
export function useWakeLock(active: boolean) {
  useEffect(() => {
    if (!active || !('wakeLock' in navigator)) return;
    let lock: WakeLockSentinel | null = null;
    let cancelled = false;
    const acquire = async () => {
      try {
        lock = await navigator.wakeLock.request('screen');
      } catch {
        /* not allowed (e.g. low battery) */
      }
    };
    void acquire();
    const onVis = () => !cancelled && document.visibilityState === 'visible' && acquire();
    document.addEventListener('visibilitychange', onVis);
    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', onVis);
      void lock?.release().catch(() => undefined);
    };
  }, [active]);
}

/** Fullscreen + landscape lock (best effort; iOS Safari does not support orientation lock). */
export async function enterLandscapeFullscreen(el: HTMLElement) {
  try {
    if (!document.fullscreenElement && el.requestFullscreen) await el.requestFullscreen({ navigationUI: 'hide' });
    const o = screen.orientation as ScreenOrientation & { lock?: (o: string) => Promise<void> };
    await o.lock?.('landscape').catch(() => undefined);
  } catch {
    /* ignore */
  }
}

export async function exitFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    (screen.orientation as ScreenOrientation & { unlock?: () => void }).unlock?.();
  } catch {
    /* ignore */
  }
}
