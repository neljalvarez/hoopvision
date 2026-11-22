import type { CameraStatus } from '../camera/useCamera';
import { IconCamera } from './icons';

export function CameraStatusPanel({ status, error, onRetry }: { status: CameraStatus; error: string | null; onRetry: () => void }) {
  if (status === 'live') return null;
  const title =
    status === 'requesting'
      ? 'Starting camera…'
      : status === 'denied'
        ? 'Camera permission needed'
        : status === 'disconnected'
          ? 'Camera disconnected'
          : status === 'unavailable'
            ? 'Camera unavailable'
            : status === 'error'
              ? 'Camera error'
              : 'Camera off';
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center bg-coal-950/90 p-6">
      <div className="max-w-sm text-center">
        <IconCamera size={40} className="mx-auto text-ember-500" />
        <div className="mt-3 font-display text-2xl font-700 uppercase">{title}</div>
        {error && <p className="mt-2 text-sm text-chalk-300">{error}</p>}
        {status === 'disconnected' && <p className="mt-2 text-sm text-chalk-300">Automatic counting is paused. Reconnect or restart the camera to continue.</p>}
        {status !== 'requesting' && (
          <button className="btn-primary mt-5" onClick={onRetry}>
            {status === 'idle' ? 'Start camera' : 'Try again'}
          </button>
        )}
        {status === 'denied' && (
          <p className="mt-4 text-xs text-chalk-400">iPhone: Settings → Safari → Camera → Allow. Android Chrome: tap the lock icon in the address bar → Permissions → Camera.</p>
        )}
      </div>
    </div>
  );
}
