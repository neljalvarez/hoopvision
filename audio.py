"""Short, distinct sounds for makes, misses and uncertain shots (no audio files needed).

Windows: winsound plays generated WAV files asynchronously. Other systems: afplay / paplay /
aplay if available, otherwise silent. Sounds never block the processing loop."""
from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np

RATE = 22050


def _tone(freqs, dur, vol, decay=6.0):
    t = np.linspace(0, dur, int(RATE * dur), endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    env = np.minimum(1, t / 0.01) * np.exp(-decay * t)
    return sig * env * vol


def make_sound(vol: float) -> np.ndarray:
    return np.concatenate([_tone([880, 1320], 0.12, vol), _tone([1175, 1760], 0.28, vol, 5)])


def miss_sound(vol: float) -> np.ndarray:
    return _tone([220, 165], 0.22, vol * 0.7, 9)


def uncertain_sound(vol: float) -> np.ndarray:
    return np.concatenate([_tone([660], 0.09, vol * 0.6, 12), np.zeros(int(RATE * 0.06)), _tone([660], 0.09, vol * 0.6, 12)])


def wav_bytes(sig: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class Audio:
    def __init__(self, enabled: bool = True, volume: float = 0.6):
        self.enabled = enabled
        self.dir = tempfile.mkdtemp(prefix="hoopvision_audio_")
        self.files = {}
        for name, fn in (("make", make_sound), ("miss", miss_sound), ("uncertain", uncertain_sound)):
            path = os.path.join(self.dir, f"{name}.wav")
            with open(path, "wb") as f:
                f.write(wav_bytes(fn(volume)))
            self.files[name] = path
        self.system = platform.system()
        self.player = None
        if self.system != "Windows":
            for cmd in ("afplay", "paplay", "aplay"):
                if shutil.which(cmd):
                    self.player = cmd
                    break

    def play(self, name: str) -> None:
        if not self.enabled or name not in self.files:
            return
        path = self.files[name]
        if self.system == "Windows":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        elif self.player:
            args = [self.player, path] if self.player != "aplay" else ["aplay", "-q", path]
            threading.Thread(target=lambda: subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), daemon=True).start()
