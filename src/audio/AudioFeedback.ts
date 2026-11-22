/**
 * Hands-free audio: synthesised cues (Web Audio API, no audio files) and spoken
 * announcements (Speech Synthesis). Announcements are queued, never overlap, and
 * low-priority lines are dropped when the queue is busy to avoid audio spam.
 */
export type Cue = 'make' | 'miss' | 'uncertain' | 'milestone' | 'start' | 'pause' | 'alert';

interface Utterance {
  text: string;
  priority: number;
}

export class AudioFeedback {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  volume = 0.8;
  muted = false;
  voice = true;
  private queue: Utterance[] = [];
  private speaking = false;
  private lastSpokenAt = 0;
  private lastCueAt = 0;
  private voiceRef: SpeechSynthesisVoice | null = null;

  /** Must be called from a user gesture (iOS/Safari autoplay policy). */
  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AC) return;
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.connect(this.ctx.destination);
    }
    void this.ctx.resume();
    this.applyVolume();
    if ('speechSynthesis' in window) {
      const pick = () => {
        const voices = speechSynthesis.getVoices();
        this.voiceRef = voices.find((v) => /en[-_](US|GB|AU)/i.test(v.lang) && /natural|enhanced|premium|google/i.test(v.name)) ?? voices.find((v) => v.lang.startsWith('en')) ?? null;
      };
      pick();
      speechSynthesis.onvoiceschanged = pick;
      // iOS needs a (silent) utterance inside the gesture to enable speech later
      const u = new SpeechSynthesisUtterance(' ');
      u.volume = 0;
      speechSynthesis.speak(u);
    }
  }

  configure(o: { volume?: number; muted?: boolean; voice?: boolean }) {
    if (o.volume !== undefined) this.volume = o.volume;
    if (o.muted !== undefined) this.muted = o.muted;
    if (o.voice !== undefined) this.voice = o.voice;
    this.applyVolume();
    if (this.muted || !this.voice) this.clearSpeech();
  }

  private applyVolume() {
    if (this.master && this.ctx) this.master.gain.setValueAtTime(this.muted ? 0 : this.volume, this.ctx.currentTime);
  }

  private tone(freq: number, start: number, dur: number, type: OscillatorType, peak: number, endFreq?: number) {
    const ctx = this.ctx!;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, start);
    if (endFreq) osc.frequency.exponentialRampToValueAtTime(endFreq, start + dur);
    g.gain.setValueAtTime(0.0001, start);
    g.gain.exponentialRampToValueAtTime(peak, start + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    osc.connect(g).connect(this.master!);
    osc.start(start);
    osc.stop(start + dur + 0.02);
  }

  private noise(start: number, dur: number, peak: number, freq: number) {
    const ctx = this.ctx!;
    const len = Math.floor(ctx.sampleRate * dur);
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass';
    f.frequency.value = freq;
    f.Q.value = 0.8;
    const g = ctx.createGain();
    g.gain.setValueAtTime(peak, start);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    src.connect(f).connect(g).connect(this.master!);
    src.start(start);
  }

  cue(kind: Cue) {
    if (!this.ctx || !this.master || this.muted) return;
    const now = this.ctx.currentTime;
    // rate limit identical rapid cues
    if (performance.now() - this.lastCueAt < 120) return;
    this.lastCueAt = performance.now();
    switch (kind) {
      case 'make':
        this.noise(now, 0.18, 0.25, 2500); // net swish
        this.tone(988, now + 0.02, 0.18, 'sine', 0.5);
        this.tone(1318, now + 0.12, 0.32, 'sine', 0.45);
        this.tone(1976, now + 0.12, 0.25, 'sine', 0.12);
        break;
      case 'miss':
        this.tone(196, now, 0.22, 'triangle', 0.35, 150);
        break;
      case 'uncertain':
        this.tone(660, now, 0.12, 'sine', 0.25);
        this.tone(660, now + 0.16, 0.12, 'sine', 0.25);
        break;
      case 'milestone':
        [784, 988, 1175, 1568].forEach((f, i) => this.tone(f, now + i * 0.09, 0.3, 'sine', 0.35));
        break;
      case 'start':
        this.tone(523, now, 0.15, 'sine', 0.35);
        this.tone(784, now + 0.15, 0.25, 'sine', 0.35);
        break;
      case 'pause':
        this.tone(784, now, 0.15, 'sine', 0.3);
        this.tone(523, now + 0.15, 0.25, 'sine', 0.3);
        break;
      case 'alert':
        this.tone(880, now, 0.15, 'square', 0.15);
        this.tone(880, now + 0.25, 0.15, 'square', 0.15);
        break;
    }
  }

  /**
   * Queue a spoken line. Priority 0 = routine (score), 1 = milestone, 2 = critical.
   * Routine lines replace older queued routine lines instead of piling up.
   */
  say(text: string, priority = 0) {
    if (!this.voice || this.muted || !('speechSynthesis' in window)) return;
    if (priority === 0) this.queue = this.queue.filter((q) => q.priority > 0);
    this.queue.push({ text, priority });
    this.queue.sort((a, b) => b.priority - a.priority);
    if (this.queue.length > 3) this.queue.length = 3;
    this.pump();
  }

  private pump() {
    if (this.speaking || this.queue.length === 0) return;
    const gap = performance.now() - this.lastSpokenAt;
    if (gap < 350) {
      setTimeout(() => this.pump(), 350 - gap);
      return;
    }
    const next = this.queue.shift()!;
    const u = new SpeechSynthesisUtterance(next.text);
    u.rate = 1.08;
    u.volume = Math.min(1, this.volume * 1.1);
    if (this.voiceRef) u.voice = this.voiceRef;
    this.speaking = true;
    const done = () => {
      this.speaking = false;
      this.lastSpokenAt = performance.now();
      this.pump();
    };
    u.onend = done;
    u.onerror = done;
    // safety net: some engines never fire onend
    setTimeout(() => this.speaking && done(), 4000);
    speechSynthesis.speak(u);
  }

  clearSpeech() {
    this.queue = [];
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    this.speaking = false;
  }

  dispose() {
    this.clearSpeech();
    void this.ctx?.close();
    this.ctx = null;
    this.master = null;
  }
}

const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];

export function numberWords(n: number): string {
  if (n <= 20) return WORDS[n] ?? String(n);
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? `-${WORDS[n % 10]}` : '');
  if (n < 1000) return `${WORDS[Math.floor(n / 100)]} hundred${n % 100 ? ` ${numberWords(n % 100)}` : ''}`;
  return String(n);
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * Milestones reached by the latest shot. `lifetimeBestStreak` is the best streak from previous
 * sessions; a new personal best is announced once per session.
 */
export function milestonesFor(
  prev: { makes: number; currentStreak: number },
  next: { makes: number; currentStreak: number; longestStreak: number },
  lifetimeBestStreak: number,
  personalBestAnnounced: boolean,
): { messages: string[]; personalBest: boolean } {
  const messages: string[] = [];
  let personalBest = false;
  if (next.currentStreak > prev.currentStreak) {
    const s = next.currentStreak;
    if (s === 3 || s === 5 || (s >= 10 && s % 5 === 0)) messages.push(`${cap(numberWords(s))} in a row!`);
    if (!personalBestAnnounced && lifetimeBestStreak > 0 && s > lifetimeBestStreak) {
      messages.push('New personal best!');
      personalBest = true;
    }
  }
  if (next.makes > prev.makes && next.makes % 25 === 0) messages.push(`${cap(numberWords(next.makes))} shots made!`);
  return { messages, personalBest };
}
