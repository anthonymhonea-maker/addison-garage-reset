# One-time helper: synthesizes the background music loop (music.wav).
import numpy as np, wave
sr = 44100
bpm = 84
beat = 60 / bpm
bar = 4 * beat
bars = 8
N = int(round(bars * bar * sr))
L = np.zeros(N); R = np.zeros(N)

def hz(m): return 440.0 * 2 ** ((m - 69) / 12)

def add(buf, t0, sig, gain=1.0):
    i = int(round(t0 * sr))
    if i >= N: return
    j = min(N, i + len(sig))
    buf[i:j] += gain * sig[: j - i]

def pad(m, dur):
    n = int((dur + 0.9) * sr); t = np.arange(n) / sr
    s = np.zeros(n)
    for cents in (-4, 4):
        f = hz(m) * 2 ** (cents / 1200)
        s += np.sin(2*np.pi*f*t) + 0.28*np.sin(2*np.pi*2*f*t) + 0.08*np.sin(2*np.pi*3*f*t)
    env = np.minimum(1, t / 0.45) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.3))
    return s * env * (1 + 0.05*np.sin(2*np.pi*0.25*t))

def pluck(m):
    n = int(1.3 * sr); t = np.arange(n) / sr
    f = hz(m)
    s = np.sin(2*np.pi*f*t) + 0.35*np.sin(2*np.pi*2*f*t)*np.exp(-t/0.12) + 0.12*np.sin(2*np.pi*3*f*t)*np.exp(-t/0.06)
    return s * np.exp(-t / 0.28) * np.minimum(1, t / 0.004)

def bass(m, dur):
    n = int(dur * sr); t = np.arange(n) / sr
    f = hz(m)
    s = np.sin(2*np.pi*f*t) + 0.2*np.sin(2*np.pi*2*f*t)
    return s * np.minimum(1, t/0.02) * np.exp(-t / (dur*0.9))

chords = [
    (36, [60, 64, 67, 71]),
    (33, [57, 60, 64, 67]),
    (41, [53, 57, 60, 64]),
    (43, [55, 59, 62, 64]),
]
patterns = [[0, 1, 2, 3, 2, 1, 2, 1], [0, 2, 1, 3, 2, 3, 1, 2]]
last = bars - 1
for b in range(bars):
    root, tones = chords[b % 4]
    t0 = b * bar
    ending = (b == last)
    pdur = bar - (1.6 if ending else 0.1)
    for k, m in enumerate(tones):
        sig = pad(m, pdur)
        add(L, t0, sig, 0.055 * (1.1 if k % 2 == 0 else 0.8))
        add(R, t0, sig, 0.055 * (0.8 if k % 2 == 0 else 1.1))
    bs = bass(root, beat * 1.8)
    for off in ((0,) if ending else (0, 2.5)):
        add(L, t0 + off*beat, bs, 0.26); add(R, t0 + off*beat, bs, 0.26)
    pat = patterns[(b // 4) % 2]
    steps = 4 if ending else 8
    for s_ in range(steps):
        m = tones[pat[s_]] + 12
        sig = pluck(m)
        t = t0 + s_ * beat / 2
        vel = 0.17 if s_ % 2 == 0 else 0.12
        pan = 0.35 if s_ % 2 == 0 else 0.65
        add(L, t, sig, vel * (1 - pan) * 2); add(R, t, sig, vel * pan * 2)
        add(R, t + 0.75*beat, sig, vel * 0.32); add(L, t + 1.5*beat, sig, vel * 0.14)

mix = np.stack([L, R], axis=1)
a = np.exp(-2*np.pi*6500/sr)
for c in range(2):
    y = np.empty(N); prev = 0.0
    x = mix[:, c]
    for i in range(N):
        prev = (1 - a) * x[i] + a * prev; y[i] = prev
    mix[:, c] = y
fade = int(0.05 * sr)
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix[:int(0.005*sr)] *= np.linspace(0, 1, int(0.005*sr))[:, None]
mix = np.tanh(mix * 1.4)
mix *= 0.7 / np.abs(mix).max()
w = wave.open("music.wav", "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
w.writeframes((mix * 32767).astype("<i2").tobytes()); w.close()
