import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np
from PIL import Image
W, H = 1024, 512
rng = np.random.default_rng(11)
def field(beta, seed):
    r = np.random.default_rng(seed)
    kx = np.fft.fftfreq(W)[None, :] * W; ky = np.fft.fftfreq(H)[:, None] * H
    k = np.sqrt(kx ** 2 + ky ** 2); k[0, 0] = 1
    spec = (r.normal(size=(H, W)) + 1j * r.normal(size=(H, W))) / k ** beta; spec[0, 0] = 0
    f = np.real(np.fft.ifft2(spec)); return (f - f.mean()) / f.std()
def hexc(h): return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
def ramp(t, stops):
    t = np.clip(t, 0, 1); out = np.zeros(t.shape + (3,))
    xs = [s for s, _ in stops]; cs = [hexc(c) for _, c in stops]
    for ch in range(3): out[..., ch] = np.interp(t, xs, [c[ch] for c in cs])
    return out
yy, xx = np.mgrid[0:H, 0:W] / H
def bands(freq, warp, seed, tilt=0.15, beta=1.6):
    w = field(beta, seed) * warp + field(2.2, seed + 1) * warp * 0.5
    return 0.5 + 0.5 * np.sin(2 * np.pi * (yy * freq + xx * tilt * freq) + w)
def veins(seed, freq=3.0, width=0.035, warp=2.5, tilt=0.37):
    w = field(1.8, seed) * warp
    v = np.abs(np.sin(2 * np.pi * (xx * freq * tilt + yy * freq) + w))
    return np.clip(1 - v / width, 0, 1) ** 1.5
def patches(seed, thr, soft, beta=1.8): return np.clip((field(beta, seed) - thr) / soft, 0, 1)
def over(c, mask, col, a=1.0): m = (mask * a)[..., None]; return c * (1 - m) + hexc(col) * m
def desat(c, keep):
    L = (c * [0.299, 0.587, 0.114]).sum(-1, keepdims=True); return L + (c - L) * keep
S = {}
# red jasper: brick/brown-red matrix in wavy bands, grey chalcedony patches, white-grey quartz veins
b = bands(6, 3.0, 1)
c = ramp(b * 0.8 + 0.1 + field(1.4, 2) * 0.12, [(0, '#5e2219'), (0.45, '#8a3426'), (0.75, '#a2442f'), (1, '#b9634b')])
c = over(c, patches(30, 1.1, 0.6), '#8f8a85', 0.6); c = over(c, patches(40, 1.5, 0.5, 2.4), '#c9c3bb', 0.45)
c = over(c, veins(3, 2.2, 0.022), '#e2dcd2'); c = over(c, veins(31, 4.0, 0.012, 2.0), '#b8b2aa', 0.8)
S[0] = desat(c, 0.9)
# orange alabaster: cream-white translucent ground, muted tan-orange clouds and bands, pale grey veins
b = bands(3.2, 2.2, 4, beta=1.9)
c = ramp(b * 0.75 + 0.12 + field(2.0, 5) * 0.2, [(0, '#c06f2e'), (0.4, '#d98a44'), (0.7, '#e9b27a'), (1, '#f4dcbc')])
c = over(c, patches(32, 0.9, 0.8, 2.2), '#f4efe8', 0.6); c = over(c, patches(43, 1.3, 0.6), '#aaa49d', 0.35)
c = over(c, veins(6, 1.6, 0.03, 3.5), '#a8a29b', 0.45); c = over(c, veins(33, 3.5, 0.015, 2.5), '#e9e4dc', 0.7)
S[1] = desat(c, 0.9)
# yellow calcite: pale honey with milky-white crystalline bands and grey seams
b = bands(8, 2.0, 7, tilt=0.05)
c = ramp(b * 0.85 + 0.08 + field(1.3, 9) * 0.08, [(0, '#b08a2a'), (0.4, '#cfa63e'), (0.75, '#e2c262'), (1, '#f1e2a8')])
c = over(c, np.clip((b - 0.82) / 0.1, 0, 1), '#f3efe6', 0.75)
c = over(c, veins(34, 2.6, 0.02, 2.0, tilt=0.05), '#8f8b84', 0.6); c = over(c, patches(41, 1.2, 0.7), '#a9a49a', 0.4)
c = over(c, np.clip((field(0.6, 8) - 2.2) / 0.6, 0, 1), '#fbf7ec', 0.3)
S[2] = desat(c, 0.9)
# verdite: grey-green mottled bands, darker green seams, white/grey talc-quartz veinlets
mo = field(1.5, 10) * 0.6 + field(2.4, 11) * 0.4; sw = bands(5.0, 3.0, 12, tilt=0.25)
c = ramp(0.5 + 0.2 * mo + 0.35 * (sw - 0.5), [(0, '#1f3a22'), (0.35, '#2f5a33'), (0.65, '#4a7a48'), (1, '#7a9e6c')])
c = over(c, patches(35, 1.0, 0.7), '#8d928b', 0.55)
c = over(c, veins(36, 3.0, 0.02, 3.0), '#d9dcd6', 0.85); c = over(c, veins(37, 5.5, 0.01, 2.0), '#a7aca4', 0.7)
S[3] = desat(c, 0.9)
# sodalite: greyish blue, white calcite veins and grey-white patches
mo = field(1.7, 14)
c = ramp(0.5 + 0.22 * mo, [(0, '#1a2a5e'), (0.45, '#26408a'), (0.8, '#3656a2'), (1, '#5a73b0')])
c = over(c, patches(17, 0.9, 0.8), '#8e939e', 0.55)
c = over(c, patches(38, 1.6, 0.4, 1.4), '#e4e5e8', 0.8)
c = over(c, veins(15, 2.8, 0.05, 3.0), '#dfe1e6'); c = over(c, veins(16, 5.0, 0.025, 2.0), '#b9bec8', 0.7)
S[4] = desat(c, 0.9)
# imperial porphyry: wine-red purple groundmass, dense small pale-pink and white feldspar crystals, no banding
c = ramp(0.5 + field(1.3, 18) * 0.10 + field(2.2, 19) * 0.08, [(0, '#4e1d27'), (0.5, '#6a2834'), (1, '#813644')])
c = c * (0.95 + 0.08 * field(0.3, 23)[..., None] * 0.5)
c = over(c, patches(42, 1.4, 0.7), '#5e4a4e', 0.35)
f1 = field(0.9, 20); f2 = field(0.9, 21); f3 = field(0.9, 24)
c = over(c, np.clip((f1 - 1.75) / 0.25, 0, 1), '#d9b4ae')        # pink plagioclase laths
c = over(c, np.clip((f2 - 2.05) / 0.25, 0, 1), '#efe4de')        # white crystals
c = over(c, np.clip((f3 - 2.2) / 0.3, 0, 1), '#9b8a8c', 0.8)     # grey specks
S[5] = desat(c, 0.92)
for k, c in S.items():
    g = np.clip(c + rng.normal(0, 0.01, c.shape), 0, 1)
    Image.fromarray((g * 255).astype(np.uint8)).save(f'{ROOT}/web/stone_{k}.jpg', quality=86)
strip = np.concatenate([np.asarray(Image.open(f'{ROOT}/web/stone_{k}.jpg').resize((512, 170))) for k in range(6)], 0)
Image.fromarray(strip).save(ROOT + '/data/stones_preview.jpg', quality=85)
