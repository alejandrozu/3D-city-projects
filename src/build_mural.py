import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Abstract flowing rainbow mural on the blank wall of 3970 17th St facing the plaza."""
import numpy as np, sys, json, pickle
sys.path.insert(0, ROOT + '/src')
from common import *
from shapes import Acc, srgb
from shapely.geometry import LineString, Point
from PIL import Image
S = pickle.load(open(ROOT + '/data/plaza_shape.pkl', 'rb'))
c0, d = S['cutline']; d = d / np.linalg.norm(d)
tri = S['tri']; PL = S['plaza']
cen = np.array(PL.centroid.coords[0]); n_in = cen - (c0 + d * np.dot(cen - c0, d)); n_in /= np.linalg.norm(n_in)   # wall normal toward the plaza
# the building wall(s) of 3970 17th St that face the plaza
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
Bs = {p['sf16_bldgid']: orient(proj(shape(p['shape'])).buffer(0), 1.0) for p in json.load(open(ROOT + '/data/bldg_sf.json'))}
g = Bs['201006.0007773']
if g.geom_type != 'Polygon': g = max(g.geoms, key=lambda q: q.area)
ring = np.array(g.exterior.coords)
TRI = tri.buffer(1.5)
segs = []
for a_, b_ in zip(ring[:-1], ring[1:]):
    e = b_ - a_; Le = np.linalg.norm(e)
    if Le < 0.3: continue
    nout = np.array([e[1], -e[0]]) / Le            # CCW polygon: outward normal on the right
    m = (a_ + b_) / 2
    if np.dot(nout, n_in) > 0.85 and TRI.contains(Point(*(m + nout * 1.0))):
        segs.append((a_, b_, nout))
print('mural wall segments', len(segs), 'total %.1f m' % sum(np.linalg.norm(b_ - a_) for a_, b_, _ in segs))
# chain the segments in order along the wall direction
segs.sort(key=lambda s_: np.dot((s_[0] + s_[1]) / 2 - c0, d))
V = []; F = []; UV = []; s_acc = 0.0; rows = []
for a_, b_, nout in segs:
    Le = np.linalg.norm(b_ - a_); n = max(2, int(Le / 0.5) + 1)
    for k, t in enumerate(np.linspace(0, 1, n)):
        p = a_ + (b_ - a_) * t + nout * 0.035
        rows.append((p, s_acc + Le * t, k == 0))
    s_acc += Le
Lw = s_acc
P = np.array([r[0] for r in rows]); S_ = np.array([r[1] for r in rows]); newseg = [r[2] for r in rows]
gnd = ground(P[:, 0], P[:, 1])
roof = sample(DSM_ROOF, P[:, 0] - n_in[0] * 1.2, P[:, 1] - n_in[1] * 1.2, order=0)
roof = np.where(np.isfinite(roof), roof, gnd + 9.0)
roof = ndimage.median_filter(roof, 5); roof = np.maximum(roof, gnd + 4.0) - 0.15
zb = gnd - 0.05
H = float((roof - zb).max())
print('mural wall %.1f m long, up to %.1f m tall' % (Lw, H))
for i in range(len(P)):
    V += [[P[i, 0], P[i, 1], zb[i]], [P[i, 0], P[i, 1], roof[i]]]
    UV += [[S_[i] / Lw, (zb[i] - zb.min()) / H], [S_[i] / Lw, (roof[i] - zb.min()) / H]]
    if i and not newseg[i]:
        a = 2 * (i - 1); F += [[a, a + 2, a + 3], [a, a + 3, a + 1]]
V = np.array(V); F = np.array(F); UV = np.array(UV)
nrm = np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]])
if np.dot(nrm[:2], n_in) < 0: F = F[:, ::-1]
UV[:, 1] = 1 - UV[:, 1]
pickle.dump(dict(V=V, F=F, UV=UV, length=Lw, height=H), open(ROOT + '/data/mural.pkl', 'wb'))

# ---------------- mural texture: flowing rainbow ribbons
TW = 2560; TH = int(TW * H / Lw) // 8 * 8
x = np.linspace(0, Lw, TW)[None, :]; y = np.linspace(H, 0, TH)[:, None]
X = np.broadcast_to(x, (TH, TW)); Y = np.broadcast_to(y, (TH, TW))
rng = np.random.default_rng(1978)
def wave_noise(X, Y, octaves, seed):
    r = np.random.default_rng(seed); out = np.zeros_like(X, dtype=np.float64)
    for k in range(octaves):
        f = 0.035 * 1.8 ** k; a = 1.0 / 1.9 ** k
        th = r.uniform(0, np.pi); ph = r.uniform(0, 6.3)
        out += a * np.sin(f * 2 * np.pi * (X * np.cos(th) + Y * np.sin(th)) + ph)
    return out
# domain warping for a liquid, flowing feel
q1 = wave_noise(X, Y, 3, 1); q2 = wave_noise(X, Y, 3, 2)
Xw = X + 3.0 * q1; Yw = Y + 1.2 * q2
r1 = wave_noise(Xw * 0.8, Yw * 0.8, 3, 3)
field = 0.25 * Yw + 0.035 * Xw + 0.22 * r1 + 0.18 * np.sin(2 * np.pi * X / 19.0)   # broad ribbons undulating along the wall
PRIDE = ['#e40303', '#ff8c00', '#ffed00', '#008026', '#004dff', '#750787']
cols = np.array([srgb(c) for c in PRIDE]) ** (1 / 2.2)
period = 1.0; nb = 6; gapf = 0.38                       # each ribbon = 6 colour bands, then a gap of background
ph = (field / period) % 1.0
inrib = ph < (1 - gapf)
band_f = np.clip(ph / (1 - gapf), 0, 0.99999) * nb
bi = np.floor(band_f).astype(int); fr = band_f - bi
# soft blend between neighbouring bands
c_a = cols[bi]; c_b = cols[np.minimum(bi + 1, nb - 1)]
blend = np.clip((fr - 0.82) / 0.18, 0, 1)[..., None]
ribbon = c_a * (1 - blend) + c_b * blend
# shading across the ribbon (rounded, painted feel) and highlights at the edges
across = np.clip(ph / (1 - gapf), 0, 1)
shade = 0.86 + 0.22 * np.sin(np.pi * across)
ribbon = ribbon * shade[..., None]
gx, gy = np.gradient(field); gm = np.hypot(gx, gy) + 1e-9
hl = np.exp(-((ph - 0.004) / 0.012) ** 2) + np.exp(-((ph - (1 - gapf) + 0.004) / 0.012) ** 2)
# background: warm off-white with faint brush swirls
bg = np.array([0.96, 0.93, 0.87])[None, None, :] * (0.95 + 0.05 * wave_noise(X * 3, Y * 3, 3, 9)[..., None] / 2)
img = np.where(inrib[..., None], ribbon, bg)
img = img * (1 - 0.85 * hl[..., None]) + 0.85 * hl[..., None] * np.array([1.0, 1.0, 1.0])
# a few small floating dots in the gaps (paint drops)
dots = wave_noise(X * 9, Y * 9, 2, 11)
dm = (~inrib) & (dots > 9.0)
img[dm] = cols[(np.floor((X[dm] * 0.7 + Y[dm] * 0.3)) % 6).astype(int)] * 0.95
# painted brick/stucco texture
grain = rng.normal(0, 0.018, img.shape[:2])[..., None]
img = np.clip(img + grain, 0, 1)
Image.fromarray((img * 255).astype(np.uint8)).save(ROOT + '/web/mural.jpg', quality=87)
Image.fromarray((img * 255).astype(np.uint8)).resize((1024, max(64, int(1024 * TH / TW)))).save(ROOT + '/data/mural_preview.jpg', quality=85)
print('texture', TW, TH)
