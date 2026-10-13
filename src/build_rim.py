import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0, ROOT + '/src')
from common import *
from landcover import landcover
from osm_load import load
from shapely.geometry import LineString, Point, box as sbox, MultiLineString
from shapely.ops import linemerge, unary_union
from shapely import prepared
from PIL import Image
from shapes import srgb
L = landcover(); SW = L['sidewalk']; RD = L['roadway']
SWp = prepared.prep(SW)
shared = SW.boundary.intersection(RD.buffer(0.08))
shared = linemerge([g for g in getattr(shared, 'geoms', [shared]) if g.geom_type == 'LineString' and g.length > 0.05]) if shared.geom_type != 'LineString' else shared
lines = [g for g in getattr(shared, 'geoms', [shared]) if g.length > 1.0]
# exclude Castro Street north of Market
nodes, ntags, ways, rels = load()
cn = [LineString([nodes[n] for n in w[0] if n in nodes]) for wid, w in ways.items() if w[1].get('name') == 'Castro Street' and 'highway' in w[1]]
cn = unary_union(cn).intersection(sbox(-60, 12, 60, 200))
EXCL = prepared.prep(cn.buffer(13.5).intersection(sbox(-60, 8, 60, 200)))
def keep_pt(p): return not EXCL.contains(Point(*p))
RIM_W = 0.90; STEP = 0.3
V = []; F = []; UV = []; tot = 0
for ln in lines:
    n = max(2, int(ln.length / STEP) + 1)
    pts = np.array([ln.interpolate(t).coords[0] for t in np.linspace(0, ln.length, n)])
    tang = np.gradient(pts, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    sgn = np.array([1 if SWp.contains(Point(*(p + q * 0.45))) else (-1 if SWp.contains(Point(*(p - q * 0.45))) else 0) for p, q in zip(pts, nrm)])
    sgn_s = np.sign(ndimage.uniform_filter1d(sgn.astype(float), 9)); sgn_s[sgn_s == 0] = 1
    nrm = nrm * sgn_s[:, None]
    nrm = ndimage.uniform_filter1d(nrm, 5, axis=0); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    ok = np.array([keep_pt(p) and (X0 <= p[0] <= X1) and (Y0 <= p[1] <= Y1) and s != 0 for p, s in zip(pts, sgn)])
    s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    i = 0
    while i < len(pts):
        if not ok[i]: i += 1; continue
        j = i
        while j < len(pts) and ok[j]: j += 1
        if j - i >= 3:
            P0 = pts[i:j] + nrm[i:j] * 0.02; P1 = pts[i:j] + nrm[i:j] * RIM_W
            P1[:, 0] = np.clip(P1[:, 0], X0, X1); P1[:, 1] = np.clip(P1[:, 1], Y0, Y1)
            z0 = ground(P0[:, 0], P0[:, 1]) + 0.025; z1 = ground(P1[:, 0], P1[:, 1]) + 0.025
            b = len(V)
            for k in range(j - i):
                V += [[P0[k, 0], P0[k, 1], z0[k]], [P1[k, 0], P1[k, 1], z1[k]]]
                UV += [[s_[i + k] / 2.4, 0.0], [s_[i + k] / 2.4, 1.0]]
                if k:
                    a = b + 2 * (k - 1)
                    F += [[a, a + 2, a + 3], [a, a + 3, a + 1]]
            tot += s_[j - 1] - s_[i]
        i = j
V = np.array(V); F = np.array(F); UV = np.array(UV)
# faces up
nz = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 2]
F[nz < 0] = F[nz < 0][:, ::-1]
pickle.dump(dict(V=V, F=F, UV=UV), open(ROOT + '/data/rim.pkl', 'wb'))
print('rim length %.0f m, %d triangles' % (tot, len(F)))
# ---------------- brick texture: 2.4 m x 0.9 m, 8 bricks of 30 cm per course, 6 courses of 15 cm, half-offset bond
TWp, THp = 1024, 384
PRIDE = ['#e40303', '#ff8c00', '#ffed00', '#008026', '#004dff', '#750787']
rng = np.random.default_rng(5)
img = np.zeros((THp, TWp, 3))
mortar = np.array([0.80, 0.78, 0.74])
for r in range(6):
    base = srgb(PRIDE[r]) ** (1 / 2.2)
    base = 0.82 * base + 0.18 * np.array([0.62, 0.55, 0.50])          # fired-clay toning
    y0 = r * 64; off = 64 if r % 2 else 0
    for b in range(9):
        x0 = b * 128 - off
        v = rng.uniform(0.84, 1.08); tint = rng.normal(0, 0.025, 3)
        xs = np.arange(max(x0, 0), min(x0 + 128, TWp)); xs_w = np.arange(x0, x0 + 128) % TWp
        blk = np.clip(base * v + tint, 0, 1)
        yy, xx = np.mgrid[0:64, 0:128]
        edge = np.minimum.reduce([xx, 127 - xx, yy, 63 - yy]).astype(float)
        sh = 0.82 + 0.18 * np.clip(edge / 6, 0, 1)
        tile = blk[None, None, :] * sh[..., None] * (1 + rng.normal(0, 0.03, (64, 128, 1)))
        img[y0:y0 + 64, xs_w] = tile
# mortar joints
for r in range(7):
    y = min(r * 64, THp - 1); img[max(y - 2, 0):y + 2] = mortar
for r in range(6):
    off = 64 if r % 2 else 0
    for b in range(9):
        x = (b * 128 - off) % TWp
        for dx in (-2, -1, 0, 1):
            img[r * 64:(r + 1) * 64, (x + dx) % TWp] = mortar
img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1)
Image.fromarray((img * 255).astype(np.uint8)).save(ROOT + '/web/brick_rim.jpg', quality=88)
