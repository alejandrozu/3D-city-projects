import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Model-wide detail: vehicles from LiDAR, traffic signals, tree grates, hydrant, mailbox."""
import numpy as np, sys, json, pickle
sys.path.insert(0, ROOT + '/src')
from common import *
from shapes import Acc, srgb, box, cyl, blob, prism
from osm_load import load
from shapely.geometry import Point
rng = np.random.default_rng(7)
S = pickle.load(open(ROOT + '/data/plaza_shape.pkl', 'rb'))
from shapely import prepared
CLEAR = prepared.prep(S['plaza'].buffer(0.3).union(S['chev']))
OUT = {}
def acc(k):
    if k not in OUT: OUT[k] = Acc()
    return OUT[k]
def gz(x, y): return float(ground(np.array([x]), np.array([y]))[0])
def place(V, x, y, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return V @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [x, y, z]

# ---------------------------------------------------------------- vehicles
PAL = [('#f2f2ef', .25), ('#1d1f22', .19), ('#b9bcc0', .17), ('#6c7075', .14), ('#2c4a7a', .08), ('#8e1c1c', .07), ('#3e5a3e', .03), ('#c9b48a', .03), ('#e6e2d6', .04)]
def pick_color():
    r = rng.random(); a = 0
    for c, p in PAL:
        a += p
        if r < a: return c
    return PAL[0][0]
def vehicle(kind, L, W, H):
    """returns dict mat -> (V,F) in vehicle frame: +X forward, z up from ground"""
    T = {}
    def add(k, V, F, col): T.setdefault(k, Acc()).add(V, F, col)
    col = srgb(pick_color()); glass = srgb('#1b242c')
    wr = {'car': 0.31, 'suv': 0.36, 'van': 0.37, 'bus': 0.5}[kind]
    hx = L / 2
    if kind == 'car':
        body = [(-hx, 0.30), (-hx, 0.72), (-hx * 0.78, 0.80), (hx * 0.62, 0.78), (hx * 0.96, 0.66), (hx, 0.50), (hx, 0.30)]
        cab = [(-hx * 0.62, 0.79), (-hx * 0.42, H), (hx * 0.12, H), (hx * 0.46, 0.79)]
    elif kind == 'suv':
        body = [(-hx, 0.36), (-hx, 0.95), (hx * 0.70, 0.95), (hx * 0.97, 0.82), (hx, 0.55), (hx, 0.36)]
        cab = [(-hx * 0.96, 0.95), (-hx * 0.90, H), (hx * 0.30, H), (hx * 0.62, 0.95)]
    else:  # van / truck
        body = [(-hx, 0.38), (-hx, H * 0.55), (hx * 0.80, H * 0.55), (hx * 0.98, H * 0.45), (hx, 0.55), (hx, 0.38)]
        cab = [(-hx * 0.98, H * 0.55), (-hx * 0.98, H), (hx * 0.62, H), (hx * 0.86, H * 0.55)]
    def side_extrude(prof, w, mat, c, inset=0.0):
        V, F = prism(np.array(prof), -w / 2, w / 2)          # profile in (x,z), extruded along y
        V = np.column_stack([V[:, 0], V[:, 2], V[:, 1]])
        if inset: V[:, 1] *= (1 - inset)
        add(mat, V, F, c)
    side_extrude(body, W * 0.98, 'carpaint', col)
    side_extrude(cab, W * 0.86, 'carglass', glass)
    # roof panel over the cabin in body colour
    xs = [p[0] for p in cab]; rx0, rx1 = min(x for x, z in cab if z >= H - 1e-6), max(x for x, z in cab if z >= H - 1e-6)
    Vr, Fr = box(((rx0 + rx1) / 2, 0, H + 0.015), (rx1 - rx0 + 0.04, W * 0.84, 0.03)); add('carpaint', Vr, Fr, col)
    # pillars (B pillar) in body colour
    Vp, Fp = box((-(hx * 0.15), 0, (0.80 + H) / 2 if kind == 'car' else (H * 0.55 + H) / 2), (0.08, W * 0.87, (H - 0.8) if kind == 'car' else H * 0.45)); add('carpaint', Vp, Fp, col)
    # wheels with hubcaps
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx = sx * (hx - max(0.75, L * 0.17)); cy = sy * (W / 2 - 0.11)
            V, F = cyl((cx, cy - 0.11, wr), (cx, cy + 0.11, wr), wr, seg=14); add('tyre', V, F, srgb('#1a1a1a'))
            V, F = cyl((cx, cy + sy * 0.111, wr), (cx, cy + sy * 0.118, wr), wr * 0.58, seg=12); add('chrome', V, F, srgb('#b0b4b8'))
    # lights + plate + bumpers
    for sy in (-1, 1):
        V, F = box((hx + 0.005, sy * W * 0.36, 0.62 if kind != 'van' else 0.75), (0.02, 0.24, 0.10)); add('headlight', V, F, srgb('#f4f1e6'))
        V, F = box((-hx - 0.005, sy * W * 0.38, 0.70 if kind != 'van' else 0.85), (0.02, 0.18, 0.12)); add('taillight', V, F, srgb('#a8141b'))
    V, F = box((hx + 0.02, 0, 0.36), (0.06, W * 0.95, 0.14)); add('trim', V, F, srgb('#2a2c2f'))
    V, F = box((-hx - 0.02, 0, 0.38), (0.06, W * 0.95, 0.14)); add('trim', V, F, srgb('#2a2c2f'))
    V, F = box((-hx - 0.055, 0, 0.52), (0.01, 0.31, 0.15)); add('plate', V, F, srgb('#e9edf0'))
    # mirrors
    for sy in (-1, 1):
        V, F = box((hx * 0.40 if kind == 'car' else hx * 0.6, sy * (W / 2 + 0.07), 0.98 if kind == 'car' else H * 0.6), (0.12, 0.10, 0.09)); add('carpaint', V, F, col)
    return {k: v.get() for k, v in T.items()}
cars = pickle.load(open(ROOT + '/data/cars.pkl', 'rb'))
nveh = 0
for c in cars:
    if CLEAR.contains(Point(c['x'], c['y'])): continue
    kind = c['kind']; L = c['L']; W = c['W']; H = c['H']
    if kind == 'car': L = float(np.clip(L, 3.9, 5.0)); H = float(np.clip(H, 1.35, 1.55))
    elif kind == 'suv': L = float(np.clip(L, 4.2, 5.2)); H = float(np.clip(H, 1.62, 1.95))
    elif kind == 'van': L = float(np.clip(L, 4.8, 9.0)); H = float(np.clip(H, 1.9, 3.3))
    W = float(np.clip(W, 1.75, 2.2 if kind != 'van' else 2.45))
    g0 = gz(c['x'], c['y'])
    for k, (V, F, C_) in vehicle(kind, L, W, H).items():
        acc(k).add(place(V, c['x'], c['y'], g0, c['yaw']), F, C_)
    nveh += 1

# ---------------------------------------------------------------- traffic signals at the two intersections
poles = pickle.load(open(ROOT + '/data/poles.pkl', 'rb'))
C = np.load(ROOT + '/data/ground_cls.npy'); NC = C.shape[0]; PXC = (X1 - X0) / NC
def cls_at(x, y):
    i = int(np.clip((x - X0) / PXC, 0, NC - 1)); j = int(np.clip((Y1 - y) / PXC, 0, NC - 1)); return int(C[j, i])
INTS = [np.array([0.0, 0.0]), np.array([22.0, -12.0])]
nsig = 0
def signal_head(p, z, face, peds=False):
    """3-aspect vehicle head (or 2-aspect ped head) facing direction `face`"""
    yaw = np.arctan2(face[1], face[0])
    if not peds:
        V, F = box((0, 0, 0), (0.28, 0.36, 1.02)); acc('signal').add(place(V, *p, z, yaw), F, srgb('#22262a'))
        V, F = box((0, 0, 0), (0.03, 0.56, 1.22)); acc('signal').add(place(V + [-0.16, 0, 0], *p, z, yaw), F, srgb('#141618'))
        for k, (colr, lit) in enumerate((('#e8241c', True), ('#e7b416', False), ('#26c25a', False))):
            zz = 0.33 - k * 0.33
            V, F = cyl((0.141, 0, zz), (0.16, 0, zz), 0.10, seg=14); acc('lamp_on' if lit else 'lamp_off').add(place(V, *p, z, yaw), F, srgb(colr) if lit else srgb(colr) * 0.25)
            V, F = box((0.2, 0, zz + 0.10), (0.12, 0.24, 0.02)); acc('signal').add(place(V, *p, z, yaw), F, srgb('#22262a'))
    else:
        V, F = box((0, 0, 0), (0.22, 0.38, 0.62)); acc('signal').add(place(V, *p, z, yaw), F, srgb('#22262a'))
        V, F = box((0.115, 0, 0.13), (0.01, 0.28, 0.24)); acc('lamp_off').add(place(V, *p, z, yaw), F, srgb('#3a1e10'))
        V, F = box((0.115, 0, -0.14), (0.01, 0.28, 0.24)); acc('lamp_on').add(place(V, *p, z, yaw), F, srgb('#f2f2ee'))
for p in poles:
    x, y = p['x'], p['y']
    if CLEAR.contains(Point(x, y)) or cls_at(x, y) == 1: continue
    dmin = min(np.hypot(*(np.array([x, y]) - c)) for c in INTS)
    if dmin > 26 or p['h'] < 4.5: continue
    c = min(INTS, key=lambda c: np.hypot(*(np.array([x, y]) - c)))
    toc = c - [x, y]; toc /= np.linalg.norm(toc)
    g0 = p['g']
    signal_head((x + toc[0] * 0.25, y + toc[1] * 0.25), g0 + 3.4, toc)
    signal_head((x - toc[1] * 0.2, y + toc[0] * 0.2), g0 + 2.6, np.array([-toc[1], toc[0]]), peds=True)
    V, F = box((0, 0, 0), (0.18, 0.12, 0.25)); acc('signal').add(place(V, x + toc[0] * 0.12, y + toc[1] * 0.12, g0 + 1.05, np.arctan2(toc[1], toc[0])), F, srgb('#c9b21a'))  # push button
    nsig += 1

# ---------------------------------------------------------------- tree grates on sidewalks
trees = pickle.load(open(ROOT + '/data/trees.pkl', 'rb')); small = pickle.load(open(ROOT + '/data/small_trees.pkl', 'rb'))
curbs = [proj(shape(p['the_geom'])) for p in json.load(open(ROOT + '/data/curbs.json')) if p['layer'] == 'CURBS']
from shapely.ops import unary_union
CU = unary_union(curbs)
ngr = 0
for (x, y) in [(t['base'][0], t['base'][1]) for t in trees if t['species']] + [(s['x'], s['y']) for s in small]:
    if not (X0 < x < X1 and Y0 < y < Y1) or CLEAR.contains(Point(x, y)) or cls_at(x, y) != 2: continue
    q = CU.interpolate(CU.project(Point(x, y))) if CU.geom_type == 'LineString' else None
    best = min(getattr(CU, 'geoms', [CU]), key=lambda l: l.distance(Point(x, y)))
    t = best.project(Point(x, y)); a = best.interpolate(max(t - 0.5, 0)); b = best.interpolate(min(t + 0.5, best.length))
    yaw = np.arctan2(b.y - a.y, b.x - a.x)
    g0 = gz(x, y)
    V, F = box((0, 0, 0), (1.22, 1.22, 0.025)); acc('grate').add(place(V, x, y, g0 + 0.012, yaw), F, srgb('#2a2622'))
    for k in range(-4, 5):   # radial slots pattern as raised ribs
        V, F = box((k * 0.12, 0, 0), (0.035, 1.1, 0.012)); acc('grate').add(place(V, x, y, g0 + 0.028, yaw), F, srgb('#3a342e'))
    ngr += 1

# ---------------------------------------------------------------- hydrant + mailbox from OSM
nodes, ntags, ways, rels = load()
for k, tg in ntags.items():
    x, y = nodes[k]
    if not (X0 < x < X1 and Y0 < y < Y1) or CLEAR.contains(Point(x, y)): continue
    g0 = gz(x, y)
    if tg.get('emergency') == 'fire_hydrant':
        V, F = cyl((x, y, g0), (x, y, g0 + 0.62), 0.12, 0.11, seg=12); acc('hydrant').add(V, F, srgb('#e9e4d6'))
        V, F = blob((x, y, g0 + 0.66), (0.13, 0.13, 0.09), 1); acc('hydrant').add(V, F, srgb('#2a5aa8'))
        for a in (0, np.pi):
            V, F = cyl((x, y, g0 + 0.45), (x + np.cos(a) * 0.2, y + np.sin(a) * 0.2, g0 + 0.45), 0.05, seg=8); acc('hydrant').add(V, F, srgb('#e9e4d6'))
    if tg.get('amenity') == 'post_box':
        V, F = box((x, y, g0 + 0.55), (0.45, 0.52, 1.1)); acc('mailbox').add(V, F, srgb('#1f3d8a'))
        V, F = cyl((x - 0.225, y, g0 + 1.1), (x + 0.225, y, g0 + 1.1), 0.26, seg=16); acc('mailbox').add(V, F, srgb('#1f3d8a'))
pickle.dump({k: v.get() for k, v in OUT.items()}, open(ROOT + '/data/extras.pkl', 'wb'))
print('vehicles', nveh, 'signal poles', nsig, 'grates', ngr, {k: len(v.get()[1]) for k, v in OUT.items()})
