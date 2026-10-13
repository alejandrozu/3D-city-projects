import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Castro triangle plaza: 24 planters in pride order, 8 rainbow benches, 6 m marble fountain with the statue pair.
Everything in local world coords: x east, y north, z = NAVD88 metres."""
import numpy as np, sys, json, pickle
sys.path.insert(0, ROOT + '/src')
from common import *
from shapes import Acc, srgb, box, cyl, blob, tube, prism
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import substring
import trimesh, triangle as tr

rng = np.random.default_rng(1969)
S = pickle.load(open(ROOT + '/data/plaza_shape.pkl', 'rb'))
PL = S['plaza']; c0, dcut = S['cutline']
FLAG = np.array([-39.4, -38.8])
PRIDE = ['#e40303', '#ff8c00', '#ffed00', '#008026', '#004dff', '#750787']   # red .. purple
PRIDE_NAMES = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
def gz(x, y): return float(ground(np.array([x]), np.array([y]))[0])

# ------------------------------------------------------------------ layout
cen = np.array(PL.centroid.coords[0])
dn = dcut / np.linalg.norm(dcut)
wallpt = c0 + dn * np.dot(cen - c0, dn)
FC = cen + (wallpt - cen) * 0.28                       # fountain centre: centre of plaza, nudged toward the wall
print('fountain centre', FC, 'dist to wall %.1f, to edge %.1f' % (np.linalg.norm(wallpt - FC), PL.exterior.distance(Point(*FC))))

ring = np.array(PL.exterior.coords)[:-1]
# boundary classification
def wall_d(p): q = p - c0; return abs(dn[0] * q[1] - dn[1] * q[0])
wall = np.array([wall_d(p) < 1.2 for p in ring])
tip_i = np.argmin(ring @ (-(wallpt - cen) / np.linalg.norm(wallpt - cen)) * -1) if False else np.argmax(((ring - wallpt) ** 2).sum(1))
TIP = ring[tip_i]
print('tip', TIP)
# walk the ring from the tip in both directions until reaching the wall -> north / south side polylines
n = len(ring)
def walk(step):
    pts = []; i = tip_i
    while True:
        i = (i + step) % n
        if wall[i]:
            pts.append(ring[i]); break
        pts.append(ring[i])
        if len(pts) > n: break
    return np.array([TIP] + pts)
sideA = walk(1); sideB = walk(-1)
north, south = (sideA, sideB) if sideA[:, 1].mean() > sideB[:, 1].mean() else (sideB, sideA)
print('north side %.1f m, south side %.1f m' % (LineString(north).length, LineString(south).length))

TIP_OPEN = 7.5        # open point of the plaza (metres of boundary from the tip kept free)
GAP = 5.0             # opening in the middle of each side
CORNER = 0.0          # planters run right up to the wall
BOX_D = 0.70; BOX_H = 0.70; BOX_GAP = 0.10; EDGE = 0.0   # planter boxes: depth, height, joint, setback from plaza edge
def inward(p):
    q = Point(*p); b = PL.exterior.project(q); a = PL.exterior.interpolate(max(b - 0.3, 0)); c = PL.exterior.interpolate(min(b + 0.3, PL.exterior.length))
    t = np.array([c.x - a.x, c.y - a.y]); t /= np.linalg.norm(t); nrm = np.array([-t[1], t[0]])
    if not PL.contains(Point(*(p + nrm * 0.5))): nrm = -nrm
    return nrm
from shapely import prepared as _prep
PLI = _prep.prep(PL.buffer(0.02))
from shapely.geometry import Polygon as _Poly
WALL_HALF = _Poly([c0 - dn * 300, c0 + dn * 300, c0 + dn * 300 + (cen - wallpt) / np.linalg.norm(cen - wallpt) * 300, c0 - dn * 300 + (cen - wallpt) / np.linalg.norm(cen - wallpt) * 300])
stretches = []
for name, side in (('north', north), ('south', south)):
    ls = LineString(side); Ls = ls.length; mid = (TIP_OPEN + Ls - CORNER) / 2
    for (a, b, red_end) in ((TIP_OPEN, mid - GAP / 2, 'start'), (mid + GAP / 2, Ls - CORNER, 'end')):
        Lb = ((b - a) - 5 * BOX_GAP) / 6
        boxes = []
        for k in range(6):
            s0 = a + k * (Lb + BOX_GAP); s1 = s0 + Lb
            p0 = np.array(ls.interpolate(s0).coords[0]); p1 = np.array(ls.interpolate(s1).coords[0])
            pm = np.array(ls.interpolate((s0 + s1) / 2).coords[0])
            t = p1 - p0; L = np.linalg.norm(t); t /= L
            nr = np.array([-t[1], t[0]])
            if np.dot(nr, inward(pm)) < 0: nr = -nr
            edge = EDGE
            at_wall = (red_end == 'end' and k == 5)
            for _ in range(40):   # push inward until the whole box sits inside the plaza
                c = (p0 + p1) / 2 + nr * (edge + BOX_D / 2)
                ext_ = 2.0 if at_wall else 0.0
                rect = _Poly([c - t * L / 2 - nr * BOX_D / 2, c + t * (L / 2 + ext_) - nr * BOX_D / 2, c + t * (L / 2 + ext_) + nr * BOX_D / 2, c - t * L / 2 + nr * BOX_D / 2])
                if at_wall: rect = rect.intersection(WALL_HALF)
                if all(PLI.contains(Point(*q)) for q in list(rect.exterior.coords)[:-1]): break
                edge += 0.03
            boxes.append(dict(c=c, t=t, n=nr, L=float(L), poly=rect, edge=edge, at_wall=at_wall))
        order = list(range(6)) if red_end == 'start' else list(range(6))[::-1]   # red toward the corner, purple toward the gap
        col = [None] * 6
        for ci, pi in enumerate(order): col[pi] = ci
        stretches.append(dict(side=name, a=a, b=b, boxes=boxes, colors=col, line=ls))
print('stretch lengths', [round(s_['b'] - s_['a'], 1) for s_ in stretches], 'box lengths', [round(s_['boxes'][0]['L'], 2) for s_ in stretches])

# ------------------------------------------------------------------ generic helpers
def lathe(profile, seg=48, phase=0.0, rmod=None):
    """profile: list of (r, z) from bottom to top; returns V,F (open surface, outward normals)"""
    pr = np.asarray(profile, float); m = len(pr)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False) + phase
    V = []
    for i, (r, z) in enumerate(pr):
        rr = r * (rmod(a, i) if rmod else 1.0)
        V.append(np.stack([np.cos(a) * rr, np.sin(a) * rr, np.full(seg, z)], 1))
    V = np.concatenate(V); F = []
    for i in range(m - 1):
        for j in range(seg):
            a0 = i * seg + j; a1 = i * seg + (j + 1) % seg; b0 = a0 + seg; b1 = a1 + seg
            F += [[a0, a1, b1], [a0, b1, b0]]
    return V, np.array(F)

def cap(z, r, seg=48, up=True, phase=0.0):
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False) + phase
    V = np.vstack([np.stack([np.cos(a) * r, np.sin(a) * r, np.full(seg, z)], 1), [[0, 0, z]]])
    F = [[seg, j, (j + 1) % seg] if up else [seg, (j + 1) % seg, j] for j in range(seg)]
    return V, np.array(F)

def xform(V, yaw=0.0, t=(0, 0, 0), s=1.0):
    c, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    return (np.asarray(V) * s) @ R.T + np.asarray(t)

def rot_to(V, d):
    """rotate geometry built along +Z to point along d"""
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    z = np.array([0, 0, 1.0]); v = np.cross(z, d); c = np.dot(z, d)
    if np.linalg.norm(v) < 1e-9: return V if c > 0 else V * [1, 1, -1]
    from scipy.spatial.transform import Rotation as Rt
    R = Rt.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()
    return V @ R.T

OUT = {}      # material key -> Acc
def acc(k):
    if k not in OUT: OUT[k] = Acc()
    return OUT[k]
UVS = {}      # material key -> list of uv arrays aligned with Acc order (water / paving)

# ------------------------------------------------------------------ planters + flowers (templates at origin)
PW, PD = 0.64, 0.60          # flower patch footprint (tiles along each box)
def flower_patch(ci):
    T = {}
    def add(k, V, F, col):
        T.setdefault(k, Acc()).add(V, F, col)
    # foliage dome: individual leaves
    lv = []
    leafc = {0: '#2f5a2a', 1: '#3d6b2c', 2: '#3c6a2b', 3: '#24411f', 4: '#2f5f3a', 5: '#5d7a5a'}[ci]
    nleaf = 130
    for k in range(nleaf):
        px_, py_ = rng.uniform(-PW / 2 + 0.02, PW / 2 - 0.02), rng.uniform(-PD / 2 + 0.02, PD / 2 - 0.02); a = rng.uniform(0, 2 * np.pi)
        u = max(abs(px_) / (PW / 2), abs(py_) / (PD / 2)) ** 2; rr = 0.0
        z = 0.04 + 0.20 * (1 - u) + rng.uniform(-0.02, 0.05)
        ln = rng.uniform(0.09, 0.16) if ci != 5 else rng.uniform(0.05, 0.09); wd = ln * (0.55 if ci not in (5,) else 0.25)
        tilt = np.radians(rng.uniform(20, 70))
        dirv = np.array([np.cos(a) * np.cos(tilt), np.sin(a) * np.cos(tilt), np.sin(tilt)])
        side = np.array([-np.sin(a), np.cos(a), 0]); nrm = np.cross(side, dirv)
        p0 = np.array([px_, py_, z])
        tip = p0 + dirv * ln; m1 = p0 + dirv * ln * 0.45
        Vl = np.array([p0, m1 + side * wd / 2 + nrm * 0.008, tip, m1 - side * wd / 2 + nrm * 0.008, m1 + nrm * 0.015])
        Fl = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        col = srgb(leafc) * rng.uniform(0.8, 1.2)
        add('leaf', Vl, Fl, col)
    flower_col = srgb(PRIDE[ci])
    def petal_ring(c, nrm, npet, r_in, r_out, wd, col, cup=0.2, twist=0.0):
        nrm = nrm / np.linalg.norm(nrm)
        a0 = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]); a0 /= np.linalg.norm(a0); b0 = np.cross(nrm, a0)
        Vp = []; Fp = []
        for j in range(npet):
            a = 2 * np.pi * j / npet + twist
            d = a0 * np.cos(a) + b0 * np.sin(a); s = -a0 * np.sin(a) + b0 * np.cos(a)
            p_in = c + d * r_in; p_mid = c + d * (r_in + r_out) / 2 + nrm * cup * (r_out - r_in) * 0.3; p_out = c + d * r_out + nrm * cup * (r_out - r_in)
            k = len(Vp)
            Vp += [p_in, p_mid + s * wd / 2, p_out, p_mid - s * wd / 2]
            Fp += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
        return np.array(Vp), np.array(Fp)
    def stem(p0, p1, r=0.004):
        V, F = cyl(p0, p1, r, r * 0.8, seg=4, cap=False); add('leaf', V, F, srgb('#3f6b2e'))
    def head_pos(k, nh, hmin, hmax, spread=0.85):
        x = rng.uniform(-PW / 2, PW / 2) * spread; y = rng.uniform(-PD / 2, PD / 2) * spread
        e = max(abs(x) / (PW / 2), abs(y) / (PD / 2))
        return np.array([x, y, rng.uniform(hmin, hmax) * (1 - 0.3 * e * e)])
    if ci == 0:     # red geraniums: umbels of five-petal florets
        for k in range(15):
            c = head_pos(k, 15, 0.24, 0.40); stem(c * [0.85, 0.85, 0] + [0, 0, -0.03], c)
            for j in range(18):
                d = rng.normal(size=3); d[2] = abs(d[2]) + 0.6; d /= np.linalg.norm(d)
                fc = c + d * 0.085
                V, F = petal_ring(fc, d, 5, 0.003, 0.039, 0.033, None, cup=0.15, twist=rng.random())
                add('flower', V, F, flower_col * rng.uniform(0.85, 1.1))
    elif ci == 1:   # orange marigolds: ruffled pompoms
        for k in range(36):
            c = head_pos(k, 36, 0.16, 0.30)
            stem(c * [0.9, 0.9, 0] + [0, 0, -0.03], c)
            for layer, (rr, cup) in enumerate(((0.068, 0.05), (0.054, 0.45), (0.038, 0.9), (0.021, 1.4))):
                V, F = petal_ring(c + [0, 0, 0.006 * layer], np.array([0, 0, 1.0]), 14 - layer * 2, 0.003, rr, rr * 0.55, None, cup=cup, twist=layer * 0.37)
                add('flower', V, F, flower_col * (1.05 - 0.08 * layer))
    elif ci == 2:   # yellow rudbeckia-type daisies with golden centres
        for k in range(28):
            c = head_pos(k, 28, 0.20, 0.38); stem(c * [0.9, 0.9, 0] + [0, 0, -0.03], c)
            nrm = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3), 1.0])
            V, F = petal_ring(c, nrm, 15, 0.021, 0.093, 0.024, None, cup=-0.15, twist=rng.random())
            add('flower', V, F, flower_col * rng.uniform(0.9, 1.05))
            Vb, Fb = blob(c + nrm / np.linalg.norm(nrm) * 0.009, (0.020, 0.020, 0.014), 1)
            add('flower', Vb, Fb, srgb('#c98a0a'))
    elif ci == 3:   # green zinnias 'Envy': double layered petals
        for k in range(27):
            c = head_pos(k, 27, 0.20, 0.38); stem(c * [0.9, 0.9, 0] + [0, 0, -0.03], c)
            nrm = np.array([rng.normal(0, 0.25), rng.normal(0, 0.25), 1.0])
            for layer, (r, cup) in enumerate(((0.084, 0.05), (0.064, 0.3), (0.040, 0.7))):
                V, F = petal_ring(c + nrm / np.linalg.norm(nrm) * 0.006 * layer, nrm, 13, 0.009, r, r * 0.42, None, cup=cup, twist=layer * 0.4)
                add('flower', V, F, srgb('#b5dc5a') * (1.0 - 0.08 * layer) * rng.uniform(0.92, 1.08))
            Vb, Fb = blob(c + nrm / np.linalg.norm(nrm) * 0.018, (0.012, 0.012, 0.009), 1); add('flower', Vb, Fb, srgb('#5a7d1c'))
    elif ci == 4:   # blue hydrangea mopheads
        for k in range(8):
            c = head_pos(k, 8, 0.16, 0.26, 0.7); R = rng.uniform(0.15, 0.19)
            stem(c * [0.9, 0.9, 0] + [0, 0, -0.03], c)
            for j in range(95):
                d = rng.normal(size=3); d[2] = abs(d[2]) * 1.2 + 0.15; d /= np.linalg.norm(d)
                fc = c + d * R
                V, F = petal_ring(fc, d, 4, 0.003, 0.031, 0.028, None, cup=0.05, twist=rng.random() * 2)
                add('flower', V, F, flower_col * rng.uniform(0.75, 1.15) + srgb('#2a1a6a') * rng.uniform(0, 0.25))
    elif ci == 5:   # purple lavender spikes
        for k in range(125):
            x = rng.uniform(-PW / 2, PW / 2) * 0.9; y = rng.uniform(-PD / 2, PD / 2) * 0.9
            base = np.array([x * 0.7, y * 0.7, 0.05])
            lean = np.array([x * 0.45, y * 0.45, 0.0])
            top = base + lean + np.array([0, 0, rng.uniform(0.44, 0.60)])
            stem(base, base + (top - base) * 0.72, 0.0025)
            p0 = base + (top - base) * 0.70
            V, F = cyl(p0, top, 0.020, 0.008, seg=6, cap=True)
            # bumpy whorls
            V = V + np.sin(V[:, 2] * 260)[:, None] * 0.002 * np.array([1, 1, 0])
            add('flower', V, F, flower_col * rng.uniform(0.85, 1.2) + srgb('#20104a') * 0.2)
    return {k: v.get() for k, v in T.items()}

def rounded_rect(L, D, r, n=6):
    pts = []
    for (cx, cy, a0) in ((L / 2 - r, D / 2 - r, 0), (-L / 2 + r, D / 2 - r, 90), (-L / 2 + r, -D / 2 + r, 180), (L / 2 - r, -D / 2 + r, 270)):
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / n); pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)
def ring_prism(outer, inner, z0, z1):
    import mapbox_earcut as me
    n1, n2 = len(outer), len(inner)
    P2 = np.vstack([outer, inner[::-1]])
    tri = me.triangulate_float64(P2, np.array([n1, n1 + n2], dtype=np.uint32)).reshape(-1, 3)
    m = len(P2)
    V = np.vstack([np.column_stack([P2, np.full(m, z1)]), np.column_stack([P2, np.full(m, z0)])])
    F = [tri, tri[:, ::-1] + m]
    for (o, cnt, flip) in ((0, n1, False), (n1, n2, False)):
        for k in range(cnt):
            a_, b_ = o + k, o + (k + 1) % cnt
            F.append(np.array([[a_ + m, b_ + m, b_], [a_ + m, b_, a_]]))
    return V, np.concatenate(F)
from shapely.geometry.polygon import orient as _orient
def _ring(g):
    g = _orient(g, 1.0); return np.array(g.exterior.coords)[:-1]
def planter_box(P, ci):
    """stainless box planter; P = footprint polygon in the box frame (x along the side, y inward), z from 0 to BOX_H"""
    T = {}
    def add(k, V, F, col): T.setdefault(k, Acc()).add(V, F, col)
    steel = srgb('#c9ccd0'); accent = srgb(PRIDE[ci]); H = BOX_H
    Pr = P.buffer(-0.03, join_style=2).buffer(0.03, join_style=1)
    V, F = prism(_ring(Pr.buffer(-0.03, join_style=2)), 0.0, 0.07); add('steel', V, F, srgb('#3a3c3f'))                 # recessed plinth (shadow gap)
    V, F = prism(_ring(Pr), 0.07, H - 0.045); add('steel', V, F, steel)                                                # body
    V, F = ring_prism(_ring(Pr.buffer(0.015, join_style=1)), _ring(Pr.buffer(-0.05, join_style=2)), H - 0.045, H)  # capping rim
    add('steel', V, F, steel * 1.04)
    V, F = prism(_ring(Pr.buffer(-0.035, join_style=2)), H - 0.11, H - 0.07); add('soil', V, F, srgb('#3b2a1e'))
    for (z0, z1, ex) in ((H - 0.26, H - 0.17, 0.005), (0.11, 0.125, 0.003), (H - 0.145, H - 0.137, 0.003)):          # colour accents
        V, F = prism(_ring(Pr.buffer(ex, join_style=1)), z0, z1); add('accent', V, F, accent)
    R_ = _ring(Pr)
    for i in range(len(R_)):                                                                                         # panel seams on long faces
        p0, p1 = R_[i], R_[(i + 1) % len(R_)]; e = p1 - p0; Le = np.linalg.norm(e)
        if Le < 1.0: continue
        u = e / Le; nout = np.array([u[1], -u[0]]); yaw = np.arctan2(u[1], u[0])
        nseam = max(1, int(round(Le / 0.9)))
        for k in range(1, nseam):
            q = p0 + u * Le * k / nseam + nout * 0.001
            V, F = box((q[0], q[1], (0.07 + H - 0.26) / 2), (0.008, 0.004, H - 0.26 - 0.07), yaw + np.pi / 2); add('seam', V, F, srgb('#7d8186'))
            V, F = box((q[0], q[1], (H - 0.137 + H - 0.045) / 2), (0.008, 0.004, 0.092), yaw + np.pi / 2); add('seam', V, F, srgb('#7d8186'))
    return {k: v.get() for k, v in T.items()}

PATCHES = {ci: flower_patch(ci) for ci in range(6)}
POTS = []
from shapely import affinity as _aff
for s_ in stretches:
    for bx, ci in zip(s_['boxes'], s_['colors']):
        c = bx['c']; t = bx['t']; nr = bx['n']
        yaw = float(np.arctan2(t[1], t[0]))
        # world footprint -> box frame (x along t, y along the inward normal)
        Wc = np.array(bx['poly'].exterior.coords)[:-1] - c
        loc = _Poly(np.column_stack([Wc @ t, Wc @ np.array([-t[1], t[0]])]))
        z = min(gz(*q) for q in np.array(bx['poly'].exterior.coords)) - 0.02
        inner = loc.buffer(-0.035, join_style=2)
        seg = inner.intersection(LineString([(-50, 0), (50, 0)]))
        xa, xb = (seg.bounds[0], seg.bounds[2]) if not seg.is_empty else (-bx['L'] / 2, bx['L'] / 2)
        span = xb - xa; npatch = max(1, int(round(span / PW))); pitch = span / npatch
        patches = [dict(x=float(xa + (k + 0.5) * pitch), yaw=float(np.pi * (k % 2)), sx=float(pitch / PW), sy=float((BOX_D - 0.07) / PD)) for k in range(npatch)]
        POTS.append(dict(pos=(float(c[0]), float(c[1]), float(z)), yaw=yaw, color=ci, side=s_['side'], L=float(span + 0.07), mesh=planter_box(loc, ci), patches=patches, soil_z=BOX_H - 0.07, odd=bool(bx['at_wall'])))
TEMPLATES = PATCHES

# ------------------------------------------------------------------ benches (template, front = +Y)
def bench_template():
    T = {}
    def add(k, V, F, col): T.setdefault(k, Acc()).add(V, F, col)
    iron = srgb('#2b2d30'); L = 1.80
    # cast-iron frame profile (side view in y-z): seat support, back support curving back, legs, armrest
    prof = [(0.30, 0.0), (0.22, 0.0), (0.16, 0.38), (-0.18, 0.40), (-0.24, 0.0), (-0.32, 0.0), (-0.25, 0.42), (-0.30, 0.86), (-0.24, 0.87), (-0.18, 0.44), (0.22, 0.44)]
    for x in (-L / 2 + 0.06, 0.0, L / 2 - 0.06):
        th = 0.045 if abs(x) > 0.1 else 0.03
        yz = np.array(prof)
        V2 = np.column_stack([yz[:, 0], yz[:, 1]])
        V, F = prism(V2, 0, th)        # extruded along z -> convert: (y, z_profile, x)
        V = np.column_stack([V[:, 2] + x - th / 2, V[:, 0], V[:, 1]])
        add('iron', V, F, iron)
    for x in (-L / 2 + 0.06, L / 2 - 0.06):   # armrests with scroll ends
        pts = [(x, -0.22, 0.62), (x, 0.0, 0.64), (x, 0.20, 0.62), (x, 0.26, 0.58), (x, 0.25, 0.53), (x, 0.21, 0.54)]
        V, F = tube(pts, 0.022, 8); add('iron', V, F, iron)
        V, F = cyl((x, 0.22, 0.44), (x, 0.22, 0.60), 0.018, seg=8); add('iron', V, F, iron)
    # rainbow slats: backrest top->bottom red, orange, yellow ; seat back->front green, blue, purple
    slats = [(-0.282, 0.80, 18), (-0.262, 0.70, 14), (-0.242, 0.60, 10), (-0.12, 0.465, 0), (0.0, 0.47, 0), (0.12, 0.465, 0)]
    for k, (y, z, tilt) in enumerate(slats):
        w = 0.095; t = 0.034
        V, F = box((0, 0, 0), (L, w, t))
        # round the long edges by scaling corners a little
        V[:, 1] *= 1.0; V[:, 2] *= 1.0
        a = np.radians(tilt)
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        if k < 3:
            V = V @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).T   # backrest boards stand up
            V = V @ R.T
        V = V + [0, y, z]
        add(f'slat{k}', V, F, srgb(PRIDE[k]))
        for x in (-L / 2 + 0.06, 0.0, L / 2 - 0.06):  # bolt heads
            Vb, Fb = cyl((x, y + (0.02 if k >= 3 else 0.022), z + (0.018 if k >= 3 else 0)), (x, y + (0.02 if k >= 3 else 0.03), z + (0.024 if k >= 3 else 0)), 0.008, seg=6)
            add('iron', Vb, Fb, srgb('#8d9094'))
    return {k: v.get() for k, v in T.items()}
BENCH = bench_template()
BENCHES = []
for s_ in stretches:
    bxs = s_['boxes']; nb = 4 if s_['side'] == 'north' else 3
    cs = np.array([bx_['c'] for bx_ in bxs])
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cs, axis=0), axis=1))]
    for i in range(nb):
        f = (i + 0.5) / nb * seglen[-1]
        k = min(int(np.searchsorted(seglen, f, side='right')) - 1, len(cs) - 2)
        w = (f - seglen[k]) / max(seglen[k + 1] - seglen[k], 1e-6)
        c_ = cs[k] * (1 - w) + cs[k + 1] * w
        nr = bxs[k]['n'] * (1 - w) + bxs[k + 1]['n'] * w; nr /= np.linalg.norm(nr)
        q = c_ + nr * (BOX_D / 2 + 0.75 + 0.33)
        yaw = np.arctan2(nr[1], nr[0]) - np.pi / 2     # front (+Y) faces into the plaza, length parallel to the side
        BENCHES.append(dict(pos=(float(q[0]), float(q[1]), gz(*q) - 0.01), yaw=float(yaw), side=s_['side']))
print('bench distance to fountain', [round(float(np.hypot(b_['pos'][0] - FC[0], b_['pos'][1] - FC[1])), 1) for b_ in BENCHES])

# ------------------------------------------------------------------ fountain
ZF = max(gz(FC[0] + 3.4 * np.cos(a), FC[1] + 3.4 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)) + 0.03
ZMIN = min(gz(FC[0] + 3.4 * np.cos(a), FC[1] + 3.4 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24))
to_flag = FLAG - FC; to_flag /= np.linalg.norm(to_flag)
TH_FLAG = np.arctan2(to_flag[1], to_flag[0])        # statues and plinth face the big flag
_wn = np.degrees(np.arctan2(dn[1], dn[0])) + 90.0     # east wall normal
_cands = [((b - (np.degrees(TH_FLAG) + k * 60)) + 180) % 360 - 180 for k in range(6) for b in (_wn, _wn + 180)]
ROT = min([c for c in _cands if -60 < c <= 0], key=lambda c: -c)       # smallest clockwise turn that puts a side parallel to the wall
TH0 = TH_FLAG + np.radians(ROT)                      # hexagonal body orientation
print('fountain rotated %.1f deg (clockwise) to align with the wall' % ROT)
R_OUT = 3.30; A_OUT = R_OUT * np.cos(np.pi / 6); RIM_W = 0.55; A_IN = A_OUT - RIM_W
Z_RIM = 0.56; Z_WATER = 0.46; Z_FLOOR = 0.06
STONES = [('Red jasper', '#a8231d'), ('Carnelian', '#d9621c'), ('Yellow jasper', '#d4a72c'), ('Nephrite jade', '#2f7a3e'), ('Sodalite', '#2c4fa3'), ('Purple porphyry', '#5b2a6e')]
def hexpt(k, apo, frac):
    """point on hexagon side k at apothem apo, frac in [0,1] from corner k to corner k+1"""
    a0 = TH0 - np.pi / 6 + k * np.pi / 3; a1 = a0 + np.pi / 3
    rc = apo / np.cos(np.pi / 6)
    p0 = np.array([np.cos(a0), np.sin(a0)]) * rc; p1 = np.array([np.cos(a1), np.sin(a1)]) * rc
    return p0 + (p1 - p0) * frac
def hex_sweep(profile, nside=12):
    """profile: list of (apothem, z); hexagonal surface of revolution"""
    loops = []
    for apo, z in profile:
        pts = []
        for k in range(6):
            for j in range(nside):
                p = hexpt(k, apo, j / nside); pts.append([p[0], p[1], z])
        loops.append(pts)
    V = np.array(loops).reshape(-1, 3); m = 6 * nside; F = []
    for i in range(len(profile) - 1):
        for j in range(m):
            a0 = i * m + j; a1 = i * m + (j + 1) % m; b0 = a0 + m; b1 = a1 + m
            F += [[a0, a1, b1], [a0, b1, b0]]
    return V, np.array(F)
FV = lambda V: np.asarray(V) + [FC[0], FC[1], ZF]
marble = srgb('#f1eee8'); marble2 = srgb('#e7e3dc')
zb = ZMIN - ZF - 0.12
# outer wall with base plinth and moulding
V, F = hex_sweep([(A_OUT + 0.07, zb), (A_OUT + 0.07, 0.10), (A_OUT + 0.035, 0.14), (A_OUT, 0.17), (A_OUT, Z_RIM - 0.10), (A_OUT + 0.025, Z_RIM - 0.09)])
acc('marble').add(FV(V), F, marble2)
# rim cap: marble top per side with a smooth seat depression; a polished coloured stone slab sits in it
NT, NW = 64, 13
for k in range(6):
    Vg = np.zeros((NT, NW, 3))
    for i, t in enumerate(np.linspace(0, 1, NT)):
        for j, w in enumerate(np.linspace(0, 1, NW)):
            apo = A_IN + 0.02 + (A_OUT + 0.025 - 0.02 - A_IN - 0.02) * w
            p = hexpt(k, apo, t)
            u = (t - 0.5) / 0.27; vv = (w - 0.5) / 0.44
            rr = u * u + vv * vv
            d = 0.045 * max(0.0, 1 - rr) ** 1.2
            Vg[i, j] = [p[0], p[1], Z_RIM - d]
    F = []
    for i in range(NT - 1):
        for j in range(NW - 1):
            a = i * NW + j; b = (i + 1) * NW + j
            F += [[a, b + 1, b], [a, a + 1, b + 1]]
    acc('marble').add(FV(Vg.reshape(-1, 3)), np.array(F), marble)
    # stone slab: ellipse following the dish, 1.5 cm proud of the dish floor, polished
    cen = hexpt(k, A_IN + 0.02 + (A_OUT + 0.005 - A_IN - 0.02) * 0.5, 0.5)
    side = hexpt(k, A_OUT, 1.0) - hexpt(k, A_OUT, 0.0); L_side = np.linalg.norm(side); tdir = side / L_side
    ndir = cen / np.linalg.norm(cen)
    ru = 0.27 * L_side * 0.80; rv = 0.44 * (A_OUT + 0.005 - A_IN - 0.02) * 0.80
    nr, na = 10, 64; Vs = [[cen[0], cen[1], Z_RIM - 0.045 + 0.012]]; Fs = []
    for ir in range(1, nr + 1):
        f = ir / nr
        for ia in range(na):
            a = 2 * np.pi * ia / na
            q = cen + tdir * np.cos(a) * ru * f + ndir * np.sin(a) * rv * f
            dd = 0.045 * max(0.0, 1 - f * f * 0.64) ** 1.2
            Vs.append([q[0], q[1], Z_RIM - dd + 0.012 * (1 - f ** 6) + 0.002])
    for ia in range(na): Fs.append([0, 1 + (ia + 1) % na, 1 + ia])
    for ir in range(1, nr):
        b0 = 1 + (ir - 1) * na; b1 = 1 + ir * na
        for ia in range(na):
            a0 = b0 + ia; a1 = b0 + (ia + 1) % na; c0 = b1 + ia; c1 = b1 + (ia + 1) % na
            Fs += [[a0, a1, c1], [a0, c1, c0]]
    acc(f'stone{k}').add(FV(np.array(Vs)), np.array(Fs), np.array([1.0, 1.0, 1.0])); UVS.setdefault(f'stone{k}', []).append(np.array(Vs)[:, :2] / 0.9)
# bullnose edges of the rim (outer + inner)
V, F = hex_sweep([(A_OUT + 0.025, Z_RIM - 0.09), (A_OUT + 0.045, Z_RIM - 0.065), (A_OUT + 0.05, Z_RIM - 0.035), (A_OUT + 0.04, Z_RIM - 0.01), (A_OUT + 0.025, Z_RIM)], 24)
acc('marble').add(FV(V), F, marble)
V, F = hex_sweep([(A_IN + 0.02, Z_RIM), (A_IN + 0.005, Z_RIM - 0.012), (A_IN, Z_RIM - 0.035), (A_IN, Z_FLOOR)], 24)
acc('marble').add(FV(V), F[:, ::-1], marble)
# basin floor
Vfl = [];
for k in range(6):
    p0 = hexpt(k, A_IN, 0); p1 = hexpt(k, A_IN, 1); Vfl += [[0, 0, Z_FLOOR], [p0[0], p0[1], Z_FLOOR], [p1[0], p1[1], Z_FLOOR]]
acc('marble_floor').add(FV(Vfl), np.arange(18).reshape(-1, 3), srgb('#cfd6d6'))
# water surface of the main basin (hex with pedestal hole) - uv planar
def hex_ring_surface(a_outer, a_inner, z, nside=12):
    o = []; i_ = []
    for k in range(6):
        for j in range(nside):
            p = hexpt(k, a_outer, j / nside); o.append([p[0], p[1], z])
            q = hexpt(k, a_inner, j / nside); i_.append([q[0], q[1], z])
    V = np.array(o + i_); m = 6 * nside; F = []
    for j in range(m):
        a0, a1 = j, (j + 1) % m; b0, b1 = a0 + m, a1 + m
        F += [[a0, a1, b1], [a0, b1, b0]]
    return V, np.array(F)[:, ::-1]
PED_A = 0.75
V, F = hex_ring_surface(A_IN - 0.004, PED_A + 0.004, Z_WATER)
acc('water').add(FV(V), F, srgb('#3f8fa2')); UVS.setdefault('water', []).append(V[:, :2] / 3.0)
# central pedestal with recessed panels and spouts
V, F = hex_sweep([(PED_A + 0.07, Z_FLOOR), (PED_A + 0.07, 0.16), (PED_A + 0.03, 0.20), (PED_A, 0.23), (PED_A, 1.0), (PED_A + 0.04, 1.03), (PED_A + 0.06, 1.07), (PED_A + 0.06, 1.12), (0.58, 1.12)], 16)
acc('marble').add(FV(V), F, marble)
for k in range(6):
    c = hexpt(k, PED_A, 0.5); nrm = c / np.linalg.norm(c); tang = np.array([-nrm[1], nrm[0]])
    # recessed panel (frame proud, panel inset)
    for (wd, ht, off, col) in ((0.52, 0.56, 0.012, marble), (0.42, 0.46, -0.004, marble2)):
        q = np.array([*(c + nrm * off), 0.62])
        Vp, Fp = box((0, 0, 0), (0.02, wd, ht)); yaw = np.arctan2(nrm[1], nrm[0])
        acc('marble').add(FV(xform(Vp, yaw, q)), Fp, col)
    # mask-like spout boss + spout
    Vb, Fb = blob((0, 0, 0), (0.06, 0.09, 0.09), 1); acc('marble').add(FV(xform(Vb, np.arctan2(nrm[1], nrm[0]), [*(c + nrm * 0.03), 0.66])), Fb, marble)
    p0 = np.array([*(c + nrm * 0.06), 0.66]); p1 = np.array([*(c + nrm * 0.16), 0.655])
    Vs, Fs = cyl(p0, p1, 0.022, 0.016, seg=10); acc('bronze').add(FV(Vs), Fs, srgb('#8a6a3a'))
# ---- six cascading pools of decreasing size, each carved from a coloured stone
POOL_STONES = [('Porphyry', '#6a2834'), ('Sodalite', '#2f4b88'), ('Verdite', '#4f7444'),
               ('Yellow calcite', '#d8b866'), ('Orange alabaster', '#d2935e'), ('Red jasper', '#8b3a2e')]   # bottom (largest) to top (smallest): red on top
POOL_R = [1.40, 1.18, 0.98, 0.82, 0.68, 0.56]
POOL_LOBES = [12, 12, 10, 10, 8, 8]
def pool_scale(R): s_ = R / 1.36; return s_, 0.35 + 0.65 * s_
def stem_r(R): return 0.10 + 0.13 * R / 1.36
# rim heights: each gap = depth of the pool above + 0.18 m of visible stem
POOL_Z = [2.05]
for i in range(1, 6):
    s_, ds = pool_scale(POOL_R[i]); POOL_Z.append(POOL_Z[-1] + 0.45 * ds + 0.18)
print('pool rims', [round(z, 2) for z in POOL_Z])
def stone_uv(V, R, zt):
    ang = (np.arctan2(V[:, 1], V[:, 0]) / (2 * np.pi)) % 1.0
    rep = max(1, int(round(2 * np.pi * R / 1.6)))
    return np.column_stack([ang * rep, (V[:, 2] - zt) / 0.8 + 0.5])
def pool(i):
    R = POOL_R[i]; zt = POOL_Z[i]; nl = POOL_LOBES[i]; s_, ds = pool_scale(R); ph = np.pi / nl * (i % 2)
    rs_above = stem_r(POOL_R[i + 1]) if i < 5 else 0.16
    lob = lambda a, k: 1 + 0.045 * np.cos(nl * (a - ph)) * (k >= 3)
    outer = [(max(0.36 * s_, stem_r(R) * 1.15), zt - 0.45 * ds), (0.55 * s_, zt - 0.39 * ds), (0.85 * s_, zt - 0.31 * ds), (1.12 * s_, zt - 0.21 * ds),
             (1.30 * s_, zt - 0.11 * ds), (1.36 * s_, zt - 0.05 * ds), (1.34 * s_, zt - 0.01 * ds), (1.27 * s_, zt), (1.22 * s_, zt - 0.04 * ds)]
    V, F = lathe(outer, 96, phase=0.0, rmod=lob)
    acc(f'pool{i}').add(FV(V), F, np.array([1.0, 1.0, 1.0])); UVS.setdefault(f'pool{i}', []).append(stone_uv(V, R, zt))
    inner = [(1.22 * s_, zt - 0.04 * ds), (1.05 * s_, zt - 0.08 * ds), (0.6 * s_, zt - 0.12 * ds), (rs_above * 1.1, zt - 0.13 * ds)]
    V, F = lathe(inner, 96, rmod=lambda a, k: 1 + 0.045 * np.cos(nl * (a - ph)) * (k == 0))
    acc(f'pool{i}').add(FV(V), F[:, ::-1], np.array([1.0, 1.0, 1.0])); UVS.setdefault(f'pool{i}', []).append(stone_uv(V, R, zt))
    # water surface
    V, F = lathe([(1.255 * s_, zt - 0.025 * ds), (rs_above * 1.12, zt - 0.025 * ds)], 96, rmod=lambda a, k: 1 + 0.045 * np.cos(nl * (a - ph)) * (k == 0))
    acc('water').add(FV(V), F[:, ::-1], srgb('#5f9fb0')); UVS['water'].append(V[:, :2] / 3.0)
    # overflow streams from each lobe tip, falling into the pool below (or the basin)
    z_land = (POOL_Z[i - 1] - 0.025 * pool_scale(POOL_R[i - 1])[1]) if i > 0 else Z_WATER
    r_lim = (1.20 * pool_scale(POOL_R[i - 1])[0]) if i > 0 else A_IN - 0.15
    nz = 22
    for k in range(nl):
        a0 = ph + 2 * np.pi * k / nl
        r0 = 1.36 * s_ * 1.045 + 0.012
        fall = zt - z_land; r1 = min(r0 + 0.16 * np.sqrt(fall / 0.6), r_lim)
        Vs = []; Fs = []; uv = []
        for j in range(nz):
            tt = j / (nz - 1); zz = zt + 0.004 - tt * (fall + 0.004)
            r = r0 + (r1 - r0) * np.sqrt(tt)
            half = (0.075 * s_ + 0.02) * (1 - 0.45 * tt) + 0.004
            for m, w in enumerate(np.linspace(-1, 1, 5)):
                aa = a0 + w * half / r
                Vs.append([np.cos(aa) * r, np.sin(aa) * r, zz]); uv.append([(w + 1) / 2 + k, tt * fall * 3.0])
            if j:
                b0 = (j - 1) * 5; b1 = j * 5
                for m in range(4): Fs += [[b0 + m, b1 + m, b1 + m + 1], [b0 + m, b1 + m + 1, b0 + m + 1]]
        acc('waterfall').add(FV(np.array(Vs)), np.array(Fs), srgb('#d8ecf2')); UVS.setdefault('waterfall', []).append(np.array(uv))
    # ripple ring where the streams land
    rr = (r0 + r1) / 2 + 0.06
    if i > 0:
        V, F = lathe([(rr + 0.07, z_land + 0.004), (rr - 0.05, z_land + 0.004)], 96)
        acc('foam').add(FV(V), F[:, ::-1], srgb('#eef6f8'))
def baluster(z0, z1, rs, knops=1):
    h = z1 - z0; prof = [(rs * 1.35, z0), (rs * 1.15, z0 + 0.03), (rs * 1.0, z0 + 0.06)]
    for k in range(knops):
        zc = z0 + h * (k + 1) / (knops + 1)
        prof += [(rs * 0.9, zc - 0.06), (rs * 1.35, zc - 0.02), (rs * 1.35, zc + 0.02), (rs * 0.9, zc + 0.06)]
    prof += [(rs * 1.0, z1 - 0.06), (rs * 1.15, z1 - 0.03), (rs * 1.35, z1)]
    V, F = lathe(prof, 48); acc('marble').add(FV(V), F, marble)
# vase baluster from the pedestal to the first pool
s1, ds1 = pool_scale(POOL_R[0]); zb1 = POOL_Z[0] - 0.45 * ds1
bal = [(0.58, 1.12), (0.60, 1.15), (0.52, 1.19), (0.42, 1.24), (0.40, 1.28), (0.45, 1.36), (0.50, 1.43), (0.47, 1.50), (0.40, zb1 - 0.06), (0.36 * s1 + 0.04, zb1 - 0.02), (0.36 * s1 + 0.02, zb1)]
V, F = lathe(bal, 64, rmod=lambda a, i: 1 + (0.035 * np.cos(16 * a) if 4 <= i <= 7 else 0)); acc('marble').add(FV(V), F, marble)
for i in range(6):
    pool(i)
    if i < 5:
        sA, dsA = pool_scale(POOL_R[i]); sB, dsB = pool_scale(POOL_R[i + 1])
        baluster(POOL_Z[i] - 0.13 * dsA, POOL_Z[i + 1] - 0.45 * dsB, stem_r(POOL_R[i + 1]), 1)
# finial stem from the top pool to the statue plinth
sT, dsT = pool_scale(POOL_R[5]); ZP = 5.30
baluster(POOL_Z[5] - 0.13 * dsT, ZP - 0.10, 0.16, 2)
V, F = lathe([(0.20, ZP - 0.10), (0.30, ZP - 0.06), (0.38, ZP - 0.02), (0.40, ZP)], 64); acc('marble').add(FV(V), F, marble)
# statue plinth: rectangular, long side across the pair, with base and crown mouldings and a bronze plaque
PL_W, PL_D = 1.05, 0.74
for (w, d, z0, z1) in ((PL_W + 0.10, PL_D + 0.10, ZP, ZP + 0.07), (PL_W + 0.05, PL_D + 0.05, ZP + 0.07, ZP + 0.12), (PL_W, PL_D, ZP + 0.12, 5.86),
                       (PL_W + 0.05, PL_D + 0.05, 5.86, 5.91), (PL_W + 0.10, PL_D + 0.10, 5.91, 6.00)):
    Vb, Fb = box((0, 0, (z0 + z1) / 2), (d, w, z1 - z0), TH_FLAG)
    acc('marble').add(FV(Vb), Fb, marble)
fwd2 = np.array([np.cos(TH_FLAG), np.sin(TH_FLAG)])
Vb, Fb = box((*(fwd2 * (PL_D / 2 + 0.006)), (ZP + 0.12 + 5.86) / 2), (0.012, 0.62, 0.30), TH_FLAG); acc('bronze').add(FV(Vb), Fb, srgb('#7a5a2e'))
Vb, Fb = box((*(fwd2 * (PL_D / 2 + 0.013)), (ZP + 0.12 + 5.86) / 2), (0.004, 0.56, 0.24), TH_FLAG); acc('bronze').add(FV(Vb), Fb, srgb('#9a7a44'))
# inner corner jets arcing toward the pedestal + pedestal spout jets arcing outward
def arc(p0, p1, h, n=28):
    pts = []
    for t in np.linspace(0, 1, n):
        p = p0 + (p1 - p0) * t; p = p.copy(); p[2] += 4 * h * t * (1 - t); pts.append(p)
    return np.array(pts)
for k in range(6):
    cnr = hexpt(k, A_IN + 0.08, 0.0)
    p0 = np.array([*cnr, Z_RIM + 0.02]); dirn = -cnr / np.linalg.norm(cnr)
    p1 = np.array([*(dirn * -(PED_A + 0.45)), Z_WATER])
    pts = arc(p0, np.array([*(cnr + dirn * (np.linalg.norm(cnr) - PED_A - 0.45)), Z_WATER]), 1.15)
    V, F = tube(pts, 0.016, 8); acc('waterfall').add(FV(V), F, srgb('#dff0f5'))
    UVS['waterfall'].append(np.column_stack([(np.arange(len(V)) % 8) / 8, np.linspace(0, 6, len(V))]))
    c = hexpt(k, PED_A, 0.5); nrm = c / np.linalg.norm(c)
    p0 = np.array([*(c + nrm * 0.16), 0.655]); p1 = np.array([*(c + nrm * 1.25), Z_WATER])
    pts = arc(p0, p1, 0.18)
    V, F = tube(pts, 0.012, 8); acc('waterfall').add(FV(V), F, srgb('#dff0f5'))
    UVS['waterfall'].append(np.column_stack([(np.arange(len(V)) % 8) / 8, np.linspace(0, 6, len(V))]))
# ripple rings where water lands (static foam rings)
for rr in (1.36 * pool_scale(POOL_R[0])[0] * 1.045 + 0.2,):
    V, F = lathe([(rr + 0.18, Z_WATER + 0.004), (rr - 0.10, Z_WATER + 0.004)], 96)
    acc('foam').add(FV(V), F[:, ::-1], srgb('#eef6f8'))

# ---- hexagonal tile courses around the basin: six glazed rainbow rings (purple inside, red outside)
#      alternating with rings of the plaza's own stone; the plaza paving continues these courses outward
COURSE = 0.30; RING0 = A_OUT + 0.10; NRING = 11; GROUT = 0.008
PAVE_HEX = '#d3cdc2'
for j in range(NRING):
    a0 = RING0 + j * COURSE + GROUT / 2; a1 = RING0 + (j + 1) * COURSE - GROUT / 2
    side_len = 2 * ((a0 + a1) / 2) * np.tan(np.pi / 6)
    glazed = (j % 2 == 0); ci = 5 - j // 2
    TL = 0.30 if glazed else 0.60
    n = max(3, int(round(side_len / TL)))
    off = 0.0 if glazed else (0.5 / n) * (j // 2 % 2)
    for k in range(6):
        for m in range(n):
            t0 = m / n + off; t1 = (m + 1) / n + off
            g = GROUT / 2 / side_len
            segs_ = [(t0, min(t1, 1.0))] + ([(0.0, t1 - 1.0)] if t1 > 1.0 else [])
            for (u0, u1) in segs_:
                if u1 - u0 < 0.01: continue
                q = np.array([hexpt(k, a0, u0 + g), hexpt(k, a0, u1 - g), hexpt(k, a1, u1 - g), hexpt(k, a1, u0 + g)]) + FC
                zg = ground(q[:, 0], q[:, 1])
                V, F = prism(q, zg + 0.004, zg + 0.030)
                if glazed: acc('tiles').add(V, F, srgb(PRIDE[ci]) * rng.uniform(0.88, 1.06))
                else: acc('hex_stone').add(V, F, srgb(PAVE_HEX) * rng.uniform(0.90, 1.05))
BAND_OUT = RING0 + NRING * COURSE

# ------------------------------------------------------------------ statues on top (pair frame -> world)
ST = pickle.load(open(ROOT + '/data/statues.pkl', 'rb'))
fwd = to_flag; left = np.array([-fwd[1], fwd[0]])
def pair2world(v):
    v = np.asarray(v)
    xy = FC + np.outer(v[:, 0], left) + np.outer(v[:, 2], fwd)
    return np.column_stack([xy, v[:, 1] + ZF + 6.0])
for who in ('A', 'B'):
    for part, (v, f) in ST[who].items():
        acc('statue_cloth' if part == 'cloth' else 'statue').add(pair2world(v), np.asarray(f), srgb('#f3f0ea'))

# ------------------------------------------------------------------ paving mesh + texture
from PIL import Image, ImageDraw, ImageFilter
bx0, by0, bx1, by1 = PL.bounds; bx0 -= 0.5; by0 -= 0.5; bx1 += 0.5; by1 += 0.5
TS = 2048; sx = TS / (bx1 - bx0); sy = TS / (by1 - by0)
def tp(x, y): return ((x - bx0) * sx, (by1 - y) * sy)
img = np.zeros((TS, TS, 3), np.float32)
xs = bx0 + (np.arange(TS) + 0.5) / sx; ys = by1 - (np.arange(TS) + 0.5) / sy
XX, YY = np.meshgrid(xs, ys)
dxy = np.stack([XX - FC[0], YY - FC[1]], -1)
r_ = np.hypot(dxy[..., 0], dxy[..., 1]); ang = np.arctan2(dxy[..., 1], dxy[..., 0]) - TH0
sector = np.floor((ang + np.pi / 6) / (np.pi / 3)) % 6
loc = ang - sector * np.pi / 3; loc = (loc + np.pi) % (2 * np.pi) - np.pi
apo = r_ * np.cos(loc); sl = r_ * np.sin(loc)                   # distance from centre across courses / along the side
course = np.floor((apo - RING0) / COURSE)
TLP = 0.60
sl_off = sl + (course % 2) * TLP / 2 + sector * 0.137
tile_i = np.floor(sl_off / TLP)
h = (np.sin(course * 12.9898 + tile_i * 78.233 + sector * 3.17) * 43758.5453) % 1
base = srgb(PAVE_HEX) ** (1 / 2.2)
img[:] = base
img *= (0.92 + 0.12 * h)[..., None]
grain = np.random.default_rng(3).random((TS, TS)).astype(np.float32)
img *= (0.94 + 0.10 * grain)[..., None]
jc = np.abs(((apo - RING0) / COURSE) % 1 - 0.5) * COURSE          # course joints
jt = np.abs((sl_off / TLP) % 1 - 0.5) * TLP                         # joints along the course
mitre = np.abs(r_ * np.sin(np.abs(loc) - np.pi / 6))                 # mitre joints on the six corner lines
joint = (COURSE / 2 - jc < 0.005) | (TLP / 2 - jt < 0.005) | (mitre < 0.006)
img[joint] *= 0.70
band = (apo > A_OUT + 0.02) & (apo < BAND_OUT + 0.004)               # grout bed under the raised ring tiles
img[band] = srgb('#55565a') ** (1 / 2.2)
# border band along the plaza edge
mask = Image.new('L', (TS, TS), 0); d = ImageDraw.Draw(mask)
d.polygon([tp(*p) for p in PL.exterior.coords], fill=255)
inner = PL.buffer(-0.40)
mi = Image.new('L', (TS, TS), 0); ImageDraw.Draw(mi).polygon([tp(*p) for p in inner.exterior.coords], fill=255)
border = (np.array(mask) > 0) & (np.array(mi) == 0)
img[border] = srgb('#6a6a6e') ** (1 / 2.2) * (0.92 + 0.12 * grain[border])[:, None]
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(ROOT + '/web/plaza_paving.jpg', quality=88)
# paving mesh: constrained triangulation with interior points, draped on terrain
ext = np.array(PL.exterior.coords)[:-1]
gx_, gy_ = np.meshgrid(np.arange(bx0, bx1, 0.6), np.arange(by0, by1, 0.6))
ins = [(a, b) for a, b in zip(gx_.ravel(), gy_.ravel()) if PL.buffer(-0.3).contains(Point(a, b))]
Vt = np.vstack([ext, np.array(ins)])
segs = np.array([[i, (i + 1) % len(ext)] for i in range(len(ext))])
T = tr.triangulate(dict(vertices=Vt, segments=segs), 'pq')
Vt = T['vertices']; Ft = T['triangles']
Zt = ground(Vt[:, 0], Vt[:, 1]) + 0.02
PAVE = (np.column_stack([Vt, Zt]), Ft, np.column_stack([(Vt[:, 0] - bx0) / (bx1 - bx0), (by1 - Vt[:, 1]) / (by1 - by0)]))

# ------------------------------------------------------------------ export
data = dict(acc={k: v.get() for k, v in OUT.items()}, uvs={k: np.concatenate(v) for k, v in UVS.items()},
            pot_templates=TEMPLATES, pots=POTS, bench=BENCH, benches=BENCHES, pave=PAVE,
            fountain=dict(center=FC.tolist(), z=ZF, stones=STONES, pool_stones=POOL_STONES, pool_z=POOL_Z, pool_r=POOL_R, th0=TH_FLAG, th_hex=TH0, top=ZF + 6.0, statue_top=ZF + 7.5),
            plaza=PL, tri=S['tri'], tip=TIP.tolist(), wallpt=wallpt.tolist())
pickle.dump(data, open(ROOT + '/data/plaza.pkl', 'wb'))
print('pots', len(POTS), 'benches', len(BENCHES), {k: len(v.get()[1]) for k, v in OUT.items()})
print('templates tris', {ci: sum(len(f) for (_, f, _) in t.values()) for ci, t in TEMPLATES.items()}, 'bench tris', sum(len(f) for (_, f, _) in BENCH.values()))
print('fountain base z %.2f (ground %.2f..%.2f)' % (ZF, ZMIN, ZF - 0.03))
