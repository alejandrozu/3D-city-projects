import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Greco-Roman figure builder on the CC0 MakeHuman base mesh.
Frame (MakeHuman): +X = figure's own left, +Y up, +Z forward. Output in metres."""
import numpy as np, json, os
from scipy.spatial.transform import Rotation as Rot
MH = ROOT + '/mh/data'

# ------------------------------------------------------------------ loading
def load_obj():
    V = []; F = []; G = []; cur = None
    for line in open(f'{MH}/3dobjs/base.obj'):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('g '):
            cur = line.split()[1]
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            F.append(idx); G.append(cur)
    return np.array(V), F, G

def load_target(name):
    p = f'{MH}/targets/macrodetails/{name}.target'
    d = {}
    for line in open(p):
        if line.startswith('#') or not line.strip(): continue
        a = line.split(); d[int(a[0])] = [float(a[1]), float(a[2]), float(a[3])]
    return d

BASE_V, BASE_F, BASE_G = load_obj()
SKEL = json.load(open(f'{MH}/rigs/default.mhskel'))
WTS = json.load(open(f'{MH}/rigs/default_weights.mhw'))['weights']

def morph(muscle, weight, height=0.75, prop=1.0, eth=(0.6, 0.2, 0.2), overdrive=1.0):
    """MakeHuman macro interpolation for a young male."""
    V = BASE_V.copy()
    def add(name, w):
        if w <= 1e-6: return
        try: t = load_target(name)
        except FileNotFoundError: return
        if not t: return
        idx = np.fromiter(t.keys(), int); off = np.array(list(t.values()))
        V[idx] += w * off
    for e, w in zip(('caucasian', 'african', 'asian'), eth):
        add(f'{e}-male-young', w)
    mw = {'averagemuscle': 1 - max(0, (muscle - .5) * 2) - max(0, (.5 - muscle) * 2),
          'maxmuscle': max(0, (muscle - .5) * 2), 'minmuscle': max(0, (.5 - muscle) * 2)}
    ww = {'averageweight': 1 - max(0, (weight - .5) * 2) - max(0, (.5 - weight) * 2),
          'maxweight': max(0, (weight - .5) * 2), 'minweight': max(0, (.5 - weight) * 2)}
    for m, a in mw.items():
        for w_, b in ww.items():
            c = a * b
            if c <= 1e-6: continue
            add(f'universal-male-young-{m}-{w_}', c * (overdrive if m == 'maxmuscle' else 1.0))
            add(f'proportions/male-young-{m}-{w_}-idealproportions', c * prop)
            if height > .5: add(f'height/male-young-{m}-{w_}-maxheight', c * (height - .5) * 2)
    return V

def heroic(V, shoulder=1.08, waist=0.97, neck=1.0):
    """broaden the shoulder girdle (rest pose, dm units). Arms translate with the shoulder."""
    V = V.copy(); x, y = V[:, 0], V[:, 1]
    def f(yy):
        up = np.clip((yy - 6.4) / (8.6 - 6.4), 0, 1); up = up * up * (3 - 2 * up)
        dn = np.clip((10.4 - yy) / (10.4 - 9.6), 0, 1)
        sh = 1 + (shoulder - 1) * up * dn
        wa = 1 + (waist - 1) * np.exp(-((yy - 6.3) / 0.8) ** 2)
        return sh * wa
    xs = 2.34
    arm = (np.abs(x) > xs) & (y > 4.2)
    dx = np.where(arm, (f(np.full_like(y, 8.83)) - 1) * xs * np.sign(x), (f(y) - 1) * x)
    V[:, 0] += dx
    return V

# ------------------------------------------------------------------ skeleton
def joints(V):
    return {k: V[v].mean(0) for k, v in SKEL['joints'].items()}

def bone_table(V):
    J = joints(V); B = {}
    for name, b in SKEL['bones'].items():
        B[name] = dict(head=J[b['head']], tail=J[b['tail']], parent=b['parent'])
    order = []; seen = set()
    def visit(n):
        if n in seen: return
        p = B[n]['parent']
        if p: visit(p)
        seen.add(n); order.append(n)
    for n in B: visit(n)
    return B, order

def align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0: return np.eye(3)
        ax = np.cross(a, [1, 0, 0]);
        if np.linalg.norm(ax) < 1e-6: ax = np.cross(a, [0, 1, 0])
        return Rot.from_rotvec(ax / np.linalg.norm(ax) * np.pi).as_matrix()
    return Rot.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()

def rotm(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return Rot.from_rotvec(axis * np.radians(deg)).as_matrix()

def pose(V, spec):
    """spec: dict bone -> dict(pre=3x3 world rot, dir=target dir, twist=deg, curl=(axis_rest, deg))
    returns posed vertices, posed joint heads, global rotations"""
    B, order = bone_table(V)
    R = {}; H = {}
    for b in order:
        p = B[b]['parent']; s = spec.get(b, {})
        if p:
            Rp = R[p]; hp = Rp @ (B[b]['head'] - B[p]['head']) + H[p]
        else:
            Rp = np.eye(3); hp = B[b]['head'].copy()
        Rb = Rp
        if 'pre' in s: Rb = s['pre'] @ Rb
        if 'dir' in s:
            d = Rb @ (B[b]['tail'] - B[b]['head'])
            Rb = align(d, np.asarray(s['dir'], float)) @ Rb
            if s.get('twist'): Rb = rotm(s['dir'], s['twist']) @ Rb
        if 'curl' in s:
            ax_rest, deg = s['curl']
            Rb = rotm(Rb @ ax_rest, deg) @ Rb
        R[b] = Rb; H[b] = hp
    # skinning
    n = len(V); acc = np.zeros((n, 3)); wsum = np.zeros(n)
    for b, lst in WTS.items():
        if b not in R: continue
        a = np.array(lst); idx = a[:, 0].astype(int); w = a[:, 1]
        P = (V[idx] - B[b]['head']) @ R[b].T + H[b]
        np.add.at(acc, idx, P * w[:, None]); np.add.at(wsum, idx, w)
    miss = wsum < 1e-6
    acc[miss] = V[miss] @ R['root'].T + (H['root'] - R['root'] @ B['root']['head'])
    wsum[miss] = 1
    return acc / wsum[:, None], H, R, B

# ------------------------------------------------------------------ geometry helpers
def body_faces(groups=('body', 'helper-l-eye', 'helper-r-eye')):
    tris = []; gid = []
    for f, g in zip(BASE_F, BASE_G):
        if g not in groups: continue
        k = groups.index(g)
        if len(f) == 4: tris += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]; gid += [k, k]
        else: tris.append(f[:3]); gid.append(k)
    return np.array(tris), np.array(gid)

def compact(V, F):
    used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F], used

def vnormals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], n)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)

def noise3(P, freq, seed=0):
    """cheap smooth value-noise sum of sines (deterministic)"""
    rng = np.random.default_rng(seed); out = np.zeros(len(P))
    for k in range(6):
        d = rng.normal(size=3); d /= np.linalg.norm(d); ph = rng.uniform(0, 6.28)
        out += np.sin(P @ d * freq * (1 + 0.37 * k) + ph) / (1 + 0.5 * k)
    return out / 2.2

def worley_bumps(P, cell, seed=0):
    """curl-like bumps: 1 - distance to nearest jittered cell point (normalised)"""
    rng = np.random.default_rng(seed)
    g = np.floor(P / cell).astype(int)
    best = np.full(len(P), 9.0); best2 = np.full(len(P), 9.0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = g + [dx, dy, dz]
                h = (c[:, 0] * 73856093 ^ c[:, 1] * 19349663 ^ c[:, 2] * 83492791 ^ seed) & 0xffffff
                jit = np.stack([(h % 1000) / 1000, ((h // 1000) % 1000) / 1000, ((h * 7) % 997) / 997], 1)
                q = (c + jit) * cell
                d = np.linalg.norm(P - q, axis=1) / cell
                upd = d < best; best2 = np.where(upd, best, np.minimum(best2, d)); best = np.where(upd, d, best)
    return best, best2

# ------------------------------------------------------------------ figure assembly
import trimesh, pyfqmr
from scipy import ndimage

def ik(S, T, L1, L2, pole):
    d = T - S; dist = np.linalg.norm(d); dist = min(dist, (L1 + L2) * 0.995); u = d / np.linalg.norm(d)
    a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist); h = np.sqrt(max(L1 * L1 - a * a, 0))
    q = pole - np.dot(pole, u) * u; q /= np.linalg.norm(q)
    E = S + a * u + h * q; W = S + u * dist
    return E, W

def hand_axes(B, side):
    lat = B[f'finger2-1.{side}']['head'] - B[f'finger5-1.{side}']['head']; lat /= np.linalg.norm(lat)
    return lat

def palm_normal_rest(B, side):
    a = B[f'finger2-1.{side}']['head']; b = B[f'finger5-1.{side}']['head']; w = B[f'wrist.{side}']['head']
    pn = np.cross(a - w, b - w); pn /= np.linalg.norm(pn)
    th = B[f'finger1-3.{side}']['tail']
    if np.dot(th - (a + b + w) / 3, pn) < 0: pn = -pn
    return pn

def orient_palm(Vr, spec, B, side, desired, iters=3):
    """adjust forearm twist so the posed palm normal points along `desired`"""
    pn = palm_normal_rest(B, side); desired = np.asarray(desired, float) / np.linalg.norm(desired)
    for _ in range(iters):
        _, H, R, _ = pose(Vr, spec)
        a = H[f'wrist.{side}'] - H[f'lowerarm01.{side}']; a /= np.linalg.norm(a)
        c = R[f'wrist.{side}'] @ pn
        cp = c - np.dot(c, a) * a; dp = desired - np.dot(desired, a) * a
        if np.linalg.norm(cp) < 1e-6 or np.linalg.norm(dp) < 1e-6: break
        cp /= np.linalg.norm(cp); dp /= np.linalg.norm(dp)
        ang = np.degrees(np.arctan2(np.dot(a, np.cross(cp, dp)), np.dot(cp, dp)))
        if abs(ang) < 1.0: break
        for bn, fr in ((f'lowerarm01.{side}', 0.4), (f'lowerarm02.{side}', 0.6)):
            sp = spec[bn]; sp['twist'] = sp.get('twist', 0) + fr * ang
    return spec

def finger_spec(B, side, curls, thumb=20, spread=0):
    """curls: dict finger index(2..5)->deg per segment; thumb curl deg.
    Curl axis = d x n, with d = finger direction and n = palm normal (toward the thumb tip)."""
    d = B[f'finger3-3.{side}']['tail'] - B[f'finger3-1.{side}']['head']; d /= np.linalg.norm(d)
    a = B[f'finger2-1.{side}']['head']; b = B[f'finger5-1.{side}']['head']; w = B[f'wrist.{side}']['head']
    pn = np.cross(a - w, b - w); pn /= np.linalg.norm(pn)
    th = B[f'finger1-3.{side}']['tail']
    if np.dot(th - (a + b + w) / 3, pn) < 0: pn = -pn
    ax = np.cross(d, pn); ax /= np.linalg.norm(ax)
    spec = {}
    for fi in (2, 3, 4, 5):
        c = curls.get(fi, 20)
        for seg in (1, 2, 3):
            spec[f'finger{fi}-{seg}.{side}'] = dict(curl=(ax, c * (0.75 if seg == 1 else 1.0)))
    tax = np.cross(B[f'finger1-2.{side}']['tail'] - B[f'finger1-1.{side}']['head'], pn); tax /= np.linalg.norm(tax)
    for seg in (2, 3):
        spec[f'finger1-{seg}.{side}'] = dict(curl=(tax, thumb * 0.6))
    return spec

def scalp_mask(Vr, B, kind):
    """rest-pose hair region (bool per vertex)"""
    hh = B['head']['head']; ht = B['head']['tail']
    C = hh + (ht - hh) * 0.45 + np.array([0, 0, -0.15])
    r = np.linalg.norm(ht - hh) * 0.62
    d = (Vr - C) / r
    near = np.linalg.norm(Vr - C, axis=1) < r * 1.6
    top = d[:, 1] > -0.02
    back = (d[:, 2] < -0.10) & (d[:, 1] > (-0.62 if kind == 'curly' else -0.80))
    side = (np.abs(d[:, 0]) > 0.46) & (d[:, 1] > -0.22) & (d[:, 2] < 0.32)
    face = (d[:, 2] > 0.28) & (d[:, 1] < 0.30)
    m = near & (top | back | side) & ~face
    return m

def cloth_mask(Vr, B, kind):
    root = B['root']['head']; hipL = B['upperleg01.L']['head']; hipR = B['upperleg01.R']['head']
    hw = abs(hipL[0] - hipR[0]) / 2
    x = (Vr[:, 0] - root[0]) / hw; y = Vr[:, 1]; z = Vr[:, 2] - root[2]
    crotch = hipL[1] - 1.25 * hw * 0.55
    top_front = root[1] + (0.75 if kind == 'wrap' else 0.95); top_back = root[1] + 0.95
    ytop = np.where(z > 0, top_front, top_back) - 0.25 * np.clip(np.abs(x) - 0.6, 0, 2)
    low_center = crotch - (0.55 if kind == 'wrap' else 0.95)
    ylow = low_center + np.clip(np.abs(x) - 0.35, 0, 3) ** 1.3 * (0.9 if kind == 'wrap' else 0.7)
    torso = (np.abs(x) < 1.9) & (np.abs(z) < 2.5)
    return torso & (y < ytop) & (y > ylow)

def region_depth(F, mask, n):
    """smooth 0..1 inside-depth of a vertex region (graph diffusion) for edge tapering"""
    import scipy.sparse as sp
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel() + 1e-9
    d = mask.astype(float)
    for _ in range(6):
        d = (A @ d) / deg; d = np.where(mask, d, 0)
    return np.clip((d - 0.35) / 0.65, 0, 1)

def shell(Vp, Np, F, fmask, offset, inner=0.02, rim=True):
    """surface offset shell over faces fmask with per-vertex offset; returns V,F with thickness"""
    sub = F[fmask]; used = np.unique(sub); remap = -np.ones(len(Vp), int); remap[used] = np.arange(len(used))
    S = remap[sub]; Vo = Vp[used] + Np[used] * offset[used, None]; Vi = Vp[used] + Np[used] * inner
    n = len(used)
    Vs = np.concatenate([Vo, Vi]); Fs = [S, S[:, ::-1] + n]
    if rim:
        e = np.sort(np.concatenate([S[:, [0, 1]], S[:, [1, 2]], S[:, [2, 0]]]), 1)
        u, c = np.unique(e, axis=0, return_counts=True); bd = set(map(tuple, u[c == 1]))
        rimf = []
        for f in S:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                if (min(a, b), max(a, b)) in bd: rimf += [[b, a, a + n], [b, a + n, b + n]]
        if rimf: Fs.append(np.array(rimf))
    return Vs, np.concatenate(Fs)

def build_figure(cfg):
    Vr = morph(cfg['muscle'], cfg['weight'], cfg.get('height', .75), 1.0, cfg.get('eth', (0.6, 0.2, 0.2)), cfg.get('overdrive', 1.0))
    Vr = heroic(Vr, cfg.get('shoulder', 1.06), cfg.get('waist', 0.97))
    B, order = bone_table(Vr)
    s_in, s_out = cfg['inner'], cfg['outer']          # 'L'/'R'
    sgn = {'L': 1, 'R': -1}
    spec = {}
    # ---- contrapposto: weight on the inner leg
    tilt = cfg.get('tilt', 4.0) * (1 if s_in == 'L' else -1)   # raise weight-side hip
    spec['root'] = dict(pre=rotm([0, 0, 1], tilt) @ rotm([0, 1, 0], cfg.get('hip_yaw', 0)))
    for k, b in enumerate(('spine04', 'spine03', 'spine02')):
        spec[b] = dict(pre=rotm([0, 0, 1], -tilt * 0.55) @ rotm([1, 0, 0], cfg.get('chest_up', -1.5)) @ rotm([0, 1, 0], cfg.get('twist', 0) / 3))
    spec['neck01'] = dict(pre=rotm([0, 0, 1], tilt * 0.3))
    spec['head'] = dict(pre=rotm([0, 1, 0], cfg.get('head_yaw', 0)) @ rotm([1, 0, 0], -cfg.get('head_up', 4)) @ rotm([0, 0, 1], cfg.get('head_roll', 0)))
    w, r = s_in, s_out
    spec[f'upperleg01.{w}'] = dict(dir=[0.05 * -sgn[w], -1, 0.0]); spec[f'upperleg02.{w}'] = dict(dir=[0.05 * -sgn[w], -1, 0.0])
    spec[f'lowerleg01.{w}'] = dict(dir=[0.01 * -sgn[w], -1, -0.03]); spec[f'lowerleg02.{w}'] = dict(dir=[0.01 * -sgn[w], -1, -0.03])
    bend = cfg.get('knee', 0.22)
    spec[f'upperleg01.{r}'] = dict(dir=[0.10 * sgn[r], -1, 0.12 + bend * 0.4]); spec[f'upperleg02.{r}'] = dict(dir=[0.10 * sgn[r], -1, 0.12 + bend * 0.4])
    spec[f'lowerleg01.{r}'] = dict(dir=[0.05 * sgn[r], -1, -bend]); spec[f'lowerleg02.{r}'] = dict(dir=[0.05 * sgn[r], -1, -bend])
    for sd, splay in ((w, 8), (r, 16)):
        fd = B[f'foot.{sd}']['tail'] - B[f'foot.{sd}']['head']
        spec[f'foot.{sd}'] = dict(dir=rotm([0, 1, 0], splay * sgn[sd]) @ fd)
    # ---- free arm
    fa = cfg['free_arm']
    if fa == 'point':
        up = np.array([0.30 * sgn[r], 0.22, 0.92]); fo = np.array([0.18 * sgn[r], 0.28, 0.95])
        spec[f'clavicle.{r}'] = dict(pre=rotm([0, 0, 1], 9 * sgn[r]))
        spec[f'upperarm01.{r}'] = dict(dir=up); spec[f'upperarm02.{r}'] = dict(dir=up)
        spec[f'lowerarm01.{r}'] = dict(dir=fo); spec[f'lowerarm02.{r}'] = dict(dir=fo)
        spec[f'wrist.{r}'] = dict(dir=fo + np.array([0, -0.05, 0]))
        spec.update(finger_spec(B, r, {2: 3, 3: 78, 4: 84, 5: 88}, thumb=30))
    else:  # relaxed
        up = np.array([0.14 * sgn[r], -1, 0.03]); fo = np.array([0.07 * sgn[r], -1, 0.22])
        spec[f'upperarm01.{r}'] = dict(dir=up); spec[f'upperarm02.{r}'] = dict(dir=up)
        spec[f'lowerarm01.{r}'] = dict(dir=fo); spec[f'lowerarm02.{r}'] = dict(dir=fo)
        spec[f'wrist.{r}'] = dict(dir=fo + np.array([-0.06 * sgn[r], -0.15, -0.12]))
        spec.update(finger_spec(B, r, {2: 22, 3: 28, 4: 32, 5: 36}, thumb=15))
    if fa == 'point':
        orient_palm(Vr, spec, B, r, [-0.3 * sgn[r], -1, 0.0])
    else:
        orient_palm(Vr, spec, B, r, [-1.0 * sgn[r], 0, -0.25])
    # provisional inner arm (hanging) to obtain shoulder position
    spec[f'upperarm01.{w}'] = dict(dir=[0.1 * sgn[w], -1, 0.05])
    _, H, R, _ = pose(Vr, spec)
    return Vr, B, spec, H

def finish_figure(Vr, B, spec, cfg, hand_target_local):
    s_in = cfg['inner']; sgn = {'L': 1, 'R': -1}; w = s_in
    _, H, R, _ = pose(Vr, spec)
    S = H[f'upperarm01.{w}']
    L1 = np.linalg.norm(B[f'lowerarm01.{w}']['head'] - B[f'upperarm01.{w}']['head'])
    L2 = np.linalg.norm(B[f'wrist.{w}']['head'] - B[f'lowerarm01.{w}']['head'])
    pole = np.array([0.6 * sgn[w], -0.2, -1.0])
    E, Wt = ik(S, hand_target_local, L1, L2, pole)
    spec[f'upperarm01.{w}'] = dict(dir=E - S); spec[f'upperarm02.{w}'] = dict(dir=E - S, twist=10 * sgn[w])
    spec[f'lowerarm01.{w}'] = dict(dir=Wt - E); spec[f'lowerarm02.{w}'] = dict(dir=Wt - E)
    hd = (Wt - E) / np.linalg.norm(Wt - E) * 0.6 + np.array([-0.30 * sgn[w], -1.0, 0.05])
    spec[f'wrist.{w}'] = dict(dir=hd)
    spec.update(finger_spec(B, w, {2: 48, 3: 55, 4: 60, 5: 64}, thumb=30))
    orient_palm(Vr, spec, B, w, cfg['inner_palm'])
    Vp, H, R, _ = pose(Vr, spec)
    return Vp, H, R

def surface_parts(Vr, Vp, B, cfg):
    Fb, gid = body_faces()
    Vr_c, Fb, used0 = compact(Vr, Fb); Vp_c = Vp[used0]
    # subdivide rest and posed meshes identically (same topology)
    mr = trimesh.Trimesh(Vr_c, Fb, process=False); mp = trimesh.Trimesh(Vp_c, Fb, process=False)
    vr2, f2 = trimesh.remesh.subdivide_loop(mr.vertices, mr.faces, iterations=1)
    vp2, _ = trimesh.remesh.subdivide_loop(mp.vertices, mp.faces, iterations=1)
    vr2 = np.asarray(vr2); vp2 = np.asarray(vp2); f2 = np.asarray(f2)
    vr2, f2c, used = compact(vr2, f2); vp2 = vp2[used]
    Np = vnormals(vp2, f2c)
    parts = {'body': (vp2, f2c)}
    # ---- hair
    hm = scalp_mask(vr2, B, cfg['hair'])
    fmask = hm[f2c].all(1)
    # refine hair region once more for fine curls
    sub = f2c[fmask]; uidx = np.unique(sub); rm = -np.ones(len(vp2), int); rm[uidx] = np.arange(len(uidx))
    hv_r, hf = trimesh.remesh.subdivide(vr2[uidx], rm[sub])
    hv_p, _ = trimesh.remesh.subdivide(vp2[uidx], rm[sub])
    hv_r = np.asarray(hv_r); hv_p = np.asarray(hv_p); hf = np.asarray(hf)
    hn = vnormals(hv_p, hf)
    # taper toward region border
    inside = np.ones(len(hv_r), bool)
    e = np.sort(np.concatenate([hf[:, [0, 1]], hf[:, [1, 2]], hf[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True); border = np.zeros(len(hv_r), bool); border[u[c == 1].ravel()] = True
    from scipy.sparse.csgraph import dijkstra
    import scipy.sparse as sp
    L = np.linalg.norm(hv_r[e[:, 0]] - hv_r[e[:, 1]], axis=1)
    G = sp.coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(hv_r),) * 2).tocsr()
    db = dijkstra(G, indices=np.nonzero(border)[0], min_only=True, limit=1.2)
    db = np.where(np.isfinite(db), db, 1.2)
    taper = np.clip(db / 0.30, 0, 1) ** 0.6
    if cfg['hair'] == 'curly':
        f1, f2_ = worley_bumps(hv_r, 0.26, seed=cfg.get('seed', 1))
        curl = np.clip(1 - f1, 0, 1) ** 1.6
        ridge = np.clip(f2_ - f1, 0, 1) ** 0.5
        off = taper * (0.13 + 0.24 * curl * ridge + 0.03 * noise3(hv_r, 3.0, 9))
    else:
        P = hv_r.copy()
        wav = np.sin(P[:, 1] * 7.0 + noise3(P, 2.5, 3) * 2.5 + P[:, 0] * 2.0)
        f1, _ = worley_bumps(hv_r * [1, 0.45, 1], 0.33, seed=7)
        lock = np.clip(1 - f1, 0, 1) ** 1.2
        off = taper * (0.15 + 0.06 * wav + 0.15 * lock)
    off = off - 0.02 * (1 - taper)
    hairV = hv_p + hn * off[:, None]
    parts['hair'] = (hairV, hf)
    parts['_hair_rest'] = (hv_r, hv_p, hn, off)
    # ---- cloth: hanging drape computed in posed space
    keep = (np.abs(vr2[:, 0]) < 2.3) & (vr2[:, 1] < 5.6)   # torso + legs only (rest pose)
    parts['_drape_input'] = (vp2, keep)
    return parts

def knot_parts(Vp, H, R, B, cfg):
    """small gathered knot + hanging ends on the outer hip (posed space)"""
    side = cfg['outer']; sg = 1 if side == 'L' else -1
    hip = H[f'upperleg01.{side}']; Rr = R['root']
    out = []
    # locate a body point near the iliac crest, outer side
    p = hip + Rr @ np.array([0.55 * sg, 0.85, 0.35])
    rng = np.random.default_rng(4)
    for k in range(5):
        c = p + Rr @ (rng.normal(size=3) * [0.12, 0.10, 0.12])
        v, f = trimesh.creation.icosphere(2).vertices, trimesh.creation.icosphere(2).faces
        out.append((v * [0.22, 0.17, 0.2] + c, f))
    for k, (dx, ln) in enumerate(((-0.12, 1.5), (0.14, 1.15))):
        top = p + Rr @ np.array([dx * sg, -0.05, 0.1])
        pts = [top + Rr @ np.array([0.05 * np.sin(t * 3 + k) * sg, -t * ln, 0.12 * t]) for t in np.linspace(0, 1, 9)]
        V = []; Fh = []
        for i, q in enumerate(pts):
            wdt = 0.22 * (1 - 0.45 * i / 8); tt = Rr @ np.array([1, 0, 0])
            for j in range(7):
                a = (j / 6 - 0.5) * wdt
                fold = 0.05 * np.sin(j * 1.8 + i * 0.4)
                V.append(q + tt * a + Rr @ np.array([0, 0, fold]))
            if i:
                b0 = (i - 1) * 7; b1 = i * 7
                for j in range(6): Fh += [[b0 + j, b1 + j, b1 + j + 1], [b0 + j, b1 + j + 1, b0 + j + 1]]
        V = np.array(V); Fh = np.array(Fh)
        out.append((V, np.concatenate([Fh, Fh[:, ::-1]])))
    return out

def merge(parts):
    V = []; F = []; n = 0
    for v, f in parts:
        V.append(v); F.append(f + n); n += len(v)
    return np.concatenate(V), np.concatenate(F)

def decimate(V, F, target):
    if len(F) <= target: return V, F
    s = pyfqmr.Simplify(); s.setMesh(np.asarray(V, np.float64), np.asarray(F, np.int32))
    s.simplify_mesh(target_count=target, aggressiveness=4, preserve_border=True, verbose=False)
    v, f, _ = s.getMesh(); return v, f

CFG = {
  'A': dict(inner_palm=[-0.55, 0, -1.0], overdrive=1.35, shoulder=1.10, waist=0.95, muscle=1.0, weight=0.35, height=0.8, inner='R', outer='L', free_arm='point', hair='curly', cloth='wrap',
            tilt=4.5, head_up=7, head_yaw=4, twist=-6, hip_yaw=-3, knee=0.24, folds=12, seed=2, eth=(0.65, 0.2, 0.15), target_h=1.50),
  'B': dict(inner_palm=[0.55, 0, 1.0], overdrive=1.0, shoulder=1.05, waist=0.97, muscle=0.72, weight=0.2, height=0.75, inner='L', outer='R', free_arm='relaxed', hair='wavy', cloth='drape',
            tilt=5.0, head_up=3, head_yaw=-8, head_roll=4, twist=5, hip_yaw=4, knee=0.28, folds=14, seed=5, eth=(0.55, 0.15, 0.3), target_h=1.45),
}

def build_pair(gap=0.46, hand_fwd=0.10, hand_h=0.40):
    """returns dict name->list of (V,F) in metres, pair frame: +Z forward, +X = left of the figures, feet at y=0"""
    pre = {}
    for k in ('A', 'B'):
        Vr, B, spec, H = build_figure(CFG[k])
        Vp, _, _, _ = pose(Vr, spec)
        h = Vp[:, 1].max() - Vp[:, 1].min(); s = CFG[k]['target_h'] / h
        pre[k] = (Vr, B, spec, H, s, Vp[:, 1].min())
    # A stands at +X (its own left, the right side as seen from the front), B at -X
    pos = {'A': np.array([gap / 2, 0, 0]), 'B': np.array([-gap / 2, 0, 0])}
    out = {}
    for k in ('A', 'B'):
        Vr, B, spec, H, s, y0 = pre[k]
        # hand meeting point in pair frame -> local units (before scale)
        hipA = pre['A'][3]['upperleg01.L'][1]; 
        hand_world = np.array([0.0, (hand_h + 0.0), hand_fwd])
        # height of meeting point: a bit below the shorter figure's hip
        hipB = (pre['B'][3]['upperleg01.L'][1] - pre['B'][5]) * pre['B'][4]
        hand_world[1] = hipB * 0.93
        hand_world = hand_world + (np.array([0.021, 0, 0.021]) if k == 'A' else np.array([-0.021, 0, -0.015]))
        local = (hand_world - pos[k]) / s + np.array([0, y0, 0])
        Vp, Hh, Rr = finish_figure(Vr, B, spec, CFG[k], local)
        y0 = Vp[:, 1].min()
        parts = surface_parts(Vr, Vp, B, CFG[k])
        vp2, keep = parts.pop('_drape_input')
        hv_r, hv_p, hn, hoff = parts.pop('_hair_rest')
        Rh = Rr['head']; Hh_ = Hh['head']; hr0 = B['head']['head']
        hair_rest_out = (hv_p + hn * hoff[:, None] - Hh_) @ Rh + hr0     # posed -> rest (rigid head)
        headmask = np.linalg.norm(Vr - (hr0 + (B['head']['tail'] - hr0) * 0.45), axis=1) < np.linalg.norm(B['head']['tail'] - hr0) * 0.9
        acc_rest = wreath(B, hair_rest_out, Vr[headmask], 'laurel' if CFG[k]['hair'] == 'curly' else 'fillet')
        acc_posed = [((v - hr0) @ Rh.T + Hh_, f) for v, f in acc_rest]
        knots = drape(vp2, keep, Hh, Rr, CFG[k])
        def T(v): return (v - np.array([0, y0, 0])) * s + pos[k]
        res = {}
        bv, bf = parts['body']; bv, bf = decimate(T(bv), bf, 48000)
        res['body'] = (bv, bf)
        hv, hf = parts['hair']; res['hair'] = decimate(T(hv), hf, 12000)
        res['cloth'] = merge([(T(v), f) for v, f in knots] + [(T(v), f) for v, f in acc_posed])
        out[k] = res
    return out

# ------------------------------------------------------------------ drapery (posed space, dm)
def drape(Vb, keep, H, R, cfg):
    """hanging cloth panels + twisted waist cord + side knot. Vb: posed body verts, keep: mask of torso/leg verts"""
    O = (H['upperleg01.L'] + H['upperleg01.R']) / 2
    Rr = R['root']; up = Rr @ [0, 1, 0]; fw = Rr @ [0, 0, 1]; lf = Rr @ [1, 0, 0]
    P = Vb[keep] - O
    P = P[np.linalg.norm(P - (P @ (R['root'] @ [0, 1, 0]))[:, None] * (R['root'] @ [0, 1, 0]), axis=1) < 2.2]
    py = P @ up; px = P @ lf; pz = P @ fw
    th_v = np.arctan2(px, pz); r_v = np.hypot(px, pz)
    outer = 1 if cfg['outer'] == 'L' else -1
    k_th = np.radians(80) * outer                     # knot angle (outer hip)
    wide = cfg['cloth'] == 'drape'
    def yw(th):  # waist line height above hip joints (slanted, high at knot)
        return (0.62 if not wide else 0.70) + 0.28 * np.cos(th - k_th)
    def rbody(th, y, win=np.radians(4), dy=0.07):
        m = (np.abs(np.angle(np.exp(1j * (th_v - th)))) < win) & (np.abs(py - y) < dy)
        return r_v[m].max() if m.any() else np.nan
    parts = []
    # ---- panels
    def panel(th0, th1, hem, nth=44, ny=34, off0=0.10, folds=9, seed=0):
        ths = np.linspace(th0, th1, nth)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 6.3, 3)
        V = np.zeros((nth, ny, 3))
        for i, th in enumerate(ths):
            ytop = yw(th); ybot = hem(th)
            ys = np.linspace(ytop, ybot, ny)
            rmax = 0.0
            for j, y in enumerate(ys):
                rb = rbody(th, y)
                if np.isfinite(rb): rmax = max(rmax, rb)
                depth = (ytop - y) / max(ytop - ybot, 1e-3)
                fold = np.abs(np.sin(th * folds + ph[0] + 0.6 * np.sin(y * 1.7 + ph[1]))) ** 0.8
                rr = rmax + off0 + (0.03 + 0.17 * depth) * fold + 0.05 * depth
                yy = y - 0.04 * depth * np.sin(th * folds * 0.5 + ph[2])
                V[i, j] = O + up * yy + (fw * np.cos(th) + lf * np.sin(th)) * rr
        # smooth across theta a little
        V = ndimage.uniform_filter1d(V, 3, axis=0, mode='nearest')
        Fp = []
        for i in range(nth - 1):
            for j in range(ny - 1):
                a = i * ny + j; b = (i + 1) * ny + j
                Fp += [[a, b, b + 1], [a, b + 1, a + 1]]
        Fp = np.array(Fp); Vf = V.reshape(-1, 3)
        # thickness: second sheet slightly inside
        nrm = vnormals(Vf, Fp)
        Vi = Vf - nrm * 0.03; n = len(Vf)
        Fall = [Fp, Fp + n]
        # hem roll
        hem_pts = V[:, -1]; parts.append(tube_path(hem_pts, 0.045, 8))
        side_pts = [V[0], V[-1]]
        for sp_ in side_pts: parts.append(tube_path(sp_, 0.03, 6))
        parts.append((np.concatenate([Vf, Vi]), np.concatenate(Fall)))
    if not wide:
        panel(np.radians(-34), np.radians(34), lambda th: -1.85 + 0.9 * (th / np.radians(34)) ** 2, folds=11, seed=1)
        panel(np.radians(180 - 30), np.radians(180 + 30), lambda th: -0.85 + 0.5 * ((th - np.pi) / np.radians(30)) ** 2, folds=9, seed=2)
    else:
        panel(np.radians(-46), np.radians(46), lambda th: -2.45 + 1.1 * (th / np.radians(46)) ** 2, folds=12, seed=3)
        panel(np.radians(180 - 42), np.radians(180 + 42), lambda th: -1.25 + 0.7 * ((th - np.pi) / np.radians(42)) ** 2, folds=10, seed=4)
    # ---- twisted waist cord all around
    ths = np.linspace(-np.pi, np.pi, 181)
    pts = []
    for th in ths:
        y = yw(th); rb = rbody(th, y, np.radians(5), 0.09)
        if not np.isfinite(rb): rb = 1.3
        pts.append(O + up * y + (fw * np.cos(th) + lf * np.sin(th)) * (rb + 0.13))
    pts = np.array(pts); pts = ndimage.uniform_filter1d(pts, 5, axis=0, mode='wrap')
    parts.append(tube_path(np.vstack([pts, pts[:1]]), 0.13, 12, twist=7.0, ridges=4, amp=0.035))
    # ---- knot on the outer hip with two hanging ends
    kp = pts[np.argmin(np.abs(ths - k_th))]
    outv = (fw * np.cos(k_th) + lf * np.sin(k_th))
    rng = np.random.default_rng(11)
    for k in range(4):
        c = kp + outv * (0.08 + 0.05 * k % 2) + up * rng.uniform(-0.12, 0.08) + np.cross(up, outv) * rng.uniform(-0.15, 0.15)
        ico = trimesh.creation.icosphere(2)
        parts.append((ico.vertices * [0.21, 0.17, 0.21] @ np.eye(3) + c, ico.faces))
    side_t = np.cross(up, outv)
    for k, (dx, ln, wd) in enumerate(((-0.10, 1.55, 0.26), (0.13, 1.15, 0.22))):
        top = kp + outv * 0.12 + side_t * dx
        V = []; Fh = []
        nseg = 12
        for i in range(nseg):
            t = i / (nseg - 1)
            q = top - up * (t * ln) + outv * (0.10 * t) + side_t * (0.06 * np.sin(t * 3 + k))
            wdt = wd * (1 - 0.35 * t)
            for j in range(7):
                a = (j / 6 - 0.5) * wdt
                V.append(q + side_t * a + outv * (0.05 * np.sin(j * 1.9 + i * 0.5)))
            if i:
                b0 = (i - 1) * 7; b1 = i * 7
                for j in range(6): Fh += [[b0 + j, b1 + j, b1 + j + 1], [b0 + j, b1 + j + 1, b0 + j + 1]]
        V = np.array(V); Fh = np.array(Fh)
        parts.append((V, Fh))
    return parts

def tube_path(pts, r, nc=8, twist=0.0, ridges=0, amp=0.0):
    pts = np.asarray(pts, float); n = len(pts)
    T = np.gradient(pts, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([0, 1.0, 0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0, 0])
    N0 = np.cross(T[0], ref); N0 /= np.linalg.norm(N0)
    Ns = [N0]
    for i in range(1, n):
        v = np.cross(Ns[-1], T[i]); Nn = np.cross(T[i], v); Nn /= np.linalg.norm(Nn) + 1e-12; Ns.append(Nn)
    Ns = np.array(Ns); Bs = np.cross(T, Ns)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    V = []; F = []
    for i in range(n):
        for j in range(nc):
            a = 2 * np.pi * j / nc
            rr = r * (1 + (amp / r) * np.sin(ridges * a + twist * s[i])) if ridges else r
            V.append(pts[i] + (Ns[i] * np.cos(a) + Bs[i] * np.sin(a)) * rr)
        if i:
            for j in range(nc):
                a0 = (i - 1) * nc + j; a1 = (i - 1) * nc + (j + 1) % nc; b0 = i * nc + j; b1 = i * nc + (j + 1) % nc
                F += [[a0, b0, b1], [a0, b1, a1]]
    return np.array(V), np.array(F)

# ------------------------------------------------------------------ head accessories (rest -> posed via head bone)
def leaf_mesh(L=0.5, W=0.17, n=7):
    """lanceolate laurel leaf along +X, folded on the midrib, normal ~ +Z"""
    V = [];
    ts = np.linspace(0, 1, n)
    for t in ts:
        w = W * np.sin(np.pi * t ** 0.8) * (1 - 0.15 * t)
        V += [[t * L, -w / 2, 0.02 * w / W], [t * L, 0, 0.05], [t * L, w / 2, 0.02 * w / W]]
    F = []
    for i in range(n - 1):
        a = i * 3; b = a + 3
        F += [[a, b, b + 1], [a, b + 1, a + 1], [a + 1, b + 1, b + 2], [a + 1, b + 2, a + 2]]
    return np.array(V), np.array(F)

def head_ring(B, hair_rest_pts, Vr_head, y_front, y_back, lift=0.05, n=120):
    hh = B['head']['head']; ht = B['head']['tail']
    C = hh + (ht - hh) * 0.45 + np.array([0, 0, -0.15])
    pts = np.concatenate([hair_rest_pts, Vr_head])
    d = pts - C; ang = np.arctan2(d[:, 0], d[:, 2]); rad = np.hypot(d[:, 0], d[:, 2])
    phis = np.linspace(-np.pi, np.pi, n, endpoint=False); ring = []
    for ph in phis:
        y = y_back + (y_front - y_back) * (0.5 + 0.5 * np.cos(ph))
        m = (np.abs(np.angle(np.exp(1j * (ang - ph)))) < 0.09) & (np.abs(pts[:, 1] - y) < 0.12)
        r = rad[m].max() if m.any() else 1.0
        ring.append(C + np.array([np.sin(ph) * (r + lift), y - C[1], np.cos(ph) * (r + lift)]))
    ring = np.array(ring)
    ring = ndimage.uniform_filter1d(ring, 7, axis=0, mode='wrap')
    return phis, ring

def wreath(B, hair_rest_pts, Vr_head, kind):
    hh = B['head']['head']; ht = B['head']['tail']; hl = np.linalg.norm(ht - hh)
    yf = hh[1] + hl * 0.62; yb = hh[1] + hl * 0.78
    phis, ring = head_ring(B, hair_rest_pts, Vr_head, yf, yb, lift=0.04 if kind == 'laurel' else 0.015)
    out = []
    if kind == 'fillet':
        out.append(tube_path(np.vstack([ring, ring[:1]]), 0.055, 6))
        back = ring[np.argmin(np.abs(np.abs(phis) - np.pi))]
        for k, (dx, ln) in enumerate(((-0.08, 1.0), (0.09, 0.8))):
            pts = [back + np.array([dx + 0.04 * np.sin(t * 4 + k), -t * ln, -0.06 * t - 0.03 * np.sin(t * 5)]) for t in np.linspace(0, 1, 10)]
            V = []; F = []
            for i, q in enumerate(pts):
                V += [q + [-0.06, 0, 0], q + [0.06, 0, 0]]
                if i: a = 2 * (i - 1); F += [[a, a + 2, a + 3], [a, a + 3, a + 1]]
            out.append((np.array(V), np.array(F)))
        return out
    # laurel: two rows of leaves sweeping back from the front centre, tied at the back
    LV, LF = leaf_mesh()
    rng = np.random.default_rng(21)
    for side in (-1, 1):
        sel = [i for i, ph in enumerate(phis) if 0.05 < side * ph < 2.55]
        sel = sorted(sel, key=lambda i: side * phis[i])
        for k, i in enumerate(sel[::2]):
            p = ring[i]; j = min(len(ring) - 1, max(0, i + side * 2)) % len(ring)
            tang = ring[(i + side) % len(ring)] - ring[(i - side) % len(ring)]; tang /= np.linalg.norm(tang)
            outw = p - np.array([B['head']['head'][0], p[1], B['head']['head'][2]]); outw /= np.linalg.norm(outw)
            upv = np.cross(outw, tang); upv /= np.linalg.norm(upv)
            if upv[1] < 0: upv = -upv
            for row in (-1, 1):
                a = np.radians(28 * row + rng.uniform(-8, 8))
                dirv = tang * np.cos(a) + upv * np.sin(a)
                nrm = np.cross(dirv, upv if row > 0 else -upv); nrm = outw * 0.8 + nrm * 0.2; nrm /= np.linalg.norm(nrm)
                side_v = np.cross(nrm, dirv); side_v /= np.linalg.norm(side_v)
                M = np.stack([dirv, side_v, nrm], 1)
                sc = 0.85 + 0.3 * rng.random()
                out.append(((LV * sc) @ M.T + p + upv * 0.02 * row, LF))
        # berries
        for i in sel[1::6]:
            ico = trimesh.creation.icosphere(1)
            out.append((ico.vertices * 0.045 + ring[i] + np.array([0, 0.0, 0]), ico.faces))
    back = ring[np.argmin(np.abs(np.abs(phis) - np.pi))]
    for k, (dx, ln) in enumerate(((-0.07, 0.9), (0.08, 0.7))):
        pts = [back + np.array([dx + 0.03 * np.sin(t * 4 + k), -t * ln, -0.05 * t]) for t in np.linspace(0, 1, 9)]
        V = []; F = []
        for i, q in enumerate(pts):
            V += [q + [-0.045, 0, 0], q + [0.045, 0, 0]]
            if i: a = 2 * (i - 1); F += [[a, a + 2, a + 3], [a, a + 3, a + 1]]
        out.append((np.array(V), np.array(F)))
    return out
