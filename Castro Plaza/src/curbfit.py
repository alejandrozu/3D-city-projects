import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0, ROOT + '/src')
from common import *
raw = R['dtm']
zmax = ndimage.maximum_filter(np.nan_to_num(raw, nan=-1e3), 3); zmin = ndimage.minimum_filter(np.nan_to_num(raw, nan=1e3), 3)
step = np.where(np.isfinite(raw), zmax - zmin, 0)
M = (step > 0.07) & (step < 0.35)
jj, ii = np.nonzero(M); PX = GX0 + (ii + 0.5) * RES; PY = GY0 + (jj + 0.5) * RES
def fit(pred, var, rng_, tol=0.7, axis='x'):
    """fit coordinate `axis` as a linear function of the other within a corridor around pred(other)"""
    o = PY if axis == 'x' else PX; c = PX if axis == 'x' else PY
    sel = (o >= rng_[0]) & (o <= rng_[1]) & (np.abs(c - pred(o)) < tol)
    a, b = np.polyfit(o[sel], c[sel], 1)
    for _ in range(4):
        r = np.abs(c[sel] - (a * o[sel] + b)); keep = r < max(0.12, np.percentile(r, 70))
        a, b = np.polyfit(o[sel][keep], c[sel][keep], 1)
    return a, b, int(sel.sum())
out = {}
out['castro_east'] = fit(lambda y: 10.2 - 0.094 * (y + 45), None, (-100, -24))
out['castro_west'] = fit(lambda y: -2.0 - 0.09 * (y + 45), None, (-100, -36))
out['17th_south_w'] = fit(lambda x: -21.3 + (x - 70) * 0.045, None, (26, 93), tol=1.0, axis='y')
out['17th_south_e'] = fit(lambda x: -17.0 + (x - 120) * 0.045, None, (110, 131), tol=1.0, axis='y')
for k, (a, b, n) in out.items(): print(k, 'slope %.4f intercept %.2f  n=%d' % (a, b, n))
json.dump({k: [a, b] for k, (a, b, n) in out.items()}, open(ROOT + '/data/curbfit.json', 'w'))
# quick check values
a, b, _ = out['castro_east']; print('castro east x at y=-30,-45,-70,-92:', [round(a * y + b, 2) for y in (-30, -45, -70, -92)])
a, b, _ = out['castro_west']; print('castro west x at y=-40,-70,-92:', [round(a * y + b, 2) for y in (-40, -70, -92)])
a, b, _ = out['17th_south_w']; print('17th south y at x=30,70,93:', [round(a * x + b, 2) for x in (30, 70, 93)])
a, b, _ = out['17th_south_e']; print('17th south y at x=110,130:', [round(a * x + b, 2) for x in (110, 130)])
