import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Resample NAIP onto the model grid (roof colours and the viewer's 'Aerial photo' layer)."""
import sys; sys.path.insert(0, ROOT + '/src')
from common import *
import pyproj
from PIL import Image
U = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
bx = list(map(float, open(ROOT + '/data/utm_bbox.txt').read().split('\n')[0].split(',')))
naip = np.asarray(Image.open(ROOT + '/data/naip.png').convert('RGB')).astype(np.float32)
sN = naip.shape[1] / (bx[2] - bx[0])
N = 1024; xs = X0 + (np.arange(N) + 0.5) * (X1 - X0) / N; ys = Y1 - (np.arange(N) + 0.5) * (Y1 - Y0) / N
X, Y = np.meshgrid(xs, ys); lo, la = ll(X.ravel(), Y.ravel()); e, n = U.transform(lo, la)
c = (np.asarray(e) - bx[0]) * sN - 0.5; r = (bx[3] - np.asarray(n)) * sN - 0.5
out = np.stack([ndimage.map_coordinates(naip[..., k], [r, c], order=3, mode='nearest').reshape(N, N) for k in range(3)], -1)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(ROOT + '/web/aerial_naip.jpg', quality=85)
