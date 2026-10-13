import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Compress the GLB and build the static viewer in docs/ (served by GitHub Pages).
Needs gltfpack (npm install gltfpack@0.22.0)."""
import os, shutil, subprocess, sys
W = ROOT + '/web/'; O = ROOT + '/docs/'; M = ROOT + '/model/'
os.makedirs(O, exist_ok=True); os.makedirs(M, exist_ok=True)
subprocess.run(['npx', '--yes', 'gltfpack@0.22.0', '-i', W + 'castro.glb', '-o', W + 'castro_q.glb', '-kn', '-km', '-ke', '-kv', '-vtf', '-vp', '16'], check=True)
subprocess.run([sys.executable, ROOT + '/src/pack_text.py', W + 'castro_q.glb', O + 'castro_model.txt'], check=True)
from PIL import Image
Image.open(ROOT + '/data/ground_tex.jpg').save(O + 'ground_tex.jpg', quality=84)
for f in ['aerial_naip.jpg', 'plaza_paving.jpg', 'mural.jpg', 'brick_rim.jpg', 'labels.json', 'views.json'] + [f'stone_{k}.jpg' for k in range(6)]:
    shutil.copy(W + f, O + f)
s = open(W + 'index.src.html').read()
s = s.replace('const TOTAL_MB=%s;' % s.split('const TOTAL_MB=')[1].split(';')[0], 'const TOTAL_MB=%.2f;' % (os.path.getsize(O + 'castro_model.txt') / 1e6))
open(W + 'index.src.html', 'w').write(s)
HEAD = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
open(O + 'index.html', 'w').write(HEAD + s.replace('__THREE__', 'https://cdn.jsdelivr.net/npm/three@0.170.0'))
open(O + '.nojekyll', 'w').close()
shutil.copy(W + 'castro.glb', M + 'castro-market-model.glb'); shutil.copy(W + 'castro_q.glb', M + 'castro-market-model-quantized.glb')
print('viewer built in docs/')
