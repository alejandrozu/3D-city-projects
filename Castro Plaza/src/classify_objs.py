import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from common import *
from PIL import Image, ImageDraw
V=np.load('data/voxels.npz'); cnt=V['cnt']
O=np.load('data/objpts.npz'); Q,H=O['Q'],O['H']
VS=0.5; OX,OY=X0-2,Y0-2
# footprint distance (all non-canopy footprints)
ny,nx=DTM.shape
img=Image.new('L',(nx,ny),0); d=ImageDraw.Draw(img)
def tp(cs): return [((x-GX0)/RES-0.5,(y-GY0)/RES-0.5) for x,y in cs]
CANOPY_IDS={'201006.0059932','201006.0158617','201006.0158847'}
for p in json.load(open('data/bldg_sf.json')):
    if p['sf16_bldgid'] in CANOPY_IDS: continue
    g=proj(shape(p['shape']))
    for gg in getattr(g,'geoms',[g]): d.polygon(tp(gg.exterior.coords),fill=1)
FPm=np.array(img).astype(bool)
FD=ndimage.distance_transform_edt(~FPm)*RES
np.save('data/fp_dist.npy',FD)
lab,n=ndimage.label(cnt>=2,structure=np.ones((3,3,3)))
I,J,K=np.nonzero(lab); L=lab[I,J,K]
x=OX+(I+0.5)*VS; y=OY+(J+0.5)*VS; h=2.3+(K+0.5)*VS
fd=sample(FD,x,y,order=0)
order=np.argsort(L); L_s=L[order]; bounds=np.searchsorted(L_s,np.arange(1,n+2))
comps=[]
for c in range(n):
    s=order[bounds[c]:bounds[c+1]]
    if len(s)==0: continue
    cx,cy,ch=x[s],y[s],h[s]
    ext=max(np.ptp(cx),np.ptp(cy))+VS; vext=np.ptp(ch)+VS
    xy=np.column_stack([cx,cy])
    if len(s)>=3:
        ev=np.sort(np.linalg.eigvalsh(np.cov(xy.T)+1e-6*np.eye(2)))[::-1]; lin=ev[0]/ev[1]
    else: lin=1
    pv=cnt[I[s],J[s],K[s]].mean()
    # lower-part width (first 2.5 m of the component)
    low=ch<=ch.min()+2.0; lw=max(np.ptp(cx[low]),np.ptp(cy[low]))+VS
    comps.append(dict(id=c+1,n=len(s),cx=float(cx.mean()),cy=float(cy.mean()),hmin=float(ch.min()),hmax=float(ch.max()),ext=float(ext),vext=float(vext),lin=float(lin),pv=float(pv),fd=float(np.median(fd[s])),fdmax=float(fd[s].max()),lw=float(lw)))
def kind(c):
    if c['fd']<1.0 and c['fdmax']<2.6: return 'appendage'
    if c['n']<=3: return 'noise'
    if c['lin']>15 and c['vext']<=1.5 and c['pv']<6: return 'wire'
    if c['lw']<=1.25 and c['vext']>=1.5 and c['ext']<=3.0: return 'pole'
    if c['ext']>=1.5 and c['vext']>=1.0 and c['n']>=12: return 'tree'
    if c['vext']>=1.0: return 'small'
    return 'misc'
import collections
for c in comps: c['kind']=kind(c)
print(collections.Counter(c['kind'] for c in comps))
json.dump(comps,open('data/components.json','w'))
np.save('data/voxlab.npy',lab)
