import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from geo import *
import shapely
from shapely.geometry import shape
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=np.load('data/rasters.npz'); gx0,gy0,res=float(R['gx0']),float(R['gy0']),float(R['res'])
inten=R['inten']; dsm=R['dsm_last']; dtm=R['dtm_f']
def proj(g): return shapely.transform(g, lambda a: np.array(TM.transform(a[:,0],a[:,1])).T)
ny,nx=inten.shape; ext=(gx0,gx0+nx*res,gy0,gy0+ny*res)
# intensity image
v=np.nan_to_num(inten,nan=0); lo,hi=np.percentile(v[v>0],[2,99.5])
img=np.clip((v-lo)/(hi-lo),0,1)
Image.fromarray((np.flipud(img)*255).astype(np.uint8)).save('data/intensity.png')
# hillshade of dsm
d=np.where(np.isfinite(dsm),dsm,dtm)
gy,gx=np.gradient(d,res); sl=np.pi/2-np.arctan(np.hypot(gx,gy)); asp=np.arctan2(-gx,gy)
az,alt=np.radians(315),np.radians(45)
hs=np.sin(alt)*np.sin(sl)+np.cos(alt)*np.cos(sl)*np.cos(az-asp)
fig,ax=plt.subplots(figsize=(16,16))
ax.imshow(np.clip(hs,0,1),extent=ext,origin='lower',cmap='gray')
for p in json.load(open('data/bldg_sf.json')):
    g=proj(shape(p['shape']))
    for gg in getattr(g,'geoms',[g]): ax.plot(*gg.exterior.xy,color='red',lw=0.7)
for p in json.load(open('data/curbs.json')):
    g=proj(shape(p['the_geom']))
    for gg in getattr(g,'geoms',[g]): ax.plot(*gg.xy,color='cyan',lw=0.8)
ax.set_xlim(X0,X1); ax.set_ylim(Y0,Y1); ax.set_aspect('equal')
plt.savefig('data/hillshade.png',dpi=70,bbox_inches='tight')
