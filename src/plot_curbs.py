import sys, json; sys.path.insert(0,'.')
from geo import *
from esri import mosaic
from shapely.geometry import shape
import shapely, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def proj(g): return shapely.transform(g, lambda a: np.array(TM.transform(a[:,0],a[:,1])).T)
im,px=mosaic(20)
x0,x1,y0,y1=[float(v) for v in sys.argv[1:5]]
a=px(x0,y1); b=px(x1,y0)
fig,ax=plt.subplots(figsize=(14,14*(y1-y0)/(x1-x0)))
ax.imshow(im.crop((int(a[0]),int(a[1]),int(b[0]),int(b[1]))),extent=(x0,x1,y0,y1))
for p in json.load(open('../data/row.json')):
    g=proj(shape(p['the_geom']))
    for gg in getattr(g,'geoms',[g]): ax.fill(*gg.exterior.xy,color='cyan',alpha=0.15); ax.plot(*gg.exterior.xy,color='cyan',lw=0.8)
for p in json.load(open('../data/curbs.json')):
    g=proj(shape(p['the_geom']))
    for gg in getattr(g,'geoms',[g]): ax.plot(*gg.xy,color='red' if p['layer']=='CURBS' else 'yellow',lw=1.2)
for p in json.load(open('../data/bldg_sf.json')):
    g=proj(shape(p['shape']))
    for gg in getattr(g,'geoms',[g]): ax.plot(*gg.exterior.xy,color='orange',lw=0.6)
ax.plot([X0,X1,X1,X0,X0],[Y0,Y0,Y1,Y1,Y0],'w-',lw=2)
ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect('equal'); ax.grid(alpha=0.3)
plt.savefig(sys.argv[5],dpi=80,bbox_inches='tight')
