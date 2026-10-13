import sys, json; sys.path.insert(0,'.')
from osm_load import *
from esri import mosaic
from shapely.geometry import shape
import shapely, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
nodes,ntags,ways,rels=load()
def proj(g): return shapely.transform(g, lambda a: np.array(TM.transform(a[:,0],a[:,1])).T)
im,px=mosaic(20)
x0,x1,y0,y1=[float(v) for v in sys.argv[1:5]] if len(sys.argv)>4 else (X0-10,X1+10,Y0-10,Y1+10)
a=px(x0,y1); b=px(x1,y0)
crop=im.crop((int(a[0]),int(a[1]),int(b[0]),int(b[1])))
fig,ax=plt.subplots(figsize=(14,14*(y1-y0)/(x1-x0)))
ax.imshow(crop,extent=(x0,x1,y0,y1),alpha=0.75)
for p in json.load(open('../data/parcels_sf.json')):
    g=proj(shape(p['shape']))
    for gg in getattr(g,'geoms',[g]): ax.plot(*gg.exterior.xy,color='yellow',lw=0.6)
for p in json.load(open('../data/bldg_sf.json')):
    g=proj(shape(p['shape']))
    for gg in getattr(g,'geoms',[g]):
        ax.plot(*gg.exterior.xy,color='red',lw=1)
    c=g.representative_point()
    if x0<c.x<x1 and y0<c.y<y1: ax.text(c.x,c.y,'%.0f'%float(p['hgt_median_m']),color='red',fontsize=7,ha='center')
col={'primary':'blue','secondary':'cyan','tertiary':'cyan','residential':'white','service':'orange','footway':'lime','pedestrian':'magenta','steps':'pink','primary_link':'blue'}
for wid,(nds,tg) in ways.items():
    pts=np.array([nodes[n] for n in nds if n in nodes])
    if len(pts)<2: continue
    if 'highway' in tg: ax.plot(pts[:,0],pts[:,1],color=col.get(tg['highway'],'gray'),lw=1.2 if tg['highway']!='footway' else 0.7)
    if tg.get('railway') in ('tram','light_rail','disused'): ax.plot(pts[:,0],pts[:,1],color='black',lw=0.8,ls='--')
    if tg.get('amenity')=='fuel': ax.plot(pts[:,0],pts[:,1],color='orange',lw=2)
for k,tg in ntags.items():
    p=nodes[k]
    if not (x0<p[0]<x1 and y0<p[1]<y1): continue
    if 'name' in tg and ('amenity' in tg or 'shop' in tg or 'tourism' in tg): ax.text(p[0],p[1],tg['name'],fontsize=6,color='k',bbox=dict(fc='white',alpha=0.6,pad=0.5))
    if tg.get('natural')=='tree': ax.plot(*p,'g^',ms=4)
for t in json.load(open('../data/trees_sf.json')):
    p=xy(float(t['longitude']),float(t['latitude'])); ax.plot(*p,'o',color='green',ms=3)
ax.plot([X0,X1,X1,X0,X0],[Y0,Y0,Y1,Y1,Y0],'k-',lw=2); ax.plot(0,0,'k+',ms=15); ax.plot(CX,CY,'kx',ms=10)
ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect('equal'); ax.grid(alpha=0.3)
plt.savefig(sys.argv[5] if len(sys.argv)>5 else '../data/plan.png',dpi=90,bbox_inches='tight')
