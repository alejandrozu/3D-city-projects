import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from landcover import landcover
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box as sbox
from shapely.ops import unary_union, nearest_points, linemerge
L=landcover()
road=L['roadway']
P={p['mapblklot']:proj(shape(p['shape'])).buffer(0) for p in json.load(open('data/parcels_sf.json'))}
chev=unary_union([P['3563023'],P['3563022']])
nonroad=sbox(X0-30,Y0-30,X1+30,Y1+30).difference(road)
blk=[g for g in getattr(nonroad,'geoms',[nonroad]) if g.intersects(chev)][0]
others=unary_union([q for k,q in P.items() if k not in ('3563023','3563022') and q.intersects(blk.buffer(1))])
# shared east lot line: part of chevron boundary touching other parcels
shared=chev.boundary.intersection(others.buffer(0.3))
segs=[g for g in getattr(shared,'geoms',[shared]) if g.length>0.5]
pts=np.concatenate([np.array(g.coords) for g in segs])
# fit line through shared boundary, extend
c=pts.mean(0); u,sv,vt=np.linalg.svd(pts-c); d=vt[0]
cut=LineString([c-d*200,c+d*200])
from shapely.ops import split
pieces=split(blk,cut)
tri=[g for g in pieces.geoms if g.intersects(chev.buffer(-1))]
tri=max(tri,key=lambda g:g.intersection(chev).area)
tri=tri.difference(others)
tri=max(getattr(tri,'geoms',[tri]),key=lambda g:g.area)
print('triangle (curb to lot line) %.1f m2, chevron lots %.1f'%(tri.area,chev.area))
# 5 m from the roadway
far=tri.difference(road.buffer(5.0,resolution=32))
far=max(getattr(far,'geoms',[far]),key=lambda g:g.area)
rounded=far.buffer(-3.0,resolution=32).buffer(3.0,resolution=32)
# the two eastern corners (at the wall) stay square: within 7 m of the wall line use the unrounded outline
wall_zone=LineString([c-d*200,c+d*200]).buffer(7.0,cap_style=2)
plaza=unary_union([rounded,far.intersection(wall_zone)]).buffer(0)
plaza=max(getattr(plaza,'geoms',[plaza]),key=lambda g:g.area).simplify(0.02)
print('plaza %.1f m2, min dist to road %.2f'%(plaza.area, plaza.distance(road)))
# wall side: plaza boundary within 0.6 m of the extended lot line
pickle.dump(dict(tri=tri,plaza=plaza,road=road,chev=chev,cutline=(c,d),others=others),open('data/plaza_shape.pkl','wb'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,9))
for g in getattr(road,'geoms',[road]): ax.fill(*g.exterior.xy,color='#555',alpha=.5)
ax.fill(*tri.exterior.xy,color='orange',alpha=.3); ax.plot(*chev.exterior.xy,'r--')
ax.fill(*plaza.exterior.xy,color='green',alpha=.5)
for q in getattr(others,'geoms',[others]): ax.plot(*q.exterior.xy,'k-',lw=.5)
for p in json.load(open('data/bldg_sf.json')):
    g=proj(shape(p['shape']))
    for gg in getattr(g,'geoms',[g]): ax.plot(*gg.exterior.xy,'b-',lw=.8)
ax.set_xlim(10,95); ax.set_ylim(-30,45); ax.set_aspect('equal'); ax.grid(alpha=.3)
plt.savefig('data/plaza_shape.png',dpi=60,bbox_inches='tight')
