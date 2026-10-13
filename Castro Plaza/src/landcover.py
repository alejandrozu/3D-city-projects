import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from geo import *
from osm_load import load
import shapely
from shapely.geometry import shape, box, LineString, Polygon, Point, MultiPolygon
from shapely.ops import unary_union, polygonize
def proj(g): return shapely.transform(g, lambda a: np.array(TM.transform(a[:,0],a[:,1])).T)
SQ=box(X0,Y0,X1,Y1)
BIG=SQ.buffer(30,join_style=2)
def curb_corrections(roadway):
    """Replace outdated curb lines with curbs measured from the 2023 LiDAR (curb-step pixels, see curbfit.py)."""
    import os
    p=ROOT + '/data/curbfit.json'
    if not os.path.exists(p): return roadway
    C=json.load(open(p))
    ae,be=C['castro_east']; aw,bw=C['castro_west']; asw,bsw=C['17th_south_w']; ase,bse=C['17th_south_e']
    xe=lambda y: ae*y+be; xw=lambda y: aw*y+bw; ys=lambda x: asw*x+bsw; ye=lambda x: ase*x+bse
    Y0b=Y0-40
    # SE block of Castro & 17th: east of the Castro east curb and south of the 17th south curb, corner rounded (r = 3 m)
    pe=Polygon([(xe(Y0b),Y0b)]+[(xe(y),y) for y in np.linspace(Y0b,-20,60)]+[(x,ys(x)) for x in np.linspace(xe(-20),94,60)]+[(94,Y0b)])
    xc=xe(ys(8.0)); 
    corner=Polygon([(xe(Y0b),Y0b)]+[(xe(y),y) for y in np.linspace(Y0b,-10,40)]+[(x,ys(x)) for x in np.linspace(xe(-10)-3,94,40)]+[(94,Y0b)])
    blockSE=pe.buffer(-3.0,join_style=1).buffer(3.0,join_style=1)
    # make sure the far (south / east) edges are not shrunk by the rounding
    blockSE=blockSE.union(Polygon([(xe(Y0b)+3,Y0b),(xe(-40)+3,-40),(90,-40),(90,Y0b)]))
    # Castro west curb south of Market: tapers back into the old curb return between y=-40 and -31
    rw=[ (g.bounds[0]) for g in getattr(roadway.intersection(LineString([(-12,-31),(6,-31)]).buffer(0.01)),'geoms',[roadway.intersection(LineString([(-12,-31),(6,-31)]).buffer(0.01))]) if not g.is_empty]
    x31=min(rw) if rw else xw(-31)
    blockW=Polygon([(-60,Y0b),(xw(Y0b),Y0b),(xw(-40),-40),(x31,-31),(-60,-31)])
    # 17th south curb east of Hartford
    blockE=Polygon([(110,ye(110)),(X1+40,ye(X1+40)),(X1+40,Y0b),(110,Y0b)])
    fix=unary_union([blockSE,blockW,blockE])
    return roadway.difference(fix)
def landcover():
    nodes,ntags,ways,rels=load()
    parcels=[proj(shape(p['shape'])) for p in json.load(open('data/parcels_sf.json'))]
    lots=unary_union([p.buffer(0) for p in parcels]).intersection(BIG)
    curbs=[]; islands=[]
    for p in json.load(open('data/curbs.json')):
        g=proj(shape(p['the_geom']))
        for gg in getattr(g,'geoms',[g]):
            if p['layer']=='ISLANDS': islands.append(gg)
            elif p['layer']=='CURBS': curbs.append(gg)
    lines=unary_union([c.intersection(BIG) for c in curbs+islands]+[BIG.boundary])
    faces=list(polygonize(lines))
    # road centerlines
    roads=[]
    for wid,(nds,tg) in ways.items():
        if tg.get('highway') in ('primary','secondary','tertiary','residential','primary_link','unclassified') and len(nds)>1:
            roads.append(LineString([nodes[n] for n in nds if n in nodes]))
    roadlines=unary_union(roads).intersection(BIG)
    road_faces=[]; other=[]
    for f in faces:
        L=f.intersection(roadlines).length
        if L>3 and f.area>150: road_faces.append(f)
        else: other.append(f)
    roadway=unary_union(road_faces)
    # islands: polygons from island rings
    isl=unary_union([Polygon(i.coords) for i in islands if i.is_ring] + [f for f in other if any(f.buffer(-0.01).within(Polygon(i.coords)) for i in islands if i.is_ring)])
    roadway=roadway.difference(isl)
    roadway=curb_corrections(roadway)
    row=BIG.difference(lots)
    sidewalk=row.difference(roadway).difference(isl)
    return dict(lots=lots,roadway=roadway,sidewalk=sidewalk,islands=isl,faces=faces,roads=roadlines)
if __name__=='__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    L=landcover()
    fig,ax=plt.subplots(figsize=(13,13))
    def fill(g,c,a=0.6):
        for gg in getattr(g,'geoms',[g]):
            if gg.geom_type!='Polygon' or gg.is_empty: continue
            ax.fill(*gg.exterior.xy,color=c,alpha=a,lw=0)
            for h in gg.interiors: ax.fill(*h.xy,color='white',alpha=1,lw=0)
    fill(L['lots'],'tan'); fill(L['sidewalk'],'lightgray',0.9); fill(L['roadway'],'dimgray'); fill(L['islands'],'green')
    for gg in getattr(L['roads'],'geoms',[L['roads']]): ax.plot(*gg.xy,'y-',lw=0.6)
    ax.plot(*SQ.exterior.xy,'k-',lw=2)
    ax.set_aspect('equal'); ax.set_xlim(X0-30,X1+30); ax.set_ylim(Y0-30,Y1+30)
    plt.savefig('data/landcover.png',dpi=65,bbox_inches='tight')
    print({k:round(v.area,1) for k,v in L.items() if hasattr(v,'area') and not isinstance(v,list)})
