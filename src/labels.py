import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from osm_load import load
from shapely.geometry import Polygon, Point, LineString, box as sbox
nodes,ntags,ways,rels=load()
meta=json.load(open('data/buildings_meta.json'))
B={p['sf16_bldgid']:proj(shape(p['shape'])).buffer(0) for p in json.load(open('data/bldg_sf.json'))}
# attach named OSM building ways to footprints
for wid,(nds,tg) in ways.items():
    if 'name' in tg and 'building' in tg and len(nds)>3:
        pg=Polygon([nodes[n] for n in nds if n in nodes]); c=pg.centroid
        for m in meta:
            if B[m['id']].contains(c) and tg['name'] not in m['names']: m['names'].insert(0,tg['name'])
json.dump(meta,open('data/buildings_meta.json','w'),indent=0)
def gz(x,y): return float(ground(np.array([x]),np.array([y]))[0])
def roof(name_sub):
    for m in meta:
        if any(name_sub in n for n in m['names']+m['addr']): return m
L=[]
def add(name,sub,x,y,z,cat):
    L.append(dict(name=name,sub=sub,x=round(float(x),2),y=round(float(y),2),z=round(float(z),2),cat=cat))
g0=gz(0,0); add('Castro St & Market St','Intersection · origin of the local grid (0, 0)',0,0,g0+0.5,'origin')
add('Square centre','30 m east of the intersection',CX,CY,gz(CX,CY)+0.5,'origin')
chev=json.load(open('data/parcels_sf.json'))
from shapely.ops import unary_union
lots=unary_union([proj(shape(p['shape'])).buffer(0) for p in chev if p['mapblklot'] in ('3563023','3563022')])
osm_chev=Polygon([nodes[n] for n in ways['256903890'][0]])
PZ=pickle.load(open('data/plaza.pkl','rb')) if True else None
fc=PZ['fountain']['center']
add('New plaza','Former Chevron triangle · %.0f m²'%PZ['plaza'].area,fc[0],fc[1],PZ['fountain']['statue_top']+0.6,'focus')
m=roof('The Café'); add('The Café','2369 Market St · bar',76.8,30.0,m['roof_top']+0.8,'poi')
m=roof('Castro Theatre')
if m: add('Castro Theatre','429 Castro St · 1922',m['cx'],m['cy'],m['roof_top']+0.8,'poi')
m=roof('Twin Peaks Tavern'); add('Twin Peaks Tavern','401 Castro St',m['cx']-8,m['cy']+4,m['roof_top']+0.8,'poi')
x,y=nodes['7166033408']; add('Rainbow flag','Harvey Milk Plaza · 25 m pole',x,y,gz(x,y)+26,'poi')
hm=Polygon([nodes[n] for n in ways['225526801'][0]]).centroid; add('Harvey Milk Plaza','Castro Station entrance',hm.x,hm.y,gz(hm.x,hm.y)+2,'place')
jw=Polygon([nodes[n] for n in ways['188964530'][0]]).centroid; add('Jane Warner Plaza','',jw.x,jw.y,gz(jw.x,jw.y)+2,'place')
add('F line terminal','17th St & Castro St',50,-17.5,gz(50,-17.5)+3.5,'place')
# street names: midpoint of the longest in-square piece of each named street
SQ=sbox(X0+6,Y0+6,X1-6,Y1-6)
streets={}
for wid,(nds,tg) in ways.items():
    if tg.get('highway') in ('primary','secondary','tertiary','residential') and 'name' in tg:
        ls=LineString([nodes[n] for n in nds if n in nodes]).intersection(SQ)
        for l in getattr(ls,'geoms',[ls]):
            if l.is_empty: continue
            if tg['name'] not in streets or l.length>streets[tg['name']].length: streets[tg['name']]=l
for nm,l in streets.items():
    if l.length<15: continue
    p=l.interpolate(0.72 if nm=='Market Street' else 0.5,normalized=True); q=l.interpolate(min(1,0.72 if nm=='Market Street' else 0.5)+0.02,normalized=True)
    ang=np.degrees(np.arctan2(q.y-p.y,q.x-p.x))
    add(nm.replace('Street','St'),'%.0f°'%ang,p.x,p.y,gz(p.x,p.y)+0.3,'street')
import pickle
A=json.load(open('data/chev_areas.json'))
st=PZ['fountain']['stones']
ps=PZ['fountain']['pool_stones']
plaza=dict(title='New plaza · former Chevron triangle',rows=[
 ['Outline','Every point 5 m from the roadway; east side is the wall of 3970 17th St'],
 ['Area','%.0f m² (old station lots %d m², curb-to-curb triangle %d m²)'%(PZ['plaza'].area,A['lots'],A['tri'])],
 ['Planters','24 stainless-steel boxes, 70 cm tall and 70 cm deep, flush with the edge, in 4 stretches of 6 with 10 cm joints; red at the corners (cut to meet the wall), purple at the openings'],
 ['Openings','The point is open, plus a 5 m gap in the middle of the north and south sides'],
 ['Flowers','Red geraniums, orange marigolds, yellow rudbeckia, green zinnias, blue hydrangeas, purple lavender'],
 ['Benches','14: four per stretch on the Market side, three on the 17th St side; parallel to the sides, facing in; slats in pride order'],
 ['Mural','Abstract flowing rainbow on the blank east wall (3970 17th St), %.0f m long'%pickle.load(open('data/mural.pkl','rb'))['length']],
 ['Fountain','6.0 m: white marble basin (56 cm seat) and pedestal, then six cascading pools, largest at the bottom'],
 ['Pool stones','Top to bottom: '+' · '.join(n for n,_ in ps[::-1])],
 ['Seat stones',' · '.join(n for n,_ in st)],
 ['Tile rings','Six glazed rainbow rings around the basin (purple inside, red outside), alternating with rings of the plaza stone; the paving radiates outward in hexagonal courses'],
 ['Statues','Two 1.5 m marble figures holding hands, facing the Harvey Milk Plaza flag; total height 7.5 m'],
 ['Grade','Fountain base %.2f m NAVD88'%PZ['fountain']['z']]],
 note='The figures are posed from the open-licence MakeHuman body; hair, wreath, ribbon and drapery are modelled for this piece.')
json.dump(dict(labels=L,cards=dict(plaza=plaza)),open('web/labels.json','w'),indent=1)
print(len(L),[l['name'] for l in L])
