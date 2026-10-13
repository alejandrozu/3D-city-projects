import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from shapes import *
from osm_load import load
from shapely.geometry import Point, LineString, Polygon, box as sbox
from shapely.ops import unary_union, nearest_points
from shapely.geometry.polygon import orient
rng=np.random.default_rng(42)
nodes,ntags,ways,rels=load()
SQ=sbox(X0,Y0,X1,Y1)
import pickle as _pk
_S=_pk.load(open(ROOT + '/data/plaza_shape.pkl','rb'))
from shapely import prepared as _prep
_CLEAR=_prep.prep(_S['plaza'].buffer(0.3).union(_S['chev']))
def cleared(x,y): return _CLEAR.contains(Point(x,y))
def inq(x,y,m=0): return X0-m<=x<=X1+m and Y0-m<=y<=Y1+m and not cleared(x,y)
def g(x,y): return float(ground(np.array([x]),np.array([y]))[0])
C=np.load('data/ground_cls.npy'); NC=C.shape[0]; PXC=(X1-X0)/NC
def cls_at(x,y):
    i=int(np.clip((x-X0)/PXC,0,NC-1)); j=int(np.clip((Y1-y)/PXC,0,NC-1)); return int(C[j,i])
curbs=[proj(shape(p['the_geom'])) for p in json.load(open('data/curbs.json')) if p['layer'] in ('CURBS','ISLANDS')]
CU=unary_union(curbs)
def curb_frame(x,y):
    """nearest curb point, unit tangent, unit normal pointing from curb toward (x,y)"""
    q=nearest_points(Point(x,y),CU)[1]; s=CU.project(q) if CU.geom_type=='LineString' else None
    # tangent via small offsets on the nearest line
    best=None
    for l in getattr(CU,'geoms',[CU]):
        d=l.distance(Point(x,y))
        if best is None or d<best[0]: best=(d,l)
    l=best[1]; t=l.project(Point(x,y)); a=l.interpolate(max(t-0.5,0)); b=l.interpolate(min(t+0.5,l.length))
    tv=np.array([b.x-a.x,b.y-a.y]); tv/=np.linalg.norm(tv)+1e-9
    nv=np.array([x-q.x,y-q.y]); dn=np.linalg.norm(nv)
    if dn<1e-6: nv=np.array([-tv[1],tv[0]])
    else: nv/=dn
    return np.array([q.x,q.y]),tv,nv,best[0]
OUT={}
def acc(name):
    if name not in OUT: OUT[name]=Acc()
    return OUT[name]
# ======================= TREES
trees=pickle.load(open('data/trees.pkl','rb')); small=pickle.load(open('data/small_trees.pkl','rb'))
greens=['#5f7f44','#4e6d3a','#6b8a4b','#55703f','#46623a','#62804a','#4a6a40']
sp_col={'Platanus':'#6e8c48','Magnolia':'#3c5a33','Lophostemon':'#4d6a3a','Ginkgo':'#93a54a','Ficus':'#3b5634','Tristaniopsis':'#4b693c','Prunus cerasifera':'#5e3f45','Prunus':'#6a7f4a','Cercis':'#7a6a48','Olea':'#7d8a6a','Pittosporum':'#55703f','Arbutus':'#4f6a3c'}
def tree_color(sp):
    for k,v in sp_col.items():
        if sp and sp.startswith(k): return v
    return greens[rng.integers(len(greens))]
def is_palm(sp): return bool(sp) and any(k in sp for k in ('Phoenix','Syagrus','Washingtonia','Trachycarpus'))
def palm(x,y,g0,top,sp):
    A=acc('trunks'); L=acc('palm_fronds')
    thick='Phoenix' in sp
    r=0.38 if thick else 0.16
    crown=top-(2.2 if thick else 1.6)
    v,f=cyl((x,y,g0-0.2),(x,y,crown),r,r*0.92,seg=10); A.add(v,f,srgb('#7a6650' if thick else '#8a8478'))
    # crown knob
    v,f=blob((x,y,crown+0.3),(r*1.6,r*1.6,0.7),1,rng,0.05); A.add(v,f,srgb('#6a5a40'))
    nfr=18 if thick else 14; Lf=3.6 if thick else 3.0
    for k in range(nfr):
        az=2*np.pi*k/nfr+rng.random()*0.3; el=np.radians(rng.uniform(15,55)) if k%2 else np.radians(rng.uniform(-10,25))
        d=np.array([np.cos(az),np.sin(az)])
        pts=[];w=[]
        for s in np.linspace(0,1,7):
            hz=s*Lf*np.cos(el)*0.95; vz=s*Lf*np.sin(el)-1.4*s*s*Lf*0.5
            pts.append((x+d[0]*hz,y+d[1]*hz,crown+0.4+vz)); w.append(0.55*np.sin(np.pi*min(s+0.08,1))+0.05)
        V=[];F=[]
        for i,(p,ww) in enumerate(zip(pts,w)):
            nrm=np.array([-d[1],d[0]])*ww/2
            V+= [(p[0]+nrm[0],p[1]+nrm[1],p[2]-0.05*ww),(p[0]-nrm[0],p[1]-nrm[1],p[2]-0.05*ww)]
            if i: a=2*(i-1); F+=[(a,a+2,a+3),(a,a+3,a+1)]
        col=srgb('#5c7a37' if thick else '#6b8a40')*(0.85+0.3*rng.random())
        L.add(np.array(V),np.array(F),col)
def blob_tree(t,col):
    A=acc('foliage'); T=acc('trunks')
    bx,by,g0=t['base']; cb=t['crown_base']
    r=max(0.07,(t['dbh']*0.0254/2*1.1) if t['dbh'] else 0.05+0.011*(t['top']-g0))
    r=min(r,0.45)
    ttop=np.array([bx,by,max(cb+0.8,g0+1.5)])
    v,f=cyl((bx,by,g0-0.2),tuple(ttop),r,r*0.6,seg=7); T.add(v,f,srgb('#5b4a3c'))
    # scaffold branches toward the largest crown masses
    bl=sorted(t['blobs'],key=lambda b:-np.prod(b[1]))[:5]
    for c,rr in bl:
        c=np.asarray(c,float); tgt=ttop+(c-ttop)*0.75
        if np.linalg.norm(tgt-ttop)<0.4: continue
        v,f=cyl(tuple(ttop-[0,0,0.3]),tuple(tgt),r*0.5,max(0.025,r*0.15),seg=5); T.add(v,f,srgb('#56463a'))
    base=srgb(col)
    for c,rr in t['blobs']:
        rr=np.array(rr); rr[2]=max(rr[2],0.5)
        v,f=blob(c,rr,1,rng,0.10)
        sh=0.72+0.38*np.clip((v[:,2]-(c[2]-rr[2]))/(2*rr[2]),0,1)
        cc=base*(0.85+0.3*rng.random())
        A.add(v,f,np.clip(sh[:,None]*cc[None,:],0,1))
for t in trees:
    bx,by,g0=t['base']
    if not inq(bx,by,0.5): continue
    sp=t['species']
    if is_palm(sp): palm(bx,by,g0,t['top'],sp); continue
    if sp.startswith('Chamaerops') or sp.startswith('Trachycarpus'):
        T=acc('trunks'); A=acc('palm_fronds')
        for k in range(3):
            off=np.array([np.cos(k*2.1),np.sin(k*2.1)])*0.35; h_=1.4+0.5*k
            top=np.array([bx+off[0],by+off[1],g0+h_])
            v,f=cyl((bx+off[0],by+off[1],g0-0.1),top,0.11,0.09,seg=6); T.add(v,f,srgb('#6b5a46'))
            for j in range(9):
                az=2*np.pi*j/9+k; d=np.array([np.cos(az),np.sin(az)])
                tip=top+np.array([d[0]*1.0,d[1]*1.0,0.25+0.3*rng.random()])
                v,f=cyl(top,tip,0.03,0.32,seg=3); A.add(v,f,srgb('#56703c'))
        continue
    if sp.startswith('Cordyline'):
        T=acc('trunks'); A=acc('palm_fronds')
        v,f=cyl((bx,by,g0-0.2),(bx,by,t['top']-0.8),0.12,0.09,seg=6); T.add(v,f,srgb('#6b5d4c'))
        for k in range(10):
            az=2*np.pi*k/10; d=np.array([np.cos(az),np.sin(az)]); top=np.array([bx,by,t['top']-0.8])
            tip=top+np.array([d[0]*0.9,d[1]*0.9,0.5+0.4*rng.random()])
            v,f=cyl(top,tip,0.12,0.01,seg=3); A.add(v,f,srgb('#55713a'))
        continue
    blob_tree(t,tree_color(sp))
for s in small:
    if not inq(s['x'],s['y'],0.5): continue
    g0=g(s['x'],s['y'])
    if is_palm(s['species']): palm(s['x'],s['y'],g0,g0+5.0,s['species']); continue
    t=dict(base=(s['x'],s['y'],g0),crown_base=g0+1.8,top=g0+4.0,dbh=s['dbh'],blobs=[((s['x'],s['y'],g0+3.0),(1.1,1.1,1.0))])
    blob_tree(t,tree_color(s['species']))
# ======================= POLES
poles=pickle.load(open('data/poles.pkl','rb'))
signs=json.load(open('data/signs.json'))
SG=[(xy(float(s['shape']['coordinates'][0]),float(s['shape']['coordinates'][1])),s) for s in signs]
poles=[p for p in poles if inq(p['x'],p['y']) and cls_at(p['x'],p['y'])!=1]
INTS=[(0.0,0.0),(4.0,-17.5),(23,-12)]
street_poles=[]
for p in poles:
    x,y,g0,top=p['x'],p['y'],p['g'],p['top']
    c=cls_at(x,y)
    wood= c in (0,7)
    P=acc('poles_wood' if wood else 'poles')
    r0=0.14 if wood else 0.12; r1=0.11 if wood else 0.075
    v,f=cyl((x,y,g0-0.2),(x,y,top),r0,r1,seg=8); P.add(v,f,srgb('#5a4632' if wood else '#3d4246'))
    if not wood: street_poles.append(p)
    if (not wood) and p['arm'] and 1.0<=p['armlen']<=5.0:
        d=np.array(p['arm']); L_=p['armlen']; z=top-0.25
        e=np.array([x+d[0]*L_,y+d[1]*L_,z+0.25])
        v,f=cyl((x,y,z),e,0.05,0.04,seg=6); P.add(v,f,srgb('#3d4246'))
        v,f=box((e[0],e[1],e[2]-0.12),(0.7,0.32,0.16),np.arctan2(d[1],d[0])); acc('lamps').add(v,f,srgb('#d9d6c8'))
# ======================= SIGNS (posts + plates)
def plate_color(code,cat):
    code=code or ''
    if code.startswith('R1-1'): return '#b3202a','oct'
    if code.startswith('D3'): return '#1f6b3a','rect'   # street name
    if code.startswith('W'): return '#e8c21a','diamond'
    if code.startswith('R3') or code.startswith('R5') or code.startswith('R10'): return '#f4f4f0','rect'
    return '#f1f0ea','rect'
seen={}
for (x,y),s in SG:
    if not inq(x,y): continue
    key=(round(x/0.4),round(y/0.4)); seen.setdefault(key,[]).append(((x,y),s))
for key,items in seen.items():
    (x,y),s=items[0]
    g0=g(x,y)
    sup=(s.get('sign_support') or '')
    near=[p for p in street_poles if np.hypot(p['x']-x,p['y']-y)<1.5]
    mast='MAST' in sup
    if near: px,py=near[0]['x'],near[0]['y']
    else:
        px,py=x,y
        if cls_at(x,y)==1 and not mast: continue
        if not mast:
            v,f=cyl((px,py,g0-0.1),(px,py,g0+3.1),0.03,seg=6); acc('poles').add(v,f,srgb('#6b6f72'))
    try: _,tv,nv,dist=curb_frame(px,py)
    except Exception: tv,nv=np.array([1,0]),np.array([0,1])
    yaw=np.arctan2(nv[1],nv[0])  # plate faces the street -> normal along nv
    for k,((sx,sy),ss) in enumerate(items[:4]):
        col,shape_=plate_color(ss.get('sign_code'),ss.get('sign_category'))
        try: w_,h_=[int(v)*0.0254 for v in (ss.get('sign_size') or '18X24').split('X')[:2]]
        except Exception: w_,h_=0.45,0.6
        w_=min(max(w_,0.3),0.95); h_=min(max(h_,0.2),0.95)
        z=g0+(5.6 if mast else 2.6)-k*0.75
        if shape_=='rect' and col=='#1f6b3a': z=g0+3.6; w_,h_=0.9,0.2
        cx,cy=px+nv[0]*0.06,py+nv[1]*0.06
        v,f=box((cx,cy,z),(0.02,w_,h_),yaw); acc('signs').add(v,f,srgb(col))
# ======================= RAILS & F-LINE CONTACT WIRE
WIRES=[]
for wid,(nds,tg) in ways.items():
    if tg.get('railway') not in ('tram','disused') or tg.get('tunnel')=='yes': continue
    pts=np.array([nodes[n] for n in nds if n in nodes])
    if len(pts)<2: continue
    ls=LineString(pts).intersection(SQ)
    for l in getattr(ls,'geoms',[ls]):
        if l.is_empty or l.length<1: continue
        n=max(2,int(l.length/1.0)+1)
        s=np.linspace(0,l.length,n); P=np.array([[l.interpolate(t).x,l.interpolate(t).y] for t in s])
        d=np.gradient(P,axis=0); d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-9; nrm=np.column_stack([-d[:,1],d[:,0]])
        for side in (-0.7175,0.7175):
            R_=P+nrm*side; z=ground(R_[:,0],R_[:,1])+0.012
            V=[];F=[]
            for i in range(len(R_)):
                for w in (-0.04,0.04): V.append((R_[i,0]+nrm[i,0]*w,R_[i,1]+nrm[i,1]*w,z[i]))
                if i: a=2*(i-1); F+=[(a,a+2,a+3),(a,a+3,a+1)]
            acc('rails' if tg['railway']=='tram' else 'rails_old').add(np.array(V),np.array(F),srgb('#9a9ea2' if tg['railway']=='tram' else '#6f6a64'))
        if tg['railway']=='tram':
            z=ground(P[:,0],P[:,1])+5.6
            WIRES.append(np.column_stack([P,z]))
# 24 trolley coach wires on Castro (from Muni route geometry)
for r in json.load(open('data/muni_routes.json')):
    if r.get('route_name')!='24': continue
    for line in r['shape']['coordinates']:
        P=np.array([xy(*c) for c in line])
        ls=LineString(P).intersection(SQ.buffer(-0.5))
        for l in getattr(ls,'geoms',[ls]):
            if l.is_empty or l.length<2: continue
            n=max(2,int(l.length/2)+1); s=np.linspace(0,l.length,n)
            Q_=np.array([[l.interpolate(t).x,l.interpolate(t).y] for t in s])
            d=np.gradient(Q_,axis=0); d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-9; nrm=np.column_stack([d[:,1],-d[:,0]])  # right-hand side
            for off in (2.0,2.6):
                R_=Q_+nrm*off; z=ground(R_[:,0],R_[:,1])+5.8
                WIRES.append(np.column_stack([R_,z]))
lw=pickle.load(open('data/wires.pkl','rb'))
for w in lw:
    p=np.array(w['pts'])
    if all(X0<=a<=X1 and Y0<=b<=Y1 for a,b,_ in p) and not any(cleared(a,b) for a,b,_ in p[::2]): WIRES.append(p)
pickle.dump(WIRES,open('data/all_wires.pkl','wb'))
# ======================= FURNITURE
# parking meters (dedupe by location)
mt=json.load(open('data/meters.json')); seenm=set()
for m in mt:
    if m.get('on_offstreet_type')!='ON': continue
    x,y=xy(float(m['longitude']),float(m['latitude']))
    if not inq(x,y): continue
    k=(round(x/0.6),round(y/0.6))
    if k in seenm: continue
    seenm.add(k); g0=g(x,y)
    if m.get('meter_type')=='MS':  # multi-space pay station
        v,f=box((x,y,g0+0.8),(0.35,0.3,1.6),0); acc('furniture').add(v,f,srgb('#3c4a5a'))
    else:
        v,f=cyl((x,y,g0-0.05),(x,y,g0+1.15),0.04,seg=6); acc('furniture').add(v,f,srgb('#55595c'))
        v,f=box((x,y,g0+1.3),(0.22,0.18,0.32),0); acc('furniture').add(v,f,srgb('#2f3336'))
# bike racks
for b in json.load(open('data/bikeracks.json')):
    try: x,y=xy(*map(float,b['shape']['coordinates']))
    except Exception: continue
    if not inq(x,y): continue
    try: _,tv,nv,_=curb_frame(x,y)
    except Exception: tv,nv=np.array([1.,0]),np.array([0,1.])
    n=int(b.get('racks') or 1); g0=g(x,y)
    for k in range(min(n,6)):
        c=np.array([x,y])+tv*(k-(n-1)/2)*0.9
        a=c+nv*0.35; e=c-nv*0.35
        pts=[(a[0],a[1],g0),(a[0],a[1],g0+0.75),(e[0],e[1],g0+0.75),(e[0],e[1],g0)]
        v,f=tube(pts,0.025,6); acc('metal').add(v,f,srgb('#9aa0a6'))
# trash cans
for t in json.load(open('data/trashcans.json')):
    x,y=xy(float(t['longitude']),float(t['latitude']))
    if not inq(x,y): continue
    g0=g(x,y); v,f=cyl((x,y,g0),(x,y,g0+1.05),0.32,0.3,seg=12); acc('furniture').add(v,f,srgb('#2f4a3a'))
# bus shelters at Muni stops flagged with a shelter
SH=[]
for s in json.load(open('data/munistops.json')):
    if str(s.get('shelter'))!='1': continue
    x,y=xy(float(s['longitude']),float(s['latitude']))
    if not inq(x,y,-2): continue
    q,tv,nv,dist=curb_frame(x,y)
    if dist>6: continue
    c=q+nv*1.35; g0=g(*c); yaw=np.arctan2(tv[1],tv[0])
    SH.append((c,yaw,s.get('stopname')))
    v,f=box((c[0],c[1],g0+2.55),(4.6,1.7,0.12),yaw); acc('shelter_roof').add(v,f,srgb('#d5d8da'))
    back=c+nv*0.75
    v,f=box((back[0],back[1],g0+1.3),(4.4,0.03,2.2),yaw); acc('glass').add(v,f,srgb('#9fb8c4'))
    for sgn in (-1,1):
        side=c+tv*sgn*2.2
        v,f=box((side[0],side[1],g0+1.3),(0.03,1.4,2.2),yaw); acc('glass').add(v,f,srgb('#9fb8c4'))
        for dn in (-0.75,0.75):
            pp=c+tv*sgn*2.25+nv*dn
            v,f=cyl((pp[0],pp[1],g0),(pp[0],pp[1],g0+2.5),0.04,seg=6); acc('metal').add(v,f,srgb('#6f777d'))
    # bench
    bc=c+nv*0.45; v,f=box((bc[0],bc[1],g0+0.45),(1.8,0.4,0.06),yaw); acc('metal').add(v,f,srgb('#6f777d'))
# public toilet kiosk by the Chevron (OSM amenity=toilets)
for k,tg in ntags.items():
    if tg.get('amenity')=='toilets':
        x,y=nodes[k]
        if not inq(x,y): continue
        q,tv,nv,dist=curb_frame(x,y); c=q+nv*2.0; g0=g(*c); yaw=np.arctan2(tv[1],tv[0])
        v,f=box((c[0],c[1],g0+1.45),(3.6,1.9,2.9),yaw); acc('kiosk').add(v,f,srgb('#2f5a45'))
        v,f=box((c[0],c[1],g0+3.0),(3.9,2.2,0.2),yaw); acc('kiosk').add(v,f,srgb('#244636'))
# ======================= FLAGS
def flag(pole_xy,g0,top,w,h,dir_xy,name):
    x,y=pole_xy; d=np.array(dir_xy)/np.linalg.norm(dir_xy)
    v,f=cyl((x,y,g0-0.2),(x,y,top),0.18 if h>3 else 0.05,0.08 if h>3 else 0.04,seg=10); acc('flagpole').add(v,f,srgb('#c9ccce'))
    v,f=blob((x,y,top+0.15),(0.18,0.18,0.18),1); acc('flagpole').add(v,f,srgb('#c8a845'))
    cols=['#e40303','#ff8c00','#ffed00','#008026','#004dff','#750787']
    nu=24
    for si,cc in enumerate(cols):
        z1=top-0.3-si*h/6; z0=z1-h/6
        V=[];F=[]
        for i in range(nu+1):
            u=i/nu; wave=0.35*np.sin(u*np.pi*2.2+0.6)*u*(h/6)
            px,py=x+d[0]*u*w-d[1]*wave,y+d[1]*u*w+d[0]*wave
            sag=-0.06*u*w*0.25
            V+=[(px,py,z0+sag),(px,py,z1+sag)]
            if i: a=2*(i-1); F+=[(a,a+2,a+3),(a,a+3,a+1)]
        acc('flag').add(np.array(V),np.array(F),srgb(cc))
for k,tg in ntags.items():
    if tg.get('man_made')!='flagpole': continue
    x,y=nodes[k]
    if not inq(x,y): continue
    g0=g(x,y)
    if tg.get('height'):
        # big Harvey Milk Plaza flag: use LiDAR top if a pole was detected nearby
        top=g0+float(tg['height'])
        flag((x,y),g0,top,9.1,6.1,(1.0,0.15),'Gilbert Baker Memorial Rainbow Flag')
    else:
        flag((x,y),g0,g0+6.0,1.5,1.0,(1.0,0.2),'pride flag')
# rainbow banners on street poles along Castro St (x within the Castro corridor)
castro=[w for w,(n,t) in ways.items() if t.get('name')=='Castro Street' and 'highway' in t]
CL=unary_union([LineString([nodes[n] for n in ways[w][0]]) for w in castro])
for p in street_poles:
    if CL.distance(Point(p['x'],p['y']))<11 and p['h']>6:
        q,tv,nv,dist=curb_frame(p['x'],p['y'])
        d=-nv  # banner sticks out over the curb toward the street
        z=p['g']+5.2
        x,y=p['x'],p['y']
        cols=['#e40303','#ff8c00','#ffed00','#008026','#004dff','#750787']
        for si,cc in enumerate(cols):
            z1=z-si*0.25; z0=z1-0.25
            V=[(x,y,z0),(x+d[0]*0.9,y+d[1]*0.9,z0),(x+d[0]*0.9,y+d[1]*0.9,z1),(x,y,z1)]
            acc('flag').add(np.array(V),np.array([(0,1,2),(0,2,3)]),srgb(cc))
pickle.dump({k:v.get() for k,v in OUT.items()},open('data/details.pkl','wb'))
print({k:(len(v.get()[0]),len(v.get()[1])) for k,v in OUT.items()})
print('wires',len(WIRES),'shelters',len(SH))
