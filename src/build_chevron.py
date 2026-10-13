import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from shapes import *
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
import mapbox_earcut as me
OUT={}
def acc(n):
    if n not in OUT: OUT[n]=Acc()
    return OUT[n]
P=np.load('data/lidar_local.npy'); P=P[P[:,3]==1]
B={p['sf16_bldgid']:p for p in json.load(open('data/bldg_sf.json'))}
def canopy(bid,fascia=0.75,stripes=True,label='canopy'):
    g=orient(proj(shape(B[bid]['shape'])).buffer(0).simplify(0.05),1.0)
    if g.geom_type!='Polygon': g=max(g.geoms,key=lambda q:q.area)
    gi=g.buffer(-0.4)
    m=(P[:,0]>g.bounds[0])&(P[:,0]<g.bounds[2])&(P[:,1]>g.bounds[1])&(P[:,1]<g.bounds[3]); Q=P[m]
    from shapely import prepared
    pg=prepared.prep(gi); ins=np.array([pg.contains(Point(a,b)) for a,b in Q[:,:2]]); Q=Q[ins]
    Hh=Q[:,2]-ground(Q[:,0],Q[:,1]); med=np.median(Hh); Q=Q[np.abs(Hh-med)<0.8]
    A_=np.column_stack([np.ones(len(Q)),Q[:,0],Q[:,1]])
    coef=np.linalg.lstsq(A_,Q[:,2],rcond=None)[0]
    for _ in range(3):
        r=np.abs(A_@coef-Q[:,2]); s=r<max(0.08,np.percentile(r,80)); coef=np.linalg.lstsq(A_[s],Q[s,2],rcond=None)[0]
    ring=np.array(g.exterior.coords)[:-1]; n=len(ring)
    zt=coef[0]+coef[1]*ring[:,0]+coef[2]*ring[:,1]; zb=zt-fascia
    tri=me.triangulate_float64(ring,np.array([n],dtype=np.uint32)).reshape(-1,3)
    V=np.concatenate([np.column_stack([ring,zt]),np.column_stack([ring,zb])])
    side=[]
    for k in range(n):
        a,b=k,(k+1)%n; side+=[[a+n,b+n,b],[a+n,b,a]]
    acc('canopy_white').add(V,np.concatenate([tri,tri[:,::-1]+n,np.array(side)]),srgb('#f2f2ef'))
    # fascia stripes (slightly proud of the fascia)
    if stripes:
        cen=np.array(g.centroid.coords[0])
        for (z0,z1,col) in [(0.12,0.24,'#1a4ea8'),(0.27,0.35,'#d22630')]:
            V=[];F=[]
            for k in range(n):
                a,b=ring[k],ring[(k+1)%n]
                oa=a+(a-cen)/np.linalg.norm(a-cen)*0.02; ob=b+(b-cen)/np.linalg.norm(b-cen)*0.02
                za,zb_=zt[k]-fascia,zt[(k+1)%n]-fascia
                base=len(V); V+=[(oa[0],oa[1],za+z0),(ob[0],ob[1],zb_+z0),(ob[0],ob[1],zb_+z1),(oa[0],oa[1],za+z1)]
                F+=[(base,base+1,base+2),(base,base+2,base+3)]
            acc('canopy_stripe').add(np.array(V),np.array(F),srgb(col))
    return g,coef,fascia
info={}
# ---------------- Chevron canopy + forecourt
g,coef,fas=canopy('201006.0059932',0.8,True)
rr=g.minimum_rotated_rectangle; c=np.array(rr.exterior.coords)[:4]
e=[c[1]-c[0],c[2]-c[1]]; L=[np.linalg.norm(v) for v in e]; ax=e[int(np.argmax(L))]/max(L)
# main rectangular body of the canopy (south part): centre of the polygon eroded
core=g.buffer(-2.0); cc=np.array(core.centroid.coords[0])
perp=np.array([-ax[1],ax[0]])
yaw=np.arctan2(ax[1],ax[0])
def zund(x,y): return coef[0]+coef[1]*x+coef[2]*y-fas
islands=[]
for k,off in enumerate((-3.4,3.4)):
    ic=cc+ax*off; g0=float(ground(np.array([ic[0]]),np.array([ic[1]]))[0])
    islands.append((ic,g0))
    v,f=box((ic[0],ic[1],g0+0.09),(4.4,1.15,0.18),yaw); acc('concrete').add(v,f,srgb('#cbc7bd'))
    # dispenser (MPD) in the middle of the island
    v,f=box((ic[0],ic[1],g0+0.18+0.9),(1.25,0.55,1.8),yaw); acc('dispenser').add(v,f,srgb('#e9e9e6'))
    v,f=box((ic[0],ic[1],g0+0.18+1.9),(1.3,0.6,0.25),yaw); acc('canopy_stripe').add(v,f,srgb('#1a4ea8'))
    for s in (-1,1):   # screens / nozzle panels on both faces
        sp=ic+perp*s*0.285
        v,f=box((sp[0],sp[1],g0+1.35),(0.9,0.02,0.55),yaw); acc('dark').add(v,f,srgb('#20262b'))
    # canopy column at island end
    colp=ic+ax*1.75; zu=zund(*colp)
    v,f=box((colp[0],colp[1],(g0+zu)/2),(0.4,0.4,zu-g0),yaw); acc('canopy_white').add(v,f,srgb('#e6e6e2'))
    # bollards at both ends
    for s in (-1,1):
        bp=ic+ax*s*2.45
        v,f=cyl((bp[0],bp[1],g0),(bp[0],bp[1],g0+1.0),0.1,seg=10); acc('bollard').add(v,f,srgb('#e3b51c'))
    # under-canopy light panels
    for s in (-1,1):
        lp=ic+perp*s*2.6; zu=zund(*lp)
        v,f=box((lp[0],lp[1],zu-0.02),(1.2,0.6,0.04),yaw); acc('light').add(v,f,srgb('#fffbe8'))
info['chevron_canopy']=dict(top_at_center=float(coef[0]+coef[1]*cc[0]+coef[2]*cc[1]),area=round(g.area,1),islands=[i[0].tolist() for i in islands])
# price sign (monument pylon) near Market St at LiDAR-detected object
px,py=36.2,10.0; g0=float(ground(np.array([px]),np.array([py]))[0])
# orient parallel to the Market St curb nearby (direction of Market ~ 38 deg)
yaw_m=np.radians(38)
v,f=box((px,py,g0+2.2),(2.0,0.45,4.4),yaw_m); acc('canopy_white').add(v,f,srgb('#f0f0ec'))
for s in (-1,1):
    n_=np.array([-np.sin(yaw_m),np.cos(yaw_m)])*s*0.235
    v,f=box((px+n_[0],py+n_[1],g0+3.45),(1.7,0.02,1.1),yaw_m); acc('canopy_stripe').add(v,f,srgb('#1a4ea8'))
    for kk in range(3):
        v,f=box((px+n_[0]*1.02,py+n_[1]*1.02,g0+2.55-kk*0.45),(1.5,0.02,0.34),yaw_m); acc('dark').add(v,f,srgb('#1b1f22'))
v,f=box((px,py,g0+0.25),(2.3,0.7,0.5),yaw_m); acc('concrete').add(v,f,srgb('#b9b4a9'))
# ---------------- Castro Gas canopies (NW corner) - plain white canopies with 2 columns
for bid in ('201006.0158617','201006.0158847'):
    g2,coef2,fas2=canopy(bid,0.6,False)
    c2=np.array(g2.centroid.coords[0]); g0=float(ground(np.array([c2[0]]),np.array([c2[1]]))[0])
    rr=g2.minimum_rotated_rectangle; cc2=np.array(rr.exterior.coords)[:4]; ax2=(cc2[1]-cc2[0])/np.linalg.norm(cc2[1]-cc2[0])
    for s in (-1,1):
        p_=c2+ax2*s*1.6; zu=coef2[0]+coef2[1]*p_[0]+coef2[2]*p_[1]-fas2
        v,f=box((p_[0],p_[1],(g0+zu)/2),(0.35,0.35,zu-g0),np.arctan2(ax2[1],ax2[0])); acc('canopy_white').add(v,f,srgb('#e6e6e2'))
    v,f=box((c2[0],c2[1],g0+0.09),(4.0,1.1,0.18),np.arctan2(ax2[1],ax2[0])); acc('concrete').add(v,f,srgb('#cbc7bd'))
    v,f=box((c2[0],c2[1],g0+1.08),(1.2,0.55,1.8),np.arctan2(ax2[1],ax2[0])); acc('dispenser').add(v,f,srgb('#e2e2de'))
pickle.dump(({k:v.get() for k,v in OUT.items()},info),open('data/chevron.pkl','wb'))
print(info,{k:len(v.get()[1]) for k,v in OUT.items()})
