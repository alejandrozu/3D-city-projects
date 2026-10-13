import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.cluster.vq import kmeans2
F=np.load('data/pointfeat.npz'); Q,H,lin,pla,sca,vz=F['Q'],F['H'],F['lin'],F['pla'],F['sca'],F['vz']
FD=np.load('data/fp_dist.npy'); fd=sample(FD,Q[:,0],Q[:,1],order=0)
wire=(lin>0.8)&(vz<0.6)&(H>3.0); pole=(lin>0.75)&(vz>0.8); plan=(pla>0.5)
veg=~(wire|pole|plan)&(H>=2.3)&(Q[:,0]>X0-1)&(Q[:,0]<X1+1)&(Q[:,1]>Y0-1)&(Q[:,1]<Y1+1)
V=Q[veg]; VH=H[veg]; VP=pla[veg]
# drop isolated veg points (wire/sign remnants): need >=6 veg neighbours within 0.6 m
from scipy.spatial import cKDTree
T=cKDTree(V[:,:3]); nn=np.array([len(x) for x in T.query_ball_point(V[:,:3],0.6,workers=-1)])
V=V[nn>=8]; VH=VH[nn>=8]; VP=VP[nn>=8]
print('veg pts',len(V))
C=np.load('data/ground_cls.npy'); NC=C.shape[0]; PXC=(X1-X0)/NC
def cls_at(x,y):
    i=int(np.clip((x-X0)/PXC,0,NC-1)); j=int(np.clip((Y1-y)/PXC,0,NC-1)); return int(C[j,i])
CR=0.5; ox,oy=X0-1,Y0-1; nx=int((X1-X0+2)/CR)+1; ny=int((Y1-Y0+2)/CR)+1
ci=((V[:,0]-ox)/CR).astype(int); cj=((V[:,1]-oy)/CR).astype(int)
chm=np.zeros((ny,nx)); np.maximum.at(chm,(cj,ci),VH)
cnt=np.zeros((ny,nx)); np.add.at(cnt,(cj,ci),1)
mask=ndimage.binary_closing(cnt>0,iterations=1)&(ndimage.maximum_filter(chm,3)>2.3)
chms=ndimage.gaussian_filter(chm,1.0)
peaks=peak_local_max(chms,min_distance=4,threshold_abs=3.0,labels=mask.astype(int))
markers=np.zeros_like(chm,int)
for k,(r,c) in enumerate(peaks): markers[r,c]=k+1
# street tree inventory as extra markers (snap to local max within 3 cells)
inv=json.load(open('data/trees_sf.json')); inv_pts=[]
for t in inv:
    x,y=xy(float(t['longitude']),float(t['latitude']))
    if not (X0-1<x<X1+1 and Y0-1<y<Y1+1): continue
    inv_pts.append((x,y,t))
lab=watershed(-chms,markers,mask=mask)
print('segments',lab.max())
pl=lab[cj,ci]
trees=[]
rng=np.random.default_rng(3)
for s in range(1,lab.max()+1):
    idx=np.nonzero(pl==s)[0]
    if len(idx)<25: continue
    pts=V[idx,:3]; hh=VH[idx]
    top=pts[:,2].max(); hmax=hh.max()
    if hmax<2.8: continue
    planar_frac=float((VP[idx]>0.5).mean()); vext=float(np.ptp(pts[:,2]))
    # tree base: inventory tree nearest to segment top location (within 3 m) else top xy
    tpk=pts[np.argmax(pts[:,2])]
    cxy=pts[:,:2].mean(0)
    best=None
    for (x,y,t) in inv_pts:
        dd=np.hypot(x-cxy[0],y-cxy[1])
        inside=np.min(np.hypot(pts[:,0]-x,pts[:,1]-y))<1.2
        if inside and (best is None or dd<best[0]): best=(dd,x,y,t)
    if best:
        bx_,by_,info=best[1],best[2],best[3]
        if planar_frac>0.4 or vext<1.2:   # segment is a vehicle/shelter roof next to a real street tree: use a generic crown
            g0_=float(ground(np.array([bx_]),np.array([by_]))[0]); d_=float(info.get('mapdbh') or 6)
            hgt=min(4+0.35*d_,12); rad=min(1.2+0.12*d_,4)
            trees.append(dict(base=(float(bx_),float(by_),g0_),top=g0_+hgt,crown_base=g0_+hgt*0.4,blobs=[((bx_,by_,g0_+hgt*0.68),(rad,rad,hgt*0.33))],
                              species=info.get('species',''),dbh=d_,n=0,extent=2*rad))
            continue
    else:
        info=None
        if len(idx)<70 or vext<1.4 or planar_frac>0.35: continue
        top3=pts[pts[:,2]>np.percentile(pts[:,2],85)]
        bx_,by_=top3[:,:2].mean(0)
        if cls_at(bx_,by_)==1 and len(idx)<500: continue
        ext_=max(np.ptp(pts[:,0]),np.ptp(pts[:,1]))
        if ext_<3.0 and cls_at(bx_,by_) in (1,2,3,4,8): continue
    g0=float(ground(np.array([bx_]),np.array([by_]))[0])
    crown_base=np.percentile(pts[:,2],3)
    # blobs
    k=int(np.clip(len(idx)/150,1,16))
    W=pts.copy()
    if k>1:
        try: cen,lb=kmeans2(W,k,minit='++',seed=rng)
        except Exception: cen,lb=W.mean(0,keepdims=True),np.zeros(len(W),int)
    else: cen,lb=W.mean(0,keepdims=True),np.zeros(len(W),int)
    blobs=[]
    for b in range(len(cen)):
        sel=W[lb==b]
        if len(sel)<6: continue
        c=sel.mean(0); sd=sel.std(0)
        r=np.maximum(sd*2.1,0.5)
        # don't let blobs extend far below the crown base
        blobs.append((c.tolist(),r.tolist()))
    sp=(info or {}).get('species','') if info else ''
    trees.append(dict(base=(float(bx_),float(by_),g0),top=float(top),crown_base=float(crown_base),blobs=blobs,
                      species=sp,dbh=float((info or {}).get('mapdbh') or 0) if info else 0.0,n=len(idx),
                      extent=float(max(np.ptp(pts[:,0]),np.ptp(pts[:,1])))))
print('trees',len(trees),'with inventory',sum(1 for t in trees if t['species']))
pickle.dump(trees,open('data/trees.pkl','wb'))
# inventory trees not matched (small trees below LiDAR threshold) -> keep as small trees
matched=set()
for t in trees:
    if t['species']: matched.add((round(t['base'][0],2),round(t['base'][1],2)))
small=[]
for (x,y,t) in inv_pts:
    if (round(x,2),round(y,2)) in matched: continue
    # is it inside some segment? check chm at that location
    i=int((x-ox)/CR); j=int((y-oy)/CR)
    if 0<=i<nx and 0<=j<ny and lab[j,i]>0: continue
    small.append(dict(x=x,y=y,species=t.get('species',''),dbh=float(t.get('mapdbh') or 3)))
print('small inventory trees',len(small))
pickle.dump(small,open('data/small_trees.pkl','wb'))
