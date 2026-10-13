import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from skimage.transform import probabilistic_hough_line
F=np.load('data/pointfeat.npz'); Q,H,lin,pla,sca,vz=F['Q'],F['H'],F['lin'],F['pla'],F['sca'],F['vz']
inq=(Q[:,0]>X0-1)&(Q[:,0]<X1+1)&(Q[:,1]>Y0-1)&(Q[:,1]<Y1+1)
from scipy.spatial import cKDTree
vegm=(sca>0.06)&(lin<0.7)&(H>2.5)
TV=cKDTree(Q[vegm,:3])
vden=np.array([len(x) for x in TV.query_ball_point(Q[:,:3],1.0,workers=-1)])
FD=np.load('data/fp_dist.npy'); fdp=sample(FD,Q[:,0],Q[:,1],order=0)
pole=(lin>0.7)&(vz>0.8)&inq&(vden<20)&(fdp>0.45)
wire=(lin>0.8)&(vz<0.6)&(H>3.5)&inq&(vden<12)
# ---- poles
P_=Q[pole]; PH=H[pole]
CR=0.25; ox,oy=X0-1,Y0-1; nx=int((X1-X0+2)/CR)+1; ny=int((Y1-Y0+2)/CR)+1
gi=((P_[:,0]-ox)/CR).astype(int); gj=((P_[:,1]-oy)/CR).astype(int)
occ=np.zeros((ny,nx),bool); occ[gj,gi]=True
lab,n=ndimage.label(ndimage.binary_dilation(occ,iterations=2))
pl=lab[gj,gi]
# all candidate points (any class) for pole top detection
poles=[]
from scipy.spatial import cKDTree
TA=cKDTree(Q[:,:2])
for s in range(1,n+1):
    idx=np.nonzero(pl==s)[0]
    if len(idx)<6: continue
    pts=P_[idx]; hh=PH[idx]
    if np.ptp(hh)<1.5: continue
    xm,ym=np.median(pts[:,0]),np.median(pts[:,1])
    around=TV.query_ball_point([xm,ym,np.median(pts[:,2])],1.6)
    if len(around)>35: continue
    x,y=np.median(pts[:,0]),np.median(pts[:,1])
    # full column: all points within 0.45 m
    col=TA.query_ball_point([x,y],0.45)
    top=max(pts[:,2].max(),Q[col,2].max() if col else 0)
    g0=float(ground(np.array([x]),np.array([y]))[0])
    # arm detection: points 0.6-3.5 m away at heights within 1.2 m of top
    ring=TA.query_ball_point([x,y],3.5)
    rp=Q[ring]; dxy=np.hypot(rp[:,0]-x,rp[:,1]-y)
    arm=(dxy>0.6)&(rp[:,2]>top-1.4)&(rp[:,2]<top+0.3)&(lin[ring]>0.5)
    armdir=None
    if arm.sum()>=5:
        v=rp[arm,:2]-[x,y]; armdir=(v.mean(0)/np.linalg.norm(v.mean(0))).tolist(); armlen=float(np.percentile(dxy[arm],90))
    poles.append(dict(x=float(x),y=float(y),g=g0,top=float(top),h=float(top-g0),arm=armdir,armlen=float(armlen) if armdir else 0.0,n=len(idx)))
# merge poles closer than 0.8 m
poles.sort(key=lambda p:-p['n']); merged=[]
for p in poles:
    if all(np.hypot(p['x']-q['x'],p['y']-q['y'])>0.8 for q in merged): merged.append(p)
poles=[p for p in merged if p['h']>2.2]
print('poles',len(poles), 'heights',np.round(np.percentile([p['h'] for p in poles],[10,50,90]),1), 'with arm',sum(1 for p in poles if p['arm']))
pickle.dump(poles,open('data/poles.pkl','wb'))
# ---- wires
W=Q[wire]
WR=0.2; wx=int((X1-X0+2)/WR)+1; wy=int((Y1-Y0+2)/WR)+1
img=np.zeros((wy,wx),bool)
ii=((W[:,0]-ox)/WR).astype(int); jj=((W[:,1]-oy)/WR).astype(int); img[jj,ii]=True
img=ndimage.binary_dilation(img,iterations=1)
segs=probabilistic_hough_line(img,threshold=10,line_length=int(4/WR),line_gap=int(2.5/WR),rng=1)
print('hough segs',len(segs))
TW=cKDTree(W[:,:2])
wires=[]; used=np.zeros(len(W),bool)
segs=sorted(segs,key=lambda s:-np.hypot(s[1][0]-s[0][0],s[1][1]-s[0][1]))
for (c0,r0),(c1,r1) in segs:
    a=np.array([ox+c0*WR,oy+r0*WR]); b=np.array([ox+c1*WR,oy+r1*WR])
    d=b-a; L=np.linalg.norm(d)
    if L<3: continue
    u=d/L; nrm=np.array([-u[1],u[0]])
    # candidate points near the infinite line within segment range extended
    cand=TW.query_ball_point((a+b)/2,L/2+3)
    cand=np.array([c for c in cand if not used[c]])
    if len(cand)<8: continue
    rel=W[cand,:2]-a; s=rel@u; t=rel@nrm
    m=(np.abs(t)<0.35)&(s>-3)&(s<L+3)
    if m.sum()<8: continue
    # wires stack vertically: cluster by z (multiple conductors) using residual of quadratic fit
    S=s[m]; Z=W[cand[m],2]; idxs=cand[m]
    # iterative split into z-layers
    remaining=np.ones(len(S),bool)
    for layer in range(4):
        if remaining.sum()<8: break
        Sr,Zr=S[remaining],Z[remaining]
        A=np.column_stack([np.ones_like(Sr),Sr,Sr**2])
        # robust fit: iterate trimming
        sel=np.ones(len(Sr),bool)
        for it in range(4):
            if sel.sum()<6: break
            coef,*_=np.linalg.lstsq(A[sel],Zr[sel],rcond=None)
            res=np.abs(A@coef-Zr); thr=max(0.12,np.percentile(res[sel],60)*1.5)
            sel=res<thr
        if sel.sum()<8: break
        s0,s1=Sr[sel].min(),Sr[sel].max()
        cov=len(np.unique(np.floor(Sr[sel]/1.0)))/max(1.0,np.ceil(s1-s0))
        if s1-s0<5.0 or cov<0.45:
            remaining[np.nonzero(remaining)[0][sel]]=False; continue
        ss=np.linspace(s0,s1,max(2,int((s1-s0)/1.0)+1))
        zz=coef[0]+coef[1]*ss+coef[2]*ss**2
        pts=[(float(a[0]+u[0]*q+0*0),float(a[1]+u[1]*q),float(z)) for q,z in zip(ss,zz)]
        # lateral offset: mean t of inliers
        tt=float(np.mean(t[m][remaining][sel]))
        pts=[(p[0]+nrm[0]*tt,p[1]+nrm[1]*tt,p[2]) for p in pts]
        wires.append(dict(pts=pts,n=int(sel.sum())))
        gi_=idxs[remaining][sel]; used[gi_]=True
        remaining[np.nonzero(remaining)[0][sel]]=False
print('wires',len(wires),'total length %.0f m'%sum(np.hypot(w['pts'][-1][0]-w['pts'][0][0],w['pts'][-1][1]-w['pts'][0][1]) for w in wires),'used pts',used.sum(),'/',len(W))
pickle.dump(wires,open('data/wires.pkl','wb'))
