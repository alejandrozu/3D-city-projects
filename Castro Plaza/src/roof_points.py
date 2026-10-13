import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, time
sys.path.insert(0,ROOT + '/src')
from common import *
from scipy.spatial import cKDTree
t0=time.time()
P=np.load('data/lidar_local.npy'); P=P[P[:,3]==1]
m=(P[:,0]>X0-3)&(P[:,0]<X1+3)&(P[:,1]>Y0-3)&(P[:,1]<Y1+3); P=P[m]
H=P[:,2]-ground(P[:,0],P[:,1]); P=P[H>1.5]
XYZ=P[:,:3].astype(np.float64); T=cKDTree(XYZ); K=16
sca=np.zeros(len(XYZ),np.float32); pla=np.zeros_like(sca)
for a in range(0,len(XYZ),250000):
    b=min(a+250000,len(XYZ)); dd,ii=T.query(XYZ[a:b],k=K,distance_upper_bound=1.0,workers=-1)
    valid=np.isfinite(dd); ii=np.where(valid,ii,np.arange(a,b)[:,None]); nb=XYZ[ii]; w=valid[...,None].astype(float); c=w.sum(1)
    mu=(nb*w).sum(1)/c; d=(nb-mu[:,None,:])*w; C=np.einsum('nki,nkj->nij',d,d)/np.maximum(c[...,None],1)
    ev=np.linalg.eigvalsh(C); l3,l2,l1=ev[:,0],ev[:,1],np.maximum(ev[:,2],1e-9)
    sca[a:b]=l3/l1; pla[a:b]=(l2-l3)/l1
veg=(sca>0.07)&(pla<0.5)
# majority filter in 3D: point is veg if most neighbours within 0.8 m are veg
vf=np.zeros(len(XYZ),np.float32)
for a in range(0,len(XYZ),250000):
    b=min(a+250000,len(XYZ)); dd,ii=T.query(XYZ[a:b],k=24,distance_upper_bound=0.8,workers=-1)
    valid=np.isfinite(dd); ii=np.where(valid,ii,0); vf[a:b]=(veg[ii]&valid).sum(1)/np.maximum(valid.sum(1),1)
vegm=vf>0.5
R_=np.load('data/rasters.npz'); gx0,gy0,res=float(R_['gx0']),float(R_['gy0']),float(R_['res']); ny,nx=R_['dsm_last'].shape
keep=~vegm
i=((P[keep,0]-gx0)/res).astype(int); j=((P[keep,1]-gy0)/res).astype(int); ok=(i>=0)&(i<nx)&(j>=0)&(j<ny)
g=np.full(nx*ny,-np.inf); np.maximum.at(g,j[ok]*nx+i[ok],P[keep][ok,2]); g[np.isinf(g)]=np.nan
np.save('data/dsm_roof.npy',g.reshape(ny,nx))
print('points',len(XYZ),'veg frac %.2f'%vegm.mean(),'%.0fs'%(time.time()-t0))
