import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys
sys.path.insert(0,ROOT + '/src')
from common import *
from scipy.spatial import cKDTree
O=np.load('data/objpts.npz'); Q,H=O['Q'],O['H']
m=H>=2.0; Q=Q[m]; H=H[m]
XYZ=Q[:,:3].astype(np.float64)
print('pts',len(XYZ))
T=cKDTree(XYZ)
K=16
lin=np.zeros(len(XYZ),np.float32); pla=np.zeros_like(lin); sca=np.zeros_like(lin); vz=np.zeros_like(lin); rad=np.zeros_like(lin)
for a in range(0,len(XYZ),200000):
    b=min(a+200000,len(XYZ))
    dd,ii=T.query(XYZ[a:b],k=K,distance_upper_bound=1.0)
    valid=np.isfinite(dd)
    ii=np.where(valid,ii,np.arange(a,b)[:,None])
    nb=XYZ[ii]  # n,K,3
    w=valid[...,None].astype(float)
    cnt_=w.sum(1)
    mu=(nb*w).sum(1)/cnt_
    d=(nb-mu[:,None,:])*w
    C=np.einsum('nki,nkj->nij',d,d)/np.maximum(cnt_[...,None],1)
    ev,evec=np.linalg.eigh(C)  # ascending
    l3,l2,l1=ev[:,0],ev[:,1],ev[:,2]; l1=np.maximum(l1,1e-9)
    lin[a:b]=(l1-l2)/l1; pla[a:b]=(l2-l3)/l1; sca[a:b]=l3/l1
    vz[a:b]=np.abs(evec[:,2,2])  # z component of principal direction
    rad[a:b]=np.sqrt(np.maximum(l1,0))
    lin[a:b][cnt_[:,0]<5]=0
np.savez_compressed('data/pointfeat.npz',Q=Q,H=H,lin=lin,pla=pla,sca=sca,vz=vz,rad=rad)
print('linear>0.85 & horizontal', ((lin>0.85)&(vz<0.5)).sum(), 'linear vertical', ((lin>0.8)&(vz>0.85)).sum(), 'scatter>0.08',(sca>0.08).sum())
