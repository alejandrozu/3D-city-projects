import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
P=np.load('data/lidar_local.npy'); P=P[(P[:,3]==1)]
m=(P[:,0]>X0)&(P[:,0]<X1)&(P[:,1]>Y0)&(P[:,1]<Y1); P=P[m]
H=P[:,2]-ground(P[:,0],P[:,1])
C=np.load('data/ground_cls.npy'); N=C.shape[0]; PX=(X1-X0)/N
ci=np.clip(((P[:,0]-X0)/PX).astype(int),0,N-1); ri=np.clip(((Y1-P[:,1])/PX).astype(int),0,N-1); cl=C[ri,ci]
sel=(H>0.35)&(H<3.6)&((cl==1)|(cl==6))
Q=P[sel]; HQ=H[sel]
CR=0.25; nx=int((X1-X0)/CR); ny=int((Y1-Y0)/CR)
i=((Q[:,0]-X0)/CR).astype(int).clip(0,nx-1); j=((Q[:,1]-Y0)/CR).astype(int).clip(0,ny-1)
occ=np.zeros((ny,nx),int); np.add.at(occ,(j,i),1)
hmax=np.zeros((ny,nx)); np.maximum.at(hmax,(j,i),HQ)
mask=ndimage.binary_closing(occ>=2,iterations=2)&(ndimage.maximum_filter(hmax,3)>0.9)
lab,n=ndimage.label(mask)
cars=[]
for k in range(1,n+1):
    jj,ii=np.nonzero(lab==k)
    if len(jj)<20: continue
    xy=np.column_stack([X0+(ii+0.5)*CR,Y0+(jj+0.5)*CR])
    c=xy.mean(0); U,S,Vt=np.linalg.svd(xy-c,full_matrices=False); ax=Vt[0]
    proj_=(xy-c)@Vt.T; L=np.ptp(proj_[:,0])+CR; W=np.ptp(proj_[:,1])+CR
    hm=np.percentile(hmax[jj,ii],95)
    # sanity: vehicle-like footprint
    if not (2.8<=L<=14 and 1.3<=W<=3.2 and 1.0<=hm<=3.8): continue
    c=c+Vt[0]*(proj_[:,0].max()+proj_[:,0].min())/2+Vt[1]*(proj_[:,1].max()+proj_[:,1].min())/2
    # which end is the front? higher cabin toward the back for sedans: use height profile along axis
    t=proj_[:,0]; hh=hmax[jj,ii]
    front_low = hh[t>np.percentile(t,70)].mean() < hh[t<np.percentile(t,30)].mean()
    yaw=np.arctan2(ax[1],ax[0]) + (0 if front_low else np.pi)
    kind='bus' if L>9 else ('van' if (hm>2.1 or L>5.6) else ('suv' if hm>1.62 else 'car'))
    cars.append(dict(x=float(c[0]),y=float(c[1]),yaw=float(yaw),L=float(min(L,13.5)),W=float(min(max(W,1.65),2.6)),H=float(hm),kind=kind))
print(len(cars),'vehicles', {k:sum(1 for c in cars if c['kind']==k) for k in ('car','suv','van','bus')})
pickle.dump(cars,open('data/cars.pkl','wb'))
