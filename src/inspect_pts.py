import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from geo import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P=np.load('data/lidar_local.npy'); P=P[(P[:,3]!=7)&(P[:,3]!=18)]
R=np.load('data/rasters.npz'); gx0,gy0,res=float(R['gx0']),float(R['gy0']),float(R['res']); dtm=R['dtm_f']
x0,x1,y0,y1=map(float,sys.argv[1:5]); out=sys.argv[5]
m=(P[:,0]>x0)&(P[:,0]<x1)&(P[:,1]>y0)&(P[:,1]<y1); Q=P[m]
i=np.clip(((Q[:,0]-gx0)/res).astype(int),0,dtm.shape[1]-1); j=np.clip(((Q[:,1]-gy0)/res).astype(int),0,dtm.shape[0]-1)
hag=Q[:,2]-dtm[j,i]
fig,axs=plt.subplots(1,3,figsize=(24,8))
o=np.argsort(Q[:,2])
sc=axs[0].scatter(Q[o,0],Q[o,1],c=np.clip(hag[o],-0.5,12),s=0.6,cmap='turbo'); axs[0].set_aspect('equal'); plt.colorbar(sc,ax=axs[0])
axs[0].set_title('top, colour=HAG'); axs[0].grid(alpha=.3)
axs[1].scatter(Q[:,0],hag,c=Q[:,1],s=0.3,cmap='viridis'); axs[1].set_title('side (x vs HAG)'); axs[1].grid(alpha=.3)
axs[2].scatter(Q[:,1],hag,c=Q[:,0],s=0.3,cmap='viridis'); axs[2].set_title('side (y vs HAG)'); axs[2].grid(alpha=.3)
plt.savefig(out,dpi=60,bbox_inches='tight')
