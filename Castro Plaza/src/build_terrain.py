import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, pyfqmr
sys.path.insert(0,ROOT + '/src')
from common import *
step=0.25
xs=np.arange(X0,X1+1e-6,step); ys=np.arange(Y0,Y1+1e-6,step)
X,Y=np.meshgrid(xs,ys); Z=ground(X,Y)
ny,nx=X.shape
V=np.stack([X.ravel(),Y.ravel(),Z.ravel()],1)
idx=np.arange(nx*ny).reshape(ny,nx)
a=idx[:-1,:-1].ravel(); b=idx[:-1,1:].ravel(); c=idx[1:,1:].ravel(); d=idx[1:,:-1].ravel()
F=np.concatenate([np.stack([a,b,c],1),np.stack([a,c,d],1)])
print('full',len(V),len(F))
s=pyfqmr.Simplify(); s.setMesh(V,F)
s.simplify_mesh(target_count=110000,aggressiveness=5,preserve_border=True,max_iterations=150,verbose=False)
v,f,n=s.getMesh()
print('decimated',len(v),len(f))
# check error: sample random points
err=np.abs(v[:,2]-ground(v[:,0],v[:,1])).max(); print('max vertex dev',err)
np.savez_compressed('data/terrain_mesh.npz',v=v,f=f)
print('z range',v[:,2].min(),v[:,2].max())
