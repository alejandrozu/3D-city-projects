import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, rasterio, pyproj
sys.path.insert(0,ROOT + '/src')
from geo import *
from scipy import ndimage
P=np.load('data/lidar_local.npy')
P=P[(P[:,3]!=7)&(P[:,3]!=18)]
RES=0.25; M=6.0  # grid covers square + 6 m margin
gx0,gy0=X0-M,Y0-M; nx=int((X1-X0+2*M)/RES); ny=int((Y1-Y0+2*M)/RES)
def idx(x,y):
    i=((x-gx0)/RES).astype(int); j=((y-gy0)/RES).astype(int); ok=(i>=0)&(i<nx)&(j>=0)&(j<ny); return i,j,ok
def grid_stat(pts,val,how):
    i,j,ok=idx(pts[:,0],pts[:,1]); i,j,v=i[ok],j[ok],val[ok]
    lin=j*nx+i
    if how=='max':
        g=np.full(nx*ny,-np.inf); np.maximum.at(g,lin,v); g[np.isinf(g)]=np.nan
    elif how=='min':
        g=np.full(nx*ny,np.inf); np.minimum.at(g,lin,v); g[np.isinf(g)]=np.nan
    else:
        s=np.bincount(lin,v,nx*ny); c=np.bincount(lin,None,nx*ny); g=np.where(c>0,s/np.maximum(c,1),np.nan)
    return g.reshape(ny,nx)
G=P[P[:,3]==2]
dtm=grid_stat(G,G[:,2],'mean')
inten=grid_stat(G,G[:,6],'mean')
last=P[(P[:,4]==P[:,5])]
dsm_last=grid_stat(last,last[:,2],'max')
dsm_first=grid_stat(P,P[:,2],'max')
single=P[(P[:,5]==1)&(P[:,3]==1)]
dsm_single=grid_stat(single,single[:,2],'max')
fr=P[(P[:,4]==1)&(P[:,3]==1)]
i_,j_,ok_=idx(fr[:,0],fr[:,1]); lin_=j_[ok_]*nx+i_[ok_]
tot=np.bincount(lin_,minlength=nx*ny).reshape(ny,nx).astype(float); mul=np.bincount(lin_,(fr[ok_,5]>1).astype(float),nx*ny).reshape(ny,nx)
multi_frac=ndimage.uniform_filter(mul,5)/np.maximum(ndimage.uniform_filter(tot,5),1e-6)
print('grid',nx,ny,'dtm coverage',np.isfinite(dtm).mean())
# fill dtm holes: coarse interpolation via nearest + smoothing at 0.5m
from scipy.interpolate import griddata
jj,ii=np.nonzero(np.isfinite(dtm))
# coarse fill: downsample to 1m for interpolation speed
def fill(g):
    m=np.isfinite(g)
    if m.all(): return g
    yy,xx=np.mgrid[0:g.shape[0],0:g.shape[1]]
    out=g.copy()
    out[~m]=griddata((yy[m][::3],xx[m][::3]),g[m][::3],(yy[~m],xx[~m]),method='linear')
    m2=np.isfinite(out)
    if not m2.all():
        d,inds=ndimage.distance_transform_edt(~m2,return_indices=True); out=out[inds[0],inds[1]]
    return out
dtm_f=fill(dtm)
np.savez_compressed('data/rasters.npz',dtm=dtm,dtm_f=dtm_f,inten=inten,dsm_last=dsm_last,dsm_first=dsm_first,dsm_single=dsm_single,multi_frac=multi_frac,gx0=gx0,gy0=gy0,res=RES)
# compare with 3DEP
src=rasterio.open('data/dem.tif'); U=pyproj.Transformer.from_crs("EPSG:4326","EPSG:32610",always_xy=True)
xs=np.linspace(X0,X1,30); ys=np.linspace(Y0,Y1,30); X,Y=np.meshgrid(xs,ys)
lo,la=ll(X.ravel(),Y.ravel()); e,n=U.transform(lo,la)
dem=np.array([v[0] for v in src.sample(zip(e,n))])
i=((X.ravel()-gx0)/RES).astype(int); j=((Y.ravel()-gy0)/RES).astype(int)
mine=dtm[j,i]; ok=np.isfinite(mine)
print('lidar dtm - 3dep: mean %.3f median %.3f std %.3f'%(np.mean(mine[ok]-dem[ok]),np.median(mine[ok]-dem[ok]),np.std(mine[ok]-dem[ok])))
print('elev range in square', np.nanmin(dtm_f), np.nanmax(dtm_f))
