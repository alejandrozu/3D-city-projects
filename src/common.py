import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from geo import *
from scipy import ndimage
import shapely
from shapely.geometry import shape
ZB=30.0   # model y = NAVD88 elevation - ZB
R=np.load(ROOT + '/data/rasters.npz')
GX0,GY0,RES=float(R['gx0']),float(R['gy0']),float(R['res'])
_d=R['dtm_f']
_d=ndimage.median_filter(_d,5); DTM=ndimage.gaussian_filter(_d,0.6)
DSM_LAST=R['dsm_last']; DSM_SINGLE=R['dsm_single'] if 'dsm_single' in R.files else R['dsm_last']; MULTI=R['multi_frac'] if 'multi_frac' in R.files else None;
import os as _os
DSM_ROOF=np.load(ROOT + '/data/dsm_roof.npy') if _os.path.exists(ROOT + '/data/dsm_roof.npy') else DSM_SINGLE
DSM_FIRST=R['dsm_first']; INTEN=R['inten']
def sample(grid,x,y,order=1):
    x=np.asarray(x,float); y=np.asarray(y,float)
    c=(x-GX0)/RES-0.5; r=(y-GY0)/RES-0.5
    return ndimage.map_coordinates(grid,[r.ravel(),c.ravel()],order=order,mode='nearest').reshape(x.shape)
def ground(x,y): return sample(DTM,x,y)
def proj(g): return shapely.transform(g, lambda a: np.array(TM.transform(a[:,0],a[:,1])).T)
def to_model(x,y,z):
    """local east/north/elev -> glTF (X right, Y up, Z toward viewer=south)"""
    return np.stack([np.asarray(x)-CX, np.asarray(z)-ZB, -(np.asarray(y)-CY)],-1)
