import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import json, subprocess, os, sys, numpy as np, laspy, pyproj
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,ROOT + '/src')
from geo import *
BASE='https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SanFrancisco_1_B23'
ept=json.loads(subprocess.run(['curl','-s',BASE+'/ept.json'],capture_output=True,text=True).stdout)
B=ept['bounds']
M=pyproj.Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
mnlon,mnlat,mxlon,mxlat=bbox_ll(15)
qx0,qy0=M.transform(mnlon,mnlat); qx1,qy1=M.transform(mxlon,mxlat)
qz0,qz1=-50,250
print('query',qx0,qy0,qx1,qy1)
import time
def get(url,path,tries=8):
    if os.path.exists(path): return path
    for i in range(tries):
        r=subprocess.run(['curl','-s','-m','120','-o',path+'.part','-w','%{http_code}',url],capture_output=True,text=True)
        if r.stdout=='200': os.rename(path+'.part',path); return path
        time.sleep(0.5+i)
    raise RuntimeError(url+' '+r.stdout)
hier={}
def bounds(key):
    d,x,y,z=map(int,key.split('-')); s=(B[3]-B[0])/2**d
    return B[0]+x*s,B[1]+y*s,B[2]+z*s,B[0]+(x+1)*s,B[1]+(y+1)*s,B[2]+(z+1)*s
def ov(k):
    b=bounds(k); return b[0]<qx1 and b[3]>qx0 and b[1]<qy1 and b[4]>qy0 and b[2]<qz1 and b[5]>qz0
def load_hier(key):
    p=get(f'{BASE}/ept-hierarchy/{key}.json',f'data/ept/h_{key}.json'); h=json.load(open(p))
    for k,v in h.items():
        if not ov(k): continue
        if v==-1: load_hier(k)
        else: hier[k]=v
load_hier('0-0-0-0')
# traverse from root only into overlapping nodes (and load sub-hierarchies lazily is done above fully? -> may be big)
print('hier nodes',len(hier))
sel=[k for k in hier if (lambda b: b[0]<qx1 and b[3]>qx0 and b[1]<qy1 and b[4]>qy0 and b[2]<qz1 and b[5]>qz0)(bounds(k))]
print('selected',len(sel),'points',sum(hier[k] for k in sel))
def dl(k): return get(f'{BASE}/ept-data/{k}.laz',f'data/ept/{k}.laz')
with ThreadPoolExecutor(16) as ex: files=list(ex.map(dl,sel))
arrs=[]
for f in files:
    las=laspy.read(f)
    x,y,z=np.asarray(las.x),np.asarray(las.y),np.asarray(las.z)
    m=(x>qx0)&(x<qx1)&(y>qy0)&(y<qy1)
    arrs.append(np.stack([x[m],y[m],z[m],np.asarray(las.classification)[m].astype(float),np.asarray(las.return_number)[m].astype(float),np.asarray(las.number_of_returns)[m].astype(float),np.asarray(las.intensity)[m].astype(float)],1))
P=np.concatenate(arrs); print('points in bbox',len(P))
lon,lat=pyproj.Transformer.from_crs("EPSG:3857","EPSG:4326",always_xy=True).transform(P[:,0],P[:,1])
lx,ly=TM.transform(lon,lat)
out=np.stack([lx,ly,P[:,2],P[:,3],P[:,4],P[:,5],P[:,6]],1).astype(np.float32)
np.save('data/lidar_local.npy',out); print(out[:3]); import collections; print(collections.Counter(out[:,3].astype(int)))
