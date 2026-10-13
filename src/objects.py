import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from common import *
from PIL import Image, ImageDraw
from shapely.geometry import box
P=np.load('data/lidar_local.npy'); P=P[(P[:,3]==1)]
m=(P[:,0]>X0-2)&(P[:,0]<X1+2)&(P[:,1]>Y0-2)&(P[:,1]<Y1+2); P=P[m]
hag=P[:,2]-ground(P[:,0],P[:,1])
# footprint mask (buffer 0.4)
ny,nx=DTM.shape
img=Image.new('L',(nx,ny),0); d=ImageDraw.Draw(img)
def tp(cs): return [((x-GX0)/RES-0.5,(y-GY0)/RES-0.5) for x,y in cs]
CANOPY_IDS={'201006.0059932','201006.0158617','201006.0158847'}
for p in json.load(open('data/bldg_sf.json')):
    g=proj(shape(p['shape'])).buffer(0.4 if p['sf16_bldgid'] not in CANOPY_IDS else 0.15)
    for gg in getattr(g,'geoms',[g]): d.polygon(tp(gg.exterior.coords),fill=1 if p['sf16_bldgid'] not in CANOPY_IDS else 2)
FP=np.array(img)
i=np.clip(((P[:,0]-GX0)/RES).astype(int),0,nx-1); j=np.clip(((P[:,1]-GY0)/RES).astype(int),0,ny-1)
inb=FP[j,i]
keep=(inb==0)&(hag>0.3)
Q=P[keep]; H=hag[keep]
np.savez_compressed('data/objpts.npz',Q=Q,H=H)
# voxelise HAG >= 2.3
VS=0.5
hm=H>=2.3; Qh=Q[hm]; Hh=H[hm]
vi=((Qh[:,0]-(X0-2))/VS).astype(int); vj=((Qh[:,1]-(Y0-2))/VS).astype(int); vk=((Hh-2.3)/VS).astype(int)
vk=np.clip(vk,0,79)
shp=(vi.max()+1,vj.max()+1,80)
cnt=np.zeros(shp,np.int32); np.add.at(cnt,(vi,vj,vk),1)
print('occupied voxels',(cnt>0).sum(),'pts/voxel pct',np.percentile(cnt[cnt>0],[10,25,50,75,90]))
for th in (1,2,3,4):
    lab,n=ndimage.label(cnt>=th,structure=np.ones((3,3,3)))
    sizes=np.bincount(lab.ravel())[1:]
    print('th',th,'components',n,'>=20 vox',(sizes>=20).sum(),'largest',sorted(sizes)[-5:])
np.savez_compressed('data/voxels.npz',cnt=cnt,vi=vi,vj=vj,vk=vk)
