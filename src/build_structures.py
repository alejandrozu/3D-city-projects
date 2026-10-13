import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle
sys.path.insert(0,ROOT + '/src')
from common import *
from shapes import *
from shapely.geometry import box as sbox, Polygon, Point
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
F=np.load('data/pointfeat.npz'); Q,H,lin,pla,sca,vz=F['Q'],F['H'],F['lin'],F['pla'],F['sca'],F['vz']
wire=(lin>0.8)&(vz<0.6); pole=(lin>0.7)&(vz>0.8)
veg=(sca>0.06)&~wire&~pole&(pla<0.45)
inq=(Q[:,0]>X0)&(Q[:,0]<X1)&(Q[:,1]>Y0)&(Q[:,1]<Y1)
C=np.load('data/ground_cls.npy'); NC=C.shape[0]; PXC=(X1-X0)/NC
ci=np.clip(((Q[:,0]-X0)/PXC).astype(int),0,NC-1); ri=np.clip(((Y1-Q[:,1])/PXC).astype(int),0,NC-1); cl=C[ri,ci]
# exclude canopy regions (modelled explicitly)
CANOPY_IDS={'201006.0059932','201006.0158617','201006.0158847'}
import pickle as _pk
_S=_pk.load(open(ROOT + '/data/plaza_shape.pkl','rb'))
cans=unary_union([proj(shape(p['shape'])).buffer(0.8) for p in json.load(open('data/bldg_sf.json')) if p['sf16_bldgid'] in CANOPY_IDS]+[_S['tri'],_S['chev'].buffer(0.5)])
cand=inq&(H>=2.3)&~wire&~pole&(cl!=1)&(cl!=3)
S=Q[cand]; SH=H[cand]; SV=veg[cand]
from shapely import prepared
pc=prepared.prep(cans)
keep=np.array([not pc.contains(Point(a,b)) for a,b in S[:,:2]])
S,SH,SV=S[keep],SH[keep],SV[keep]
CR=0.5; nx=int((X1-X0)/CR); ny=int((Y1-Y0)/CR)
i=((S[:,0]-X0)/CR).astype(int).clip(0,nx-1); j=((S[:,1]-Y0)/CR).astype(int).clip(0,ny-1)
lin_=j*nx+i
cnt=np.bincount(lin_,minlength=nx*ny); vegc=np.bincount(lin_,SV.astype(float),nx*ny)
top=np.full(nx*ny,-np.inf); np.maximum.at(top,lin_,S[:,2]); bot=np.full(nx*ny,np.inf); np.minimum.at(bot,lin_,S[:,2])
ok=(cnt>=4)&(vegc/np.maximum(cnt,1)<0.35)
ok=ok.reshape(ny,nx); ok=ndimage.binary_opening(ok,iterations=1)
top=top.reshape(ny,nx); bot=bot.reshape(ny,nx)
lab,n=ndimage.label(ok)
A=Acc(); groups={}
FD=np.load('data/fp_dist.npy')
for k in range(1,n+1):
    jj,ii=np.nonzero(lab==k)
    if len(jj)<3: continue
    xs=X0+(ii+0.5)*CR; ys=Y0+(jj+0.5)*CR
    fd=np.median(sample(FD,xs,ys,order=0))
    tz=np.median(top[jj,ii]); 
    for a,b in zip(jj,ii):
        t=top[a,b]; bt=bot[a,b]
        if fd<1.5: th=float(np.clip(t-bt,0.4,4.0))   # building appendage (bay window / eave)
        else: th=0.3                                  # free-standing canopy / awning
        key=(k,round(t/0.3)*0.3,round(th/0.5)*0.5)
        groups.setdefault(key,[]).append(sbox(X0+b*CR,Y0+a*CR,X0+(b+1)*CR,Y0+(a+1)*CR))
nprism=0
for (k,t,th),cells in groups.items():
    u=unary_union(cells).simplify(0.01)
    for p in getattr(u,'geoms',[u]):
        if p.geom_type!='Polygon' or p.area<0.5: continue
        p=orient(p,1.0)
        ext=list(p.exterior.coords)
        if p.interiors: p=Polygon(ext)  # ignore holes
        V,Fc=prism(ext,t-th,t); A.add(V,Fc,srgb('#cfc8bb')); nprism+=1
pickle.dump(A.get(),open('data/structures.pkl','wb'))
print('components',n,'prisms',nprism,'tris',len(A.get()[1]))
