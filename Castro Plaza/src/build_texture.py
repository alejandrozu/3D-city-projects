import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json
sys.path.insert(0,ROOT + '/src')
from common import *
from landcover import landcover
from osm_load import load
from PIL import Image, ImageDraw, ImageFilter
import rasterio, pyproj
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely.ops import unary_union
N=4096; PX=(X1-X0)/N
def to_px(coords): return [((x-X0)/PX,(Y1-y)/PX) for x,y in coords]
def raster(geom,val,img):
    d=ImageDraw.Draw(img)
    for g in getattr(geom,'geoms',[geom]):
        if g.is_empty or g.geom_type!='Polygon': continue
        d.polygon(to_px(g.exterior.coords),fill=val)
        for h in g.interiors: d.polygon(to_px(h.coords),fill=0)
    return img
L=landcover()
nodes,ntags,ways,rels=load()
cls=Image.new('L',(N,N),0)   # 0 lot
raster(L['sidewalk'],2,cls); raster(L['roadway'],1,cls); raster(L['islands'],3,cls)
def osm_poly(wid):
    nds,tg=ways[wid]; return Polygon([nodes[n] for n in nds])
# plazas
jwp=osm_poly('188964530'); hmp=osm_poly('225526801'); hmp_ped=osm_poly('1184663412')
gardens=[osm_poly(w) for w,(n,t) in ways.items() if t.get('leisure')=='garden' and len(n)>3]
parking=[osm_poly(w) for w,(n,t) in ways.items() if t.get('amenity')=='parking' and len(n)>3]
chev=osm_poly('256903890')
for p in parking: raster(p,6,cls)
import pickle as _pk
_S=_pk.load(open(ROOT + '/data/plaza_shape.pkl','rb'))
raster(_S['tri'],2,cls)
raster(jwp.difference(L['roadway']),4,cls); raster(hmp,8,cls); raster(hmp_ped,4,cls)
for g in gardens: raster(g,7,cls)
raster(_S['plaza'],9,cls)
C=np.array(cls)
# coordinates of texel centres
xs=X0+(np.arange(N)+0.5)*PX; ys=Y1-(np.arange(N)+0.5)*PX
# intensity upsampled
from scipy import ndimage
I=np.nan_to_num(INTEN,nan=np.nanmedian(INTEN))
bg=ndimage.median_filter(I,size=13)
rel=(I-bg)
def up(grid):
    c=(xs-GX0)/RES-0.5; r=(ys-GY0)/RES-0.5
    return ndimage.map_coordinates(grid,np.meshgrid(r,c,indexing='ij'),order=1,mode='nearest').astype(np.float32)
Iu=up(I); BGu=up(bg)
# sharper intensity: splat raw ground points onto the texture grid with a small Gaussian kernel
P=np.load('data/lidar_local.npy'); G_=P[(P[:,3]==2)&(P[:,0]>X0-1)&(P[:,0]<X1+1)&(P[:,1]>Y0-1)&(P[:,1]<Y1+1)]
ci=((G_[:,0]-X0)/PX).astype(int); ri=((Y1-G_[:,1])/PX).astype(int); ok=(ci>=0)&(ci<N)&(ri>=0)&(ri<N)
num=np.zeros((N,N),np.float32); den=np.zeros((N,N),np.float32)
np.add.at(num,(ri[ok],ci[ok]),G_[ok,6]); np.add.at(den,(ri[ok],ci[ok]),1)
sig=0.11/PX
num=ndimage.gaussian_filter(num,sig); den=ndimage.gaussian_filter(den,sig)
Is=np.where(den>1e-3,num/np.maximum(den,1e-6),Iu)
Is=np.where(den>0.05,Is,Iu)
RELu=Is-BGu
print('splat done')
print('intensity pct',np.percentile(I,[5,25,50,75,95,99]))
# NAIP colours for lots
U=pyproj.Transformer.from_crs("EPSG:4326","EPSG:32610",always_xy=True)
bx=list(map(float,open('data/utm_bbox.txt').read().split('\n')[0].split(',')))
naip=np.asarray(Image.open('data/naip.png').convert('RGB')).astype(np.float32)/255
sN=naip.shape[1]/(bx[2]-bx[0])
# sample NAIP at 1/8 resolution then upsample
k=8; xs8=xs[::k]; ys8=ys[::k]; XX,YY=np.meshgrid(xs8,ys8)
lo,la=ll(XX.ravel(),YY.ravel()); e,n=U.transform(lo,la)
ci=(np.asarray(e)-bx[0])*sN-0.5; ri=(bx[3]-np.asarray(n))*sN-0.5
nai=np.stack([ndimage.map_coordinates(naip[...,ch],[ri,ci],order=1,mode='nearest').reshape(XX.shape) for ch in range(3)],-1)
nai=np.stack([ndimage.zoom(nai[...,ch],k,order=1)[:N,:N] for ch in range(3)],-1)
# base colours
def hexc(h): return np.array([int(h[i:i+2],16)/255 for i in (1,3,5)],np.float32)
base={9:hexc('#cfc9bd'),0:hexc('#8d8a7c'),1:hexc('#4a4b4f'),2:hexc('#b9b5ab'),3:hexc('#5f6e45'),4:hexc('#a99f92'),5:hexc('#8a8780'),6:hexc('#55565a'),7:hexc('#5d7445'),8:hexc('#9c7f6e')}
img=np.zeros((N,N,3),np.float32)
for c,col in base.items(): img[C==c]=col
# texture modulation by LiDAR intensity
nI=np.clip(Iu/np.percentile(I,60),0.55,1.6)
mod=0.78+0.22*nI
img*=mod[...,None]
# lots: blend NAIP colour (yards, gardens, lots)
lotm=(C==0)|(C==7)
img[lotm]=0.35*img[lotm]+0.65*np.clip(nai[lotm]*1.05,0,1)
# paint markings on roadway & plazas & lots(parking): bright relative intensity
thr=np.percentile(rel,97.2)
road=((C==1)|(C==6)|(C==4)|(C==5))
paint=(RELu>thr*0.95)&road
paint=ndimage.binary_opening(paint,iterations=1); paint=ndimage.binary_closing(paint,iterations=2)
lab_,nl=ndimage.label(paint); sz=ndimage.sum(paint,lab_,range(1,nl+1)); small=np.isin(lab_,np.nonzero(sz<40)[0]+1); paint[small]=False
soft=ndimage.gaussian_filter(paint.astype(np.float32),0.7)
# yellow centre lines for two-way streets: near OSM centerline of non-oneway roads
yellow=np.zeros((N,N),bool)
ymask=Image.new('L',(N,N),0); d=ImageDraw.Draw(ymask)
for wid,(nds,tg) in ways.items():
    if tg.get('highway') in ('secondary','tertiary','residential') and tg.get('oneway')!='yes':
        pts=[nodes[n] for n in nds if n in nodes]
        if len(pts)>1: d.line(to_px(pts),fill=255,width=int(1.6/PX))
YM=np.array(ymask)>0
pc=np.where(YM[...,None],hexc('#e9c13a'),hexc('#efefea'))
img=img*(1-soft[...,None])+pc*soft[...,None]
# detectable warning surfaces at curb ramps (yellow pads)
ramps=json.load(open('data/curbramps.json'))
curbs=[proj(shape(p['the_geom'])) for p in json.load(open('data/curbs.json')) if p['layer']=='CURBS']
cu=unary_union(curbs)
from shapely.ops import nearest_points
pad=Image.new('L',(N,N),0); d=ImageDraw.Draw(pad)
for r in ramps:
    x,y=xy(float(r['longitude']),float(r['latitude']))
    if not (X0<x<X1 and Y0<y<Y1): continue
    q=nearest_points(Point(x,y),cu)[1]
    dx,dy=q.x-x,q.y-y; dl=np.hypot(dx,dy)
    if dl>4: continue
    ux,uy=(dx/dl,dy/dl) if dl>1e-3 else (1,0)
    # pad 1.2 m wide (along curb) x 0.9 m deep, its outer edge at the curb
    cx,cy=q.x-ux*0.5,q.y-uy*0.5; tx,ty=-uy,ux
    poly=[(cx+tx*0.6+ux*0.45,cy+ty*0.6+uy*0.45),(cx-tx*0.6+ux*0.45,cy-ty*0.6+uy*0.45),(cx-tx*0.6-ux*0.45,cy-ty*0.6-uy*0.45),(cx+tx*0.6-ux*0.45,cy+ty*0.6-uy*0.45)]
    d.polygon(to_px(poly),fill=255)
P_=np.array(pad)>0
img[P_]=hexc('#d9b23a')*(0.9+0.1*nI[P_,None])
# subtle sidewalk scoring joints (1.5 m grid) on sidewalks
# sidewalk scoring joints aligned with the nearest curb (1.5 m squares)
ang_img=Image.new('F',(N,N),np.nan) if False else None
amap=np.full((N,N),np.nan,np.float32)
from PIL import ImageDraw as _D
im_=Image.new('I',(N,N),0); dd=_D.Draw(im_)
segs=[]
for cl_ in curbs:
    cs=np.array(cl_.coords)
    for a,b in zip(cs[:-1],cs[1:]):
        ang=np.arctan2(b[1]-a[1],b[0]-a[0])%np.pi; segs.append(ang)
        dd.line(to_px([tuple(a),tuple(b)]),fill=len(segs),width=3)
lab_img=np.array(im_)
dist,(ii,jj)=ndimage.distance_transform_edt(lab_img==0,return_indices=True)
near=lab_img[ii,jj]; angs=np.array([0.0]+segs,np.float32)[near]
XX,YY=np.meshgrid(xs,ys)
u=XX*np.cos(angs)+YY*np.sin(angs); v=-XX*np.sin(angs)+YY*np.cos(angs)
J=((np.abs(((u/1.5)+0.5)%1-0.5)*1.5<PX*0.6)|(np.abs(((v/1.5)+0.5)%1-0.5)*1.5<PX*0.6))&(C==2)
img[J]*=0.88
img=np.clip(img,0,1)
out=Image.fromarray((img*255).astype(np.uint8))
out.save('data/ground_tex.jpg',quality=86)
np.save('data/ground_cls.npy',C)
print('saved', out.size)
