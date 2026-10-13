import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, hashlib
sys.path.insert(0,ROOT + '/src')
from common import *
from osm_load import load
import triangle, pyfqmr
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from shapely import prepared
from PIL import Image, ImageDraw
import pyproj
SQ=box(X0,Y0,X1,Y1)
ny,nx=DSM_LAST.shape
cx_=GX0+(np.arange(nx)+0.5)*RES; cy_=GY0+(np.arange(ny)+0.5)*RES
GT=DTM  # ground grid (smoothed)
# NAIP for roof colours
U=pyproj.Transformer.from_crs("EPSG:4326","EPSG:32610",always_xy=True)
bx=list(map(float,open('data/utm_bbox.txt').read().split('\n')[0].split(',')))
naip=np.asarray(Image.open('data/naip.png').convert('RGB')).astype(np.float32)/255
sN=naip.shape[1]/(bx[2]-bx[0])
def naip_at(x,y):
    lo,la=ll(np.asarray(x),np.asarray(y)); e,n=U.transform(lo,la)
    c=(np.asarray(e)-bx[0])*sN-0.5; r=(bx[3]-np.asarray(n))*sN-0.5
    return np.stack([ndimage.map_coordinates(naip[...,k],[r,c],order=1,mode='nearest') for k in range(3)],-1)
def mask_of(poly):
    """boolean mask on raster grid for polygon (cell centres)"""
    img=Image.new('L',(nx,ny),0); d=ImageDraw.Draw(img)
    def tp(cs): return [((x-GX0)/RES-0.5,(y-GY0)/RES-0.5) for x,y in cs]
    for g in getattr(poly,'geoms',[poly]):
        d.polygon(tp(g.exterior.coords),fill=1)
        for h in g.interiors: d.polygon(tp(h.coords),fill=0)
    return np.array(img).astype(bool)
def densify(coords,step=0.5):
    out=[]
    for (x0,y0),(x1,y1) in zip(coords[:-1],coords[1:]):
        L=np.hypot(x1-x0,y1-y0); n=max(1,int(np.ceil(L/step)))
        for k in range(n): out.append((x0+(x1-x0)*k/n,y0+(y1-y0)*k/n))
    return out
def roof_field(poly,fallback):
    m=mask_of(poly)
    if m.sum()<2: return None,None
    js,is_=np.nonzero(m); j0,j1,i0,i1=js.min()-4,js.max()+5,is_.min()-4,is_.max()+5
    j0=max(j0,0); i0=max(i0,0)
    sub=m[j0:j1,i0:i1]; z=DSM_ROOF[j0:j1,i0:i1].copy(); g=GT[j0:j1,i0:i1]
    z[~sub]=np.nan
    if MULTI is not None and np.isfinite(z[sub]).sum()>10:
        mf=MULTI[j0:j1,i0:i1]
        base=np.nanmedian(z[sub])
        cand=sub&np.isfinite(z)&(z>base+2.0)
        lab_,nl=ndimage.label(ndimage.binary_dilation(cand,iterations=1))
        for k in range(1,nl+1):
            reg=(lab_==k)&sub
            if reg.sum()>=4 and np.nanmean(mf[reg])>0.2:
                z[ndimage.binary_dilation(reg,iterations=2)&sub&(mf>0.12)]=np.nan
    valid=np.isfinite(z)&sub
    if valid.sum()<3: z=np.where(sub,fallback,np.nan); valid=sub.copy()
    # nearest fill of holes inside footprint
    _,ind=ndimage.distance_transform_edt(~valid,return_indices=True); z=z[ind[0],ind[1]]
    # robust denoise: median 5x5 (1.25 m) restricted to footprint values
    z=ndimage.median_filter(z,5)
    z=ndimage.grey_closing(z,size=(7,7))
    dist=ndimage.distance_transform_edt(np.pad(sub,1))[1:-1,1:-1]*RES
    band=0.8
    deep=sub&(dist>=band)
    if deep.sum()>=4:
        _,ind=ndimage.distance_transform_edt(~deep,return_indices=True)
        bad=~deep
        z[bad]=z[ind[0][bad],ind[1][bad]]
    # suppress isolated spikes (poles/flags/trees): cap at 95th pct + 2 m
    zin=z[sub]; cap=np.percentile(zin,97)+1.5
    z=np.minimum(z,cap)
    z=np.maximum(z,g+2.2)  # roofs at least 2.2 m above ground
    return z,(j0,i0)
def build_one(poly,fallback,others=None,commercial=False,zoff=-0.4,underside=False):
    z,(j0,i0)=roof_field(poly,fallback)
    if z is None: return None
    def zs(x,y):
        c=(np.asarray(x)-GX0)/RES-0.5-i0; r=(np.asarray(y)-GY0)/RES-0.5-j0
        return ndimage.map_coordinates(z,[r,c],order=1,mode='nearest')
    # PSLG
    verts=[]; segs=[]; holes=[]; rings=[]
    for ring in [poly.exterior]+list(poly.interiors):
        pts=densify(list(ring.coords),0.5)
        if len(pts)<3: continue
        b=len(verts); verts+=pts; n=len(pts)
        segs+=[(b+k,b+(k+1)%n) for k in range(n)]; rings.append((b,n))
    for h in poly.interiors:
        hp=Polygon(h).representative_point(); holes.append((hp.x,hp.y))
    # interior grid points
    minx,miny,maxx,maxy=poly.bounds
    gx=np.arange(np.floor(minx/0.5)*0.5,maxx,0.5); gy=np.arange(np.floor(miny/0.5)*0.5,maxy,0.5)
    GXX,GYY=np.meshgrid(gx,gy); pp=prepared.prep(poly.buffer(-0.3))
    inner=[(a,b) for a,b in zip(GXX.ravel(),GYY.ravel()) if pp.contains(Point(a,b))]
    nb=len(verts)
    A=dict(vertices=np.array(verts+inner),segments=np.array(segs))
    if holes: A['holes']=np.array(holes)
    T=triangle.triangulate(A,'pQ')
    V=T['vertices']; F=T['triangles']
    Z=zs(V[:,0],V[:,1])
    V3=np.column_stack([V,Z])
    if len(F)>60:
        s=pyfqmr.Simplify(); s.setMesh(V3.astype(np.float64),F.astype(np.int32))
        s.simplify_mesh_lossless(epsilon=LOSSLESS_EPS,preserve_border=True,verbose=False,max_iterations=100)
        V3b,Fb,_=s.getMesh()
    else: V3b,Fb=V3,F
    # walls from the (unchanged) boundary rings
    W=[];WF=[]; WUV=[]; WUV1=[]; CV=[]; CF=[]
    def sweep(pa,pb,za,zb,nrm,prof):
        n0=len(CV); m=len(prof)
        for (o,dz) in prof: CV.append((pa[0]+nrm[0]*o,pa[1]+nrm[1]*o,za+dz))
        for (o,dz) in prof: CV.append((pb[0]+nrm[0]*o,pb[1]+nrm[1]*o,zb+dz))
        for i in range(m):
            a0=n0+i; a1=n0+(i+1)%m; b0=a0+m; b1=a1+m
            CF.extend([(a0,b0,b1),(a0,b1,a1)])
        CF.extend([(n0,n0+i+1,n0+i) for i in range(1,m-1)]); CF.extend([(n0+m,n0+m+i,n0+m+i+1) for i in range(1,m-1)])
    CORN=[(0,-0.50),(0.07,-0.50),(0.07,-0.42),(0.15,-0.36),(0.20,-0.30),(0.20,-0.24),(0.30,-0.14),(0.32,-0.04),(0.32,0.02),(0,0.02)]
    BELT=[(0,-0.10),(0.07,-0.10),(0.09,-0.06),(0.09,0.06),(0,0.06)]
    for b,n in rings:
        ring=np.array(verts[b:b+n]+[verts[b]])
        zt=zs(ring[:,0],ring[:,1]); gr=ground(ring[:,0],ring[:,1]); zb=gr+zoff; zt=np.maximum(zt,zb+0.5)
        s_=np.concatenate([[0],np.cumsum(np.hypot(np.diff(ring[:,0]),np.diff(ring[:,1])))])
        for k in range(n):
            d=ring[k+1]-ring[k]; L=np.hypot(*d)
            kind=0.0
            if L>1e-6:
                nrm=np.array([d[1],-d[0]])/L; m=(ring[k]+ring[k+1])/2; tp=m+nrm*0.7
                if not (X0<tp[0]<X1 and Y0<tp[1]<Y1): kind=0.0
                elif others is not None and others.contains(Point(*tp)): kind=0.0
                elif PLAZA_TRI.contains(Point(*tp)): kind=0.0          # blank wall facing the plaza (mural)
                else:
                    c=cls_at(*tp)
                    kind=(1.0 if commercial else 0.75) if c in (1,2,3,4,8) else 0.5
            if kind>=0.74 and L>1.0 and min(zt[k]-gr[k],zt[k+1]-gr[k+1])>4.5:
                d_=d/L; pa=ring[k]-d_*0.06; pb=ring[k+1]+d_*0.06
                sweep(pa,pb,zt[k],zt[k+1],nrm,CORN)
                if kind>0.9: sweep(pa,pb,gr[k]+3.62,gr[k+1]+3.62,nrm,BELT)
                elif min(zt[k]-gr[k],zt[k+1]-gr[k+1])>7.5: sweep(pa,pb,gr[k]+0.9+3.2-0.08,gr[k+1]+0.9+3.2-0.08,nrm,BELT)
            base=len(W)
            W+= [(ring[k,0],ring[k,1],zb[k]),(ring[k+1,0],ring[k+1,1],zb[k+1]),(ring[k+1,0],ring[k+1,1],zt[k+1]),(ring[k,0],ring[k,1],zt[k])]
            WUV+=[(s_[k],zoff),(s_[k+1],zoff),(s_[k+1],zt[k+1]-gr[k+1]),(s_[k],zt[k]-gr[k])]
            WUV1+=[(kind,zt[k]-gr[k]),(kind,zt[k+1]-gr[k+1]),(kind,zt[k+1]-gr[k+1]),(kind,zt[k]-gr[k])]
            WF+=[(base,base+1,base+2),(base,base+2,base+3)]
    if underside:      # soffit of an overhanging bay
        Vu=np.column_stack([V,ground(V[:,0],V[:,1])+zoff]); b0=len(W)
        W+=[tuple(v) for v in Vu]; WUV+=[(0.0,0.0)]*len(Vu); WUV1+=[(0.0,0.0)]*len(Vu)
        WF+=[(b0+f[0],b0+f[2],b0+f[1]) for f in F]
    return dict(cornice_v=np.array(CV).reshape(-1,3),cornice_f=np.array(CF,dtype=int).reshape(-1,3),roof_v=V3b,roof_f=Fb,wall_v=np.array(W),wall_f=np.array(WF),wall_uv=np.array(WUV),wall_uv1=np.array(WUV1),zfield=(z,j0,i0))
LOSSLESS_EPS=float(sys.argv[1]) if len(sys.argv)>1 else 1e-3
# ---------- main
nodes,ntags,ways,rels=load()
pois=[(nodes[k],t) for k,t in ntags.items() if 'name' in t and any(a in t for a in ('amenity','shop','tourism','leisure','office','craft'))]
parcels=[(proj(shape(p['shape'])).buffer(0),p) for p in json.load(open('data/parcels_sf.json'))]
B=json.load(open('data/bldg_sf.json'))
out=[]; meta=[]
CANOPY_IDS={'201006.0059932','201006.0158617','201006.0158847','201006.0074707'}
Cg=np.load('data/ground_cls.npy'); NC=Cg.shape[0]; PXC=(X1-X0)/NC
def cls_at(x,y):
    i=int(np.clip((x-X0)/PXC,0,NC-1)); j=int(np.clip((Y1-y)/PXC,0,NC-1)); return int(Cg[j,i])
import pickle as _pk
_S=_pk.load(open('data/plaza_shape.pkl','rb')); PLAZA_TRI=prepared.prep(_S['tri'].buffer(-0.3))
LOTS=unary_union([pg for pg,_ in parcels]).buffer(0.05)
BAY_H=3.0; BAY_D=0.9
ALLP={p['sf16_bldgid']:proj(shape(p['shape'])).buffer(0) for p in B if p['sf16_bldgid'] not in CANOPY_IDS}
zoning={}
for pg,pp in parcels: pass
for p in B:
    if p['sf16_bldgid'] in CANOPY_IDS: continue
    g=proj(shape(p['shape'])).buffer(0)
    if not g.intersects(SQ): continue
    gc=g.intersection(SQ)
    parts=[q for q in getattr(gc,'geoms',[gc]) if q.geom_type=='Polygon' and q.area>2.0]
    if not parts: continue
    gnd=float(p['gnd_min_m']); hmed=float(p['hgt_median_m'])
    bid=p['sf16_bldgid']; mblr=p.get('mblr','')
    pieces=[]
    others=prepared.prep(unary_union([v for k,v in ALLP.items() if k!=bid and v.intersects(g.buffer(2))]) if any(k!=bid and v.intersects(g.buffer(2)) for k,v in ALLP.items()) else Polygon())
    zc=[pp.get('zoning_code','') for pg,pp in parcels if pg.intersects(g) and pg.intersection(g).area>0.3*min(g.area,pg.area)]
    commercial=any(z_ and (z_.startswith('NC') or z_.startswith('C-') or z_.startswith('UMU')) for z_ in zc) or bool([1 for (pt,t) in pois if g.buffer(1.5).contains(Point(pt))])
    # ground floor stands on the property line; up to 0.9 m of bay overhang above 3 m (SF bay-window limit)
    main_g=g.intersection(LOTS)
    if main_g.area>0.3*g.area and hmed>5.0:
        bay_g=g.intersection(LOTS.buffer(BAY_D)).difference(main_g).buffer(-0.02).buffer(0.02)
        plan=[(main_g.intersection(SQ),dict(zoff=-0.4)),(bay_g.intersection(SQ),dict(zoff=BAY_H,underside=True))]
    else:
        plan=[(gc,dict(zoff=-0.4))]
    for geom,kw in plan:
        for q in getattr(geom,'geoms',[geom]):
            if q.geom_type!='Polygon' or q.area<(2.0 if kw['zoff']<0 else 0.4): continue
            q=orient(q.simplify(0.05),1.0)
            r=build_one(q,gnd+hmed,others,commercial,**kw)
            if r: pieces.append(r)
    if not pieces: continue
    # metadata
    addr=[]
    for pg,pp in parcels:
        if pg.intersects(g) and pg.intersection(g).area>0.3*min(g.area,pg.area):
            a=f"{pp.get('from_address_num','')}{'-'+pp['to_address_num'] if pp.get('to_address_num') not in (None,pp.get('from_address_num')) else ''} {str(pp.get('street_name','')).title()} {str(pp.get('street_type','')).title()}".strip()
            if a and a not in addr: addr.append(a)
    names=[t['name'] for (pt,t) in pois if g.buffer(1.5).contains(Point(pt))]
    rv=np.concatenate([q['roof_v'] for q in pieces]); 
    hag=rv[:,2]-ground(rv[:,0],rv[:,1])
    inner=g.buffer(-1.0)
    if inner.is_empty: inner=g
    rp=inner.representative_point()
    samp=[inner.representative_point()]
    xsS=np.random.default_rng(0).uniform(*g.bounds[0::2],400); ysS=np.random.default_rng(1).uniform(*g.bounds[1::2],400)
    pi=prepared.prep(inner); sel=[(a,b) for a,b in zip(xsS,ysS) if pi.contains(Point(a,b))][:120]
    col=np.median(naip_at([s[0] for s in sel] or [rp.x],[s[1] for s in sel] or [rp.y]),axis=0)
    out.append(dict(id=bid,pieces=pieces,roofcol=col.tolist()))
    meta.append(dict(id=bid,mblr=mblr,addr=addr,names=names,ground=round(gnd,2),h_med=round(hmed,2),
        h_max=round(float(p['hgt_maxcm'])/100,2),roof_top=round(float(rv[:,2].max()),2),area=round(g.area,1),
        clipped=bool(gc.area<g.area-0.5),commercial=bool(commercial),zoning=sorted(set(zc)),cx=round(g.centroid.x,2),cy=round(g.centroid.y,2)))
print('buildings',len(out),'tris',sum(len(q['roof_f'])+len(q['wall_f']) for o in out for q in o['pieces']))
import pickle; pickle.dump(out,open('data/buildings.pkl','wb')); json.dump(meta,open('data/buildings_meta.json','w'),indent=0)
