import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import numpy as np, sys, json, pickle, hashlib, io
sys.path.insert(0,ROOT + '/src')
from common import *
from glb import GLB, vnormals, flat
G=GLB()
def h01(s,k=0): return int(hashlib.md5((s+str(k)).encode()).hexdigest()[:8],16)/2**32
# ---------- terrain
T=np.load('data/terrain_mesh.npz'); v,f=T['v'],T['f']
uv=np.column_stack([(v[:,0]-X0)/(X1-X0),(v[:,1]-Y0)/(Y1-Y0)])
tex=G.texture(open('data/ground_tex.jpg','rb').read())
mg=G.material('ground',(1,1,1,1),rough=0.92,tex=tex)
P=to_model(v[:,0],v[:,1],v[:,2]); N=vnormals(P,f)
G.node('terrain',[G.prim(P,f,N,uv=np.column_stack([uv[:,0],1-uv[:,1]]),material=mg)],group='ground')
# skirts: boundary edges of terrain
from collections import Counter
E=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),1); cnt=Counter(map(tuple,E))
bd=[e for e,c in cnt.items() if c==1]
# order not needed: each boundary edge -> quad down to base
BASE=v[:,2].min()-4.0
SV=[];SF=[]
for a,b in bd:
    pa,pb=v[a],v[b]
    q=[(pa[0],pa[1],pa[2]),(pb[0],pb[1],pb[2]),(pb[0],pb[1],BASE),(pa[0],pa[1],BASE)]
    k=len(SV); SV+=q; SF+=[(k,k+1,k+2),(k,k+2,k+3)]
SV=np.array(SV); SP=to_model(SV[:,0],SV[:,1],SV[:,2]); Pf,If,Nf,_=flat(SP,np.array(SF))
ms=G.material('soil',(0.36,0.30,0.25,1),rough=1.0,double=True)
# base plate
bp=np.array([[X0,Y0,BASE],[X1,Y0,BASE],[X1,Y1,BASE],[X0,Y1,BASE]]); BP=to_model(bp[:,0],bp[:,1],bp[:,2])
G.node('terrain_skirt',[G.prim(Pf,If,Nf,material=ms),G.prim(BP,[0,2,1,0,3,2],np.tile([0,-1,0],(4,1)),material=ms)],group='ground')
# ---------- buildings
B=pickle.load(open('data/buildings.pkl','rb')); meta={m['id']:m for m in json.load(open('data/buildings_meta.json'))}
aer=G.texture(open('web/aerial_naip.jpg','rb').read())
mr=G.material('roof',(1,1,1,1),rough=0.85,tex=aer); mw=G.material('wall',(1,1,1,1),rough=0.9)
mcor=G.material('cornice',(1,1,1,1),rough=0.75)
pal=['#e9e3d5','#d8d2c4','#cfc6b3','#e6dccb','#b9c3c9','#a9b7b1','#c9b8a7','#d9c9b0','#bfb7ad','#e2d6c2','#b3a796','#c7cfd3','#d6c7c9','#cdd5c0','#efe9df','#9fa9ae']
def hexc(h): return np.array([int(h[i:i+2],16)/255 for i in (1,3,5)])
def lin(c): c=np.asarray(c,float); return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
for b in B:
    bid=b['id']; m=meta[bid]
    wc=hexc(pal[int(h01(bid)*len(pal))])*(0.92+0.1*h01(bid,1))
    rc=np.clip(np.array(b['roofcol'])*0.95,0,1)
    prims=[]
    rv=[];rf=[];wv=[];wf=[];wuv=[];wuv1=[];cv=[];cf=[]
    for q in b['pieces']:
        if len(q.get('cornice_f',[])): o3=sum(len(x) for x in cv); cv.append(q['cornice_v']); cf.append(q['cornice_f']+o3)
        o=len(rv) if False else sum(len(x) for x in rv)
        rv.append(q['roof_v']); rf.append(q['roof_f']+o)
        o2=sum(len(x) for x in wv); wv.append(q['wall_v']); wf.append(q['wall_f']+o2); wuv.append(q['wall_uv']); wuv1.append(q['wall_uv1'])
    rv=np.concatenate(rv); rf=np.concatenate(rf); wv=np.concatenate(wv); wf=np.concatenate(wf); wuv=np.concatenate(wuv); wuv1=np.concatenate(wuv1)
    RP=to_model(rv[:,0],rv[:,1],rv[:,2]); Pr,Ir,Nr,_=flat(RP,rf)
    WP=to_model(wv[:,0],wv[:,1],wv[:,2]); Pw,Iw,Nw,UVw,UV1w=flat(WP,wf,wuv,wuv1)
    ruv=np.column_stack([(rv[rf.ravel(),0]-X0)/(X1-X0),(Y1-rv[rf.ravel(),1])/(Y1-Y0)])
    tint=np.clip(lin(rc)/max(lin(rc).mean(),1e-3),0.85,1.15)*0.0+1.0
    prims.append(G.prim(Pr,Ir,Nr,uv=ruv,colors=np.tile(tint*0.97,(len(Pr),1)),material=mr))
    if cv:
        cv=np.concatenate(cv); cf=np.concatenate(cf); CP=to_model(cv[:,0],cv[:,1],cv[:,2]); Pc,Ic,Nc,_=flat(CP,cf)
        trim=np.clip(lin(wc)*1.12+0.06,0,1)
        prims.append(G.prim(Pc,Ic,Nc,colors=np.tile(trim,(len(Pc),1)),material=mcor))
    prims.append(G.prim(Pw,Iw,Nw,uv=UVw,uv1=UV1w,colors=np.tile(lin(wc),(len(Pw),1)),material=mw))
    G.node('bldg_'+bid,prims,extras=m,group='buildings')
# ======================================================= details
G2=G  # same writer
def lin3(c): return c  # details colours already linear
def add_acc(name,data,mat,group,smooth=False):
    if data is None: return
    V,F,Cc=data
    P_=to_model(V[:,0],V[:,1],V[:,2])
    if smooth:
        N_=vnormals(P_,F); G.node(name,[G.prim(P_,F,N_,colors=Cc,material=mat)],group=group)
    else:
        Pf,If,Nf,Cf=flat(P_,F,Cc); G.node(name,[G.prim(Pf,If,Nf,colors=Cf,material=mat)],group=group)
D=pickle.load(open('data/details.pkl','rb'))
M=dict(
 foliage=G.material('foliage',(1,1,1,1),rough=0.95), trunks=G.material('bark',(1,1,1,1),rough=0.95),
 palm_fronds=G.material('frond',(1,1,1,1),rough=0.9,double=True),
 poles=G.material('pole_metal',(1,1,1,1),rough=0.55,metal=0.3), poles_wood=G.material('pole_wood',(1,1,1,1),rough=0.95),
 lamps=G.material('lamp',(1,1,1,1),rough=0.4,emissive=(0.25,0.24,0.2)), signs=G.material('sign',(1,1,1,1),rough=0.5,double=True),
 rails=G.material('rail',(1,1,1,1),rough=0.35,metal=0.9), rails_old=G.material('rail_old',(1,1,1,1),rough=0.6,metal=0.6),
 furniture=G.material('furniture',(1,1,1,1),rough=0.6,metal=0.2), metal=G.material('steel',(1,1,1,1),rough=0.4,metal=0.7),
 shelter_roof=G.material('shelter_roof',(1,1,1,1),rough=0.5), glass=G.material('glass',(1,1,1,0.35),rough=0.1,alpha='BLEND',double=True),
 kiosk=G.material('kiosk',(1,1,1,1),rough=0.6), flagpole=G.material('flagpole',(1,1,1,1),rough=0.35,metal=0.8),
 flag=G.material('flag',(1,1,1,1),rough=0.9,double=True))
grp=dict(foliage='vegetation',trunks='vegetation',palm_fronds='vegetation',flag='landmarks',flagpole='landmarks')
for k,data in D.items():
    add_acc(k,data,M[k],grp.get(k,'streetscape'),smooth=k in ('foliage','trunks','flagpole'))
S=pickle.load(open('data/structures.pkl','rb'))
add_acc('lidar_structures',S,G.material('structure',(1,1,1,1),rough=0.85),'structures')
# ======================================================= plaza
PZ=pickle.load(open('data/plaza.pkl','rb'))
PM=dict(marble=G.material('marble',(1,1,1,1),rough=0.32),marble_floor=G.material('marble_floor',(1,1,1,1),rough=0.6),
        water=G.material('water',(1,1,1,0.62),rough=0.04,alpha='BLEND'),waterfall=G.material('waterfall',(1,1,1,0.55),rough=0.1,alpha='BLEND',double=True),
        foam=G.material('foam',(1,1,1,0.6),rough=0.6,alpha='BLEND'),bronze=G.material('bronze',(1,1,1,1),rough=0.35,metal=0.85),
        statue=G.material('statue_marble',(1,1,1,1),rough=0.36),statue_cloth=G.material('statue_marble_cloth',(1,1,1,1),rough=0.4,double=True))
PM['tiles']=G.material('glazed_tile',(1,1,1,1),rough=0.22); PM['hex_stone']=G.material('paving_stone',(1,1,1,1),rough=0.75); PM['grout']=G.material('grout',(1,1,1,1),rough=0.95)
STEX=[G.texture(open(f'web/stone_{k}.jpg','rb').read()) for k in range(6)]
for k in range(6):
    PM[f'stone{k}']=G.material(f'stone_{k}',(1,1,1,1),rough=0.22,tex=STEX[k])
    PM[f'pool{k}']=G.material(f'pool_stone_{5-k}',(1,1,1,1),rough=0.28,tex=STEX[5-k])   # pool 0 = bottom = purple, pool 5 = top = red
for k,(V,F,Cc) in PZ['acc'].items():
    P_=to_model(V[:,0],V[:,1],V[:,2])
    if k in ('water','waterfall'):
        uvk=PZ['uvs'][k]; N_=vnormals(P_,F)
        G.node('fountain_'+k,[G.prim(P_,F,N_,uv=uvk,colors=Cc,material=PM[k])],group='plaza')
    elif k.startswith('statue'):
        N_=vnormals(P_,F); G.node(k,[G.prim(P_,F,N_,colors=Cc,material=PM[k])],group='plaza',extras=dict(kind='statue'))
    elif k.startswith('pool') or k.startswith('stone'):
        Pf,If,Nf,UVf,Cf=flat(P_,F,PZ['uvs'][k],Cc) if not k.startswith('pool') else (lambda N_: (P_,F,N_,PZ['uvs'][k],Cc))(vnormals(P_,F))
        G.node('fountain_'+k,[G.prim(Pf,If,Nf,uv=UVf,colors=Cf,material=PM[k])],group='plaza')
    elif k=='marble':
        Pf,If,Nf,Cf=flat(P_,F,Cc); G.node('fountain_marble',[G.prim(Pf,If,Nf,colors=Cf,material=PM[k])],group='plaza')
    else:
        Pf,If,Nf,Cf=flat(P_,F,Cc); G.node('fountain_'+k,[G.prim(Pf,If,Nf,colors=Cf,material=PM.get(k,PM['marble']))],group='plaza')
# paving
pv,pf,puv=PZ['pave']; pav=G.texture(open('web/plaza_paving.jpg','rb').read())
mpav=G.material('paving',(1,1,1,1),rough=0.7,tex=pav)
PP=to_model(pv[:,0],pv[:,1],pv[:,2]); G.node('plaza_paving',[G.prim(PP,pf,vnormals(PP,pf),uv=puv,material=mpav)],group='plaza')
# planters (one mesh per colour, instanced) and benches
MPOT=dict(steel=G.material('stainless',(1,1,1,1),rough=0.28,metal=1.0),accent=G.material('pot_accent',(1,1,1,1),rough=0.35,metal=0.3),
          soil=G.material('soil_top',(1,1,1,1),rough=1.0),leaf=G.material('leaf',(1,1,1,1),rough=0.8,double=True),flower=G.material('petal',(1,1,1,1),rough=0.7,double=True))
def zup_to_model(V): return np.column_stack([V[:,0],V[:,2],-V[:,1]])
MPOT['seam']=G.material('steel_seam',(1,1,1,1),rough=0.5,metal=0.8)
patch_mesh={}
for ci,T in PZ['pot_templates'].items():
    prims=[]
    for k,(V,F,Cc) in T.items():
        P_=zup_to_model(V); Pf,If,Nf,Cf=flat(P_,F,Cc); prims.append(G.prim(Pf,If,Nf,colors=Cf,material=MPOT[k]))
    patch_mesh[ci]=G.mesh(f'flowers_{ci}',prims)
NAMES=['red','orange','yellow','green','blue','purple']
for i,p in enumerate(PZ['pots']):
    prims=[]
    for k,(V,F,Cc) in p['mesh'].items():
        P_=zup_to_model(V); Pf,If,Nf,Cf=flat(P_,F,Cc); prims.append(G.prim(Pf,If,Nf,colors=Cf,material=MPOT[k]))
    bm=G.mesh(f'planter_box_{i}',prims)
    kids=[G.inst(f'flowers_{NAMES[p["color"]]}_{i}_{j}',patch_mesh[p['color']],(q['x'],p['soil_z'],0.0),q['yaw'],scale=(q['sx'],1.0,q['sy']),top=False,
                 extras=dict(kind='planter',color=NAMES[p['color']],side=p['side'])) for j,q in enumerate(p['patches'])]
    x,y,z=p['pos']; t=to_model(np.array([x]),np.array([y]),np.array([z]))[0]
    G.inst(f'planter_{NAMES[p["color"]]}_{i}',bm,t,p['yaw'],extras=dict(kind='planter',color=NAMES[p['color']],side=p['side'],length=round(p['L'],2)),group='plaza',children=kids)
MB=dict(iron=G.material('cast_iron',(1,1,1,1),rough=0.5,metal=0.6))
for k in range(6): MB[f'slat{k}']=G.material(f'bench_{NAMES[k]}',(1,1,1,1),rough=0.35)
prims=[]
for k,(V,F,Cc) in PZ['bench'].items():
    P_=zup_to_model(V); Pf,If,Nf,Cf=flat(P_,F,Cc); prims.append(G.prim(Pf,If,Nf,colors=Cf,material=MB[k]))
bmesh=G.mesh('bench',prims)
for i,b in enumerate(PZ['benches']):
    x,y,z=b['pos']; t=to_model(np.array([x]),np.array([y]),np.array([z]))[0]
    G.inst(f'bench_{i}',bmesh,t,b['yaw'],extras=dict(kind='bench',side=b['side']),group='plaza')
# ======================================================= extras (vehicles, signals, grates)
EX=pickle.load(open('data/extras.pkl','rb'))
ME=dict(carpaint=G.material('carpaint',(1,1,1,1),rough=0.25,metal=0.4),carglass=G.material('carglass',(1,1,1,1),rough=0.08,metal=0.3),
        tyre=G.material('tyre',(1,1,1,1),rough=0.9),chrome=G.material('chrome',(1,1,1,1),rough=0.2,metal=1.0),
        headlight=G.material('headlight',(1,1,1,1),rough=0.2,emissive=(0.3,0.3,0.28)),taillight=G.material('taillight',(1,1,1,1),rough=0.3,emissive=(0.25,0.02,0.02)),
        trim=G.material('trim',(1,1,1,1),rough=0.7),plate=G.material('plate',(1,1,1,1),rough=0.5),
        signal=G.material('signal_housing',(1,1,1,1),rough=0.6),lamp_on=G.material('signal_lit',(1,1,1,1),rough=0.3,emissive=(1,1,1)),
        lamp_off=G.material('signal_dark',(1,1,1,1),rough=0.3),grate=G.material('grate',(1,1,1,1),rough=0.8,metal=0.4),
        hydrant=G.material('hydrant',(1,1,1,1),rough=0.5),mailbox=G.material('mailbox',(1,1,1,1),rough=0.45))
CARS=('carpaint','carglass','tyre','chrome','headlight','taillight','trim','plate')
for k,data in EX.items():
    if k in CARS: continue
    add_acc(k,data,ME[k],'vehicles' if k in ('carpaint','carglass','tyre','chrome','headlight','taillight','trim','plate') else 'streetscape')
# ======================================================= mural + brick rim
MU=pickle.load(open('data/mural.pkl','rb'))
mmu=G.material('mural',(1,1,1,1),rough=0.85,tex=G.texture(open('web/mural.jpg','rb').read()))
P_=to_model(MU['V'][:,0],MU['V'][:,1],MU['V'][:,2]); G.node('mural',[G.prim(P_,MU['F'],vnormals(P_,MU['F']),uv=MU['UV'],material=mmu)],group='plaza',extras=dict(kind='mural'))
RI=pickle.load(open('data/rim.pkl','rb'))
mri=G.material('brick_rim',(1,1,1,1),rough=0.8,tex=G.texture(open('web/brick_rim.jpg','rb').read()))
P_=to_model(RI['V'][:,0],RI['V'][:,1],RI['V'][:,2]); G.node('brick_rim',[G.prim(P_,RI['F'],vnormals(P_,RI['F']),uv=RI['UV'],material=mri)],group='streetscape')
# wires as GL lines
W=pickle.load(open('data/all_wires.pkl','rb'))
LV=[];LI=[]
CUT_R=50.0
def densify(w,step=0.25):
    out=[w[0]]
    for a_,b_ in zip(w[:-1],w[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b_[:2]-a_[:2])/step)))
        for k in range(1,n+1): out.append(a_+(b_-a_)*k/n)
    return np.array(out)
nkept=0; ncut=0
for w in W:
    w=densify(np.asarray(w,float)); keep=np.hypot(w[:,0]-CX,w[:,1]-CY)>=CUT_R
    # split into runs of kept points
    i=0
    while i<len(w):
        if not keep[i]: i+=1; ncut+=1; continue
        j=i
        while j<len(w) and keep[j]: j+=1
        if j-i>=2:
            seg=w[i:j]; P_=to_model(seg[:,0],seg[:,1],seg[:,2]); b=sum(len(x) for x in LV); LV.append(P_)
            LI+= [(b+k,b+k+1) for k in range(len(P_)-1)]; nkept+=1
        i=j
print('wire runs kept',nkept,'points removed',ncut)
mwire=G.material('wire',(0.05,0.05,0.05,1),rough=0.6)
G.node('overhead_wires',[G.prim(np.concatenate(LV),np.array(LI),material=mwire,mode=1)],group='wires')
G.save('web/castro.glb',scene_extras=dict(origin_lat=O_LAT,origin_lon=O_LON,center_east_m=CX,zbase=ZB))
import os; print('glb MB',os.path.getsize('web/castro.glb')/1e6)
