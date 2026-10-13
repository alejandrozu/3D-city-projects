import numpy as np
class Acc:
    """accumulate triangle meshes with per-vertex colours (linear RGB)"""
    def __init__(self): self.V=[];self.F=[];self.C=[];self.n=0
    def add(self,V,F,col):
        V=np.asarray(V,float); F=np.asarray(F,int)
        self.V.append(V); self.F.append(F+self.n); c=np.asarray(col,float)
        self.C.append(np.tile(c,(len(V),1)) if c.ndim==1 else c); self.n+=len(V)
    def get(self):
        if not self.V: return None
        return np.concatenate(self.V),np.concatenate(self.F),np.concatenate(self.C)
def srgb(h):
    c=np.array([int(h[i:i+2],16)/255 for i in (1,3,5)]); return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def box(c,size,yaw=0.0):
    """box centred at c=(x,y,z_center), size=(sx,sy,sz), yaw radians about z"""
    sx,sy,sz=np.asarray(size)/2
    v=np.array([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])
    R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    v=v@R.T+np.asarray(c)
    f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v,f
def cyl(p0,p1,r0,r1=None,seg=8,cap=True):
    r1=r0 if r1 is None else r1
    p0=np.asarray(p0,float); p1=np.asarray(p1,float); d=p1-p0; L=np.linalg.norm(d); d=d/L
    a=np.array([1,0,0]) if abs(d[0])<0.9 else np.array([0,1,0]); u=np.cross(d,a); u/=np.linalg.norm(u); w=np.cross(d,u)
    t=np.linspace(0,2*np.pi,seg,endpoint=False)
    ring=np.cos(t)[:,None]*u+np.sin(t)[:,None]*w
    V=np.concatenate([p0+r0*ring,p1+r1*ring]); F=[]
    for k in range(seg):
        a_,b_=k,(k+1)%seg; F+=[[a_,b_,seg+b_],[a_,seg+b_,seg+a_]]
    if cap:
        V=np.concatenate([V,[p0,p1]]); c0,c1=2*seg,2*seg+1
        for k in range(seg):
            a_,b_=k,(k+1)%seg; F+=[[c0,b_,a_],[c1,seg+a_,seg+b_]]
    return V,np.array(F)
def _ico():
    t=(1+5**0.5)/2
    v=np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]],float)
    f=np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
    v/=np.linalg.norm(v,axis=1,keepdims=True); return v,f
def _subdiv(v,f):
    v=list(map(tuple,v)); cache={}; nf=[]
    def mid(a,b):
        k=(min(a,b),max(a,b))
        if k not in cache:
            m=(np.array(v[a])+np.array(v[b]))/2; m/=np.linalg.norm(m); v.append(tuple(m)); cache[k]=len(v)-1
        return cache[k]
    for a,b,c in f:
        ab,bc,ca=mid(a,b),mid(b,c),mid(c,a); nf+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    return np.array(v),np.array(nf)
ICO=[_ico()]
for _ in range(2): ICO.append(_subdiv(*ICO[-1]))
def blob(c,r,level=1,rng=None,noise=0.12):
    v,f=ICO[level]; v=v.copy()
    if rng is not None: v*=(1+noise*(rng.random(len(v))[:,None]-0.5)*2)
    return v*np.asarray(r)+np.asarray(c),f.copy()
def tube(pts,r,seg=4):
    """polyline tube (for wires)"""
    pts=np.asarray(pts,float); V=[];F=[]
    for i in range(len(pts)-1):
        v,f=cyl(pts[i],pts[i+1],r,seg=seg,cap=False); F.append(f+sum(len(x) for x in V)); V.append(v)
    return np.concatenate(V),np.concatenate(F)
def prism(poly_xy,z0,z1):
    """extrude simple polygon (CCW list of xy) between z0,z1 (scalars or per-vertex arrays)"""
    import mapbox_earcut as me
    p=np.asarray(poly_xy,float)
    if np.allclose(p[0],p[-1]): p=p[:-1]
    # ensure CCW
    if np.sum((p[:,0]-np.roll(p[:,0],1))*(p[:,1]+np.roll(p[:,1],1)))>0: p=p[::-1]
    n=len(p); z0=np.broadcast_to(z0,(n,)); z1=np.broadcast_to(z1,(n,))
    tri=me.triangulate_float64(p,np.array([n],dtype=np.uint32)).reshape(-1,3)
    V=np.concatenate([np.column_stack([p,z1]),np.column_stack([p,z0])])
    F=[tri,tri[:,::-1]+n]
    side=[]
    for k in range(n):
        a,b=k,(k+1)%n; side+=[[a+n,b+n,b],[a+n,b,a]]
    F.append(np.array(side))
    return V,np.concatenate(F)
