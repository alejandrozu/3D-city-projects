import numpy as np, json, struct
class GLB:
    def __init__(self):
        self.bin=bytearray(); self.bufferViews=[]; self.accessors=[]; self.meshes=[]; self.nodes=[]; self.materials=[]; self.matidx={}
        self.images=[]; self.textures=[]; self.samplers=[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}]
        self.groups={}  # group name -> list of node indices
    def _view(self,data,target=None):
        while len(self.bin)%4: self.bin.append(0)
        off=len(self.bin); self.bin+=data
        v={"buffer":0,"byteOffset":off,"byteLength":len(data)}
        if target: v["target"]=target
        self.bufferViews.append(v); return len(self.bufferViews)-1
    def _acc(self,arr,ctype,typ,target=None,normalized=False,minmax=False):
        arr=np.ascontiguousarray(arr)
        bv=self._view(arr.tobytes(),target)
        a={"bufferView":bv,"componentType":ctype,"count":int(arr.shape[0]),"type":typ}
        if normalized: a["normalized"]=True
        if minmax:
            a["min"]=[float(x) for x in arr.min(0)]; a["max"]=[float(x) for x in arr.max(0)]
        self.accessors.append(a); return len(self.accessors)-1
    def texture(self,jpeg_bytes,mime="image/jpeg"):
        bv=self._view(jpeg_bytes); self.images.append({"bufferView":bv,"mimeType":mime})
        self.textures.append({"sampler":0,"source":len(self.images)-1}); return len(self.textures)-1
    def material(self,key,color=(1,1,1,1),rough=0.9,metal=0.0,tex=None,double=False,alpha=None,emissive=None,extras=None):
        if key in self.matidx: return self.matidx[key]
        m={"name":key,"pbrMetallicRoughness":{"baseColorFactor":list(map(float,color)),"metallicFactor":metal,"roughnessFactor":rough}}
        if tex is not None: m["pbrMetallicRoughness"]["baseColorTexture"]={"index":tex}
        if double: m["doubleSided"]=True
        if alpha: m["alphaMode"]=alpha
        if emissive: m["emissiveFactor"]=list(emissive)
        if extras: m["extras"]=extras
        self.materials.append(m); self.matidx[key]=len(self.materials)-1; return self.matidx[key]
    def prim(self,pos,idx=None,normals=None,uv=None,colors=None,material=None,mode=4,uv1=None):
        att={"POSITION":self._acc(np.asarray(pos,np.float32),5126,"VEC3",34962,minmax=True)}
        if normals is not None: att["NORMAL"]=self._acc(np.asarray(normals,np.float32),5126,"VEC3",34962)
        if uv is not None: att["TEXCOORD_0"]=self._acc(np.asarray(uv,np.float32),5126,"VEC2",34962)
        if uv1 is not None: att["TEXCOORD_1"]=self._acc(np.asarray(uv1,np.float32),5126,"VEC2",34962)
        if colors is not None:
            c=np.asarray(colors)
            if c.dtype!=np.uint8: c=np.clip(np.round(c*255),0,255).astype(np.uint8)
            if c.shape[1]==3: c=np.column_stack([c,np.full(len(c),255,np.uint8)])
            att["COLOR_0"]=self._acc(c,5121,"VEC4",34962,normalized=True)
        p={"attributes":att,"mode":mode}
        if idx is not None:
            idx=np.asarray(idx).ravel()
            if idx.max()<65535: p["indices"]=self._acc(idx.astype(np.uint16),5123,"SCALAR",34963)
            else: p["indices"]=self._acc(idx.astype(np.uint32),5125,"SCALAR",34963)
        if material is not None: p["material"]=material
        return p
    def mesh(self,name,prims):
        self.meshes.append({"name":name,"primitives":prims}); return len(self.meshes)-1
    def inst(self,name,mesh,t=(0,0,0),yaw=0.0,extras=None,group=None,children=None,scale=None,top=True):
        import math
        q=[0.0,math.sin(yaw/2),0.0,math.cos(yaw/2)]   # rotation about +Y (model up)
        n={"name":name,"translation":[float(x) for x in t],"rotation":q}
        if mesh is not None: n["mesh"]=mesh
        if scale is not None: n["scale"]=[float(x) for x in scale]
        if children: n["children"]=list(children)
        if extras: n["extras"]=extras
        self.nodes.append(n); i=len(self.nodes)-1
        if top: self.groups.setdefault(group or 'misc',[]).append(i)
        return i
    def node(self,name,prims,extras=None,group=None):
        self.meshes.append({"name":name,"primitives":prims})
        n={"name":name,"mesh":len(self.meshes)-1}
        if extras: n["extras"]=extras
        self.nodes.append(n); i=len(self.nodes)-1
        self.groups.setdefault(group or 'misc',[]).append(i); return i
    def save(self,path,scene_extras=None):
        # group parent nodes
        roots=[]
        for g,ch in self.groups.items():
            self.nodes.append({"name":g,"children":ch}); roots.append(len(self.nodes)-1)
        gl={"asset":{"version":"2.0","generator":"castro-market-model"},"scene":0,"scenes":[{"nodes":roots,**({"extras":scene_extras} if scene_extras else {})}],
            "nodes":self.nodes,"meshes":self.meshes,"materials":self.materials,"accessors":self.accessors,"bufferViews":self.bufferViews,
            "buffers":[{"byteLength":len(self.bin)}]}
        if self.images: gl.update(images=self.images,textures=self.textures,samplers=self.samplers)
        js=json.dumps(gl,separators=(',',':')).encode()
        while len(js)%4: js+=b' '
        while len(self.bin)%4: self.bin.append(0)
        total=12+8+len(js)+8+len(self.bin)
        with open(path,'wb') as f:
            f.write(struct.pack('<III',0x46546C67,2,total)); f.write(struct.pack('<II',len(js),0x4E4F534A)); f.write(js)
            f.write(struct.pack('<II',len(self.bin),0x004E4942)); f.write(bytes(self.bin))
def vnormals(V,F):
    V=np.asarray(V,np.float64); F=np.asarray(F)
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    N=np.zeros_like(V)
    for k in range(3): np.add.at(N,F[:,k],n)
    l=np.linalg.norm(N,axis=1,keepdims=True); l[l==0]=1; return N/l
def flat(V,F,extra=None,extra2=None):
    """unindex mesh -> flat normals"""
    ex2=None if extra2 is None else np.asarray(extra2)[np.asarray(F).ravel()]
    V=np.asarray(V)[np.asarray(F).ravel()]
    tri=V.reshape(-1,3,3); n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); l=np.linalg.norm(n,axis=1,keepdims=True); l[l==0]=1
    N=np.repeat(n/l,3,axis=0)
    ex=None if extra is None else np.asarray(extra)[np.asarray(F).ravel()]
    if extra2 is not None: return V,np.arange(len(V)),N,ex,ex2
    return V,np.arange(len(V)),N,ex
