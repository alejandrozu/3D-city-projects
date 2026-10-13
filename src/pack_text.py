import json,struct,gzip,base64,sys
src,dst=sys.argv[1],sys.argv[2]
b=open(src,'rb').read()
jl=struct.unpack('<I',b[12:16])[0]; g=json.loads(b[20:20+jl]); off=20+jl; bl=struct.unpack('<I',b[off:off+4])[0]; binc=b[off+8:off+8+bl]
img_views={im['bufferView'] for im in g.get('images',[])}
for m in g['materials']: m.get('pbrMetallicRoughness',{}).pop('baseColorTexture',None)
for k in ('images','textures','samplers'): g.pop(k,None)
for k in ('extensionsUsed','extensionsRequired'):
    if k in g:
        g[k]=[e for e in g[k] if e!='KHR_texture_transform']
        if not g[k]: g.pop(k)
newbin=bytearray(); remap={}; views=[]
for i,v in enumerate(g['bufferViews']):
    if i in img_views: continue
    while len(newbin)%4: newbin.append(0)
    o=v.get('byteOffset',0); nv=dict(v); nv['byteOffset']=len(newbin); newbin+=binc[o:o+v['byteLength']]; remap[i]=len(views); views.append(nv)
g['bufferViews']=views
for a in g['accessors']:
    if 'bufferView' in a: a['bufferView']=remap[a['bufferView']]
g['buffers']=[{'byteLength':len(newbin)}]
js=json.dumps(g,separators=(',',':')).encode()
while len(js)%4: js+=b' '
while len(newbin)%4: newbin.append(0)
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(newbin))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(newbin),0x004E4942)+bytes(newbin)
gz=gzip.compress(out,9); b64=base64.b64encode(gz)
open(dst,'wb').write(b64)
print('glb %.2f MB  gzip %.2f MB  text %.2f MB'%(len(out)/1e6,len(gz)/1e6,len(b64)/1e6), 'ext',g.get('extensionsUsed'))
