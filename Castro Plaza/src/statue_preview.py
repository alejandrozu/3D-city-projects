import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import sys, pickle, numpy as np
sys.path.insert(0,ROOT + '/src')
from glb import GLB, vnormals
P=pickle.load(open('data/statues.pkl','rb'))
G=GLB()
mm=G.material('marble',(0.93,0.92,0.89,1),rough=0.38)
mc=G.material('marble_cloth',(0.93,0.92,0.89,1),rough=0.42,double=True)
mg=G.material('floor',(0.5,0.5,0.52,1),rough=0.9)
for k,r in P.items():
    for n,(v,f) in r.items():
        v=np.asarray(v,float); f=np.asarray(f)
        G.node(f'{k}_{n}',[G.prim(v,f,vnormals(v,f),material=mc if n=='cloth' else mm)],group='statues')
fl=np.array([[-1.2,0,-1.2],[1.2,0,-1.2],[1.2,0,1.2],[-1.2,0,1.2]],float)
G.node('floor',[G.prim(fl,[0,2,1,0,3,2],np.tile([0,1,0],(4,1)),material=mg)],group='floor')
G.save('web/statue_test.glb')
