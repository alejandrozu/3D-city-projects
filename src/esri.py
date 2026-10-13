import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import math, sys
sys.path.insert(0,ROOT + '/src')
from geo import *
from PIL import Image
def mosaic(z):
    tx0,ty0,nx,ny=map(int,open(f'{ROOT}/data/esri_z{z}.txt').read().split())
    im=Image.open(f'{ROOT}/data/esri_z{z}.jpg')
    def px(x,y):
        lon,lat=ll(x,y); n=2**z
        X=(lon+180)/360*n; lr=math.radians(lat); Y=(1-math.asinh(math.tan(lr))/math.pi)/2*n
        return (X-tx0)*256,(Y-ty0)*256
    return im,px
