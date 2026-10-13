import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
"""Download the base sources for the square: OpenStreetMap, USGS 3DEP 1 m terrain, USGS NAIP imagery,
DataSF building footprints / parcels / street trees / Muni routes. (LiDAR: ept_fetch.py; other DataSF layers: fetch_sf.py)"""
import sys, json, subprocess, urllib.parse
sys.path.insert(0, ROOT + '/src')
from geo import *
import pyproj
D = ROOT + '/data/'
def curl(url, out=None):
    args = ['curl', '-s', '-m', '180'] + (['-o', out] if out else []) + [url]
    return subprocess.run(args, capture_output=True, text=True).stdout
mnlon, mnlat, mxlon, mxlat = bbox_ll()
curl('https://api.openstreetmap.org/api/0.6/map?bbox=-122.4375,37.7605,-122.4320,37.7650', D + 'osm_raw.xml')
U = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
xs = []; ys = []
for lo in (mnlon, mxlon):
    for la in (mnlat, mxlat):
        a, b = U.transform(lo, la); xs.append(a); ys.append(b)
bx = (min(xs), min(ys), max(xs), max(ys)); W = int(round(bx[2] - bx[0])); H = int(round(bx[3] - bx[1]))
open(D + 'utm_bbox.txt', 'w').write(','.join(map(str, bx)) + f'\n{W},{H}\n')
curl("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={},{},{},{}&bboxSR=32610&imageSR=32610&size={},{}&format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image".format(*bx, W, H), D + 'dem.tif')
curl("https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage?bbox={},{},{},{}&bboxSR=32610&imageSR=32610&size={},{}&format=png&f=image".format(*bx, W * 2, H * 2), D + 'naip.png')
wkt = f"POLYGON(({mnlon} {mnlat}, {mxlon} {mnlat}, {mxlon} {mxlat}, {mnlon} {mxlat}, {mnlon} {mnlat}))"
def soda(ds, where, out):
    d = json.loads(curl(f"https://data.sf.gov/resource/{ds}.json?$limit=50000&$where=" + urllib.parse.quote(where)))
    json.dump(d, open(D + out, 'w')); print(ds, len(d))
soda('ynuv-fyni', f"intersects(shape,'{wkt}')", 'bldg_sf.json')
soda('acdm-wktn', f"intersects(shape,'{wkt}') AND active='true'", 'parcels_sf.json')
soda('tkzw-k3nq', f"latitude::number > {mnlat} AND latitude::number < {mxlat} AND longitude::number > {mnlon} AND longitude::number < {mxlon}", 'trees_sf.json')
soda('9exe-acju', f"intersects(shape,'{wkt}')", 'muni_routes.json')
