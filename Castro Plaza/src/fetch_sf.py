import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import sys, json, subprocess, urllib.parse
sys.path.insert(0,ROOT + '/src')
from geo import *
mnlon,mnlat,mxlon,mxlat=bbox_ll()
def curl(u): return subprocess.run(['curl','-s','-m','180',u],capture_output=True,text=True).stdout
def fetch(ds,out):
    meta=json.loads(curl(f'https://data.sf.gov/api/views/{ds}.json'))
    cols=[(c['fieldName'],c['dataTypeName']) for c in meta['columns']]
    geo=[c for c in cols if c[1] in ('point','multipolygon','polygon','line','multiline','location','multipoint')]
    names=[c[0] for c in cols]
    if geo: w=f"within_box({geo[0][0]},{mxlat},{mnlon},{mnlat},{mxlon})"
    else:
        la=[n for n in names if n.lower() in ('latitude','lat','y_lat','point_y')]; lo=[n for n in names if n.lower() in ('longitude','lon','long','x_long','point_x')]
        w=f"{la[0]}::number > {mnlat} AND {la[0]}::number < {mxlat} AND {lo[0]}::number > {mnlon} AND {lo[0]}::number < {mxlon}"
    d=json.loads(curl(f"https://data.sf.gov/resource/{ds}.json?$limit=20000&$where="+urllib.parse.quote(w)))
    if isinstance(d,dict): print(ds,meta['name'],'ERR',str(d)[:300], cols); return
    json.dump(d,open(out,'w')); print(ds, meta['name'], len(d), 'geo=',geo, [n for n in names if not n.startswith(':')][:25])
for ds,out in [('emxt-b6yg','curbs.json'),('h8n7-e4ns','row.json'),('ybh5-27n2','signals.json'),('m48z-6ji4','signs.json'),('8vzz-qzz9','meters.json'),('hn4j-6fx5','bikeracks.json'),('5buk-jp7b','trashcans.json'),('g9zy-srvv','crosswalks.json'),('i28k-bkz6','munistops.json'),('ch9w-7kih','curbramps.json'),('jczu-j2ku','parklets.json'),('3psu-pn9h','streets.json'),('4g86-grxu','sidewalkwidths.json')]:
    try: fetch(ds,out)
    except Exception as e: print(ds,'EXC',e)
