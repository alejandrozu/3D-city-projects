import os as _os; ROOT = _os.environ.get('CASTRO_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # project root
import xml.etree.ElementTree as ET, sys
sys.path.insert(0,ROOT + '/src')
from geo import *
def load(path=ROOT + '/data/osm_raw.xml'):
    t=ET.parse(path).getroot()
    nodes={}; ntags={}
    for n in t.iter('node'):
        nodes[n.get('id')]=xy(float(n.get('lon')),float(n.get('lat')))
        tg={x.get('k'):x.get('v') for x in n.findall('tag')}
        if tg: ntags[n.get('id')]=tg
    ways={}
    for w in t.iter('way'):
        ways[w.get('id')]=([nd.get('ref') for nd in w.findall('nd')],{x.get('k'):x.get('v') for x in w.findall('tag')})
    rels={}
    for r in t.iter('relation'):
        rels[r.get('id')]=([(m.get('type'),m.get('ref'),m.get('role')) for m in r.findall('member')],{x.get('k'):x.get('v') for x in r.findall('tag')})
    return nodes,ntags,ways,rels
def inside(p,m=0):
    return X0-m<=p[0]<=X1+m and Y0-m<=p[1]<=Y1+m
