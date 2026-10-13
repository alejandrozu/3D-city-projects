import xml.etree.ElementTree as ET, sys, math, collections
t=ET.parse('data/osm_raw.xml').getroot()
nodes={}; ntags={}
for n in t.iter('node'):
    nodes[n.get('id')]=(float(n.get('lat')),float(n.get('lon')))
    tg={x.get('k'):x.get('v') for x in n.findall('tag')}
    if tg: ntags[n.get('id')]=tg
ways={}
for w in t.iter('way'):
    ways[w.get('id')]=([nd.get('ref') for nd in w.findall('nd')],{x.get('k'):x.get('v') for x in w.findall('tag')})
# find intersection Castro & Market
def wn(name): return [ (wid,w) for wid,w in ways.items() if w[1].get('name')==name and 'highway' in w[1]]
cs=set(); ms=set()
for wid,w in wn('Castro Street'): cs|=set(w[0]); print('Castro', wid, w[1].get('highway'), w[1].get('lanes'), w[1].get('width'), len(w[0]))
for wid,w in wn('Market Street'): ms|=set(w[0]); print('Market', wid, w[1].get('highway'), w[1].get('lanes'),w[1].get('oneway'), w[1].get('width'), len(w[0]))
for wid,w in wn('17th Street'): print('17th', wid, w[1].get('highway'), w[1].get('lanes'), w[1].get('oneway'), len(w[0]))
inter=cs&ms
print('intersection nodes', [(i,nodes[i],ntags.get(i)) for i in inter])
for k,v in ntags.items():
    s=str(v).lower()
    if 'chevron' in s or 'café' in s or 'cafe' in s or 'fuel' in s or 'castro gas' in s:
        print('NODE',k,nodes[k],v)
for wid,w in ways.items():
    s=str(w[1]).lower()
    if 'chevron' in s or 'fuel' in s or 'the cafe' in s or 'café' in s or 'castro gas' in s or 'theatre' in s or 'milk' in s:
        print('WAY',wid,w[1], len(w[0]))
