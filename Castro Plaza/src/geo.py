import pyproj
# Origin: midpoint of the two OSM signal nodes where Castro St crosses the two Market St carriageways
O_LAT=(37.76271+37.7626317)/2; O_LON=(-122.4352027+-122.435195)/2
TM=pyproj.Transformer.from_crs("EPSG:4326", f"+proj=tmerc +lat_0={O_LAT} +lon_0={O_LON} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m", always_xy=True)
INV=pyproj.Transformer.from_crs(f"+proj=tmerc +lat_0={O_LAT} +lon_0={O_LON} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m","EPSG:4326", always_xy=True)
def xy(lon,lat): return TM.transform(lon,lat)
def ll(x,y): return INV.transform(x,y)
# square: centred 30 m east of origin, 200x200 m
CX,CY=30.0,0.0; HALF=100.0
X0,X1,Y0,Y1=CX-HALF,CX+HALF,CY-HALF,CY+HALF
MARGIN=40.0
def bbox_ll(m=MARGIN):
    a=ll(X0-m,Y0-m); b=ll(X1+m,Y1+m)
    return a[0],a[1],b[0],b[1]  # minlon,minlat,maxlon,maxlat
