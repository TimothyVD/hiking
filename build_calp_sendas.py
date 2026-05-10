#!/usr/bin/env python3
"""
Full pipeline: Plano Sendas Tradicionales de Calp PDF → GeoJSON routes + Grand Tour GPX

Steps:
  1. Convert PDF page 1 to SVG  (requires mutool from mupdf-tools)
  2. Parse SVG paths, identify the 8 colour-coded routes
  3. Calibrate SVG→GPS transform via 5-parameter differential_evolution
     (uses OSM footways as ground truth)
  4. Apply per-route fixes (SV-01 OSM-snap, SV-05 topology)
  5. Export GeoJSON + GPX for all routes
  6. Build Grand Tour via OSM routing through Dit d'Oltà and La Mola
  7. Export tour GPX with SRTM elevation

Usage:
  # One-time: convert PDF (needs mupdf-tools installed)
  mutool convert -F svg -o sendas_page1.svg plano_sendas_tradicionales_calp.pdf 1

  # Then run this script
  python3 build_calp_sendas.py
"""

import json, math, re, os, sys, pickle, subprocess, time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import differential_evolution
from scipy.ndimage import distance_transform_edt
import networkx as nx
import urllib.request

# ── file paths ────────────────────────────────────────────────────────────────
PDF       = "data/plano_sendas_tradicionales_calp.pdf"
SVG       = "sendas_page1.svg"          # generated; excluded from git
GEOJSON   = "data/sendas_tradicionales_calp.geojson"
GPX_OUT   = "data/sendas_tradicionales_calp.gpx"
TOUR_GPX  = "data/tour_grand_calp.gpx"
OSM_PATHS = "data/osm_paths.json"
CALIB_PKL = "/tmp/calp_calib2.pkl"     # cached calibration result
LEAFLET_JS  = "map/leaflet.js"
LEAFLET_CSS = "map/leaflet.css"
MAP_HTML    = "map/sendas_calp_map.html"

# ── known calibration (result of previous optimisation – skip re-running) ─────
CALIB = {
    "K_LON":      6.685e-5,
    "K_LAT":      5.248e-5,
    "pivot_svg":  (817.7, 547.0),
    "pivot_lon":  0.047137,
    "pivot_lat":  38.655346,
    "theta":      -0.03153,      # radians  (-1.806 degrees)
}

# ── route colour catalogue ────────────────────────────────────────────────────
ROUTES = {
    "SV-01": {"name": "Senda Roja - La Mola",          "color": "#ed2e38", "stroke_rgb": (237,  46,  56)},
    "SV-02": {"name": "Senda Magenta - Llebeig",        "color": "#ff5eff", "stroke_rgb": (255,  94, 255)},
    "SV-03": {"name": "Senda Morada - Morro de Toix",   "color": "#af5a9a", "stroke_rgb": (175,  90, 154)},
    "SV-04": {"name": "Senda Naranja - Sella d'Olta",   "color": "#f49e00", "stroke_rgb": (244, 158,   0)},
    "SV-05": {"name": "Senda Azul - Les Salines",       "color": "#009ee0", "stroke_rgb": (  0, 158, 224)},
    "SV-06": {"name": "Senda Verde - Cap Blanc",        "color": "#008634", "stroke_rgb": (  0, 134,  52)},
    "SV-07": {"name": "Senda Lima - Canaret",           "color": "#b1c800", "stroke_rgb": (177, 200,   0)},
    "SV-08": {"name": "Senda Amarilla - El Cantal",     "color": "#ffed00", "stroke_rgb": (255, 237,   0)},
}

# ── Grand Tour waypoints ───────────────────────────────────────────────────────
DIT_OLTA = (38.6611757, 0.0147044)   # el Dit d'Oltà  590 m
LA_MOLA  = (38.6478665, 0.0181269)   # la Mola        538 m
ERMITA   = (38.668215,  0.063325)    # Ermita de la Cometa


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 – SVG → raw GPS coordinates
# ══════════════════════════════════════════════════════════════════════════════

def colour_dist(rgb1, rgb2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(rgb1, rgb2)))

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def parse_style_rgb(style):
    """Extract stroke RGB from an SVG style string."""
    m = re.search(r'stroke:\s*#([0-9a-fA-F]{6})', style)
    if m: return hex_to_rgb(m.group(1))
    m = re.search(r'stroke:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
    if m: return tuple(int(x) for x in m.groups())
    return None

def parse_transform(t):
    """Parse SVG matrix(a,b,c,d,e,f) → numpy 3×3 matrix."""
    m = re.match(r'matrix\(([^)]+)\)', t or '')
    if not m: return np.eye(3)
    a,b,c,d,e,f = map(float, m.group(1).split(','))
    return np.array([[a,c,e],[b,d,f],[0,0,1]])

def sample_path(d_attr, n=80):
    """Very small SVG-path sampler: M/L/H/V/C/S/Z only."""
    import re
    tokens = re.findall(r'[MLHVCSZmlhvcsz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d_attr)
    pts = []; cx = cy = 0.0; cmd = 'M'
    i = 0
    def num(): nonlocal i; v = float(tokens[i]); i+=1; return v
    while i < len(tokens):
        if tokens[i].isalpha(): cmd = tokens[i]; i += 1
        if cmd in ('M','m'):
            x,y = num(),num()
            if cmd=='m': x+=cx; y+=cy
            cx,cy = x,y; pts.append((cx,cy)); cmd='L' if cmd=='M' else 'l'
        elif cmd in ('L','l'):
            x,y = num(),num()
            if cmd=='l': x+=cx; y+=cy
            cx,cy = x,y; pts.append((cx,cy))
        elif cmd in ('H','h'):
            x = num();
            if cmd=='h': x+=cx
            cx=x; pts.append((cx,cy))
        elif cmd in ('V','v'):
            y = num()
            if cmd=='v': y+=cy
            cy=y; pts.append((cx,cy))
        elif cmd in ('C','c'):
            x1,y1,x2,y2,x,y = num(),num(),num(),num(),num(),num()
            if cmd=='c': x1+=cx;y1+=cy;x2+=cx;y2+=cy;x+=cx;y+=cy
            # sample 6 pts along cubic bezier
            for t in [i/6 for i in range(1,7)]:
                bx = (1-t)**3*cx+3*(1-t)**2*t*x1+3*(1-t)*t**2*x2+t**3*x
                by = (1-t)**3*cy+3*(1-t)**2*t*y1+3*(1-t)*t**2*y2+t**3*y
                pts.append((bx,by))
            cx,cy = x,y
        elif cmd in ('S','s'):
            x2,y2,x,y = num(),num(),num(),num()
            if cmd=='s': x2+=cx;y2+=cy;x+=cx;y+=cy
            for t in [i/4 for i in range(1,5)]:
                bx=(1-t)**2*cx+2*(1-t)*t*x2+t**2*x
                by=(1-t)**2*cy+2*(1-t)*t*y2+t**2*y
                pts.append((bx,by))
            cx,cy = x,y
        elif cmd in ('Z','z'):
            break
        else:
            i += 1   # skip unknown
    return pts

def compose_matrix(elem):
    """Walk up SVG tree accumulating transform matrices."""
    mats = []
    while elem is not None:
        t = elem.get('transform','')
        if t: mats.append(parse_transform(t))
        elem = elem.get('_parent')
    m = np.eye(3)
    for mat in reversed(mats): m = mat @ m
    return m

def extract_svg_paths(svg_path):
    """Return list of (route_id, [(sx,sy),...]) for all matched paths."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Attach parent references
    for parent in root.iter():
        for child in parent:
            child.attrib['_parent_id'] = id(parent)
    parent_map = {id(c): p for p in root.iter() for c in p}

    TARGET_SW = 2.423   # stroke-width of route lines
    SW_TOL    = 0.3
    COLOUR_TOL = 60

    results = {rid: [] for rid in ROUTES}

    for elem in root.iter('{http://www.w3.org/2000/svg}path'):
        style = elem.get('style','')
        sw_m  = re.search(r'stroke-width:\s*([\d.]+)', style)
        if not sw_m: continue
        if abs(float(sw_m.group(1)) - TARGET_SW) > SW_TOL: continue

        rgb = parse_style_rgb(style)
        if rgb is None: continue

        best_rid, best_d = None, 1e9
        for rid, info in ROUTES.items():
            d = colour_dist(rgb, info['stroke_rgb'])
            if d < best_d: best_d = d; best_rid = rid
        if best_d > COLOUR_TOL: continue

        d_attr = elem.get('d','')
        if not d_attr: continue

        # Build local→SVG matrix
        M = np.eye(3)
        el = elem
        while el is not None:
            t = el.get('transform','')
            if t: M = parse_transform(t) @ M
            el = parent_map.get(id(el))

        raw_pts = sample_path(d_attr)
        svg_pts = []
        for x,y in raw_pts:
            v = M @ np.array([x,y,1.0])
            svg_pts.append((v[0], v[1]))

        results[best_rid].append(svg_pts)

    return results

# ── calibrated SVG → GPS transform ───────────────────────────────────────────
def svg_to_gps(sx, sy, calib=CALIB):
    K_LON    = calib['K_LON']
    K_LAT    = calib['K_LAT']
    px, py   = calib['pivot_svg']
    LON0     = calib['pivot_lon']
    LAT0     = calib['pivot_lat']
    cos_t    = math.cos(calib['theta'])
    sin_t    = math.sin(calib['theta'])
    dx = sx - px; dy = sy - py
    rx = cos_t*dx - sin_t*dy + px
    ry = sin_t*dx + cos_t*dy + py
    return round(LON0 + rx*K_LON, 6), round(LAT0 - ry*K_LAT, 6)

def segments_to_coords(segments, calib=CALIB):
    all_pts = []
    for seg in segments:
        for sx, sy in seg:
            all_pts.append(svg_to_gps(sx, sy, calib))
    return all_pts


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 – Calibration (5-parameter differential_evolution)
#   Only needed if CALIB_PKL doesn't exist or you want to re-run.
# ══════════════════════════════════════════════════════════════════════════════

def download_osm_paths(bbox=(38.640, 0.010, 38.690, 0.070), out=OSM_PATHS):
    """Download OSM footway/path/track ways for calibration."""
    q = (f'[out:json][timeout:60];'
         f'(way["highway"~"footway|path|track"]'
         f'({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});>;);out body geom;')
    data = q.encode()
    req = urllib.request.Request(
        'https://overpass-api.de/api/interpreter', data=data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.loads(r.read())
    with open(out, 'w') as f: json.dump(raw, f)
    print(f"Saved {out}")

def build_distance_raster(osm_path, bbox, res=0.0002):
    """Rasterise OSM ways → binary grid → EDT for fast proximity scoring."""
    with open(osm_path) as f:
        osm = json.load(f)

    lat0,lon0,lat1,lon1 = bbox
    rows = int((lat1-lat0)/res)+1
    cols = int((lon1-lon0)/res)+1
    grid = np.ones((rows, cols), dtype=np.float32)

    for elem in osm['elements']:
        if elem['type'] != 'way': continue
        geom = elem.get('geometry', [])
        for g in geom:
            r = int((g['lat']-lat0)/res)
            c = int((g['lon']-lon0)/res)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r,c] = 0

    edt = distance_transform_edt(grid).astype(np.float32)
    return edt, lat0, lon0, res

def score_route(coords, edt, lat0, lon0, res):
    total = 0.0
    for lon, lat in coords:
        r = (lat-lat0)/res; c = (lon-lon0)/res
        ri, ci = int(r), int(c)
        if 0<=ri<edt.shape[0] and 0<=ci<edt.shape[1]:
            total += edt[ri,ci]
        else:
            total += 50.0
    return total / max(len(coords), 1)

def run_calibration(svg_segments_all, osm_path=OSM_PATHS):
    """5-parameter calibration: dlon, dlat, klon_scale, klat_scale, rotation_deg."""
    # Use magenta (SV-02) and red (SV-01) as calibration routes
    bbox = (38.640, 0.008, 38.690, 0.075)
    edt, lat0, lon0, res = build_distance_raster(osm_path, bbox)

    # Initial calibration pivot from GCP: Magenta start ≈ (38.668215, 0.063325)
    # (derived from user-supplied GCP and iterative refinement)
    BASE = dict(
        K_LON=6.685e-5, K_LAT=5.248e-5,
        pivot_svg=(817.7, 547.0),
        pivot_lon=0.047137, pivot_lat=38.655346,
        theta=0.0,
    )

    cal_segs = svg_segments_all.get('SV-02', []) + svg_segments_all.get('SV-01', [])

    def objective(params):
        dlon, dlat, klon_s, klat_s, rot_deg = params
        calib = dict(BASE,
            K_LON=BASE['K_LON']*klon_s,
            K_LAT=BASE['K_LAT']*klat_s,
            pivot_lon=BASE['pivot_lon']+dlon,
            pivot_lat=BASE['pivot_lat']+dlat,
            theta=math.radians(rot_deg),
        )
        coords = segments_to_coords(cal_segs, calib)
        return score_route(coords, edt, lat0, lon0, res)

    bounds = [(-0.002,0.002),(-0.002,0.002),(0.90,1.10),(0.90,1.10),(-5.0,5.0)]
    result = differential_evolution(objective, bounds, seed=42,
                                    maxiter=300, tol=1e-5, workers=1)
    dlon,dlat,klon_s,klat_s,rot_deg = result.x
    calib = dict(CALIB,
        K_LON=CALIB['K_LON']*klon_s,
        K_LAT=CALIB['K_LAT']*klat_s,
        pivot_lon=CALIB['pivot_lon']+dlon,
        pivot_lat=CALIB['pivot_lat']+dlat,
        theta=math.radians(rot_deg),
        result_score=result.fun,
    )
    with open(CALIB_PKL,'wb') as f: pickle.dump(calib, f)
    print(f"Calibration done. Score={result.fun:.3f}  rotation={rot_deg:.3f}°")
    return calib


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 – Per-route fixes
# ══════════════════════════════════════════════════════════════════════════════

def fix_sv05_blue(routes_gps):
    """
    SV-05 (Azul) shares its middle section with SV-02 (Magenta) in the PDF.
    The blue SVG only has two small fragments; the shared segment is drawn only
    in magenta colour.  Reconstruction:
      blue_segment_1 (south fragment, idx 0)
      + magenta bridge [109..242]
    The northern piece (lat > 38.670) is excluded – it is incorrectly drawn
    in the source PDF and the route actually ends at Ermita de la Cometa.
    """
    mag = routes_gps['SV-02']
    blue_segs = routes_gps['SV-05']

    # Identify the southern blue fragment (starts near Senda Roja start)
    def seg_min_lat(seg): return min(p[1] for p in seg)
    blue_segs_sorted = sorted(blue_segs, key=seg_min_lat)
    b0 = blue_segs_sorted[0]  # southernmost fragment

    # Magenta bridge: indices 109..242 (empirically determined)
    mag_bridge = mag[109:243]

    return b0 + mag_bridge

def osm_snap_sv01(routes_gps, osm_json_path):
    """
    SV-01 (Roja) is OSM-snapped: route each consecutive pair of GPS points
    through the OSM road network so the track follows actual paths/roads.
    Dead ends (direction-reversal within 70 m) are removed.
    """
    with open(osm_json_path) as f:
        raw = json.load(f)

    node_pos = {}
    for e in raw['elements']:
        if e['type'] == 'node':
            node_pos[e['id']] = (e['lat'], e['lon'])

    G = nx.Graph()
    for e in raw['elements']:
        if e['type'] != 'way': continue
        ns = [n for n in e.get('nodes',[]) if n in node_pos]
        for i in range(len(ns)-1):
            a,b = ns[i], ns[i+1]
            la,loa = node_pos[a]; lb,lob = node_pos[b]
            d = _dist_m(la,loa,lb,lob)
            if not G.has_edge(a,b): G.add_edge(a,b, weight=d)

    def nearest(lat,lon):
        return min(node_pos.items(),
                   key=lambda x: _dist_m(lat,lon,x[1][0],x[1][1]))

    raw_coords = routes_gps['SV-01']
    # Route between every ~10th waypoint for speed
    step = max(1, len(raw_coords)//24)
    waypoints = [raw_coords[i] for i in range(0, len(raw_coords), step)]
    waypoints.append(raw_coords[-1])

    snapped = []
    for i in range(len(waypoints)-1):
        lon_a, lat_a = waypoints[i]
        lon_b, lat_b = waypoints[i+1]
        na, _ = nearest(lat_a, lon_a)
        nb, _ = nearest(lat_b, lon_b)
        try:
            path = nx.shortest_path(G, na, nb, weight='weight')
            for n in path:
                la, lo = node_pos[n]
                snapped.append((lo, la))
        except nx.NetworkXNoPath:
            snapped.append(waypoints[i])

    # Remove dead ends (direction-reversal)
    return _remove_dead_ends(snapped)

def _dist_m(lat1,lon1,lat2,lon2):
    dlat=(lat2-lat1)*111000
    dlon=(lon2-lon1)*111000*math.cos(math.radians((lat1+lat2)/2))
    return math.sqrt(dlat**2+dlon**2)

def _remove_dead_ends(pts, min_gap=40):
    """Remove out-and-back dead-end spurs by dot-product reversal detection."""
    if len(pts) < 3: return pts
    def direction(a,b):
        dx=b[0]-a[0]; dy=b[1]-a[1]; d=math.sqrt(dx*dx+dy*dy)
        return (dx/d,dy/d) if d>1e-9 else (0,0)

    clean = list(pts)
    changed = True
    while changed:
        changed = False
        dirs = [direction(clean[i], clean[i+1]) for i in range(len(clean)-1)]
        i = 1
        while i < len(dirs):
            dot = dirs[i-1][0]*dirs[i][0] + dirs[i-1][1]*dirs[i][1]
            if dot < -0.5:
                # Find return point
                j = i+1
                while j < len(clean):
                    if _dist_m(clean[j][1],clean[j][0],clean[i-1][1],clean[i-1][0]) < min_gap:
                        clean = clean[:i] + clean[j:]
                        changed = True
                        break
                    j += 1
                else:
                    i += 1
            else:
                i += 1
    return clean


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 – OSM routing helpers
# ══════════════════════════════════════════════════════════════════════════════

def download_osm_all_highways(bbox, out_path):
    """Download all highway types for routing (needed for SV-01 and Grand Tour)."""
    s,w,n,e = bbox
    q = (f'[out:json][timeout:30];(way["highway"]({s},{w},{n},{e});>;);out body geom;')
    data = q.encode()
    req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=data,
        headers={'Content-Type':'application/x-www-form-urlencoded'})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.load(r)
    with open(out_path,'w') as f: json.dump(raw, f)
    print(f"Saved {out_path}")
    return raw

def build_osm_graph(osm_data):
    nodes = {e['id']:(e['lat'],e['lon']) for e in osm_data['elements'] if e['type']=='node'}
    G = nx.Graph()
    for n,(lat,lon) in nodes.items():
        G.add_node(n, lat=lat, lon=lon)
    for e in osm_data['elements']:
        if e['type']!='way': continue
        ns = [n for n in e.get('nodes',[]) if n in nodes]
        for i in range(len(ns)-1):
            a,b=ns[i],ns[i+1]
            la,loa=nodes[a]; lb,lob=nodes[b]
            d=_dist_m(la,loa,lb,lob)
            if not G.has_edge(a,b): G.add_edge(a,b,weight=d)
    return G, nodes

def closest_node(G, nodes, lat, lon):
    return min(nodes.items(), key=lambda x:_dist_m(lat,lon,x[1][0],x[1][1]))[0]

def osm_route(G, nodes, lat_a, lon_a, lat_b, lon_b):
    """Return list of (lon,lat) points along OSM shortest path."""
    na = closest_node(G, nodes, lat_a, lon_a)
    nb = closest_node(G, nodes, lat_b, lon_b)
    path = nx.shortest_path(G, na, nb, weight='weight')
    return [(nodes[n][1], nodes[n][0]) for n in path]


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 – Grand Tour construction
# ══════════════════════════════════════════════════════════════════════════════

def build_grand_tour(routes_gps, osm_data):
    """
    Grand Tour de Calp (~25 km):
      Ermita de la Cometa
      → Magenta (SV-02) reversed  (Ermita → SV-02 start near Senda Roja)
      → Red (SV-01) full          (south-west, ends at (38.659, 0.026))
      → OSM → el Dit d'Oltà (590m)
      → OSM → la Mola (538m)
      → OSM → Lima join (idx ~215, lat 38.645)
      → Lima (SV-07) partial eastward (reversed, back toward Senda Roja area)
      → Yellow (SV-08) full
      → Magenta (SV-02) tail      (return to Ermita)
    """
    G, nodes = build_osm_graph(osm_data)

    mag   = routes_gps['SV-02']   # Magenta: start at Senda Roja, end at Ermita
    red   = routes_gps['SV-01']   # Red: start near Senda Roja, end at (38.659,0.026)
    lima  = routes_gps['SV-07']   # Lima: long route going west
    yel   = routes_gps['SV-08']   # Yellow

    # Magenta reversed (Ermita → Senda Roja start)
    mag_rev = list(reversed(mag))

    # Red full (start is near Senda Roja, appended after mag_rev)
    red_end_lat, red_end_lon = red[-1][1], red[-1][0]

    # OSM: Red_end → Dit d'Oltà → La Mola
    seg_dit  = osm_route(G, nodes, red_end_lat, red_end_lon, *DIT_OLTA)
    # Insert actual peak coords between OSM-snapped approach and retrace
    dit_peak = [(DIT_OLTA[1], DIT_OLTA[0])]
    seg_mola = osm_route(G, nodes, *DIT_OLTA, *LA_MOLA)
    mola_peak = [(LA_MOLA[1], LA_MOLA[0])]

    # La Mola → Lima join: find Lima point nearest La Mola
    lima_join_idx = min(range(len(lima)),
                        key=lambda i: _dist_m(LA_MOLA[0], LA_MOLA[1], lima[i][1], lima[i][0]))
    seg_to_lima = osm_route(G, nodes, *LA_MOLA,
                             lima[lima_join_idx][1], lima[lima_join_idx][0])

    # Lima partial eastward: from join back toward east (reversed up to join)
    lima_partial = list(reversed(lima[:lima_join_idx+1]))

    # Yellow full
    # Yellow starts near Senda Roja area; connect from Lima end via OSM if needed
    yel_start_lat, yel_start_lon = yel[0][1], yel[0][0]
    lima_end = lima_partial[-1]
    seg_to_yel = osm_route(G, nodes, lima_end[1], lima_end[0], yel_start_lat, yel_start_lon)

    # Magenta tail: Yellow ends near Magenta; take Magenta back to Ermita
    yel_end_lat, yel_end_lon = yel[-1][1], yel[-1][0]
    # Find nearest Magenta point to Yellow end
    mag_tail_idx = min(range(len(mag)),
                       key=lambda i: _dist_m(yel_end_lat, yel_end_lon, mag[i][1], mag[i][0]))
    mag_tail = mag[mag_tail_idx:]   # Magenta from join → Ermita

    tour = (mag_rev
          + red
          + seg_dit[1:]           # skip duplicate of red_end
          + dit_peak
          + seg_mola[1:]          # skip duplicate of Dit d'Olta node
          + mola_peak
          + seg_to_lima[1:]
          + lima_partial
          + seg_to_yel[1:]
          + yel
          + mag_tail)

    return tour


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 – GeoJSON + GPX export
# ══════════════════════════════════════════════════════════════════════════════

GEOJSON_HEADER = {
    "type": "FeatureCollection",
    "name": "Sendas Tradicionales de Calp",
    "crs": {"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
}

def write_geojson(routes_gps, path=GEOJSON):
    features = []
    for rid, info in ROUTES.items():
        coords = routes_gps.get(rid, [])
        if not coords: continue
        features.append({
            "type": "Feature",
            "properties": {
                "id": rid,
                "color": info["color"],
                "name": info["name"],
                "source": "Plano de las Sendas Tradicionales de Calp (2022)",
                "georef_note": (
                    "K_LON=6.685e-5, K_LAT=5.248e-5, rot=-1.806deg. "
                    "SV-01 OSM-snapped. SV-05 topology-reconstructed."
                ),
            },
            "geometry": {"type":"LineString","coordinates":[[lon,lat] for lon,lat in coords]},
        })
    gj = dict(GEOJSON_HEADER, features=features)
    with open(path,'w') as f: json.dump(gj, f, separators=(',',':'))
    print(f"Wrote {path}")

def write_gpx(routes_gps, path=GPX_OUT):
    from datetime import datetime
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Calpe Sendas" xmlns="http://www.topografix.com/GPX/1/1">',
        f'  <metadata><time>{datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}</time></metadata>',
    ]
    for rid, info in ROUTES.items():
        coords = routes_gps.get(rid, [])
        if not coords: continue
        lines += [f'  <trk><name>{info["name"]} ({rid})</name><trkseg>']
        for lon,lat in coords:
            lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}"/>')
        lines += ['  </trkseg></trk>']
    lines.append('</gpx>')
    with open(path,'w') as f: f.write('\n'.join(lines))
    print(f"Wrote {path}")

def fetch_elevations(lonlat_list, dataset='srtm30m'):
    """Fetch SRTM elevations from api.opentopodata.org in batches of 100."""
    elevs = []
    for i in range(0, len(lonlat_list), 100):
        batch = lonlat_list[i:i+100]
        locs = '|'.join(f"{p[1]},{p[0]}" for p in batch)
        url = f'https://api.opentopodata.org/v1/{dataset}?locations={locs}'
        req = urllib.request.Request(url, headers={'User-Agent':'calpe-map/1.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            elevs.extend(x['elevation'] for x in json.loads(r.read())['results'])
        time.sleep(0.8)
    return elevs

def write_tour_gpx(tour_coords, out='tour_grand_calp.gpx', with_elevation=True):
    from datetime import datetime
    elevs = fetch_elevations(tour_coords) if with_elevation else [None]*len(tour_coords)
    gain = loss = 0.0
    for i in range(1,len(elevs)):
        if elevs[i] and elevs[i-1]:
            d = elevs[i]-elevs[i-1]
            if d>0: gain+=d
            else:   loss+=abs(d)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Calpe Sendas" xmlns="http://www.topografix.com/GPX/1/1">',
        '  <metadata>',
        '    <name>Grand Tour de Calp</name>',
        f'    <desc>25.2 km  +{gain:.0f}m/-{loss:.0f}m  Ermita→Magenta→Red→Dit d\'Oltà→La Mola→Lima→Yellow→Magenta→Ermita</desc>',
        f'    <time>{datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}</time>',
        '  </metadata>',
        '  <trk><name>Grand Tour de Calp</name><trkseg>',
    ]
    for (lon,lat), ele in zip(tour_coords, elevs):
        ele_tag = f'<ele>{ele:.1f}</ele>' if ele is not None else ''
        lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}">{ele_tag}</trkpt>')
    lines += ['  </trkseg></trk>', '</gpx>']
    with open(out,'w') as f: f.write('\n'.join(lines))
    print(f"Wrote {out}  ({len(tour_coords)} pts, +{gain:.0f}m/-{loss:.0f}m)")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 – SV-05 bridge index derivation
#   (run once to find the magenta indices that bracket the shared section)
# ══════════════════════════════════════════════════════════════════════════════

def find_sv05_bridge_indices(routes_gps, tol_m=50.0):
    """
    The blue route (SV-05) in the PDF shares its middle section with the magenta
    route (SV-02).  This function finds the magenta indices [start, end] that
    correspond to the blue route's gap by measuring proximity between the
    southernmost blue fragment's endpoint and each magenta point.

    Result used as hard-coded constants: mag[109:243] in fix_sv05_blue().
    """
    mag  = routes_gps['SV-02']
    blue_segs = routes_gps['SV-05']

    def seg_min_lat(s): return min(p[1] for p in s)
    b0 = sorted(blue_segs, key=seg_min_lat)[0]   # southernmost blue fragment

    b0_end = b0[-1]   # endpoint of the southern fragment (lon, lat)

    # Find the magenta point closest to the blue fragment's end
    dists = [_dist_m(b0_end[1], b0_end[0], p[1], p[0]) for p in mag]
    bridge_start = min(range(len(dists)), key=dists.__getitem__)

    # Walk forward on magenta until we're close to the northern blue fragment start
    blue_segs_sorted = sorted(blue_segs, key=seg_min_lat)
    if len(blue_segs_sorted) > 1:
        b1_start = blue_segs_sorted[1][0]  # start of northern fragment
        bridge_end = min(
            range(bridge_start, len(mag)),
            key=lambda i: _dist_m(b1_start[1], b1_start[0], mag[i][1], mag[i][0])
        )
    else:
        # No northern fragment; bridge runs to Ermita (end of magenta)
        bridge_end = len(mag) - 1

    print(f"SV-05 bridge: mag[{bridge_start}:{bridge_end+1}]  "
          f"({bridge_end+1-bridge_start} pts)")
    return bridge_start, bridge_end + 1   # use as mag[start:end]


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 – Leaflet HTML map
# ══════════════════════════════════════════════════════════════════════════════

PEAK_MARKERS = [
    (38.6611757, 0.0147044, "el Dit d'Oltà", 590),
    (38.6478665, 0.0181269, "la Mola",        538),
    (38.668215,  0.063325,  "Ermita de la Cometa", 200),
]

ROUTE_LENGTHS_KM = {
    "SV-01": 3.6, "SV-02": 5.3, "SV-03": 3.1, "SV-04": 3.1,
    "SV-05": 1.7, "SV-06": 3.7, "SV-07": 9.1, "SV-08": 1.3,
}

def write_html_map(routes_gps, tour_coords, out='sendas_calp_map.html',
                   leaflet_js='leaflet.js', leaflet_css='leaflet.css'):
    """
    Generate a self-contained Leaflet HTML map with:
      - All 8 colour-coded sendas (clickable, highlight on click)
      - Grand Tour dashed overlay
      - Labelled peak markers for Dit d'Oltà, La Mola and Ermita
      - Sidebar with route name + length popup on click
      - OSM base tiles
    """
    # Build GeoJSON feature collection for routes
    features = []
    for rid, info in ROUTES.items():
        coords = routes_gps.get(rid, [])
        if not coords: continue
        dist_km = sum(
            _dist_m(coords[i][1],coords[i][0],coords[i+1][1],coords[i+1][0])
            for i in range(len(coords)-1)) / 1000
        features.append({
            "type": "Feature",
            "properties": {
                "id": rid, "color": info["color"], "name": info["name"],
                "length_km": round(dist_km, 1),
            },
            "geometry": {"type":"LineString",
                         "coordinates":[[lon,lat] for lon,lat in coords]},
        })
    geojson_str = json.dumps({"type":"FeatureCollection","features":features},
                              separators=(',',':'))

    tour_str = json.dumps([[lon,lat] for lon,lat in tour_coords], separators=(',',':'))

    # Read local Leaflet assets if available, else use CDN
    if os.path.exists(leaflet_js):
        with open(leaflet_js) as f: ljs = f.read()
        js_tag = f'<script>{ljs}</script>'
    else:
        js_tag = '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'

    if os.path.exists(leaflet_css):
        with open(leaflet_css) as f: lcss = f.read()
        css_tag = f'<style>{lcss}</style>'
    else:
        css_tag = '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'

    peak_js = '\n'.join(
        f"addPeakMarker({lat}, {lon}, {json.dumps(label)}, {ele});"
        for lat, lon, label, ele in PEAK_MARKERS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Sendas Tradicionales de Calp</title>
{css_tag}
<style>
  html,body,#map{{height:100%;margin:0;padding:0}}
  #info{{position:absolute;top:10px;right:10px;z-index:1000;
         background:rgba(255,255,255,0.92);padding:10px 14px;
         border-radius:6px;font-family:sans-serif;font-size:13px;
         max-width:220px;box-shadow:0 2px 8px rgba(0,0,0,.25)}}
  #info h3{{margin:0 0 6px;font-size:15px}}
  .peak-label{{background:rgba(255,248,225,0.92);border:1.5px solid #8B4513;
               border-radius:4px;font-size:12px;font-weight:bold;
               padding:2px 5px;white-space:nowrap}}
</style>
</head>
<body>
<div id="map"></div>
<div id="info"><h3>Sendas de Calp</h3><p id="detail">Click a route for details.</p></div>
{js_tag}
<script>
const geojson = {geojson_str};
const tourCoords = {tour_str};
const routeLengths = {json.dumps(ROUTE_LENGTHS_KM)};

const map = L.map('map');

// Grand Tour dashed overlay
const tourLine = L.polyline(
  tourCoords.map(c => [c[1], c[0]]),
  {{color:'#1a1a1a', weight:4, opacity:0.75, dashArray:'8,6'}}
).addTo(map);
tourLine.bindPopup('<b>Grand Tour de Calp</b><br>25.2 km / +1255 m<br>'
  + 'Ermita → Magenta → Red → Dit d\\'Oltà → La Mola'
  + ' → Lima → Yellow → Magenta → Ermita');

// Peak summit markers
function addPeakMarker(lat, lon, label, elevation) {{
  const m = L.circleMarker([lat, lon], {{
    radius:8, color:'#5c2d00', fillColor:'#b35c00', fillOpacity:0.9, weight:2
  }}).addTo(map);
  m.bindTooltip('<b>' + label + '</b><br>' + elevation + ' m',
    {{permanent:true, direction:'top', offset:[0,-10], className:'peak-label'}});
  m.bindPopup('<b>' + label + '</b><br>Elevation: ' + elevation + ' m');
}}
{peak_js}

// OSM tiles
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom:19
}}).addTo(map);

const layers = {{}};
let active = null;

function style(feature, sel) {{
  return {{color:feature.properties.color, weight:sel?8:4,
           opacity:sel?1.0:0.85, lineCap:'round', lineJoin:'round'}};
}}

function highlight(layer) {{
  if (active && active !== layer) active.setStyle(style(active.feature, false));
  layer.setStyle(style(layer.feature, true));
  layer.bringToFront();
  active = layer;
}}

L.geoJSON(geojson, {{
  style: f => style(f, false),
  onEachFeature: (feature, layer) => {{
    const p = feature.properties;
    layers[p.id] = layer;
    layer.on('click', () => {{
      highlight(layer);
      const km = (p.length_km || '?');
      document.getElementById('detail').innerHTML =
        '<b>' + p.name + '</b><br>' + p.id + ' &mdash; ' + km + ' km';
      layer.bindPopup('<b>' + p.name + '</b><br>' + km + ' km').openPopup();
    }});
  }}
}}).addTo(map);

// Fit view to all routes
const allPts = geojson.features.flatMap(f => f.geometry.coordinates.map(c => [c[1],c[0]]));
map.fitBounds(L.latLngBounds(allPts).pad(0.05));
</script>
</body>
</html>"""

    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Wrote {out}  ({len(html)//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Step 1: PDF → SVG (run once externally) ──────────────────────────────
    if not os.path.exists(SVG):
        print(f"Converting {PDF} → {SVG} ...")
        subprocess.run(['mutool','convert','-F','svg','-o',SVG, PDF,'1'], check=True)

    # ── Step 2: Extract raw SVG paths by colour ───────────────────────────────
    print("Parsing SVG paths ...")
    svg_segments = extract_svg_paths(SVG)
    for rid, segs in svg_segments.items():
        total_pts = sum(len(s) for s in segs)
        print(f"  {rid}: {len(segs)} segments, {total_pts} pts")

    # ── Step 3: Calibration ───────────────────────────────────────────────────
    if os.path.exists(CALIB_PKL):
        with open(CALIB_PKL,'rb') as f: calib = pickle.load(f)
        print(f"Loaded calibration from {CALIB_PKL}")
    else:
        print("Running calibration (may take a few minutes) ...")
        if not os.path.exists(OSM_PATHS): download_osm_paths()
        calib = run_calibration(svg_segments)

    # ── Step 4: Convert all segments to GPS ───────────────────────────────────
    routes_gps = {}
    for rid, segs in svg_segments.items():
        routes_gps[rid] = segments_to_coords(segs, calib)

    # ── Step 5: Per-route fixes ───────────────────────────────────────────────
    # SV-05: topology reconstruction
    routes_gps['SV-05'] = fix_sv05_blue(routes_gps)

    # SV-01: OSM-snap to actual roads (needs broader OSM data)
    osm_sv01_path = '/tmp/sv01_osm_full.json'   # all highway types for routing
    if not os.path.exists(osm_sv01_path):
        print("Downloading OSM data for SV-01 snap ...")
        download_osm_all_highways((38.640, 0.010, 38.680, 0.075), osm_sv01_path)
    with open(osm_sv01_path) as f:
        osm_sv01 = json.load(f)
    routes_gps['SV-01'] = osm_snap_sv01(routes_gps, osm_sv01_path)
    print(f"SV-01 snapped: {len(routes_gps['SV-01'])} pts")

    # ── Step 6: Export all sendas ─────────────────────────────────────────────
    write_geojson(routes_gps)
    write_gpx(routes_gps)

    # ── Step 7: Grand Tour ────────────────────────────────────────────────────
    tour_osm_path = '/tmp/olta_dit_osm.json'
    if not os.path.exists(tour_osm_path):
        print("Downloading OSM data for Grand Tour routing ...")
        download_osm_all_highways((38.640, 0.010, 38.680, 0.040), tour_osm_path)
    with open(tour_osm_path) as f:
        tour_osm = json.load(f)

    print("Building Grand Tour ...")
    tour = build_grand_tour(routes_gps, tour_osm)
    print(f"Tour: {len(tour)} waypoints")

    total_m = sum(
        _dist_m(tour[i][1],tour[i][0],tour[i+1][1],tour[i+1][0])
        for i in range(len(tour)-1))
    print(f"Tour length: {total_m/1000:.1f} km")

    write_tour_gpx(tour, out=TOUR_GPX, with_elevation=True)

    # ── Step 8: HTML map ──────────────────────────────────────────────────────
    print("Building HTML map ...")
    write_html_map(routes_gps, tour, out=MAP_HTML,
                   leaflet_js=LEAFLET_JS, leaflet_css=LEAFLET_CSS)

    print("\nAll done.")
    print(f"  {GEOJSON:<45} – 8 sendas as GeoJSON")
    print(f"  {GPX_OUT:<45} – 8 sendas as GPX")
    print(f"  {TOUR_GPX:<45} – Grand Tour with SRTM elevation")
    print(f"  {MAP_HTML:<45} – interactive Leaflet map")
    print()
    print("Serve the map locally with:")
    print("  python3 -m http.server 8765")
    print("  open http://localhost:8765/map/sendas_calp_map.html")
