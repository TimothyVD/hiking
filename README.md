# Sendas Tradicionales de Calp

Interactive map and GPX tracks for the 8 traditional hiking routes (*sendas*) of Calpe (Comunitat Valenciana, Spain), extracted from the official municipality PDF and georeferenced against OpenStreetMap.

Includes a **Grand Tour de Calp** linking all major routes into a single 25.2 km loop with +1255 m elevation gain.

---

## Directory layout

```
.
├── build_calp_sendas.py          # full reproducible pipeline (see below)
├── data/
│   ├── plano_sendas_tradicionales_calp.pdf   # source document (Calpe 2022)
│   ├── osm_paths.json                        # OSM footways used for calibration
│   ├── sendas_tradicionales_calp.geojson     # 8 routes as GeoJSON LineStrings
│   ├── sendas_tradicionales_calp.gpx         # 8 routes as GPX tracks
│   ├── tour_grand_calp.geojson               # Grand Tour as GeoJSON
│   └── tour_grand_calp.gpx                   # Grand Tour as GPX (with SRTM elevation)
└── map/
    ├── sendas_calp_map.html                  # interactive Leaflet map
    ├── leaflet.js                            # Leaflet 1.9 (offline-capable)
    └── leaflet.css
```

---

## The routes

| ID | Name | Colour | Length |
|----|------|--------|--------|
| SV-01 | Senda Roja – La Mola | red | 3.6 km |
| SV-02 | Senda Magenta – Llebeig | magenta | 5.3 km |
| SV-03 | Senda Morada – Morro de Toix | purple | 3.1 km |
| SV-04 | Senda Naranja – Sella d'Olta | orange | 3.1 km |
| SV-05 | Senda Azul – Les Salines | blue | 1.7 km |
| SV-06 | Senda Verde – Cap Blanc | green | 3.7 km |
| SV-07 | Senda Lima – Canaret | lime | 9.1 km |
| SV-08 | Senda Amarilla – El Cantal | yellow | 1.3 km |

---

## Grand Tour de Calp

**25.2 km · +1255 m / −1255 m · circular**

Start/end: Ermita de la Cometa (200 m)

```
Ermita de la Cometa
  → Senda Magenta (reversed, south)
  → Senda Roja (full, south-west)
  → OSM path → el Dit d'Oltà  (590 m)  ★
  → OSM path → la Mola        (538 m)  ★
  → OSM path → Senda Lima join
  → Senda Lima (partial, eastward)
  → Senda Amarilla (full)
  → Senda Magenta (tail, back to Ermita)
```

Both el Dit d'Oltà and la Mola are dead-end spurs on the trail network; the route visits each summit and retraces to continue.

---

## View the map

Serve locally (required for OSM tiles):

```bash
python3 -m http.server 8765
# open http://localhost:8765/map/sendas_calp_map.html
```

Click any route to highlight it and see its name and length.

---

## Reproduce from scratch

### Requirements

```bash
# Debian/Ubuntu
sudo apt install mupdf-tools

pip install numpy scipy networkx
```

### Steps

```bash
# 1. Convert PDF page 1 to SVG (one-time, produces ~13 MB file)
mutool convert -F svg -o sendas_page1.svg \
    data/plano_sendas_tradicionales_calp.pdf 1

# 2. Run the full pipeline
python3 build_calp_sendas.py
```

The script will:
1. Parse the 3767 SVG paths and identify routes by stroke colour
2. Apply the pre-computed 5-parameter georeferencing calibration  
   *(K_LON = 6.685 × 10⁻⁵, K_LAT = 5.248 × 10⁻⁵, rotation = −1.806°)*
3. Fix SV-01 by snapping to OSM road network (NetworkX shortest-path)
4. Fix SV-05 topology (shared section reconstructed from SV-02)
5. Download OSM data and build the Grand Tour via OSM routing
6. Fetch SRTM elevation for tour GPX from api.opentopodata.org
7. Write all output files

To force re-calibration, delete `/tmp/calp_calib2.pkl`.

---

## How georeferencing works

The PDF was produced with Adobe Illustrator CS5. The SVG coordinate system uses a `matrix(1,0,0,-1,tx,ty)` y-axis flip applied through nested group transforms.

The calibration transform is:

```
dx = sx - pivot_x
dy = sy - pivot_y
rx = cos(θ)·dx − sin(θ)·dy + pivot_x
ry = sin(θ)·dx + cos(θ)·dy + pivot_y

lon = LON₀ + rx · K_LON
lat = LAT₀ − ry · K_LAT
```

Parameters were optimised with `scipy.optimize.differential_evolution` (5 free variables: Δlon, Δlat, scale_lon, scale_lat, θ) by minimising the mean distance of the Magenta and Red route traces to an OSM distance-transform raster.
