#!/usr/bin/env bash
# Rebuild the whole model from the public sources. Run from the repository root.
# Python 3.11+ with: pip install -r requirements.txt   (and Node.js for gltfpack in the last step)
set -euo pipefail
cd "$(dirname "$0")"
P="python3"
step(){ echo "== $1"; $P "src/$1.py"; }
# 1. sources
step fetch_sources        # OSM, USGS 3DEP terrain, NAIP, DataSF footprints/parcels/trees/Muni routes
step fetch_sf             # DataSF curbs, right of way, signals, signs, meters, racks, cans, stops, ramps, streets
step ept_fetch            # USGS 3DEP LiDAR (CA_SanFrancisco_1_B23, 2023) for the square, ~5.6 M points
# 2. ground, curbs, plaza outline
step rasters              # 25 cm DTM / DSM / intensity grids
step roof_points          # vegetation-free roof surface
step curbfit              # real curb lines on Castro and 17th St measured from LiDAR curb steps
step plaza_shape          # former Chevron triangle -> plaza outline 5 m from the roadway, square east corners
step build_terrain
step build_texture        # ground texture: markings from LiDAR intensity, sidewalks, plaza
step make_aerial
# 3. buildings and street objects
step build_buildings      # footprints + LiDAR roofs, bay overhangs, cornices, facade classes
step objects; step classify_objs; step pointfeat
step build_veg            # tree segmentation + crowns
step build_poles_wires
step detect_cars; step build_extras   # signals, tree grates, hydrant, mailbox (vehicles are not exported)
step build_details        # trees, poles, signs, rails, wires, furniture, flags
step build_structures     # awnings and bays seen in the LiDAR
# 4. the proposal: plaza, fountain, statues, mural, brick rim
step build_statues
step stone_tex
step build_plaza
step build_mural
step build_rim
# 5. assemble and publish
step labels
step assemble             # -> web/castro.glb
step build_viewer         # -> docs/ (GitHub Pages) and model/
