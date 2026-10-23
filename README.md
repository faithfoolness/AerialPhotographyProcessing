# Aerial Road Markings

Extracts road markings from aerial orthophotos (GeoTIFF) and turns them into
GIS-ready vector lines: solid and dashed markings as separate, measured
features in the coordinate system of the source image.

Built for road survey work, where markings have to be inventoried along
hundreds of kilometres of highway and digitising them by hand takes days.

![Web interface](docs/web-ui.jpg)

## What it does

- **Input:** an RGB orthophoto, either a GeoTIFF or a plain TIFF with a world file (`.tfw`) and optional `.prj`.
- **Output:** a GeoPackage with the layers `markings`, `segments` and `vehicles`, a zipped Shapefile and GeoJSON.
- **Attributes of every line:** type (`solid`, `dashed`, `dash`), length in metres, number of joined pieces, mean paint width.
- **Interfaces:** a command-line tool for batch processing and a web page where you upload an image and inspect the result on a satellite map.

## How it works

```mermaid
flowchart LR
    A[GeoTIFF / TIFF+TFW] --> B[Read image, CRS<br/>and ground resolution]
    B --> C[Detect vehicles<br/>YOLO11-OBB, tiled]
    B --> D[Paint mask<br/>white top-hat + colour filters]
    C -- exclude --> D
    D --> E[Connected components<br/>shape filter]
    E --> F[Skeleton → ordered<br/>centre lines]
    F --> G[Join pieces:<br/>solid across gaps,<br/>strokes into dashed lines]
    G --> H[GPKG · SHP · GeoJSON]
```

1. **Paint mask.** A morphological white top-hat keeps only stripes that are brighter than their surroundings and narrower than the widest marking. Large bright areas such as concrete, roofs and glare drop out. Pixels must also be bright, nearly colourless and not dominated by green, which removes sunlit foliage.
2. **Vehicles.** A white car is indistinguishable from paint by colour. A YOLO oriented-bounding-box model trained on aerial imagery (DOTA) finds cars and trucks, and their footprints are removed from the mask. The image is resampled to the model's training resolution and processed in overlapping tiles, so detection works on orthophotos of any size.
3. **Shape filter.** Each connected blob is measured in metres: length, mean width and elongation. A marking is long and thin; leaves, specks and car roofs are not.
4. **Centre lines.** Every blob is skeletonised and the skeleton is traced as a graph (longest path first, then long branches). This yields ordered polylines instead of a cloud of points.
5. **Joining.** Pieces are chained end to end when the gap is short, the directions match and the next piece lies on the continuation of the current one. Solid lines are restored across shadows and worn paint; individual strokes become one dashed line.

All thresholds are set in metres (`src/aerial_markings/config.py`) and converted with the image's ground resolution, so the same settings work for 3 cm and 15 cm imagery.

## Quick start

```bash
git clone https://github.com/faithfoolness/AerialPhotographyProcessing.git
cd AerialPhotographyProcessing
python -m venv .venv
.venv\Scripts\activate             # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e .[vehicles]         # without [vehicles] the YOLO step is skipped
```

Try it on a generated test image:

```bash
aerial-markings demo -o data/synthetic_road.tif
aerial-markings extract data/synthetic_road.tif -o results/
```

```
synthetic_road.tif: 6 lines in 5.3 s -> results
  dashed     2 lines      269.7 m
  solid      4 lines      586.3 m
  files: synthetic_road_markings.gpkg, synthetic_road_markings_shp.zip, synthetic_road_markings.geojson, stats.json
```

Web interface:

```bash
aerial-markings serve            # then open http://127.0.0.1:8000
```

Your own imagery:

```bash
aerial-markings extract photos/*.tif -o results/
aerial-markings extract tile.tif --crs EPSG:32637   # if the image carries no CRS
```

The YOLO weights (`yolo11n-obb.pt`, 5.5 MB) download automatically on first run.

## Project structure

```
src/aerial_markings/
  raster.py      reading GeoTIFF / TIFF+TFW, CRS resolution, ground resolution
  vehicles.py    tiled YOLO-OBB vehicle detection
  mask.py        paint mask (top-hat, colour and vegetation filters)
  vectorize.py   components, shape filter, skeleton tracing
  merge.py       chaining pieces into solid and dashed lines
  pipeline.py    the end-to-end run with timings and statistics
  export.py      GeoPackage, Shapefile, GeoJSON, web preview
  cli.py         command-line interface
  web/           FastAPI app and the map page (Leaflet)
  synthetic.py   synthetic test scene
tests/           pytest: pipeline, CRS handling, web API
```

```bash
pip install -e .[dev]
pytest
```

## Project history

The first version was a prototype that stored every white pixel in PostgreSQL
and grouped pixels with a breadth-first search in Python. It worked, but it
had three problems: about 13 minutes per image, zig-zag lines, and noise from
trees and white cars. Version 2 is a rewrite of the processing core:

| | v1 (prototype) | v2 |
|---|---|---|
| Pixel grouping | BFS over a database table, ~12 min | in-memory connected components, < 1 s |
| Paint detection | fixed threshold R, G, B > 200 | local contrast (top-hat) and colour filters |
| Noise from trees | many false lines | removed by shape filter |
| Vehicles | COCO model on the whole image shrunk to 640 px, found nothing | aerial OBB model on native-scale tiles |
| Line geometry | skeleton pixels joined row by row, zig-zags | skeleton traced as a graph |
| CRS | always labelled EPSG:4326 | read from the image or given explicitly |
| Dependencies | PostgreSQL server required | none besides Python packages |

## Limitations

- Designed for nadir RGB orthophotos with a resolution of about 2–15 cm per pixel.
- Yellow (temporary) markings are filtered out as coloured; enable them by raising `max_saturation`.
- Pedestrian crossings and arrows come out as short pieces of `dash` type; classifying them separately is a natural next step.
- The vehicle model is generic. Fine-tuning it on local imagery would improve recall for dark and partially occluded vehicles.

## License

MIT
