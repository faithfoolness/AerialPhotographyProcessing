"""Command line interface.

    aerial-markings extract photo.tif -o results/
    aerial-markings extract photo.tif --crs EPSG:32637 --no-vehicles
    aerial-markings serve --port 8000
    aerial-markings demo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def _extract(args) -> None:
    from .config import Params
    from .export import write_outputs
    from .pipeline import run

    params = Params()
    params.vehicles.enabled = not args.no_vehicles
    for image in args.images:
        out = Path(args.output) / Path(image).stem if len(args.images) > 1 else Path(args.output)
        result = run(image, params, crs=args.crs)
        files = write_outputs(result, out, formats=args.formats.split(","), preview=False)
        s = result.stats
        print(f"{Path(image).name}: {s['lines']} lines in {s['total_s']} s -> {out}")
        for t, v in s["by_type"].items():
            print(f"  {t:7s} {v['count']:4d} lines  {v['length_m']:9.1f} m")
        print("  files: " + ", ".join(files.values()))


def _serve(args) -> None:
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(Path(args.workdir)), host=args.host, port=args.port)


def _demo(args) -> None:
    from .synthetic import write_scene

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    truth = write_scene(out)
    print(f"Synthetic orthophoto written to {out} (contains {json.dumps(truth)})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="aerial-markings",
                                 description="Extract road markings from aerial orthophotos.")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="process one or more GeoTIFFs")
    e.add_argument("images", nargs="+")
    e.add_argument("-o", "--output", default="results")
    e.add_argument("--crs", help="CRS of the image if it has none, e.g. EPSG:32637")
    e.add_argument("--formats", default="gpkg,shp,geojson")
    e.add_argument("--no-vehicles", action="store_true", help="skip vehicle detection")
    e.set_defaults(func=_extract)

    s = sub.add_parser("serve", help="run the web interface")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workdir", default="runs")
    s.set_defaults(func=_serve)

    d = sub.add_parser("demo", help="write a synthetic test orthophoto")
    d.add_argument("-o", "--output", default="data/synthetic_road.tif")
    d.set_defaults(func=_demo)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
