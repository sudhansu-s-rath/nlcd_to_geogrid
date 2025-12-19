#!/usr/bin/env python3
"""
NLCD to Geogrid: Create WPS_GEOG-compatible LANDUSEF tiles by overlaying NLCD urban data on MODIS landuse.

This script:
1. Extracts urban classes from raw NLCD GeoTIFF.
2. Reprojects to EPSG:4326.
3. Maps NLCD urban to MODIS codes.
4. Tiles into WPS_GEOG format.
5. Merges each tile with corresponding MODIS base landuse tiles.
6. Outputs merged tiles in a WPS_GEOG dataset folder.

Usage: python nlcd_to_geogrid.py --raw_nlcd <path_to_raw_nlcd_tif> --modis_dir <path_to_modis_tiles> --out_dir <output_dataset_dir>
"""

import argparse
from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
import rioxarray as rxr
import shutil
import re

# ----------------------------
# MODIS-style WPS grid settings
# ----------------------------
DX = 0.00833333  # degrees (30 arc-seconds)
DY = 0.00833333  # degrees
TILE_X = 1200    # pixels
TILE_Y = 1200    # pixels
TILE_DEG = TILE_X * DX  # 10 degrees

# Origin from your MODIS index (gridpoint centers for (1,1))
KNOWN_LON = -179.99583
KNOWN_LAT = -89.99583

# Convert center-origin to edge-origin for convenience
GLOBAL_WEST_EDGE = KNOWN_LON - DX / 2.0
GLOBAL_SOUTH_EDGE = KNOWN_LAT - DY / 2.0

def extract_urban(nlcd_path: Path, urban_out: Path) -> None:
    """Extract urban classes (21-24) from NLCD GeoTIFF."""
    with rasterio.open(nlcd_path) as srcf:
        data = srcf.read(1)
        profile = srcf.profile
        urban_classes = [21, 22, 23, 24]
        mask = np.isin(data, urban_classes)
        urban = np.where(mask, data, 0).astype(data.dtype)
        profile.update({"compress": "lzw", "tiled": True})
        with rasterio.open(urban_out, "w", **profile) as dst:
            dst.write(urban, 1)
    print(f"Extracted urban to {urban_out}")

def reproject_to_4326(urban_path: Path, proj_out: Path) -> None:
    """Reproject urban GeoTIFF to EPSG:4326."""
    nlcd = rxr.open_rasterio(str(urban_path))
    nlcd_proj = nlcd.rio.reproject("EPSG:4326")
    nlcd_proj.rio.to_raster(str(proj_out))
    vals = np.unique(nlcd_proj.values)
    print(f"Reprojected to {proj_out}, unique values: {vals}")

def map_to_modis(proj_path: Path, modis_out: Path) -> None:
    """Map NLCD urban to MODIS codes (21,22->31; 23->32; 24->33)."""
    def map_nlcd_to_modis(arr):
        modis_arr = np.zeros_like(arr, dtype=np.int16)
        modis_arr[arr == 21] = 31
        modis_arr[arr == 22] = 31
        modis_arr[arr == 23] = 32
        modis_arr[arr == 24] = 33
        return modis_arr

    with rasterio.open(proj_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=np.int16, count=1)
        with rasterio.open(modis_out, 'w', **profile) as dst:
            for ji, window in src.block_windows(1):
                arr = src.read(1, window=window)
                modis_arr = map_nlcd_to_modis(arr)
                dst.write(modis_arr, 1, window=window)
                if ji == (0, 0):
                    print("Unique after map (first window):", np.unique(modis_arr))
    print(f"Mapped to {modis_out}")

def tile_edges_from_rowcol(row: int, col: int) -> tuple[float, float, float, float]:
    west = GLOBAL_WEST_EDGE + (col - 1) * TILE_DEG
    east = west + TILE_DEG
    south = GLOBAL_SOUTH_EDGE + (row - 1) * TILE_DEG
    north = south + TILE_DEG
    return west, south, east, north

def rowcol_range_for_bounds(bounds: rasterio.coords.BoundingBox) -> tuple[int, int, int, int]:
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    col_min = math.floor((left - GLOBAL_WEST_EDGE) / TILE_DEG) + 1
    col_max = math.floor((right - GLOBAL_WEST_EDGE) / TILE_DEG) + 1
    row_min = math.floor((bottom - GLOBAL_SOUTH_EDGE) / TILE_DEG) + 1
    row_max = math.floor((top - GLOBAL_SOUTH_EDGE) / TILE_DEG) + 1
    return row_min, row_max, col_min, col_max

def write_index(out_dir: Path, wordsize: int = 2) -> None:
    idx = f"""type=categorical
category_min=0
category_max=33
missing_value=0
projection=regular_ll
dx={DX}
dy={DY}
known_x=1.0
known_y=1.0
known_lat={KNOWN_LAT}
known_lon={KNOWN_LON}
wordsize={wordsize}
tile_x={TILE_X}
tile_y={TILE_Y}
tile_z=1
units="category"
description="MODIS landuse with NLCD urban overlay: 31/32/33 urban; 0-20/21-30 MODIS"
mminlu="MODIFIED_IGBP_MODIS_NOAH"
"""
    (out_dir / "index").write_text(idx)

def read_modis_tile(path: Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.uint8)
    if a.size != 1200*1200:
        raise RuntimeError(f"Bad MODIS tile size for {path}: {a.size}")
    return a.reshape(1200, 1200)

def main() -> None:
    ap = argparse.ArgumentParser(description="Create WPS_GEOG tiles with NLCD urban overlay on MODIS landuse")
    ap.add_argument("--raw_nlcd", help="Path to raw NLCD GeoTIFF (will process to urban, reproject, map)")
    ap.add_argument("--nlcd_input", help="Path to pre-processed NLCD urban GeoTIFF in EPSG:4326 (values 0,31,32,33)")
    ap.add_argument("--modis_dir", required=True, help="Directory with MODIS base landuse tiles")
    ap.add_argument("--out_dir", required=True, help="Output WPS_GEOG dataset folder")
    args = ap.parse_args()

    modis_dir = Path(args.modis_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_nlcd:
        raw_path = Path(args.raw_nlcd)
        # Extract year from filename, e.g., _2008_
        match = re.search(r'_(\d{4})_', raw_path.name)
        if match:
            year = match.group(1)
        else:
            year = "unknown"
        urban_path = out_dir / f"intermediate_01_urban_{year}.tif"
        proj_path = out_dir / f"intermediate_02_4326_{year}.tif"
        modis_path = out_dir / f"intermediate_03_modis_{year}.tif"
        extract_urban(raw_path, urban_path)
        reproject_to_4326(urban_path, proj_path)
        map_to_modis(proj_path, modis_path)
        nlcd_path = modis_path
    elif args.nlcd_input:
        nlcd_path = Path(args.nlcd_input)
    else:
        raise ValueError("Provide either --raw_nlcd or --nlcd_input")

    # Write index
    write_index(out_dir, wordsize=1)  # uint8 for merged

    # Copy all MODIS tiles to output directory
    print(f"[INFO] Copying all MODIS base tiles to {out_dir}")
    for modis_file in modis_dir.glob("*"):
        if modis_file.is_file():
            shutil.copy(modis_file, out_dir / modis_file.name)

    # Read NLCD source
    with rasterio.open(nlcd_path) as src:
        if src.crs is None or src.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            raise RuntimeError(f"Input must be EPSG:4326 (got {src.crs})")

        row_min, row_max, col_min, col_max = rowcol_range_for_bounds(src.bounds)

        print(f"[INFO] Input bounds: {src.bounds}")
        print(f"[INFO] Tiling rows {row_min}..{row_max}, cols {col_min}..{col_max} (MODIS-style 10° tiles)")
        print(f"[INFO] Output folder: {out_dir}")

        tiles_written = 0
        tiles_skipped = 0

        # Loop through intersecting global tiles
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                west, south, east, north = tile_edges_from_rowcol(row, col)

                # Build destination transform
                dst_transform = Affine(DX, 0.0, west, 0.0, -DY, north)
                dst = np.zeros((TILE_Y, TILE_X), dtype=np.int16)

                # Reproject NLCD into tile grid
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )

                # Check if tile has urban pixels
                has_urban = np.any((dst == 31) | (dst == 32) | (dst == 33))
                if not has_urban:
                    tiles_skipped += 1
                    continue

                # Tile name
                x_start = (col - 1) * TILE_X + 1
                x_end = col * TILE_X
                y_start = (row - 1) * TILE_Y + 1
                y_end = row * TILE_Y
                tile_name = f"{x_start:05d}-{x_end:05d}.{y_start:05d}-{y_end:05d}"
                modis_tile = modis_dir / tile_name
                out_tile = out_dir / tile_name

                if not modis_tile.exists():
                    print(f"[WARNING] No MODIS tile for {tile_name}, skipping")
                    continue

                # Read MODIS base
                m = read_modis_tile(modis_tile)

                # Merge: overlay NLCD urban on MODIS
                merged = m.copy().astype(np.uint8)
                urban_mask = (dst == 31) | (dst == 32) | (dst == 33)
                merged[urban_mask] = dst[urban_mask].astype(np.uint8)

                # Write merged tile
                merged.tofile(out_tile)
                tiles_written += 1

        print(f"[DONE] Tiles written: {tiles_written}")
        print(f"[DONE] Tiles skipped (no urban): {tiles_skipped}")
        print(f"[DONE] Index written: {out_dir / 'index'}")

    # Write index
    write_index(out_dir, wordsize=1)  # uint8 for merged

    # Read NLCD source
    with rasterio.open(nlcd_path) as src:
        if src.crs is None or src.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            raise RuntimeError(f"Input must be EPSG:4326 (got {src.crs})")

        row_min, row_max, col_min, col_max = rowcol_range_for_bounds(src.bounds)

        print(f"[INFO] Input bounds: {src.bounds}")
        print(f"[INFO] Tiling rows {row_min}..{row_max}, cols {col_min}..{col_max} (MODIS-style 10° tiles)")
        print(f"[INFO] Output folder: {out_dir}")

        tiles_written = 0
        tiles_skipped = 0

        # Loop through intersecting global tiles
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                west, south, east, north = tile_edges_from_rowcol(row, col)

                # Build destination transform
                dst_transform = Affine(DX, 0.0, west, 0.0, -DY, north)
                dst = np.zeros((TILE_Y, TILE_X), dtype=np.int16)

                # Reproject NLCD into tile grid
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )

                # Check if tile has urban pixels
                has_urban = np.any((dst == 31) | (dst == 32) | (dst == 33))
                if not has_urban:
                    tiles_skipped += 1
                    continue

                # Tile name
                x_start = (col - 1) * TILE_X + 1
                x_end = col * TILE_X
                y_start = (row - 1) * TILE_Y + 1
                y_end = row * TILE_Y
                tile_name = f"{x_start:05d}-{x_end:05d}.{y_start:05d}-{y_end:05d}"
                modis_tile = modis_dir / tile_name
                out_tile = out_dir / tile_name

                if not modis_tile.exists():
                    print(f"[WARNING] No MODIS tile for {tile_name}, skipping")
                    continue

                # Read MODIS base
                m = read_modis_tile(modis_tile)

                # Merge: overlay NLCD urban on MODIS
                merged = m.copy().astype(np.uint8)
                urban_mask = (dst == 31) | (dst == 32) | (dst == 33)
                merged[urban_mask] = dst[urban_mask].astype(np.uint8)

                # Write merged tile
                merged.tofile(out_tile)
                tiles_written += 1

        print(f"[DONE] Tiles merged: {tiles_written}")
        print(f"[DONE] Tiles skipped (no urban): {tiles_skipped}")
        print(f"[DONE] All MODIS tiles copied to {out_dir}")
        print(f"[DONE] Index written: {out_dir / 'index'}")

if __name__ == "__main__":
    main()