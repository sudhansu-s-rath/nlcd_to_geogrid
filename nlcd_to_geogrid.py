#!/usr/bin/env python3
"""
NLCD to Geogrid: Create WPS_GEOG-compatible LANDUSEF tiles by overlaying NLCD urban data on MODIS landuse.

FINAL FIXED VERSION - Corrected endianness issue for urban fraction tiles.
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
    nlcd_proj = nlcd.rio.reproject("EPSG:4326", resampling='nearest')
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
                arr = np.round(arr).astype(np.int16)  # ensure categorical integrity
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
category_min=1
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
description="MODIS landuse with NLCD urban overlay: 31/32/33 urban; 1-21 MODIS base"
mminlu="MODIFIED_IGBP_MODIS_NOAH"
"""
    (out_dir / "index").write_text(idx)

def write_index_urbfrac(out_dir: Path, wordsize: int = 1) -> None:
    """
    Write index file for urban fraction tiles.
    Uses uint8 (0-100) with scale_factor=0.01 for reliable WPS/geogrid compatibility.
    """
    idx = f"""type=continuous
projection=regular_ll
dx={DX}
dy={DY}
known_x=1.0
known_y=1.0
known_lat={KNOWN_LAT}
known_lon={KNOWN_LON}
wordsize={wordsize}
scale_factor=0.01
tile_x={TILE_X}
tile_y={TILE_Y}
tile_z=1
units="fraction"
description="Urban fraction (0-1) from NLCD urban mask area-mean on MODIS 30s grid"
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
        year = "input"
    else:
        raise ValueError("Provide either --raw_nlcd or --nlcd_input")

    # Create urban mask (binary: 0.0 or 1.0)
    urbmask_path = out_dir / f"intermediate_04_urbmask_{year}.tif"
    print(f"[INFO] Creating urban mask from {nlcd_path}")
    with rasterio.open(nlcd_path) as src:
        profile = src.profile.copy()
        # CRITICAL: Use float32 with no nodata value
        profile.update(dtype=rasterio.float32, count=1, nodata=None)
        with rasterio.open(urbmask_path, "w", **profile) as dst:
            for ji, window in src.block_windows(1):
                arr = src.read(1, window=window)
                # Create binary mask: 1.0 for urban (31,32,33), 0.0 for everything else
                mask = ((arr == 31) | (arr == 32) | (arr == 33)).astype(np.float32)
                dst.write(mask, 1, window=window)
    
    # Verify the urban mask
    print(f"[INFO] Verifying urban mask: {urbmask_path}")
    with rasterio.open(urbmask_path) as check:
        sample = check.read(1, window=rasterio.windows.Window(0, 0, min(1000, check.width), min(1000, check.height)))
        unique_vals = np.unique(sample)
        print(f"[DEBUG] Urbmask unique values: {unique_vals}")
        print(f"[DEBUG] Urbmask nodata: {check.nodata}")
        print(f"[DEBUG] Urbmask dtype: {check.dtypes[0]}")
        print(f"[DEBUG] Urbmask resolution: {check.res}")
        if check.nodata is not None:
            print(f"[WARNING] Urbmask has nodata value {check.nodata} - this may cause issues!")
        if not np.all(np.isin(unique_vals, [0.0, 1.0])):
            print(f"[WARNING] Urbmask contains values other than 0.0 and 1.0: {unique_vals}")
    
    # ========================================================================
    # OPTION 1: Enhance urbmask with MODIS category 13 (composite urban mask)
    # This adds MODIS-detected urban (cat 13) where NLCD doesn't provide coverage
    # Maps MODIS 13 -> treated as urban in urbfrac computation
    # TOGGLE: Comment out the entire block below to disable this enhancement
    # ========================================================================
    print(f"[INFO] Enhancing urban mask with MODIS category 13 (where NLCD has no urban)...")
    with rasterio.open(nlcd_path) as modis_src:
        with rasterio.open(urbmask_path, 'r+') as urbmask_dst:
            modis_13_count = 0
            for ji, window in modis_src.block_windows(1):
                # Read current urbmask (NLCD-derived)
                urbmask_data = urbmask_dst.read(1, window=window).astype(np.float32)
                
                # Read MODIS/NLCD-mapped layer
                modis_data = modis_src.read(1, window=window).astype(np.int16)
                
                # Find MODIS cat 13 pixels that don't already have NLCD urban
                modis_13_mask = (modis_data == 13) & (urbmask_data == 0.0)
                
                # Set these to 1.0 (urban) in the urbmask
                urbmask_data[modis_13_mask] = 1.0
                modis_13_count += np.count_nonzero(modis_13_mask)
                
                # Write back
                urbmask_dst.write(urbmask_data.astype(np.float32), 1, window=window)
            
            print(f"[INFO] Added {modis_13_count} MODIS cat 13 pixels to urbmask (treated as urban in urbfrac)")
    # ========================================================================"

    # Extract year from nlcd_path
    match = re.search(r'(\d{4})', nlcd_path.name)
    if match:
        year = match.group(1)
    else:
        year = "unknown"

    # Set urbfrac output directory (GEOGRID.TBL expects urbfrac_nlcdYYYY)
    urbfrac_out_dir = out_dir / f"urbfrac_nlcd{year}"
    urbfrac_out_dir.mkdir(parents=True, exist_ok=True)
    write_index_urbfrac(urbfrac_out_dir, wordsize=1)  # wordsize=1 for uint8 with scale_factor

    # Set final output directory with year
    final_out_dir = out_dir / f"merged_nlcd_modis_{year}"
    final_out_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for NLCD tiles with year
    temp_nlcd_dir = out_dir / f"nlcd_tiles_{year}"
    temp_nlcd_dir.mkdir(exist_ok=True)

    # Write index
    write_index(final_out_dir, wordsize=1)  # uint8 for merged

    # Read NLCD source
    with rasterio.open(nlcd_path) as src:
        if src.crs is None or src.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            raise RuntimeError(f"Input must be EPSG:4326 (got {src.crs})")

        row_min, row_max, col_min, col_max = rowcol_range_for_bounds(src.bounds)

        print(f"[INFO] Input bounds: {src.bounds}")
        print(f"[INFO] Tiling rows {row_min}..{row_max}, cols {col_min}..{col_max} (MODIS-style 10° tiles)")
        print(f"[INFO] Output folder: {final_out_dir}")

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

                # Save NLCD tile to temp directory
                nlcd_tile_path = temp_nlcd_dir / tile_name
                dst_to_write = np.flipud(dst).astype('>i2')  # big-endian int16
                dst_to_write.tofile(nlcd_tile_path)

                tiles_written += 1

        print(f"[DONE] NLCD tiles written: {tiles_written}")
        print(f"[DONE] Tiles skipped (no urban): {tiles_skipped}")

    # Tile urban fraction
    print(f"[INFO] Creating urban fraction tiles from {urbmask_path}")
    with rasterio.open(urbmask_path) as srcu:
        print(f"[DEBUG] Source for urban fraction:")
        print(f"  Resolution: {srcu.res}")
        print(f"  Shape: {srcu.shape}")
        print(f"  NoData: {srcu.nodata}")
        print(f"  Dtype: {srcu.dtypes[0]}")
        
        frac_tiles_written = 0
        
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                west, south, east, north = tile_edges_from_rowcol(row, col)
                dst_transform = Affine(DX, 0.0, west, 0.0, -DY, north)
                dst_frac = np.zeros((TILE_Y, TILE_X), dtype=np.float32)

                # Reproject with AVERAGE resampling to compute fractions
                reproject(
                    source=rasterio.band(srcu, 1),
                    destination=dst_frac,
                    src_transform=srcu.transform,
                    src_crs=srcu.crs,
                    src_nodata=None,  # No nodata in source
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=None,  # No nodata in destination
                    resampling=Resampling.average,  # Average 0.0 and 1.0 values
                )

                # Validate range and clip if necessary
                frac_min = dst_frac.min()
                frac_max = dst_frac.max()
                
                if frac_max > 1.0 or frac_min < 0.0:
                    print(f"[WARNING] Tile ({row},{col}): Invalid range [{frac_min:.6f}, {frac_max:.6f}] - clipping to [0, 1]")
                    dst_frac = np.clip(dst_frac, 0.0, 1.0)

                # Tile name
                x_start = (col - 1) * TILE_X + 1
                x_end = col * TILE_X
                y_start = (row - 1) * TILE_Y + 1
                y_end = row * TILE_Y
                tile_name = f"{x_start:05d}-{x_end:05d}.{y_start:05d}-{y_end:05d}"

                # CRITICAL: Convert fraction [0.0, 1.0] to uint8 [0, 100] for WPS compatibility
                # WPS will apply scale_factor=0.01 to convert back to [0.0, 1.0]
                frac_tile_path = urbfrac_out_dir / tile_name
                dst_frac_scaled = (np.flipud(dst_frac) * 100.0).clip(0, 100).astype(np.uint8)
                dst_frac_scaled.tofile(frac_tile_path)
                
                frac_tiles_written += 1
                
                # Debug output for first tile
                if frac_tiles_written == 1:
                    print(f"[DEBUG] First fraction tile ({row},{col}) statistics:")
                    print(f"  Range: [{dst_frac.min():.6f}, {dst_frac.max():.6f}]")
                    print(f"  Mean: {dst_frac.mean():.6f}")
                    print(f"  Non-zero pixels: {np.count_nonzero(dst_frac)} ({100*np.count_nonzero(dst_frac)/(TILE_X*TILE_Y):.2f}%)")
                    unique_vals = np.unique(dst_frac)
                    print(f"  Unique values (first 20): {unique_vals[:20]}")
                    print(f"  Total unique values: {len(unique_vals)}")
                    if len(unique_vals) <= 10:
                        print(f"  [WARNING] Only {len(unique_vals)} unique values - expected more for fractional data!")

        print(f"[DONE] Urban fraction tiles written: {frac_tiles_written} to {urbfrac_out_dir}")
    
    print("[INFO] Copying all MODIS tiles to output directory...")
    for modis_file in modis_dir.iterdir():
        if modis_file.is_file() and modis_file.name != "index":
            shutil.copy2(modis_file, final_out_dir / modis_file.name)
    print(f"[DONE] All MODIS tiles copied to {final_out_dir}")

    # Now, merge NLCD overlays
    print("[INFO] Merging NLCD urban overlays...")
    merges_done = 0
    for nlcd_tile in temp_nlcd_dir.iterdir():
        tile_name = nlcd_tile.name
        out_tile = final_out_dir / tile_name

        # Read NLCD tile (big-endian int16)
        nlcd_arr = np.fromfile(nlcd_tile, dtype='>i2').reshape(1200, 1200)

        # Read MODIS base from final_out_dir (already copied)
        m = read_modis_tile(out_tile)

        # Merge: overlay NLCD urban on MODIS
        merged = m.copy()
        urban_mask = (nlcd_arr == 31) | (nlcd_arr == 32) | (nlcd_arr == 33)
        merged[urban_mask] = nlcd_arr[urban_mask].astype(np.uint8)
        
        # ====================================================================
        # ENHANCED MERGE: Convert remaining MODIS cat 13 -> 32 (medium urban)
        # This applies ONLY to MODIS pixels with cat 13 that don't overlap
        # with NLCD urban coverage (where NLCD didn't provide urban classes).
        # Category 32 = Medium-density urban (NLCD-style)
        # TOGGLE: Comment out the block below to disable this conversion
        # ====================================================================
        remaining_cat13_mask = (m == 13) & ~urban_mask
        merged[remaining_cat13_mask] = 32
        if np.any(remaining_cat13_mask):
            cat13_pixels = np.count_nonzero(remaining_cat13_mask)
            # Uncomment line below for verbose per-tile logging:
            # print(f"[DEBUG] Tile {tile_name}: Converted {cat13_pixels} MODIS cat 13 -> 32")
        # ====================================================================

        # Write merged tile back
        merged.tofile(out_tile)
        merges_done += 1

    print(f"[DONE] Merges performed: {merges_done}")

    # Do not remove temp
    print(f"[INFO] NLCD tiles kept in {temp_nlcd_dir}")

    # Write index
    write_index(final_out_dir, wordsize=1)  # uint8 for merged
    print(f"[DONE] Index written: {final_out_dir / 'index'}")
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Urban fraction tiles: {urbfrac_out_dir}")
    print(f"Merged NLCD+MODIS tiles: {final_out_dir}")
    print(f"\nIMPORTANT: Index file settings (following NLCD 2011 format):")
    print(f"  - wordsize=1 (uint8)")
    print(f"  - scale_factor=0.01")
    print(f"  - Tiles store fractions as 0-100, WPS converts to 0.0-1.0")
    print(f"\nTo verify urban fraction quality, run:")
    print(f"  python diagnose_urban_fraction.py --urbfrac_dir {urbfrac_out_dir}")

if __name__ == "__main__":
    main()