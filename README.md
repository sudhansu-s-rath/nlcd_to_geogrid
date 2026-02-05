# NLCD to WPS_GEOG Pipeline

This script creates WPS_GEOG-compatible LANDUSEF tiles by overlaying NLCD urban data on MODIS landuse data for WRF/WPS integration. It also generates urban fraction tiles for FRC_URB2D.

## Features

- **Complete Pipeline**: Processes raw NLCD GeoTIFF to WPS_GEOG tiles
- **Urban Extraction**: Extracts urban classes (21-24) from NLCD
- **Reprojection**: Converts to EPSG:4326
- **Mapping**: Maps NLCD urban to MODIS codes (31/32/33)
- **Tiling**: Creates 10° tiles in WPS_GEOG format
- **Merging**: Overlays urban on base MODIS landuse
- **Urban Fraction**: Computes and tiles urban fraction (0-1) using area-mean resampling
- **Full Dataset**: Copies all MODIS tiles, merges urban where present, outputs separate urban fraction dataset
- **WPS-Compatible Format**: Produces tiles in correct endianness and data types for WPS geogrid
- **Correct Value Ranges**: Ensures urban fraction values are in 0-1 range, LU categories in proper integer ranges
- **Separate Outputs**: Generates both LU category tiles (merged_nlcd_modis_<year>) and urban fraction tiles (urbfrac_<year>) independently

## Usage

```bash
python nlcd_to_geogrid.py --raw_nlcd <path_to_raw_nlcd_tif> --modis_dir <path_to_modis_tiles> --out_dir <output_dataset_dir>
```

Or with pre-processed NLCD:

```bash
python nlcd_to_geogrid.py --nlcd_input <path_to_processed_nlcd_tif> --modis_dir <path_to_modis_tiles> --out_dir <output_dataset_dir>
```

## Example (2014 NLCD)

- Raw NLCD: `/data/mgeorge7/sudhansu_WORK/geo_em/nlcd_yearly_tif/Annual_NLCD_LndCov_2014_CU_C1V1.tif`
- MODIS dir: `/data/mgeorge7/sudhansu_WORK/WPS_GEOG/modis_landuse_20class_30s_with_lakes`
- Output dir: `/data/mgeorge7/sudhansu_WORK/geo_em/nlcd_to_geogrid/test_output_2014`

Command:

```bash
python nlcd_to_geogrid.py --raw_nlcd /data/mgeorge7/sudhansu_WORK/geo_em/nlcd_yearly_tif/Annual_NLCD_LndCov_2014_CU_C1V1.tif --modis_dir /data/mgeorge7/sudhansu_WORK/WPS_GEOG/modis_landuse_20class_30s_with_lakes --out_dir /data/mgeorge7/sudhansu_WORK/geo_em/nlcd_to_geogrid/test_output_2014
```

This produces:
- `test_output_2014/merged_nlcd_modis_2014/` (LU categories)
- `test_output_2014/urbfrac_2014/` (urban fraction)

Copy these to WPS_GEOG and rename urbfrac_2014 to urbfrac_nlcd2014 for GEOGRID.TBL compatibility.

## Requirements

- Python 3.8+
- rasterio
- numpy
- rioxarray
- affine
- scipy
- shutil (standard library)

## Output

- **LU Category Dataset** (`merged_nlcd_modis_<year>`): Complete WPS_GEOG LANDUSEF dataset with all tiles and index file
  - Format: uint8 tiles with proper index (wordsize=1)
  - Values: 1-33 (MODIS + NLCD urban categories)
- **Urban Fraction Dataset** (`urbfrac_<year>`): Separate urban fraction tiles for FRC_URB2D
  - Format: float32 big-endian tiles with proper index (wordsize=4, signed=yes, endian=big)
  - Values: 0.0-1.0 (urban fraction)
- Intermediate files (optional, saved in out_dir for debugging/verification)

## WPS Integration

The script produces tiles in the exact format expected by WPS geogrid:

- **LU Categories**: uint8 format compatible with categorical data reading
- **Urban Fraction**: float32 big-endian format compatible with continuous data reading
- **Index Files**: Properly configured with correct wordsize, endianness, and metadata

After running the script, copy the appropriate dataset directories to your WPS_GEOG folder:
- `merged_nlcd_modis_<year>` for LANDUSEF in GEOGRID.TBL
- `urbfrac_<year>` for FRC_URB2D in GEOGRID.TBL (rename to `urbfrac_nlcd<year>` if needed)

## Author

Sudhansu S. Rath