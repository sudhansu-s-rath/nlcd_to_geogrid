# NLCD to WPS_GEOG Pipeline

This script creates WPS_GEOG-compatible LANDUSEF tiles by overlaying NLCD urban data on MODIS landuse data for WRF/WPS integration.

## Features

- **Complete Pipeline**: Processes raw NLCD GeoTIFF to WPS_GEOG tiles
- **Urban Extraction**: Extracts urban classes (21-24) from NLCD
- **Reprojection**: Converts to EPSG:4326
- **Mapping**: Maps NLCD urban to MODIS codes (31/32/33)
- **Tiling**: Creates 10° tiles in WPS_GEOG format
- **Merging**: Overlays urban on base MODIS landuse
- **Full Dataset**: Copies all MODIS tiles, merges urban where present

## Usage

```bash
python nlcd_to_geogrid.py --raw_nlcd <path_to_raw_nlcd_tif> --modis_dir <path_to_modis_tiles> --out_dir <output_dataset_dir>
```

Or with pre-processed NLCD:

```bash
python nlcd_to_geogrid.py --nlcd_input <path_to_processed_nlcd_tif> --modis_dir <path_to_modis_tiles> --out_dir <output_dataset_dir>
```

## Example (1985 NLCD)

- Raw NLCD: `/data/mgeorge7/sudhansu_WORK/geo_em/nlcd_yearly_tif/Annual_NLCD_LndCov_1985_CU_C1V1.tif`
- MODIS dir: `/data/mgeorge7/sudhansu_WORK/WPS_GEOG/modis_landuse_20class_30s_with_lakes`
- Output dir (includes intermediates): `/data/mgeorge7/sudhansu_WORK/geo_em/nlcd_to_geogrid/test_output`

Command:

```bash
python nlcd_to_geogrid.py --raw_nlcd /data/mgeorge7/sudhansu_WORK/geo_em/nlcd_yearly_tif/Annual_NLCD_LndCov_1985_CU_C1V1.tif --modis_dir /data/mgeorge7/sudhansu_WORK/WPS_GEOG/modis_landuse_20class_30s_with_lakes --out_dir /data/mgeorge7/sudhansu_WORK/geo_em/nlcd_to_geogrid/test_output
```

After the run, copy the needed WPS_GEOG tiles and `index` file from `test_output` into your WPS_GEOG directory (e.g., `/data/mgeorge7/sudhansu_WORK/WPS_GEOG/merged_nlcd_modis_1985`) and keep intermediates in `test_output` as needed.

## Requirements

- Python 3.8+
- rasterio
- numpy
- rioxarray
- affine
- shutil (standard library)

## Output

- Complete WPS_GEOG dataset with all tiles
- Index file for WPS configuration
- Intermediate files (optional, saved in out_dir)

## Author

Sudhansu S. Rath