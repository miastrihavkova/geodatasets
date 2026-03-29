# Geodatasets

A modular Python package for converting the **Sentinel-2 Cloud Mask Catalogue** into machine learning dataset formats, optimised for fast loading during model training.

Source dataset: https://zenodo.org/records/4172871

---

## What this package does

The Sentinel-2 Cloud Mask Catalogue consists of 513 subscenes, each 1022×1022 pixels with 13 spectral bands and a pixel-wise cloud segmentation mask (CLEAR, CLOUD, CLOUD_SHADOW). The raw data is stored as `.npy` files.

This package allows to:

- Download the raw dataset from Zenodo
- Tile each 1022×1022 subscene into 512×512 tiles (padded to 1024×1024 before tiling, producing a 2×2 grid of 4 tiles per subscene)
- Convert float32 TOA reflectance values to uint16
- Convert one-hot boolean masks to uint8 class ID masks
- Save tiles in either GeoTIFF or HDF5 format
- Generate tile level and dataset level metadata
- Assign official splits from the `classification_tags.csv`

---

## Architecture

```
src/geodatasets/
├── scripts/
│   ├── download.py
│   └── process_s2_catalogue.py
├── preprocessing/
│   └── tiling.py
├── writers/
│   ├── base.py
│   ├── geotiff.py
│   └── hdf5.py
├── datasets/
│   ├── base.py
│   ├── geotiff.py
│   └── hdf5.py
├── metadata/
│   ├── base.py
│   └── s2.py
├── datamodules/
│   └── s2.py
└── schemas/
    ├── bands.py
    ├── tiles.py
    └── datamodule.py
```

New dataset sources can be added by writing a new processing script and a new metadata writer subclass, the writers, tiling utilities, and `DataModule` are reusable without modification.

---

## Installation

This package uses [Poetry](https://python-poetry.org) for dependency management.

**Install Poetry** if you don't have it:

```bash
pip install poetry
```

**Clone the repository and install all dependencies:**

```bash
git clone https://github.com/miastrihavkova/geodatasets.git
cd geodatasets
poetry install
```

---

## Dataset download

```bash
poetry run geodatasets-download --output-dir data/raw
```

Edit `SKIP_FILES` in `scripts/download.py` to skip any files you have already present.

---

## Processing the raw data

Convert the raw `.npy` subscenes into tiles ready for training. Run once per format.

```bash
# GeoTIFF — one file per tile
poetry run geodatasets-process-s2 \
  --input     data/raw \
  --output    data/processed/geotiff \
  --bands-csv data/bands.csv \
  --format    geotiff \
  --tile-size 512

# HDF5 — all tiles in two shared files (images.h5 / masks.h5)
poetry run geodatasets-process-s2 \
  --input     data/raw \
  --output    data/processed/hdf5 \
  --bands-csv data/bands.csv \
  --format    hdf5 \
  --tile-size 512
```

Output structure:

```
data/processed/{format}/
├── images/
├── masks/
├── tiles.csv
└── dataset.json 
```

---

## Output formats

### GeoTIFF

Each tile is saved as an individual `.tif` file with LZW compression and `interleave=band` layout. rasterio's `src.read([4, 3, 2])` loads only the requested bands from disk and the other bands are never decompressed. This is the recommended format for training on a local machine.

Ref: https://rasterio.readthedocs.io/en/stable/

### HDF5

All tiles are stored in two shared files (`images.h5`, `masks.h5`). Each tile is a named dataset keyed by its product id and grid position. Bands are chunked as `(1, 256, 256)`, meaning one chunk per band, so loading 3 bands decompresses exactly 3 chunks rather than all 13. HDF5 is the better choice for cloud storage or when minimising the number of files matters.

Ref: https://docs.h5py.org/en/stable/

---

## Key library choices

**Pydantic**: all data (tile records, band info, `DataModule` config, tiling parameters) are Pydantic models. This means invalid configs and malformed data are caught with a clear error message before any processing starts, rather than failing silently or producing corrupt output.

**PyTorch Lightning**: the `S2DataModule` wraps the datasets in a `LightningDataModule`, which is the standard interface for training pipelines. It handles train/val/test split setup, `DataLoader` configuration, and integrates directly with Lightning `Trainer`.

**rasterio**: the standard Python interface for geospatial raster data. Used both for writing GeoTIFFs with correct spatial metadata and for efficient reading during training.

**h5py**: the standard HDF5 interface for Python. Used with SWMR (Single Writer Multiple Reader) mode so multiple `DataLoader` worker processes can read from the same file simultaneously without conflicts.

**Abstract base classes**: `BaseWriter`, `BaseDataset`, and `BaseMetadataWriter` define contracts that concrete implementations must satisfy. Adding a new format (like Zarr) means writing one new class that implements the ABC, so no changes to the processing script or `DataModule`.

---

## Metadata

- `tiles.csv` as one row per tile, containing:

| Field | Description |
|---|---|
| `tile_id` | UUID unique across all tiles |
| `product_id` | Sentinel-2 product ID of the source subscene |
| `image_filename` | filename of the saved image tile |
| `mask_filename` | filename of the saved mask tile |
| `subscene_row/col` | top-left pixel coordinates in the original subscene |
| `tile_grid_row/col` | position in the 2×2 tile grid |
| `cloud_percent` | percentage of CLOUD pixels in this tile |
| `has_padding` | whether this tile overlaps the zero-padded region |
| `split` | train / val / test |

- `dataset.json` containing band information (name, wavelength, bandwidth, GSD) and class ID mapping.

---

## Splits

Splits are assigned from the catalogue's own dataset column as `MAIN` → train (453 subscenes), `VALIDATION` → val (50 subscenes), `CALIBRATION` → test (10 subscenes), not randomly, to preserve the intended evaluation protocol.

---

## Notebooks

### EDA

For initial exploration if the Sentinel-2 Catalogue, you can see __notebooks/eda.ipynb__.

### Format comparison

For comparison of hdf5 and GeoTIFF formats, such as disk usage and loading speed, run __notebooks/compare_formats.ipynb__

---

## Running tests

There are also minimal tests implemented, mainly to cover tiling logic and schema validation.

```bash
poetry run pytest
```

---

## Code quality

You can run these commands to ensure standard quality of code.

```bash
# Lint
poetry run ruff check src/ tests/

# Format
poetry run ruff format src/ tests/

# Type check
poetry run mypy src/
```