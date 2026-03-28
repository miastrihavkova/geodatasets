# Geodatasets
This repository serves as package for creating datasets from __Sentinel-2 Cloud Mask Catalogue__, which is available for download from __zenodo__: https://zenodo.org/records/4172871.

For any machine learning project concerning large formats as images or even multispectral data, storing format is crutial for effective loading and tensor handling during the training.


## Package installation

```bash
poetry install
```

## Dataset download

```bash
poetry run geodatasets-download
```


## Provided formats

### HDF5
https://docs.h5py.org/en/stable/

### GeoTIFF
https://geotiff.io
https://rasterio.readthedocs.io/en/stable/