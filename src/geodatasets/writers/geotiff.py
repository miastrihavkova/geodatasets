"""Writer for GeoTIFF format using rasterio."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from .base import BaseWriter


class GeoTIFFWriter(BaseWriter):
    """Writes one GeoTIFF file per tile with LZW compression."""

    def write_tile(self, tile_id: str, image: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
        """Writes single tile.

        Args:
            tile_id: Unique identifier for the tile, used as filename (without extension).
            image: Image data to write.
            mask: Mask data to write.

        Returns:
            tuple[str, str]: Paths to the written image and mask files.
        """
        H, W, C = image.shape
        transform = from_bounds(0, 0, W, H, W, H)
        crs = CRS.from_epsg(4326)

        img_path = self.images_dir / f"{tile_id}.tif"
        mask_path = self.masks_dir / f"{tile_id}.tif"

        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=C,
            dtype="uint16",
            crs=crs,
            transform=transform,
            compress="lzw",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            interleave="band",
        ) as dst:
            dst.write(image.transpose(2, 0, 1))

        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(mask[np.newaxis, ...])

        return f"{tile_id}.tif", f"{tile_id}.tif"

    def finalise(self) -> None:
        """No finalisation needed for GeoTIFFWriter."""
