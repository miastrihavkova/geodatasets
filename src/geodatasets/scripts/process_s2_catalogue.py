"""Process the raw S2 Cloud Mask Catalogue into tiled GeoTIFF or HDF5 format, with metadata."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from geodatasets.metadata.s2 import S2CatalogueMetadataWriter
from geodatasets.preprocessing.tiling import (
    mask_onehot_to_classid,
    reflectance_to_uint16,
    tile_array,
)
from geodatasets.writers.geotiff import GeoTIFFWriter
from geodatasets.writers.hdf5 import HDF5Writer

log = logging.getLogger(__name__)

WRITER_MAP = {"geotiff": GeoTIFFWriter, "hdf5": HDF5Writer}
SPLIT_MAP = {"MAIN": "train", "VALIDATION": "val", "CALIBRATION": "test"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Process S2 Cloud Mask Catalogue")
    p.add_argument("--input", type=Path, required=True, help="Raw dataset root")
    p.add_argument("--output", type=Path, required=True, help="Output dataset root")
    p.add_argument("--bands-csv", type=Path, required=True, help="Path to bands.csv")
    p.add_argument("--format", type=str, default="geotiff", choices=["geotiff", "hdf5"])
    p.add_argument("--tile-size", type=int, default=512)
    return p.parse_args()


def assign_splits(tiles_csv: Path, tags_csv: Path) -> None:
    """Assign train/val/test splits to tiles based on the classification tags.

    Args:
        tiles_csv: Path to the CSV file describing the tiles and their metadata.
        tags_csv: Path to the CSV file containing the classification and split tags.

    Raises:
        ValueError: If there are unknown dataset values in the tags CSV.
    """
    tags = pd.read_csv(tags_csv)[["scene", "dataset", "shadows_marked"]]
    tags["split"] = tags["dataset"].str.upper().map(SPLIT_MAP)

    unknown = tags[tags["split"].isna()]["dataset"].unique().tolist()
    if unknown:
        raise ValueError(f"unknown dataset values in tags CSV: {unknown}")

    df = pd.read_csv(tiles_csv)
    df = df.merge(
        tags[["scene", "split", "shadows_marked"]],
        left_on="product_id",
        right_on="scene",
        how="left",
    )
    df = df.drop(columns=["scene"])

    n_missing = df["split"].isna().sum()
    if n_missing > 0:
        log.warning("%d tiles had no matching scene in tags CSV — defaulting to train", n_missing)
        df["split"] = df["split"].fillna("train")

    df.to_csv(tiles_csv, index=False)
    log.info(
        "splits assigned — train: %d  val: %d  test: %d tiles",
        df["split"].eq("train").sum(),
        df["split"].eq("val").sum(),
        df["split"].eq("test").sum(),
    )


def process(input_dir: Path, output_dir: Path, bands_csv: Path, tile_size: int, fmt: str) -> None:
    """Process the raw S2 Cloud Mask Catalogue into tiled GeoTIFF or HDF5 format, with metadata.

    Args:
        input_dir: The directory containing the raw S2 Cloud Mask Catalogue data.
        output_dir: The directory where the processed data will be saved.
        bands_csv: The path to the CSV file containing band information.
        tile_size: The size of each tile in pixels.
        fmt: The format to save the processed data in (either "geotiff" or "hdf5").
    """
    subscene_dir = input_dir / "subscenes"
    mask_dir = input_dir / "masks"
    tags_csv = input_dir / "classification_tags.csv"

    writer = WRITER_MAP[fmt](output_dir)
    metadata = S2CatalogueMetadataWriter(output_dir, bands_csv)
    metadata.save_dataset_level()
    log.info("dataset-level metadata saved to %s", output_dir)

    subscene_paths = sorted(subscene_dir.glob("*.npy"))
    log.info("found %d subscenes in %s", len(subscene_paths), subscene_dir)

    for sub_path in tqdm(subscene_paths, desc="Processing subscenes"):
        product_id = sub_path.stem
        mask_path = mask_dir / f"{product_id}.npy"

        if not mask_path.exists():
            log.warning("no mask for %s — skipping", product_id)
            continue

        subscene = np.load(sub_path)
        mask_raw = np.load(mask_path)
        image = reflectance_to_uint16(subscene)
        mask = mask_onehot_to_classid(mask_raw)

        img_tiles, descriptors = tile_array(image, tile_size, pad_value=0)
        mask_tiles, _ = tile_array(mask, tile_size, pad_value=0)

        for img_tile, mask_tile, desc in zip(img_tiles, mask_tiles, descriptors):
            stem = f"{product_id}_r{desc.grid_row}c{desc.grid_col}"
            img_filename, mask_filename = writer.write_tile(stem, img_tile, mask_tile)
            metadata.record_tile(
                descriptor=desc,
                mask_tile=mask_tile,
                product_id=product_id,
                image_filename=img_filename,
                mask_filename=mask_filename,
            )

    writer.finalise()
    metadata.save()
    assign_splits(output_dir / "tiles.csv", tags_csv)
    log.info("done — %d tiles written to %s", metadata.tile_count, output_dir)


def main() -> None:
    """Main entry point for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    args = parse_args()
    process(args.input, args.output, args.bands_csv, args.tile_size, args.format)


if __name__ == "__main__":
    main()
