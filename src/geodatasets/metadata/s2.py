from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from geodatasets.schemas import BandInfoSchema, DatasetMetaSchema, TileDescriptor, TileRecordSchema

from .base import BaseMetadataWriter

CLASS_MAP = {0: "CLEAR", 1: "CLOUD", 2: "CLOUD_SHADOW"}


class S2CatalogueMetadataWriter(BaseMetadataWriter):
    TILES_FILENAME = "tiles.csv"
    DATASET_FILENAME = "dataset.json"

    def __init__(self, output_dir: Path, bands_csv: Path) -> None:
        super().__init__(output_dir)
        self._bands_csv = Path(bands_csv)
        self._records: list[TileRecordSchema] = []

    def record_tile(
        self,
        *,
        descriptor: TileDescriptor,
        mask_tile: np.ndarray,
        product_id: str,
        image_filename: str,
        mask_filename: str,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        record = TileRecordSchema(
            tile_id=str(uuid.uuid4()),
            image_filename=image_filename,
            mask_filename=mask_filename,
            product_id=product_id,
            subscene_row=min(descriptor.row_offset, descriptor.orig_h - 1),
            subscene_col=min(descriptor.col_offset, descriptor.orig_w - 1),
            tile_grid_row=descriptor.grid_row,
            tile_grid_col=descriptor.grid_col,
            tile_size=descriptor.tile_size,
            cloud_percent=float(np.round((mask_tile == 1).mean() * 100, 2)),
            has_padding=descriptor.contains_padding(),
            bbox_west=bbox[0] if bbox else None,
            bbox_east=bbox[1] if bbox else None,
            bbox_north=bbox[2] if bbox else None,
            bbox_south=bbox[3] if bbox else None,
        )
        self._records.append(record)

    def save(self) -> None:
        if not self._records:
            raise RuntimeError("no tile records to save — call record_tile() first")
        out = self.output_dir / self.TILES_FILENAME
        pd.DataFrame([r.model_dump() for r in self._records]).to_csv(out, index=False)
        print(f"saved {len(self._records)} tile records → {out}")

    def save_dataset_level(self) -> None:
        df = pd.read_csv(self._bands_csv, comment="#")
        bands = [BandInfoSchema(**row) for row in df.to_dict(orient="records")]
        meta = DatasetMetaSchema(class_map={str(k): v for k, v in CLASS_MAP.items()}, bands=bands)
        out = self.output_dir / self.DATASET_FILENAME
        out.write_text(meta.model_dump_json(indent=2))
        print(f"saved dataset-level metadata → {out}")

    @property
    def tile_count(self) -> int:
        return len(self._records)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self._records])
