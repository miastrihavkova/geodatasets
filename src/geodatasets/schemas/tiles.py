"""Schemas for tile descriptors and tile metadata records."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class TileDescriptor(BaseModel):
    """Descriptor for a tile to be extracted from a larger image."""

    tile_index: int = Field(ge=0)
    row_offset: int = Field(ge=0)
    col_offset: int = Field(ge=0)
    tile_size: int = Field(gt=0)
    padded_h: int = Field(gt=0)
    padded_w: int = Field(gt=0)
    orig_h: int = Field(gt=0)
    orig_w: int = Field(gt=0)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def offsets_within_padded(self) -> TileDescriptor:
        """Validates that the tile fits within the padded dimensions."""
        if self.row_offset + self.tile_size > self.padded_h:
            raise ValueError("row_offset + tile_size exceeds padded_h")
        if self.col_offset + self.tile_size > self.padded_w:
            raise ValueError("col_offset + tile_size exceeds padded_w")
        return self

    @property
    def grid_row(self) -> int:
        """Returns the row index of the tile in the grid of tiles."""
        return self.row_offset // self.tile_size

    @property
    def grid_col(self) -> int:
        """Returns the column index of the tile in the grid of tiles."""
        return self.col_offset // self.tile_size

    @property
    def slice_rows(self) -> slice:
        """Returns the slice for the rows of the tile."""
        return slice(self.row_offset, self.row_offset + self.tile_size)

    @property
    def slice_cols(self) -> slice:
        """Returns the slice for the columns of the tile."""
        return slice(self.col_offset, self.col_offset + self.tile_size)

    def contains_padding(self) -> bool:
        """Returns True if the tile contains any padding, False if fully within original image."""
        return (
            self.row_offset + self.tile_size > self.orig_h
            or self.col_offset + self.tile_size > self.orig_w
        )


class TileRecordSchema(BaseModel):
    """Record for a single tile, to be saved in the tiles.csv metadata file."""

    tile_id: str
    image_filename: str
    mask_filename: str
    product_id: str
    subscene_row: int = Field(ge=0)
    subscene_col: int = Field(ge=0)
    tile_grid_row: int = Field(ge=0)
    tile_grid_col: int = Field(ge=0)
    tile_size: int = Field(gt=0)
    cloud_percent: float = Field(ge=0.0, le=100.0)
    has_padding: bool
    bbox_west: float | None = None
    bbox_east: float | None = None
    bbox_north: float | None = None
    bbox_south: float | None = None

    @model_validator(mode="after")
    def filenames_share_stem(self) -> TileRecordSchema:
        """Validates that the image and mask filenames share the same stem."""
        img_stem = Path(self.image_filename).stem
        mask_stem = Path(self.mask_filename).stem

        # HDF5 stores all tiles in shared files so stems won't match and that's fine
        if img_stem == mask_stem:
            return self

        shared_file_stems = {"images", "masks"}
        if img_stem in shared_file_stems and mask_stem in shared_file_stems:
            return self

        raise ValueError(
            f"image and mask filenames must share the same stem, "
            f"got '{self.image_filename}' vs '{self.mask_filename}'"
        )

    @model_validator(mode="after")
    def bbox_all_or_none(self) -> TileRecordSchema:
        """Validates that either all four bbox fields are set or all are None."""
        fields = [self.bbox_west, self.bbox_east, self.bbox_north, self.bbox_south]
        if sum(v is not None for v in fields) not in (0, 4):
            raise ValueError("all four bbox fields must be set together or all None")
        return self

    @model_validator(mode="after")
    def bbox_valid_range(self) -> TileRecordSchema:
        """Validates that bbox coordinates are within valid ranges and south < north."""
        if self.bbox_west is None or self.bbox_east is None:
            return self
        if self.bbox_south is None or self.bbox_north is None:
            return self
        if not (-180 <= self.bbox_west <= 180 and -180 <= self.bbox_east <= 180):
            raise ValueError("bbox longitude out of [-180, 180]")
        if not (-90 <= self.bbox_south <= 90 and -90 <= self.bbox_north <= 90):
            raise ValueError("bbox latitude out of [-90, 90]")
        if self.bbox_south >= self.bbox_north:
            raise ValueError("bbox_south must be less than bbox_north")
        return self


class TilingConfig(BaseModel):
    """Configuration for tiling large images into smaller tiles."""

    tile_size: int = Field(default=512, gt=0)
    pad_value: int = Field(default=0, ge=0, le=65535)

    @field_validator("tile_size")
    @classmethod
    def must_be_power_of_two(cls, v: int) -> int:
        """Validates that tile_size is a power of two."""
        if v & (v - 1) != 0:
            raise ValueError(f"tile_size must be a power of two, got {v}")
        return v
