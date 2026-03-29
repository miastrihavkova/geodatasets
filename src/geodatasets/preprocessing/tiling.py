"""Utilities for tiling large geospatial arrays into smaller patches, with padding and metadata."""

from __future__ import annotations

import numpy as np

from geodatasets.schemas import TileDescriptor


def pad_to_tile_multiple(
    array: np.ndarray,
    tile_size: int,
    pad_value: int | float = 0,
) -> np.ndarray:
    """Pads the input array so that its height and width are multiples of the tile size.

    Args:
        array: The input array to be padded, of shape (H, W) or (H, W, C).
        tile_size: The desired tile size in pixels (e.g., 512).
        pad_value: The value to use for padding. Defaults to 0.

    Returns:
        np.ndarray: The padded array.
    """
    h, w = array.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    if pad_h == 0 and pad_w == 0:
        return array

    pad_width = [(0, pad_h), (0, pad_w)]
    if array.ndim == 3:
        pad_width.append((0, 0))

    return np.pad(array, pad_width, mode="constant", constant_values=pad_value)


def compute_tile_descriptors(orig_h: int, orig_w: int, tile_size: int) -> list[TileDescriptor]:
    """Compute descriptors for every non-overlapping tile in a padded grid.

    Args:
        orig_h: Height of the original unpadded array in pixels.
        orig_w: Width of the original unpadded array in pixels.
        tile_size: Tile height and width in pixels.

    Returns:
        A list of TileDescriptor objects, one per tile, in row-major order.
    """
    pad_h = (tile_size - orig_h % tile_size) % tile_size
    pad_w = (tile_size - orig_w % tile_size) % tile_size
    padded_h = orig_h + pad_h
    padded_w = orig_w + pad_w

    return [
        TileDescriptor(
            tile_index=idx,
            row_offset=row,
            col_offset=col,
            tile_size=tile_size,
            padded_h=padded_h,
            padded_w=padded_w,
            orig_h=orig_h,
            orig_w=orig_w,
        )
        for idx, (row, col) in enumerate(
            (r, c) for r in range(0, padded_h, tile_size) for c in range(0, padded_w, tile_size)
        )
    ]


def tile_array(
    array: np.ndarray,
    tile_size: int,
    pad_value: int | float = 0,
) -> tuple[list[np.ndarray], list[TileDescriptor]]:
    """Pad an array and split it into non-overlapping tiles.

    Args:
        array: Input array of shape (H, W) or (H, W, C).
        tile_size: Tile height and width in pixels. Must be a power of two.
        pad_value: Fill value for the zero-padded region.

    Returns:
        A tuple of (tiles, descriptors) where tiles is a list of arrays
        each of shape (tile_size, tile_size[, C]), and descriptors contains
        the spatial metadata for each tile in the same order.
    """
    orig_h, orig_w = array.shape[:2]
    padded = pad_to_tile_multiple(array, tile_size, pad_value)
    descriptors = compute_tile_descriptors(orig_h, orig_w, tile_size)
    tiles = [padded[d.slice_rows, d.slice_cols] for d in descriptors]
    return tiles, descriptors


def reflectance_to_uint16(array: np.ndarray) -> np.ndarray:
    """Convert a float32 array of reflectance values in [0, 1] to uint16 in [0, 65535]."""
    return np.clip(array * 10_000, 0, 65535).astype(np.uint16)


def mask_onehot_to_classid(mask_onehot: np.ndarray) -> np.ndarray:
    """Convert a one-hot encoded mask (H, W, num_classes) to a class ID mask (H, W)."""
    return np.argmax(mask_onehot, axis=-1).astype(np.uint8)
