import numpy as np
import pytest

from geodatasets.preprocessing.tiling import (
    compute_tile_descriptors,
    mask_onehot_to_classid,
    pad_to_tile_multiple,
    reflectance_to_uint16,
    tile_array,
)
from geodatasets.schemas import TileDescriptor, TileRecordSchema, TilingConfig

# Tiling


def test_pad_adds_correct_size():
    arr = np.zeros((1022, 1022, 13), dtype=np.uint16)
    out = pad_to_tile_multiple(arr, tile_size=512)
    assert out.shape == (1024, 1024, 13)


def test_pad_does_nothing_when_already_multiple():
    arr = np.zeros((512, 512), dtype=np.uint8)
    out = pad_to_tile_multiple(arr, tile_size=512)
    assert out.shape == (512, 512)


def test_pad_fills_with_zero_by_default():
    arr = np.ones((3, 3), dtype=np.uint8)
    out = pad_to_tile_multiple(arr, tile_size=4)
    assert out[3, 3] == 0


def test_tile_descriptors_count():
    descs = compute_tile_descriptors(1022, 1022, 512)
    assert len(descs) == 4  # 2x2 grid after padding to 1024


def test_tile_descriptors_grid_positions():
    descs = compute_tile_descriptors(1022, 1022, 512)
    positions = {(d.grid_row, d.grid_col) for d in descs}
    assert positions == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_tile_array_shapes():
    arr = np.zeros((1022, 1022, 13), dtype=np.uint16)
    tiles, descs = tile_array(arr, tile_size=512)
    assert len(tiles) == 4
    for t in tiles:
        assert t.shape == (512, 512, 13)


def test_reflectance_to_uint16_dtype():
    arr = np.random.rand(10, 10, 13).astype(np.float32)
    out = reflectance_to_uint16(arr)
    assert out.dtype == np.uint16


def test_reflectance_to_uint16_clamps():
    arr = np.array([[[10.0]]], dtype=np.float32)  # 10 * 10000 >> 65535
    out = reflectance_to_uint16(arr)
    assert out[0, 0, 0] == 65535


def test_mask_onehot_to_classid_shape():
    onehot = np.zeros((100, 100, 3), dtype=bool)
    onehot[..., 1] = True  # all CLOUD
    out = mask_onehot_to_classid(onehot)
    assert out.shape == (100, 100)
    assert out.dtype == np.uint8
    assert out.max() == 1


# Schemas


def test_tile_descriptor_valid():
    d = TileDescriptor(
        tile_index=0,
        row_offset=0,
        col_offset=0,
        tile_size=512,
        padded_h=1024,
        padded_w=1024,
        orig_h=1022,
        orig_w=1022,
    )
    assert d.grid_row == 0
    assert d.grid_col == 0
    assert d.contains_padding() is False


def test_tile_descriptor_immutable():
    d = TileDescriptor(
        tile_index=0,
        row_offset=0,
        col_offset=0,
        tile_size=512,
        padded_h=1024,
        padded_w=1024,
        orig_h=1022,
        orig_w=1022,
    )
    with pytest.raises(Exception):
        d.tile_index = 99


def test_tile_descriptor_invalid_offset():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TileDescriptor(
            tile_index=0,
            row_offset=600,
            col_offset=0,  # 600 + 512 > 1024
            tile_size=512,
            padded_h=1024,
            padded_w=1024,
            orig_h=1022,
            orig_w=1022,
        )


def test_tiling_config_rejects_non_power_of_two():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="power of two"):
        TilingConfig(tile_size=300)


def test_tile_record_rejects_bad_cloud_percent():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TileRecordSchema(
            tile_id="x",
            image_filename="t.tif",
            mask_filename="t.tif",
            product_id="S2A",
            subscene_row=0,
            subscene_col=0,
            tile_grid_row=0,
            tile_grid_col=0,
            tile_size=512,
            cloud_percent=110.0,  # > 100
            has_padding=False,
        )


def test_tile_record_geotiff_valid():
    r = TileRecordSchema(
        tile_id="abc",
        image_filename="tile_0000.tif",
        mask_filename="tile_0000.tif",
        product_id="S2A",
        subscene_row=0,
        subscene_col=0,
        tile_grid_row=0,
        tile_grid_col=0,
        tile_size=512,
        cloud_percent=42.5,
        has_padding=False,
    )
    assert r.cloud_percent == 42.5


def test_tile_record_hdf5_valid():
    r = TileRecordSchema(
        tile_id="abc",
        image_filename="images.h5",
        mask_filename="masks.h5",
        product_id="S2A",
        subscene_row=0,
        subscene_col=0,
        tile_grid_row=0,
        tile_grid_col=0,
        tile_size=512,
        cloud_percent=10.0,
        has_padding=False,
    )
    assert r.image_filename == "images.h5"
