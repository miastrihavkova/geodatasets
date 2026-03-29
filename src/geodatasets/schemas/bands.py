from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class BandInfoSchema(BaseModel):
    band_id: int | str
    name: str
    center_wavelength_nm: int = Field(gt=0)
    bandwidth_nm: int = Field(gt=0)
    gsd_m: int = Field(gt=0)

    model_config = {"frozen": True}


class DatasetMetaSchema(BaseModel):
    class_map: dict[str, str]
    bands: list[BandInfoSchema]

    @field_validator("class_map")
    @classmethod
    def keys_are_int_strings(cls, v: dict) -> dict:
        for k in v:
            if not k.isdigit():
                raise ValueError(f"class_map key must be an integer string, got '{k}'")
        return v

    @field_validator("bands")
    @classmethod
    def bands_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("bands list must not be empty")
        return v
