"""Tunable parameters of the extraction pipeline.

All sizes are in metres so the same settings work for imagery of any
resolution: they are converted to pixels using the ground sampling
distance (GSD) of the input raster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MaskParams:
    # Markings are thin bright stripes; a white top-hat with a window wider
    # than the widest marking keeps them and removes large bright areas
    # (concrete, roofs, sun glare on the asphalt).
    max_marking_width_m: float = 0.45
    tophat_threshold: int = 30          # min local contrast, 0-255 luminance
    min_brightness: int = 120           # absolute luminance floor
    max_saturation: int = 70            # markings are white/grey, not coloured
    exclude_vegetation: bool = True     # drop pixels where green clearly dominates


@dataclass
class ShapeParams:
    min_length_m: float = 0.8           # shorter blobs are noise
    min_elongation: float = 3.0         # length / width
    max_width_m: float = 0.6            # wider stripes are not line markings
    dash_max_length_m: float = 7.0      # longer pieces are solid lines
    isolated_min_length_m: float = 2.0  # a lone stroke that joins nothing must be at least this long


@dataclass
class VehicleParams:
    enabled: bool = True
    model: str = "yolo11n-obb.pt"       # YOLO OBB trained on DOTA (aerial imagery)
    target_gsd_m: float = 0.15          # imagery is resampled to roughly DOTA scale
    tile_px: int = 1024
    overlap_px: int = 128
    confidence: float = 0.25
    buffer_m: float = 0.5               # extra margin around each vehicle


@dataclass
class MergeParams:
    solid_max_gap_m: float = 4.0
    dash_max_gap_m: float = 14.0
    max_angle_deg: float = 12.0
    max_lateral_offset_m: float = 0.6
    simplify_tolerance_px: float = 1.5


@dataclass
class Params:
    mask: MaskParams = field(default_factory=MaskParams)
    shape: ShapeParams = field(default_factory=ShapeParams)
    vehicles: VehicleParams = field(default_factory=VehicleParams)
    merge: MergeParams = field(default_factory=MergeParams)

    def to_dict(self) -> dict:
        return asdict(self)
