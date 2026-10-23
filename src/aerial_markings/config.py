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
class RoadParams:
    # Paint is only searched for on the pavement found here.
    enabled: bool = True
    work_gsd_m: float = 0.25            # the mask is computed at this resolution
    texture_window_m: float = 1.0
    max_texture: float = 9.0            # std of brightness; asphalt is smooth, grass is not
    max_saturation: int = 45
    max_excess_green: int = 10          # 2G - R - B
    min_brightness: int = 35
    max_brightness: int = 200
    min_road_width_m: float = 2.5
    min_area_m2: float = 300.0
    margin_m: float = 0.6               # edge lines lie on the border of the asphalt


@dataclass
class ShapeParams:
    min_length_m: float = 0.8           # shorter blobs are noise
    min_elongation: float = 3.0         # length / width
    max_width_m: float = 0.6            # wider stripes are not line markings
    dash_max_length_m: float = 7.0      # longer pieces are solid lines
    keep_single_strokes: bool = False   # strokes that are not part of a dashed line
    isolated_min_length_m: float = 2.0  # ...and if they are kept, their minimum length
    min_straightness: float = 0.9       # chord / path length for strokes; grass blades bend
    straight_check_max_m: float = 30.0  # longer lines may follow a curved road


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
    dash_max_gap_m: float = 22.0        # one stroke may be hidden by a shadow or a car
    max_angle_deg: float = 12.0
    max_lateral_offset_m: float = 0.4
    min_dashes: int = 4                 # fewer aligned strokes are not a dashed line
    min_dashed_length_m: float = 15.0   # nor is a shorter one (a stroke + gap cycle is 4-12 m)
    rhythm_tolerance: float = 0.4       # strokes and gaps of a dashed line within ±40 % of the median...
    min_steady_share: float = 0.6       # ...for at least this share of them
    guide_min_length_m: float = 30.0    # solid lines used as the reference direction
    guide_radius_m: float = 20.0
    guide_check_max_m: float = 40.0     # longer dashed lines are trusted as they are
    simplify_tolerance_px: float = 1.5


@dataclass
class Params:
    road: RoadParams = field(default_factory=RoadParams)
    mask: MaskParams = field(default_factory=MaskParams)
    shape: ShapeParams = field(default_factory=ShapeParams)
    vehicles: VehicleParams = field(default_factory=VehicleParams)
    merge: MergeParams = field(default_factory=MergeParams)

    def to_dict(self) -> dict:
        return asdict(self)
