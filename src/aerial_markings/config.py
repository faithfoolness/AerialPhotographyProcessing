"""Tunable parameters of the extraction pipeline.

All sizes are in metres so the same settings work for imagery of any
resolution: they are converted to pixels using the ground sampling
distance (GSD) of the input raster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MaskParams:
    # Markings are thin bright stripes: brighter than the surface on BOTH
    # sides. Comparing each pixel with points 20-35 cm away across it finds
    # paint up to ~0.5 m wide and ignores slab edges, joints and shoulders.
    ridge_offsets_m: tuple = (0.2, 0.35)
    ridge_threshold: int = 16           # min brightness above both sides, 0-255
    min_brightness: int = 120           # absolute luminance floor
    max_saturation: int = 70            # white/grey paint...
    max_yellow_saturation: int = 220    # ...or yellow paint
    yellow_b_threshold: float = 8.0     # Lab b* above neutral that makes a line "yellow"
    exclude_vegetation: bool = True     # drop pixels where green clearly dominates


@dataclass
class RoadParams:
    # Paint is only searched for on the pavement found here.
    enabled: bool = True
    work_gsd_m: float = 0.25            # the mask is computed at this resolution
    texture_window_m: float = 1.0
    max_texture: float = 12.0           # std of brightness; asphalt is smooth, grass is not
    max_saturation: int = 45
    max_excess_green: int = 15          # 2G - R - B
    min_brightness: int = 35
    max_brightness: int = 250           # only blown-out pixels; asphalt can be very light
    fill_holes_m: float = 2.0           # joints, patches and paint inside the carriageway
    min_road_width_m: float = 2.5
    min_area_m2: float = 300.0
    margin_m: float = 0.6               # edge lines lie on the border of the asphalt


@dataclass
class SurfaceParams:
    # Paint lies on pavement: at least one side of a line must be asphalt or
    # concrete. Guardrails and debris stand on soil, gravel or grass.
    enabled: bool = True
    max_saturation: int = 50            # bluish dark asphalt ~40, orange soil ~57
    max_excess_green: int = 12
    min_value: int = 50
    side_offsets_m: tuple = (0.5, 0.8, 1.1)   # beyond the second rail of a guardrail
    sample_step_m: float = 0.5
    min_share: float = 0.6              # of positions along the line


@dataclass
class ShapeParams:
    min_length_m: float = 0.8           # shorter blobs are noise
    min_elongation: float = 3.0         # length / width
    max_width_m: float = 0.45           # wider stripes are not line markings (paint is 0.1-0.25 m + blur)
    max_spread_m: float = 0.8           # area per metre of axis; specks stuck to a line are tolerated
    max_holes_per_10m: float = 0.5      # enclosed gaps: paint has none, a guardrail is a ladder
    dash_max_length_m: float = 7.0      # longer pieces are solid lines
    keep_single_strokes: bool = False   # strokes that are not part of a dashed line
    isolated_min_length_m: float = 2.0  # ...and if they are kept, their minimum length
    min_straightness: float = 0.9       # chord / path length for strokes; grass blades bend
    straight_check_max_m: float = 30.0  # longer lines may follow a curved road...
    wiggle_window_m: float = 5.0        # ...but any 5 m of paint is nearly straight
    max_wiggle_m: float = 0.2           # median deviation of such a window from its chord


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
    lateral_per_gap: float = 0.05       # + 5 cm of offset per metre of gap (~3 degrees)
    min_dashes: int = 4                 # fewer aligned strokes are not a dashed line
    min_dashed_length_m: float = 25.0   # nor is a shorter one (two stroke + gap cycles)
    rhythm_tolerance: float = 0.4       # strokes and gaps of a dashed line within ±40 % of the median...
    min_steady_share: float = 0.6       # ...for at least this share of them
    guide_min_length_m: float = 60.0    # long solid lines used as the reference direction
    short_solid_max_m: float = 40.0     # shorter solid lines must be parallel to a reference line
    guide_radius_m: float = 20.0
    guide_check_max_m: float = 40.0     # longer dashed lines are trusted as they are
    simplify_tolerance_px: float = 1.5
    smooth_window_m: float = 4.0        # moving average along the final lines; highway bends are far gentler
    smooth_step_m: float = 0.5


@dataclass
class Params:
    road: RoadParams = field(default_factory=RoadParams)
    mask: MaskParams = field(default_factory=MaskParams)
    surface: SurfaceParams = field(default_factory=SurfaceParams)
    shape: ShapeParams = field(default_factory=ShapeParams)
    vehicles: VehicleParams = field(default_factory=VehicleParams)
    merge: MergeParams = field(default_factory=MergeParams)

    def to_dict(self) -> dict:
        return asdict(self)
