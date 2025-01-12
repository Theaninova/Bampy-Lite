from slicer_syntax import SlicerSyntax, Slicer
from extrusion import Extrusion
import open3d
import numpy as np
from dataclasses import dataclass


@dataclass
class NonPlanarSurface:
    boundary: open3d.t.geometry.LineSet
    surface: open3d.t.geometry.LineSet
    geometry: np.ndarray
    min_z: float
    max_z: float
    withheld: list[Extrusion]


class ProcessorContext:
    syntax: SlicerSyntax
    config_block: dict[str, str] = {}
    model_dir: str
    position_hint: tuple[float, float] | None = None
    gcode: list[str]
    gcode_line = 0

    line_type: str = ""

    last_p: tuple[float, float, float] = (0, 0, 0)
    last_e: float = 0
    last_contoured_z: float | None = None

    exclude_object: dict[str, list[open3d.t.geometry.RaycastingScene]] = {}
    active_object: list[open3d.t.geometry.RaycastingScene] | None = None

    non_planar_surfaces: list[NonPlanarSurface] = []
    pending_surfaces: list[NonPlanarSurface] = []

    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    extrusion: list[Extrusion] = []
    last_travel: Extrusion | None = None

    layer = 0
    z: float = 0
    height: float = 0
    width: float = 0
    wipe: bool = False

    relative_extrusion: bool = False
    relative_positioning: bool = False

    progress_percent: float = 0
    progress_remaining_minutes: float = 0

    def __init__(self, gcode: list[str], model_dir: str):
        self.gcode = gcode
        self.model_dir = model_dir
        self.syntax = SlicerSyntax(Slicer.detect(self.gcode))

        is_in_config = False
        for l in gcode:
            if not is_in_config and l.startswith(self.syntax.config_block_start):
                is_in_config = True
            elif is_in_config and l.startswith(self.syntax.config_block_end):
                break
            elif is_in_config:
                key, value = l.removeprefix(";").split("=", maxsplit=1)
                self.config_block[key.strip()] = value.strip()

    @property
    def line(self):
        return self.gcode[self.gcode_line]
