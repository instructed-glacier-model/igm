"""Ice-ocean geometry of the bmb process: masks, ice shelves and plume paths."""

from .geometry import Geometry, compute_geometry, floating_fraction
from .plume import basal_slope, grounding_line_depth
from .shelves import ice_rises, shelf_distances, shelf_labels
