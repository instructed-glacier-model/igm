from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureConstants:
    topg: Tuple[float, float] = (4151.99267578125, 0.0)
    water: Tuple[float, float] = (1.0, 0.0)
    vx: Tuple[float, float] = (3.5998013782501204, -4.1134811401367175)
    thk: Tuple[float, float] = (125.9672691345204, 0.0)
    prec: Tuple[float, float] = (463.0692443847656, 2.575910019874573)
    ndvi: Tuple[float, float] = (221.4811553955078, -0.6113666892051697)
    vy: Tuple[float, float] = (6.441854915618933, -1.314163446426393)
    temp: Tuple[float, float] = (22.149993896484375, -8.147315979003906)


@dataclass(frozen=True)
class ImageConstants:
    red: Tuple[float, float] = (255.0, 0.0)
    blue: Tuple[float, float] = (255.0, 0.0)
    green: Tuple[float, float] = (255.0, 0.0)