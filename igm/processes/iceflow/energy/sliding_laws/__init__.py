from .sliding_law import sliding_law_XY

from .weertman import weertman, WeertmanParams, Weertman

from abc import ABC, abstractmethod
class SlidingLaw(ABC):
    @abstractmethod
    def __call__():
        pass