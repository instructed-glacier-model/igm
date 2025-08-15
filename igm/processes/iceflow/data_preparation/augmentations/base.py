from abc import ABC, abstractmethod


class Augmentation(ABC):
    @abstractmethod
    def apply(self, x):
        pass
