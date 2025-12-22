from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class AbstractSignal(ABC):
    @property
    @abstractmethod
    def start_sec(self) -> float:
        pass

    @property
    @abstractmethod
    def end_sec(self) -> float:
        pass

    @property
    @abstractmethod
    def n_channels(self) -> float:
        pass

    @property
    @abstractmethod
    def sfreq(self) -> float:
        pass

    @abstractmethod
    def load_from_config(self, json: Dict[str, Any]) -> "AbstractSignal":
        pass

    @abstractmethod
    def get_raw(self, start: float, end: float) -> np.floating:
        """
        Returns raw signal (2D Numpy array). np.nan indicates a missing value

        :param start: start in seconds
        :param end: end in seconds
        :return: np.array of size (self.n_channels, self.sfreq * (end - start))
        """
        pass
