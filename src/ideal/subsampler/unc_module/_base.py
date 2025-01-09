from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class UncModuleConfigBase(BaseModel, ABC):
    @abstractmethod
    def create_unc_module(self) -> UncModuleBase:
        pass

class UncModuleBase(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def unc_predict(
        self, *args, **kwargs
    ) -> tuple[float, np.ndarray] | tuple[np.ndarray, np.ndarray]:
        pass
