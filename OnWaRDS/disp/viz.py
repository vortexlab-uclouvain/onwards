from __future__ import annotations
from typing import TYPE_CHECKING

import logging 
lg = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot  as plt

if TYPE_CHECKING:
    from ..farm import Farm

class Viz():
    def __init__(self, farm: Farm):
        self.farm = farm
        # -------------------------------------------------------------------- #

    def update(self):
        pass
        # -------------------------------------------------------------------- #

    def plot(self):
        pass
        # -------------------------------------------------------------------- #
