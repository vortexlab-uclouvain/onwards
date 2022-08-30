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
        self._was_exported = False
        # -------------------------------------------------------------------- #

    def update(self):
        pass
        # -------------------------------------------------------------------- #

    def reset(self):
        self._was_exported = False
        # -------------------------------------------------------------------- #

    def export(self):
        self._was_exported = True
        # -------------------------------------------------------------------- #

    def plot(self):
        pass
        # -------------------------------------------------------------------- #

    def savefig(self, fid:str, *args, **kwargs):
        if self.farm.out_dir:
            plt.savefig(f'{self.farm.out_dir}/{fid}', *args, **kwargs)
        # -------------------------------------------------------------------- #
