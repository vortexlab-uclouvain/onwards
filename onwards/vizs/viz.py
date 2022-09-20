from __future__ import annotations
from typing import TYPE_CHECKING, Any

import logging 
lg = logging.getLogger(__name__)

import os
import numpy as np
import matplotlib.pyplot  as plt

if TYPE_CHECKING:
    from ..farm import Farm

class Viz():
    viz_type = None
    
    def __init__(self, farm: Farm):
        """ Inits a Viz object

        Prototype class for user defined Viz objects. 
        Viz objects facilitate the export and plot of the OnWaRDS simulations 
        results.

        Parameters
        ----------
        farm : Farm
            Parent Farm object.
        """        
        self.farm = farm
        self.data = {}

        self._was_exported = False
        self._was_clean    = False 
        # -------------------------------------------------------------------- #

    def reset(self):
        """
        Resets the Viz data
        """
        self._was_exported = False
        # -------------------------------------------------------------------- #

    def update(self):
        """
        Updates the Viz data
        """        
        pass
        # -------------------------------------------------------------------- #

    def __dirgen__(self):
        """ 
        Generates the output directory name 
        """
        export_dir = f'{self.farm.out_dir}/{self.viz_type}'
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        return export_dir
        # -------------------------------------------------------------------- #

    def __savefig__(self, fid:str, *args, **kwargs):
        """ Saves the current figure 

        Parameters
        ----------
        fid : str
            Name of the output file.
        """        
        if self.farm.out_dir:
            plt.savefig(f'{self.__dirgen__()}/{fid}', *args, **kwargs)
        # -------------------------------------------------------------------- #


    def __savenpy__(self, fid:str, data:Any, *args, **kwargs):
        """ Saves the requested data as a npy file 

        Parameters
        ----------
        fid : str
            Name of the output file.
        data: Any
            Data that should be saved
        """        
        if self.farm.out_dir:
            np.save(f'{self.__dirgen__()}/{fid}', data, *args, **kwargs)
        # -------------------------------------------------------------------- #

    def __data_clean__(self):
        """ 
        Cleans the Viz data (wrapper for ``_data_clean``)
        """        
        pass
        if not self._was_clean:
            self._was_clean = True
            return self._data_clean()
        # -------------------------------------------------------------------- #

    def data_get(self, *args, **kwargs):
        """ 
        Wrapper for ``_data_get``
        """       
        self.__data_clean__()
        return self._data_get(*args, **kwargs)
        # -------------------------------------------------------------------- #

    def export(self):
        """ 
        Exports de the Viz data (wrapper for ``_export``)

        Note
        ----
        ``export`` is automatically triggered when calling ``farm.__exit__`` 
        method.
        """        
        self.__data_clean__()
        if not self._was_exported:
            self._export()
        # -------------------------------------------------------------------- #

    def plot(self):
        """ 
        Plots the Viz data (wrapper for ``_plot``)
        """        
        self.__data_clean__()
        return self._plot()
        # -------------------------------------------------------------------- #

    def _data_clean(self):
        """ 
        Clean the Viz data (should be implemented by the child class)
        """        
        raise NotImplementedError('No _data_clean method defined for Viz object.')
        # -------------------------------------------------------------------- #

    def _data_get(self, *args, **kwargs):
        """ 
        Viz data getter (should be implemented by the child class)
        """   
        raise NotImplementedError('No _data_get method defined for Viz object.')
        # -------------------------------------------------------------------- #

    def _export(self):
        """ 
        Exports the Viz data (should be implemented by the child class)
        """        
        raise NotImplementedError('No _export method defined for Viz object.')
        # -------------------------------------------------------------------- #

    def _plot(self):
        """ 
        Plots the Viz data (should be implemented by the child class)
        """        
        raise NotImplementedError('No _plot method defined for Viz object.')
        # -------------------------------------------------------------------- #


