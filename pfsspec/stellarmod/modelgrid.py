import logging
import itertools
import numpy as np
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.gridaxis import GridAxis

class ModelGrid():
    """Wraps an array or RBF grid, optionally PCA-compressed, and implements logic
    to interpolate spectra."""

    def __init__(self, config, grid_type, orig=None):
        if isinstance(orig, ModelGrid):
            self.config = config if config is not None else orig.config
            self.grid = self.create_grid(grid_type) if grid_type is not None else orig.grid
        else:
            self.config = config
            self.grid = self.create_grid(grid_type)

    def create_grid(self, grid_type):
        return grid_type(self.config)

    def allocate_values(self):
        self.grid.allocate_values()

    def get_axes(self):
        return self.grid.get_axes()

    def set_axes(self, axes):
        self.grid.set_axes(axes)

    def build_axis_indexes(self):
        self.grid.build_axis_indexes()

    def add_args(self, parser):
        self.grid.add_args(parser)

    @property
    def preload_arrays(self):
        return self.grid.preload_arrays

    @preload_arrays.setter
    def preload_arrays(self, value):
        self.grid.preload_arrays = value

    def init_from_args(self, args):
        self.grid.init_from_args(args)
        
    def get_wave(self):
        return self.grid.get_wave()

    def set_wave(self, wave):
        self.grid.set_wave(wave)

    def set_flux(self, flux, cont=None, **kwargs):
        self.grid.set_flux(flux, cont=cont, **kwargs)

    def set_flux_at(self, index, flux, cont=None):
        self.grid.set_flux_at(index, flux, cont=cont)

    def is_value_valid(self, name, value):
        return self.grid.is_value_valid(name, value)

    def load(self, filename, s=None, format=None):
        self.grid.load(filename, s=s, format=format)

    def save(self, filename=None, format=None):
        self.grid.save(filename or self.grid.filename, format=format or self.grid.fileformat)

    def get_nearest_model(self, **kwargs):
        return self.grid.get_nearest_model(**kwargs)

    def interpolate_model(self, interpolation=None, **kwargs):
        raise NotImplementedError()

    def get_model_at(self, idx):
        return self.grid.get_model_at(idx)
   