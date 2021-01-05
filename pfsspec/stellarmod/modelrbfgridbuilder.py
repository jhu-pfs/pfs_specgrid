import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.data.arraygrid import ArrayGrid
from pfsspec.data.rbfgrid import RbfGrid
from pfsspec.data.pcagrid import PcaGrid
from pfsspec.data.rbfgridbuilder import RbfGridBuilder
from pfsspec.stellarmod.modelgrid import ModelGrid

class ModelRbfGridBuilder(RbfGridBuilder):
    def __init__(self, config, grid=None, orig=None):
        super(ModelRbfGridBuilder, self).__init__(orig=orig)

        if isinstance(orig, ModelRbfGridBuilder):
            self.config = config if config is not None else orig.config
        else:
            self.config = config

    def add_args(self, parser):
        super(ModelRbfGridBuilder, self).add_args(parser)

        # Axes of input grid can be used as parameters to filter the range
        grid = self.create_grid()
        grid.add_args(parser)

    def parse_args(self):
        super(ModelRbfGridBuilder, self).parse_args()

        # Axes limits will be parsed only after the input grid is opened

    def create_grid(self):
        # It doesn't really matter if the input is already a PCA grid or just a direct
        # array because RBF interpolation is the same. On the other hand,
        # when we want to slice a PCA grid in wavelength, we have to load the
        # eigenvectors so this needs to be extended here.
        config = self.config
        if self.pca is not None and self.pca:
            config = type(config)(pca=True)
        grid = ModelGrid(config, ArrayGrid)
        return grid

    def create_rbf_grid(self):
        config = self.config
        if self.pca is not None and self.pca:
            config = type(config)(pca=True)
        grid = ModelGrid(config, RbfGrid)
        return grid

    def open_data(self, input_path, output_path):
        self.open_input_grid(input_path)
        self.open_output_grid(output_path)

    def open_input_grid(self, input_path):
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid = self.create_grid()
        self.input_grid.load(fn, format='h5')
        self.input_grid.init_from_args(self.args)
        self.input_grid.build_axis_indexes()

        # Source indexes
        if isinstance(self.input_grid.grid, PcaGrid):
            index = self.input_grid.grid.grid.get_value_index_unsliced('flux')
        else:
            index = self.input_grid.grid.get_value_index_unsliced('flux')
        self.input_grid_index = np.array(np.where(index))
        if self.top is not None:
            self.input_grid_index = self.input_grid_index[:, :min(self.top, self.input_grid_index.shape[1])]

    def open_output_grid(self, output_path):
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid = self.create_rbf_grid()
        
        # Pad the output axes. This automatically takes the parameter ranges into
        # account since the grid is sliced.
        
        # TODO: Now we take a slice of the grid, then pad it instead of using the
        #       existing surrounding values. We might need a parameter to turn
        #       padding on and off etc.
        #       But what to do if we take a subcube that's on the edge of the grid?
        #       Then padding should be done in one direction and not in the others
        #       but this is a nightmare to figure out automatically.

        orig_axes = self.input_grid.get_axes()
        padded_axes = ArrayGrid.pad_axes(orig_axes)
        self.output_grid.set_axes(padded_axes)

        # DEBUG
        self.output_grid.preload_arrays = True
        self.output_grid.grid.preload_arrays = True
        # END DEBUG

        self.output_grid.filename = fn
        self.output_grid.fileformat = 'h5'

    def save_data(self, output_path):
        self.output_grid.save(self.output_grid.filename, format=self.output_grid.fileformat)

    def run(self):
        self.output_grid.set_constants(self.input_grid.get_constants())
        self.output_grid.set_wave(self.input_grid.get_wave())

        # Copy eigv if PCA
        if isinstance(self.input_grid.grid, PcaGrid):
            for k in self.input_grid.grid.eigs:
                if self.input_grid.grid.eigs[k] is not None:
                    self.output_grid.grid.eigs[k] = self.input_grid.grid.eigs[k]
                    self.output_grid.grid.eigv[k] = self.input_grid.grid.eigv[k][self.input_grid.wave_slice or slice(None)]

            grid = self.input_grid.grid.grid
            wave_slice = None                       # Do not slice PCs
        else:
            grid = self.input_grid.grid
            wave_slice = self.input_grid.wave_slice

        # Fit RBF
        for name in grid.values:
            if grid.has_value(name):
                padded_value, padded_axes = grid.get_value_padded(name, interpolation='ijk', s=wave_slice)
                rbf = self.fit_rbf(padded_value, padded_axes)
                self.output_grid.grid.set_value(name, rbf)




#######################


#region RBF interpolation

    # TODO: is it used for anything?
    # def get_slice_rbf(self, s=None, interpolation='xyz', padding=True, **kwargs):
    #     # Interpolate the continuum and flux in a wavelength slice `s` and parameter
    #     # slices defined by kwargs using RBF. The input RBF is padded with linearly extrapolated
    #     # values to make the interpolation smooth

    #     if padding:
    #         flux, axes = self.get_value_padded('flux', s=s, interpolation=interpolation, **kwargs)
    #         cont, axes = self.get_value_padded('cont', s=s, interpolation=interpolation, **kwargs)
    #     else:
    #         flux = self.get_value('flux', s=s, **kwargs)
    #         cont = self.get_value('cont', s=s, **kwargs)

    #         axes = {}
    #         for p in self.axes.keys():
    #             if p not in kwargs:            
    #                 if interpolation == 'ijk':
    #                     axes[p] = GridAxis(p, np.arange(self.axes[p].values.shape[0], dtype=np.float64))
    #                 elif interpolation == 'xyz':
    #                     axes[p] = self.axes[p]

    #     # Max nans and where the continuum is zero
    #     mask = ~np.isnan(cont) & (cont != 0)
    #     if mask.ndim > len(axes):
    #         mask = np.all(mask, axis=-(mask.ndim - len(axes)))

    #     # Rbf must be generated on a uniform grid
    #     if padding:
    #         aa = {p: GridAxis(p, np.arange(axes[p].values.shape[0]) - 1.0) for p in axes}
    #     else:
    #         aa = {p: GridAxis(p, np.arange(axes[p].values.shape[0])) for p in axes}

    #     rbf_flux = self.fit_rbf(flux, aa, mask=mask)
    #     rbf_cont = self.fit_rbf(cont, aa, mask=mask)

    #     return rbf_flux, rbf_cont, axes

#endregion