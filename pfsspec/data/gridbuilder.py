import os
import numpy as np
import time
from tqdm import tqdm
import multiprocessing

from pfsspec.common.pfsobject import PfsObject

class GridBuilder(PfsObject):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(GridBuilder, self).__init__()

        if isinstance(orig, GridBuilder):
            self.parallel = orig.parallel
            self.threads = orig.threads

            self.input_grid = input_grid if input_grid is not None else orig.input_grid
            self.output_grid = output_grid if output_grid is not None else orig.output_grid
            self.input_grid_index = None
            self.output_grid_index = None
            self.grid_shape = None

            self.top = orig.top
        else:
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2

            self.input_grid = input_grid
            self.output_grid = output_grid
            self.input_grid_index = None
            self.output_grid_index = None
            self.grid_shape = None

            self.top = None

    def add_args(self, parser):
        parser.add_argument('--top', type=int, default=None, help='Limit number of results')

        # Axes of input grid can be used as parameters to filter the range
        grid = self.create_input_grid()
        grid.add_args(parser)

    def parse_args(self):
        self.top = self.get_arg('top', self.top)

    def create_input_grid(self):
        raise NotImplementedError()

    def create_output_grid(self):
        raise NotImplementedError()

    def open_data(self, input_path, output_path):
        # Open and preprocess input
        if input_path is not None:
            self.open_input_grid(input_path)
            self.input_grid.init_from_args(self.args)
            self.input_grid.build_axis_indexes()
            self.grid_shape = self.input_grid.get_shape()

        # Open and preprocess output
        if output_path is not None:
            self.open_output_grid(output_path)

        self.build_data_index()
        self.verify_data_index()

    def build_data_index(self):
        raise NotImplementedError()

    def verify_data_index(self):
        # Make sure all data indices have the same shape
        assert(self.input_grid_index.shape[-1] == self.output_grid_index.shape[-1])

    def open_input_grid(self, input_path):
        raise NotImplementedError()

    def open_output_grid(self, output_path):
        raise NotImplementedError()

    def save_data(self, output_path):
        self.output_grid.save(self.output_grid.filename, format=self.output_grid.fileformat)

    def get_input_count(self):
        # Return the number of data vectors
        input_count = self.input_grid_index.shape[1]
        if self.top is not None:
            input_count = min(self.top, input_count)
        return input_count

    def init_process(self):
        pass

    def run(self):
        raise NotImplementedError()