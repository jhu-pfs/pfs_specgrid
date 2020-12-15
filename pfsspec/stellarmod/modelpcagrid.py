import numpy as np

from pfsspec.data.pcagrid import PCAGrid

class ModelPCAGrid(PCAGrid):
    def __init__(self, orig=None):
        super(ModelPCAGrid, self).__init__(orig=orig)

        if isinstance(orig, ModelPCAGrid):
            self.wave = orig.wave
        else:
            self.wave = None
    
    def init_axes(self):
        super(ModelPCAGrid, self).init_axes()
        self.init_axis('Fe_H')
        self.init_axis('T_eff')
        self.init_axis('log_g')

    def init_values(self):
        super(ModelPCAGrid, self).init_values()
        self.init_value('params')
        self.init_value('rbf')

    def allocate_values(self):
        super(ModelPCAGrid, self).allocate_values()
        self.allocate_value('params')
        self.allocate_value('rbf')

    def save_items(self):
        self.save_item('wave', self.wave)

        super(ModelPCAGrid, self).save_items()

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray)

        super(ModelPCAGrid, self).load_items(s=s)