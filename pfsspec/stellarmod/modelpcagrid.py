import numpy as np

from pfsspec.data.pcagrid import PCAGrid

class ModelPCAGrid(PCAGrid):
    def __init__(self, orig=None):
        super(ModelPCAGrid, self).__init__(orig=orig)

        if isinstance(orig, ModelPCAGrid):
            self.wave = orig.wave
            self.params_rbf = orig.params_rbf
            self.coeffs_rbf = orig.coeffs_rbf
        else:
            self.wave = None
            self.params_rbf = None
            self.coeffs_rbf = None
    
    def init_axes(self):
        super(ModelPCAGrid, self).init_axes()
        self.init_axis('Fe_H')
        self.init_axis('T_eff')
        self.init_axis('log_g')

    def init_values(self):
        super(ModelPCAGrid, self).init_values()
        self.init_value('params')

    def allocate_values(self):
        super(ModelPCAGrid, self).allocate_values()
        self.allocate_value('params')

    def save_items(self):
        self.save_item('wave', self.wave)
        self.save_item('params_rbf', self.params_rbf)
        self.save_item('coeffs_rbf', self.coeffs_rbf)

        super(ModelPCAGrid, self).save_items()

    def load_items(self, s=None):
        super(ModelPCAGrid, self).load_items(s=s)
        
        self.wave = self.load_item('wave', np.ndarray)
        self.params_rbf = self.load_item('params_rbf', np.ndarray)
        self.coeffs_rbf = self.load_item('coeffs_rbf', np.ndarray)

        