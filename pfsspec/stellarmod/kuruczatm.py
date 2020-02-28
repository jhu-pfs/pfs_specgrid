import numpy as np

from pfsspec.stellarmod.atm import Atm

class KuruczAtm(Atm):
    ABUNDANCE_ELEMENTS = 99
    ATM_LAYERS = 72

    def __init__(self):
        super(KuruczAtm, self).__init__()

        self.title = None
        self.ABUNDANCE = np.empty(99)
        self.RHOX = np.empty(KuruczAtm.ATM_LAYERS)
        self.T = np.empty(KuruczAtm.ATM_LAYERS)
        self.P = np.empty(KuruczAtm.ATM_LAYERS)
        self.XNE = np.empty(KuruczAtm.ATM_LAYERS)
        self.ABROSS = np.empty(KuruczAtm.ATM_LAYERS)
        self.ACCRAD = np.empty(KuruczAtm.ATM_LAYERS)
        self.VTURB = np.empty(KuruczAtm.ATM_LAYERS)
        self.FLXCNV = np.empty(KuruczAtm.ATM_LAYERS)
        self.VCONV = np.empty(KuruczAtm.ATM_LAYERS)
        self.VELSND = np.empty(KuruczAtm.ATM_LAYERS)