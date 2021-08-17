import os
import re
import glob
import numpy as np
from scipy.optimize import curve_fit

from pfsspec.scripts.import_ import Import
from pfsspec.obsmod.pcapsf import PcaPsf
from pfsspec.obsmod.gausspsf import GaussPsf

class ImportPsf(Import):
    # TODO: convert into an import module
    # TODO: take these values from the detector config
    NPIX = 4096
    NARM = 3

    def __init__(self):
        super(ImportPsf, self).__init__()

    def add_args(self, parser):
        parser.add_argument("--in", type=str, required=True, help="gspfs binary output file\n")
        parser.add_argument("--out", type=str, required=True, help="Output text file\n")
        parser.add_argument("--arm", type=int, required=True, help='Spectrograph arm.\n')
        parser.add_argument("--wave", type=float, nargs=2, required=True, help="Wavelength limits\n")
        super(Import, self).add_args(parser)

    def run(self):
        kernels, w = self.load_psf()

        pca = PcaPsf()
        pca.wave = self.get_wave()
        pca.mean, pca.eigs = self.run_pca(kernels)
        pca.pc = self.get_pc(kernels, pca.mean, pca.eigs)
        pca.save(os.path.join(self.outdir, 'pca.h5'), format='h5')

        gauss = GaussPsf()
        gauss.wave = self.get_wave()
        gauss.sigma = self.fit_sigma(kernels)
        gauss.save(os.path.join(self.outdir, 'gauss.h5'), format='h5')

    def load_psf(self):
        arm = self.args['arm']
        psf = np.fromfile(self.args['in'], dtype=float, count=-1)
        w = psf.shape[0] // ImportPsf.NARM // ImportPsf.NPIX
        psf = psf.reshape(ImportPsf.NARM, ImportPsf.NPIX, w)
        return psf[arm, :, :].squeeze(), w

    def get_pix(self, w):
        # The 0- part is important! It's not equal to -w//2!
        pix = np.array(range(0 - w // 2, (w + 1) // 2), dtype=np.float)
        return pix

    def get_wave(self):
        # Central wavelength of pixels assuming a fully linear wavelength solution
        wave = np.linspace(self.args['wave'][0], self.args['wave'][1], ImportPsf.NPIX + 1)
        wave = 0.5 * (wave[:-1] + wave[1:])
        return wave

    def run_pca(self, psf, rank=5):
        mean = np.mean(psf, axis=0)
        X = psf - mean
        C = np.matmul(X.transpose(), X)
        U, S, V = np.linalg.svd(C)
        return mean, V[:rank, :]

    def get_pc(self, psf, mean, eigs):
        pc = np.matmul(psf - mean, eigs.transpose())
        return pc

    def fit_sigma(self, psf):
        pix = self.get_pix(psf.shape[1])
        s = np.zeros(psf.shape[0])
        for i in range(0, s.shape[0]):
            p, cov = curve_fit(ImportPsf.g2, pix, psf[i, :], p0=[1.5])
            s[i] = p[0]
        return s

    @staticmethod
    def g1(x, A, m, s):
        return A / np.sqrt(2 * np.pi) / s * np.exp(-0.5 * ((x - m) / s) ** 2)

    @staticmethod
    def g2(x, s):
        return 1 / np.sqrt(2 * np.pi) / s * np.exp(-0.5 * (x / s) ** 2)

    def execute_notebooks(self):
        super(ImportPsf, self).execute_notebooks()

        self.execute_notebook('eval_psf', parameters={
                                  'PSF_PATH': self.args['out']
                              })


def main():
    script = ImportPsf()
    script.execute()

if __name__ == "__main__":
    main()