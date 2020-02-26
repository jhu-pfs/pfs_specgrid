from test.test_base import TestBase
import os

from pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

class TestKuruczSpectrumReader(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/atlas9/gridm05aodfnew/fm05ak2odfnew.pck')
        with open(filename) as f:
            r = KuruczSpectrumReader(f)
            spec = r.read()

    def test_read_all(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/atlas9/gridm05aodfnew/fm05ak2odfnew.pck')
        with open(filename) as f:
            r = KuruczSpectrumReader(f)
            specs = r.read_all()
