from test.test_base import TestBase
import os

from pfsspec.stellarmod.kuruczatmreader import KuruczAtmReader

class TestKuruczAtmReader(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/apogee/all/mm25cp00op00/amm25cp00op00t3750g35v20.mod')
        with open(filename) as f:
            r = KuruczAtmReader(f)
            atm = r.read()

            self.assertIsNotNone(atm)
            self.assertEqual(atm.abundance.shape, (99,))
            self.assertEqual(atm.RHOX.shape, (72,))

            pass
