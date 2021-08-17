import os
from test.test_base import TestBase

from pfsspec.util.parallel import SmartParallel

class TestSmartParallel(TestBase):
    def initializer(self):
        pass

    def worker(self, i):
        return i

    def test_map_parallel(self):
        res = []
        with SmartParallel(self.initializer, verbose=True, parallel=True) as pool:
            for i in pool.map(self.worker, range(100)):
                res.append(i)
        self.assertEqual(100, len(res))
    