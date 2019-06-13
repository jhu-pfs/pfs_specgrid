from unittest import TestCase
import os

class TestBase(TestCase):
    @classmethod
    def setUpClass(self):
        self.PFSSPEC_DATA_PATH = os.environ['PFSSPEC_DATA_PATH']