import os
import numpy as np
import pandas as pd
from astropy.io import fits
from SciServer import Authentication, CasJobs

from pfsspec.io.spectrumreader import SpectrumReader
from pfsspec.survey.dataset import DataSet
from pfsspec.survey.sdssspectrum import SdssSpectrum

class SdssSpectrumReader(SpectrumReader):

    def __init__(self):
        super(SdssSpectrumReader, self).__init__()

    def read(self, file):
        loglambda0 = file[0].header['COEFF0  ']
        loglambda1 = file[0].header['COEFF1  ']
        numbins = file[0].data.shape[1]
        logwave = loglambda0 + loglambda1 * np.arange(0, numbins)

        spec = SdssSpectrum()
        spec.wave = wave = 10 ** logwave
        spec.flux = file[0].data[0, :]
        return spec

    def get_filename(mjd, plate, fiber, das='das2', ver='1d_26'):
        # .../das2/spectro/1d_26/0288/1d/spSpec-52000-0288-005.fit
        return '{:s}/spectro/{:s}/{:04d}/1d/spSpec-{:5d}-{:04d}-{:03d}.fit'.format(das, ver, int(plate), int(mjd), int(plate), int(fiber))

    def authenticate(self, username, password):
        self.sciserver_token = Authentication.login(username, password)

    def execute_query(self, sql, context='DR7'):
        return CasJobs.executeQuery(sql=sql, context=context, format="pandas")

    def find_stars(self, top=None, mjd=None, plate=None):

        where = ''
        if mjd is not None:
            where += "AND s.mjd = {:d} \n".format(mjd)
        if plate is not None:
            where += "AND s.plate = {:d} \n".format(plate)

        sql = \
        """
        SELECT {} 
        s.specObjID, s.mjd, s.plate, s.fiberID, s.ra, s.dec, s.z
        FROM SpecObjAll s
            INNER JOIN sppParams spp ON spp.specobjID = s.specObjID
        WHERE specClass = 1 AND zConf > 0.98 {}
        ORDER BY s.mjd, s.plate, s.fiberID
        """.format('' if top is None else 'TOP {:d}'.format(top),
                   where)

        return self.execute_query(sql)

    def load_dataset(self, path, params):
        dataset = DataSet()
        dataset.params = params
        dataset.spectra = []

        for index, row in params.iterrows():
            filename = SdssSpectrumReader.get_filename(row['mjd'], row['plate'], row['fiberID'])
            filename = os.path.join(path, filename)
            with fits.open(filename) as hdus:
                spec = self.read(hdus)
                dataset.spectra.append(spec)

        return dataset

