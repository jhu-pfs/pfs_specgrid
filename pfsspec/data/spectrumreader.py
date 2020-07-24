class SpectrumReader():
    def __init__(self, verbose, parallel):
        self.verbose = verbose
        self.parallel = parallel

    def read(self):
        raise NotImplementedError()