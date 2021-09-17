class SpectrumReader():
    """
    Implements functions to read one a more files from a spectrum file, stored
    in a format depending on the derived classes' implementation.
    """

    def __init__(self, orig=None):
        pass

    def read(self):
        raise NotImplementedError()

    def read_all(self):
        return [self.read(),]