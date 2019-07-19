from pfsspec.scripts.predict import Predict
from pfsspec.stellarmod.kuruczregressionalaugmenter import KuruczRegressionalAugmenter

class PredictKurucz(Predict):
    def __init__(self):
        super(PredictKurucz, self).__init__()

    def create_generators(self):
        self.prediction_generator = KuruczRegressionalAugmenter(self.dataset, self.labels, self.coeffs)

def main():
    script = PredictKurucz()
    script.execute()

if __name__ == "__main__":
    main()