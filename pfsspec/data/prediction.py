import pickle
import pandas as pd

from pfsspec.data.dataset import Dataset

class Prediction(Dataset):
    def __init__(self, orig=None):
        super(Prediction, self).__init__(orig=orig)
        if isinstance(orig, Prediction):
            self.predictions = orig.predictions
        else:
            self.predictions = None

    def save_items(self, f):
        super(Prediction, self).save_items(f)
        pickle.dump(self.predictions, f)

    def load_items(self, f):
        super(Prediction, self).load_items(f)
        self.predictions = pickle.load(f)

    def split(self, split_value):
        split_index, a, b = super(Prediction, self).split(split_value)
        a_range, b_range = self.get_split_ranges(split_index)

        a.predictions = self.predictions.iloc[a_range]
        self.reset_index(a.predictions)
        b.predictions = self.predictions.iloc[b_range]
        self.reset_index(b.predictions)

        return split_index, a, b

    def filter(self, f):
        a, b = super(Prediction, self).filter(f)

        a.predictions = self.predictions.ix[f]
        self.reset_index(a.predictions)
        b.predictions = self.predictions.ix[~f]
        self.reset_index(b.predictions)

        return a, b

    def merge(self, b):
        a = super(Prediction, self).merge(b)

        a.predictions = pd.concat([self.predictions, b.predictions], axis=0)
        self.reset_index(a.predictions)

        return a