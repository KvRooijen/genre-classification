import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from typing import List, Tuple, BinaryIO
from scipy.io import wavfile

class WAVReader(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X:List[BinaryIO]) -> List[Tuple[int, np.ndarray]]:
        return [wavfile.read(x) for x in X]
