import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from python_speech_features import mfcc

from typing import List, Tuple

class MFCCTransformer(BaseEstimator, TransformerMixin):
    """Transform wav file into its MFCC transformed values"""

    def __init__(self, subsample_factor=1) -> None:
        self.subsample_factor = subsample_factor

    def fit(self, X, y=None):
        return self

    def transform_single(self, X: Tuple[int, np.ndarray]) -> np.ndarray:
        samplerate, wavedata = X[0], X[1]
        return mfcc(
            wavedata[::self.subsample_factor],
            int(samplerate/self.subsample_factor),
            nfft=int(1024/self.subsample_factor)
        )

    def transform(self, X: List[Tuple[int, np.ndarray]]) -> List[np.ndarray]:
        X_transformed = [self.transform_single(x) for x in X]
        return X_transformed

class TimeDimReducer(BaseEstimator, TransformerMixin):
    """Reduce an array of MFCC transformed values along the time dimension"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X:List[np.ndarray]) -> pd.DataFrame:
        reduced = [np.median(x, axis=0, keepdims=True) for x in X]
        return pd.DataFrame(np.concatenate(reduced, axis=0))
