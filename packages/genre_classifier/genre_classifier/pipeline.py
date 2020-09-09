from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from genre_classifier.processing import features, preprocessing
from genre_classifier.config import config

import logging

_logger = logging.getLogger(__name__)

pipe = Pipeline(
    [
        (
            "WAV_reader",
             preprocessing.WAVReader(),
        ),
        (
            "MFCC_transformer",
            features.MFCCTransformer(subsample_factor=config.SUBSAMPLE_FACTOR),
        ),
        (
            "time_dim_reducer",
            features.TimeDimReducer()
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "Classifier",
            KNeighborsClassifier(n_neighbors=config.NNEIGHBORS)
        )
    ]
)
