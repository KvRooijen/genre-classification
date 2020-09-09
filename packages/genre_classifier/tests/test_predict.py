import math
import numpy as np

from genre_classifier.predict import make_prediction
from genre_classifier.processing.data_management import load_dataset, json_serialize

from genre_classifier.config import config

import logging
_logger = logging.getLogger(__name__)

def test_make_single_prediction():
    # Given
    test_data, test_labels = load_dataset(base_path=config.DATASET_DIR)

    # When
    subject = make_prediction(input_data = test_data[:1])

    # Then
    assert subject is not None
    assert isinstance(subject['predictions'], np.ndarray)
