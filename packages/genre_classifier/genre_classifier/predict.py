import numpy as np
import pandas as pd

from genre_classifier.processing.data_management import load_pipeline, json_restore
from genre_classifier.config import config
from genre_classifier import __version__ as _version

import logging
from typing import List, Tuple, BinaryIO

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: List[BinaryIO]) -> dict:

    prediction = _pipe.predict(input_data)
    results = {'predictions':prediction, 'version':_version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Predictions: {prediction}"
    )

    return results
