import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.pipeline import Pipeline

from genre_classifier.config import config
from genre_classifier import __version__ as _version

from typing import List, Tuple, BinaryIO

import logging


_logger = logging.getLogger(__name__)

def load_dataset(*, dataset_folder: str) -> Tuple[List[BinaryIO], List[str]]:
    X, y = [], []
    files = os.listdir(dataset_folder)

    for file_path in files:
        genre = file_path.split('.')[0]
        X += [open(os.path.join(dataset_folder, file_path), 'rb')]
        y += [genre]
    
    return X,y

def close_dataset(*, data:List[BinaryIO]) -> None:
    for f in data:
        f.close()

    """
def load_dataset(*, base_path: str) -> Tuple[List[Tuple[np.ndarray, int]], List[str]]:
    X, y = [], []

    genres = os.listdir(base_path)
    for genre in genres:
        genre_path = os.path.join(base_path, genre)
        file_paths = os.listdir(genre_path)
        for file_path in file_paths:

            samplerate, wavedata = wavfile.read(os.path.join(genre_path, file_path))
            X += [(wavedata, samplerate)]
            y += [genre]
    return X,y
    """


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def json_serialize(*, data: List[Tuple[np.ndarray, int]]) -> str:
    return json.dumps(data, cls=NumpyEncoder)

def json_restore(*, json: str) -> List[Tuple[np.ndarray, int]]:
    print(json)
    return None
