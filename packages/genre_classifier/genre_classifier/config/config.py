import pathlib

import genre_classifier

import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(genre_classifier.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

DATASET_TRAIN_DIR = PACKAGE_ROOT / "datasets" / "train"
DATASET_TEST_DIR = PACKAGE_ROOT / "datasets" / "test"
DATASET_DATAVALIDATION_DIR = PACKAGE_ROOT / "datasets" / "data_validation" 

ACCEPTED_FORMATS = ['wav']
SUBSAMPLE_FACTOR = 10
NNEIGHBORS = 20

PIPELINE_NAME = "knearestneighbors"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"
