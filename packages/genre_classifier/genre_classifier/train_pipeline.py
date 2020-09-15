import numpy as np
from sklearn.model_selection import train_test_split

from genre_classifier import pipeline
from genre_classifier.processing.data_management import load_dataset, save_pipeline, close_dataset
from genre_classifier.config import config
from genre_classifier import __version__ as _version

import logging
_logger = logging.getLogger('genre_classifier')

def run_training() -> None:

    # read training data
    data_X, data_y = load_dataset(dataset_folder = config.DATASET_TRAIN_DIR)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.2, random_state=0, stratify=data_y
    )

    pipeline.pipe.fit(X_train, y_train)
    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist = pipeline.pipe)

    close_dataset(data=data_X)

if __name__ == "__main__":
    run_training()
