from genre_classifier.config import config as model_config
from genre_classifier.processing.data_management import load_dataset
from genre_classifier import __version__ as _version

import json
import math


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data, test_labels = load_dataset(base_path=model_config.DATASET_DIR)

    # When
    data = {}
    data['file_0'] = (test_data[0], test_data[0].name)
    data['file_1'] = (test_data[1], test_data[1].name)
    response = flask_test_client.post('/v1/predict/knn',
                                      data=data,
                                      content_type='multipart/form-data')#
    # Then
    assert response.status_code == 200

    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']

    assert prediction == ['metal', 'country']
    assert response_version == _version
