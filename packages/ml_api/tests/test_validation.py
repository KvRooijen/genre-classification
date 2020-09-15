from genre_classifier.config import config as model_config
from genre_classifier.processing.data_management import load_dataset, close_dataset
from api.config import get_logger

import json

_logger = get_logger(logger_name=__name__)

def test_prediction_endpoint_validation(flask_test_client):
    #Given
    test_data, test_labels = load_dataset(dataset_folder = model_config.DATASET_DATAVALIDATION_DIR)

    # When
    data = {}
    data['wavfile'] = (test_data[0], test_data[0].name)
    data['aupfile'] = (test_data[1], test_data[1].name)
    data['mp3file'] = (test_data[2], test_data[2].name)

    response = flask_test_client.post('/v1/predict/knn',
                                      data=data,
                                      content_type='multipart/form-data')#
    close_dataset(data=test_data)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)

    # Check correct number of errors removed
    assert len(response_json.get('predictions')) + len(
        response_json.get('errors')) == len(test_data)

    _logger.info(response_json.get('predictions'))
    assert response_json.get('predictions') == ['disco']

    _logger.info(response_json.get('errors'))
    assert 'aupfile' in response_json.get('errors').keys()
    assert 'mp3file' in response_json.get('errors').keys()
