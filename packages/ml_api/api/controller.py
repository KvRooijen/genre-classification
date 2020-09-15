from flask import Blueprint, request, jsonify
from genre_classifier.predict import make_prediction

from api.config import get_logger

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'

@prediction_app.route('/v1/predict/knn', methods=['POST'])
def predict():
    if request.method == 'POST':

        audio_files = list(request.files.values())
        _logger.info(f'Number of input files: {len(audio_files)}')

        result = make_prediction(input_data=audio_files)
        _logger.info(f'Outputs: {result}')

        return jsonify({
            'predictions':list(result['predictions']),
            'version':result['version']
        })
