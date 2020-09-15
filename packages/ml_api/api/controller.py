from flask import Blueprint, request, jsonify
from genre_classifier.predict import make_prediction
from genre_classifier import __version__ as _version

from api.config import get_logger
from api import __version__ as api_version
from api.validation import validate_inputs


_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'

@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})

@prediction_app.route('/v1/predict/knn', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract POST data from request body
        input_data = request.files

        # Validate the input data
        validated_data, errors = validate_inputs(request.files)
        _logger.debug(f'Number of validated files: {len(validated_data.keys())}')
        _logger.debug(f'Number of erroneous files: {len(errors.keys())}')

        # Model prediction
        audio_files = list(validated_data.values())
        result = make_prediction(input_data=audio_files)
        _logger.debug(f'Outputs: {result}')

        return jsonify({
            'predictions':list(result['predictions']),
            'version':result['version'],
            'errors': errors
        })
