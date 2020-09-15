from genre_classifier.config import config as model_config

from typing import List, BinaryIO, Tuple

from api.config import get_logger
_logger = get_logger(logger_name=__name__)

def validate_inputs(input_data):
    """ make sure that every file is in an allowed format """

    errors = {}
    validated_data = {}

    for k,v in input_data.items():
        if v.filename.split('.')[-1] in model_config.ACCEPTED_FORMATS:
            validated_data[k] = v
        else:
            errors[k] = 'File extension not accepted'

    return validated_data, errors
