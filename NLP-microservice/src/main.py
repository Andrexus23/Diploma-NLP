from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import numpy as np
from flask_restful import Api, Resource, reqparse
import json
from flasgger.utils import swag_from
from urllib.parse import unquote
from flask_swagger_ui import get_swaggerui_blueprint
from unidecode import unidecode

SWAGGER_URL = '/api/docs'  # URL для размещения SWAGGER_UI
API_URL = '/static/swagger.json'

app = Flask(__name__)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
)

app.register_blueprint(swaggerui_blueprint)

dataset_post_args = reqparse.RequestParser()
api = Api(app)

dataset_post_args.add_argument("dataset", required=True)


@app.route("/api/docs/train-w2v/uploadDataset", methods=['POST'])
def train_model():
    TRAIN_PATH = './posted/train.json'
    if len(request.files):
        dataset = request.files.get("file")
        dataset.save(TRAIN_PATH)
        dataset.close()
        file = json.load(open(TRAIN_PATH))
        return jsonify(file)

    return {"Error": "Couldn't load file"}



    # try:
    #     with request.files[]
    return {}


if __name__ == '__main__':
    app.run(port=5000, debug=True)
