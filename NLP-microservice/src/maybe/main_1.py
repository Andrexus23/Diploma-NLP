from flask import Flask, jsonify
from flasgger import Swagger
import pandas as pd
import numpy as np
from flask_restful import Api, Resource
import json
from flasgger.utils import swag_from
from urllib.parse import unquote

from flask_swagger_ui import get_swaggerui_blueprint
from unidecode import unidecode

app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)
#
# df = pd.read_json('../data/match.json')
#
#
# @app.route('/pairs/<int:pair_number>')
# def get(pair_number):  # чтобы обрабатывать некоторый запрос, нужен метод с таким же названием
#     if pair_number > 0 and pair_number < len(df):
#         encoded = json.dumps(df.iloc[pair_number].to_dict(), ensure_ascii=False)
#         return encoded
#     else:
#         return json.dumps(df.to_dict(), ensure_ascii=False)
#
#
# # @app.route("/api/pairs/<int:pair_number>", endpoint='Main.get', methods=['GET', 'POST'])
# # api.add_resource(Main, "/api/pairs/<int:pair_number>")  # обработка нужного url-адреса
# # api.init_app(app)
# # if __name__ == "__main__":
# app.run(debug=True)
# class UploadTrainingDataset(Schema):
#     res = fields.String(description="form")


# # создаем приложение на Flask
app = Flask(__name__)

# конфигурация swagger
SWAGGER_URL = '/train-model'
API_URL = '/swagger'
SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "NLP microservice"
    }
)

app.register_blueprint(SWAGGER_BLUEPRINT, url_prefix=SWAGGER_URL)

app.run(debug=True)
