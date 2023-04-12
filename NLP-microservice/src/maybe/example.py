from flasgger import Swagger
from flask import Flask, request, jsonify, make_response
from datetime import datetime

from flask_restful import Api
from flask_swagger_ui import get_swaggerui_blueprint
from marshmallow import Schema, fields
from flask import Blueprint, current_app, json, request
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import Blueprint, current_app, json, request
from marshmallow import Schema, fields


class InputSchema(Schema):
    number = fields.Int(description="Число", required=True, example=5)
    power = fields.Int(description="Степень", required=True, example=2)


class OutputSchema(Schema):
    result = fields.Int(description="Результат", required=True, example=25)


def create_tags(spec):
    """ Создаем теги.

   :param spec: объект APISpec для сохранения тегов
   """
    tags = [{'name': 'math', 'description': 'Математические функции'}]

    for tag in tags:
        print(f"Добавляем тег: {tag['name']}")
        spec.tag(tag)


blueprint_power = Blueprint(name="power", import_name=__name__)


@blueprint_power.route('/power')
def power():
    """
   ---
   get:
     summary: Возводит число в степень
     parameters:
       - in: query
         schema: InputSchema
     responses:
       '200':
         description: Результат возведения в степень
         content:
           application/json:
             schema: OutputSchema
       '400':
         description: Не передан обязательный параметр
         content:
           application/json:
             schema: ErrorSchema
     tags:
       - math
   """
    args = request.args

    number = args.get('number')
    if number is None:
        return current_app.response_class(
            response=json.dumps(
                {'error': 'Не передан параметр number'}
            ),
            status=400,
            mimetype='application/json'
        )

    power = args.get('power')
    if power is None:
        return current_app.response_class(
            response=json.dumps(
                {'error': 'Не передан параметр power'}
            ),
            status=400,
            mimetype='application/json'
        )

    return current_app.response_class(
        response=json.dumps(
            {'response': int(number) ** int(power)}
        ),
        status=200,
        mimetype='application/json'
    )


def load_docstrings(spec, app):
    """ Загружаем описание API.

   :param spec: объект APISpec, куда загружаем описание функций
   :param app: экземпляр Flask приложения, откуда берем описание функций
   """
    for fn_name in app.view_functions:
        if fn_name == 'static':
            continue
        print(f'Загружаем описание для функции: {fn_name}')
        view_fn = app.view_functions[fn_name]
        spec.path(view=view_fn)


class ErrorSchema:
    pass


def get_apispec(app):
    """ Формируем объект APISpec.

   :param app: объект Flask приложения
   """
    spec = APISpec(
        title="My App",
        version="1.0.0",
        openapi_version="3.0.3",
        plugins=[FlaskPlugin(), MarshmallowPlugin()],
    )

    spec.components.schema("Input", schema=InputSchema)
    spec.components.schema("Output", schema=OutputSchema)
    spec.components.schema("Error", schema=ErrorSchema)

    create_tags(spec)

    load_docstrings(spec, app)

    return spec


app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)


SWAGGER_URL = '/docs'
API_URL = '/swagger'

swagger_ui_blueprint = get_swaggerui_blueprint(
   SWAGGER_URL,
   API_URL,
   config={
       'app_name': 'My App'
   }
)

app.run(debug=True)