from routes import *

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with app.app_context():
            load()
            swaggerui_blueprint = get_swaggerui_blueprint(
                g.SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
                g.API_URL,
                config={  # Swagger UI config overrides
                    'app_name': "Test application"
                },
            )
            app.register_blueprint(swaggerui_blueprint)
            api = Api(app)
            CORS(app, support_credentials=True)
            app.run(port=g.server_port, host='0.0.0.0', debug=True)
