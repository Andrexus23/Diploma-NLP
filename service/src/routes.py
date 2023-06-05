import logging
import sys
import time
import traceback

import flask
import pymorphy2
import sklearn.utils
from flask import Flask, jsonify, request, g
import pandas as pd
from flask_restful import Api, Resource, reqparse
import json
import configparser
from flask_swagger_ui import get_swaggerui_blueprint
from nltk.corpus import stopwords
import Common
import ModelResearcher as MR
from flask_cors import CORS

punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()

app = Flask(__name__)

with app.app_context():
    @app.before_request
    def load():
        g.config = configparser.ConfigParser()
        g.config.read(sys.argv[-1])
        g.SWAGGER_URL = g.config['path']['SWAGGER_URL']  # URL для размещения SWAGGER_UI
        g.API_URL = g.config['path']['API_URL']
        g.TRAIN_PATH = g.config['path']['TRAIN_PATH']
        g.PREPROCESSED_PATH = g.config['path']['PREPROCESSED_PATH']
        g.MODELS_GENSIM_PATH = g.config['path']['MODELS_GENSIM_PATH']
        g.MODELS_TRANSFORMER_PATH = g.config['path']['MODELS_TRANSFORMER_PATH']
        g.UPLOADED = g.config['path']['UPLOADED']
        g.IMAGE_PATH = g.config['path']['IMAGE_PATH']
        g.ALLOWED_MODELS_GENSIM = json.loads(g.config['models']['gensim'])
        g.ALLOWED_MODELS_TRANSFORMER = json.loads(g.config['models']['transformer'])
        MR.model_types = json.loads(g.config['models']['model_types'])
        MR.redis_host = g.config['hosts']['redis_host']
        MR.redis_port = int(g.config['ports']['redis_port'])
        g.server_host = g.config['hosts']['server_host']
        g.server_port = int(g.config['ports']['server_port'])



    @app.route("/api/docs/train/uploadDataset", methods=['POST'])
    def upload_train_data():
        if len(request.files):
            dataset = request.files.get("file")
            dataset.save(g.TRAIN_PATH)
            dataset.close()
            return {"Success": "File's been successfully uploaded"}
        return {"Error": "Couldn't load file"}

    @app.route("/api/docs/get-image/<string:name>", methods=['GET'])
    def get_image(name):
        try:
            res = flask.send_file(path_or_file=g.IMAGE_PATH + name, mimetype="image/png")
            return res
        except FileNotFoundError:
            return {"Error": "No such image"}

    @app.route("/api/docs/train/<string:name>", methods=['GET'])
    def train_model(name):
        res = None
        modelResearcher = MR.ModelResearcher()
        preprocessed_texts = None
        try:
            preprocessed_texts = pd.read_json(g.PREPROCESSED_PATH)
        except FileNotFoundError:
            start = time.perf_counter()
            texts = pd.read_json(g.TRAIN_PATH)
            preprocessed_texts = Common.preprocess_and_save(texts, g.PREPROCESSED_PATH)
            end = time.perf_counter()
            res = f'Preprocessing time: {end - start:0.4f} secs'
            print(res)

        if name not in g.ALLOWED_MODELS_GENSIM:
            return jsonify({"Error": "Incorrect model name"})
        try:
            start = time.perf_counter()
            modelResearcher.train(preprocessed_texts, model=name, model_path=g.MODELS_GENSIM_PATH)
            end = time.perf_counter()
            res = f'Model training time: {end - start:0.4f} secs'
            return jsonify({"Success": res})
        except Exception as e:
            logging.error(traceback.format_exc())
            return jsonify({"Error": "Cannot train model"})


    @app.route("/api/docs/match2texts/<string:name>", methods=['POST'])
    def match2texts(name):
        modelResearcher = None
        path = None
        if (name not in g.ALLOWED_MODELS_GENSIM) and \
                (name not in g.ALLOWED_MODELS_TRANSFORMER):
            return jsonify({"Error": "No such model in service"})
        modelResearcher = MR.ModelResearcher()
        type_ = None
        if name in g.ALLOWED_MODELS_GENSIM:
            path = g.MODELS_GENSIM_PATH + name
            type_ = "gensim"
        elif name in g.ALLOWED_MODELS_TRANSFORMER:
            path = g.MODELS_TRANSFORMER_PATH + name
            type_ = "transformer"
        else:
            return jsonify({"Error": "Something went wrong"})
        try:
            values = request.values.values()
            first = next(values)
            second = next(values)
            exists = modelResearcher.load(path, name, type_)
            if not exists:
                return {"Error": "No model: you should train or download it"}
            sim = None
            if type_ == "gensim":
                first = Common.preprocess(first, stop_words, punctuation_marks, morph)
                second = Common.preprocess(second, stop_words, punctuation_marks, morph)
                sim = round(
                    modelResearcher.predict_sentences_similarity(pd.Series([first]), pd.Series([second]),
                                                                 model_name=name)[
                        0], 4)
            elif type_ == "transformer":
                sim = modelResearcher.predict_transfomer_two_texts(first, second, path, 4)

            if sim < 0:
                sim = 0.0
            return jsonify({"Texts' similarity": str(sim)})
        except Exception as e:
            logging.error(traceback.format_exc())
            return jsonify({"Error": "Something went wrong"})


    @app.route("/api/docs/maximize-f1-score/<string:name>", methods=['POST'])
    def maximize_f1_score(name):
        if (name not in g.ALLOWED_MODELS_GENSIM) and \
                (name not in g.ALLOWED_MODELS_TRANSFORMER):
            return jsonify({"Error": "No such model in service"})

        if name in g.ALLOWED_MODELS_GENSIM:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_GENSIM_PATH + name
            type_ = "gensim"
        else:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_TRANSFORMER_PATH + name
            type_ = "transformer"
        exists = modelResearcher.load(path, name, model_type=type_)

        if not exists:
            return {"Error": "No model: you should train or download it"}
        dataset = request.files['file']
        filename = g.UPLOADED + "/" + dataset.filename
        dataset.save(filename)
        dataset.close()
        try:
            start = time.perf_counter()
            df = pd.read_json(filename)
            if name in g.ALLOWED_MODELS_GENSIM:
                df = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
                res = modelResearcher.maximize_f1_score(df["preprocessed_text_rp"], df["preprocessed_text_proj"], df,
                                                        image_path=g.IMAGE_PATH,
                                                        model_name=name,
                                                        model_type="gensim",
                                                        step=0.02)
            else:
                res = modelResearcher.maximize_f1_score(df["text_rp"], df["text_proj"], df,
                                                        image_path=g.IMAGE_PATH,
                                                        model_name=name,
                                                        model_type="transformer",
                                                        step=0.02)
            end = time.perf_counter()
            print("time: " + str(end - start))

            return res
        except Exception as e:
            logging.error(traceback.format_exc())
            return jsonify({"Error": "Something went wrong"})


    @app.route("/api/docs/maximize-f1-score-crossvalid-loo/<string:name>", methods=['POST'])
    def maximize_f1_score_loo(name):
        if (name not in g.ALLOWED_MODELS_GENSIM) and \
                (name not in g.ALLOWED_MODELS_TRANSFORMER):
            return jsonify({"Error": "No such model in service"})

        if name in g.ALLOWED_MODELS_GENSIM:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_GENSIM_PATH + name
            type_ = "gensim"
        else:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_TRANSFORMER_PATH + name
            type_ = "transformer"
        exists = modelResearcher.load(path, name, model_type=type_)

        if not exists:
            return {"Error": "No model: you should train or download it"}
        df = get_df('file')
        try:
            if name in g.ALLOWED_MODELS_GENSIM:
                df = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
                res = modelResearcher.maximize_f1_score_loo(df["preprocessed_text_rp"], df["preprocessed_text_proj"],
                                                            df,
                                                            model_name=name,
                                                            model_type="gensim",
                                                            step=0.02)
            else:
                res = modelResearcher.maximize_f1_score_loo(df["text_rp"], df["text_proj"], df,
                                                            model_name=name,
                                                            model_type="transformer",
                                                            step=0.02)
            return res
        except Exception as e:
            logging.error(traceback.format_exc())
            return jsonify({"Error": "Something went wrong"})


    @app.route("/api/docs/maximize-f1-score-crossvalid-train-test/<string:name>", methods=['POST'])
    def maximize_f1_score_train_test(name):
        if (name not in g.ALLOWED_MODELS_GENSIM) and \
                (name not in g.ALLOWED_MODELS_TRANSFORMER):
            return jsonify({"Error": "No such model in service"})

        if name in g.ALLOWED_MODELS_GENSIM:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_GENSIM_PATH + name
            type_ = "gensim"
        else:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_TRANSFORMER_PATH + name
            type_ = "transformer"
        exists = modelResearcher.load(path, name, model_type=type_)

        if not exists:
            return {"Error": "No model: you should train or download it"}
        res = None

        try:
            df = get_df('file')
            df_train_f1, df_test_f1 = modelResearcher.get_train_test_dfs_for_f1(df)
            if name in g.ALLOWED_MODELS_GENSIM:
                df_train_f1 = modelResearcher.preprocess_and_save_pairs(df_train_f1, 'text_rp', 'text_proj')
                df_test_f1 = modelResearcher.preprocess_and_save_pairs(df_test_f1, 'text_rp', 'text_proj')
                res = modelResearcher.maximize_f1_score_train_test(df_train_f1, df_test_f1,
                                                                   image_path=g.IMAGE_PATH,
                                                                   model_name=name,
                                                                   model_type="gensim",
                                                                   field_1="preprocessed_text_rp",
                                                                   field_2="preprocessed_text_proj",
                                                                   step=0.02)
            elif name in g.ALLOWED_MODELS_TRANSFORMER:
                res = modelResearcher.maximize_f1_score_train_test(df_train_f1, df_test_f1,
                                                                   image_path=g.IMAGE_PATH,
                                                                   model_name=name,
                                                                   model_type="transformer",
                                                                   field_1="text_rp",
                                                                   field_2="text_proj",
                                                                   step=0.02)

            return res
        except Exception as e:
            logging.error(traceback.format_exc())
            return jsonify({"Error": "Something went wrong"})

    @app.route("/api/docs/match_texts_from_corpus/<string:name>", methods=['POST'])
    def match_two_texts_from_corpus(name):
        if (name not in g.ALLOWED_MODELS_GENSIM) and \
                (name not in g.ALLOWED_MODELS_TRANSFORMER):
            return jsonify({"Error": "No such model in service"})

        if name in g.ALLOWED_MODELS_GENSIM:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_GENSIM_PATH + name
            type_ = "gensim"
        else:
            modelResearcher = MR.ModelResearcher()
            path = g.MODELS_TRANSFORMER_PATH + name
            type_ = "transformer"
        exists = modelResearcher.load(path, name, model_type=type_)

        if not exists:
            return {"Error": "No model: you should train or download it"}
        df = get_df('file')
        res = None
        if name in g.ALLOWED_MODELS_GENSIM:
            df_preprocessed = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
            res = modelResearcher.match_texts_from_corpus(df_preprocessed,
                                                          model_name=name,
                                                          model_type="gensim",
                                                          field_1="preprocessed_text_rp",
                                                          field_2="preprocessed_text_proj")

        elif name in g.ALLOWED_MODELS_TRANSFORMER:
            res = modelResearcher.match_texts_from_corpus(df,
                                                          model_name=name,
                                                          model_type="transformer",
                                                          field_1="text_rp",
                                                          field_2="text_proj")
        df.insert(loc=4, column='score', value=res)
        return df.to_json(orient="records", force_ascii=False)

    def get_df(name):
        dataset = request.files[name]
        filename = g.UPLOADED + "/" + dataset.filename
        dataset.save(filename)
        dataset.close()
        return pd.read_json(filename)


    @app.route("/api/docs/get-ROC-AUC", methods=["POST"])
    def get_roc_auc():
        modelResearcher = MR.ModelResearcher()
        df = get_df('file')

        for model_name in g.ALLOWED_MODELS_GENSIM:
            modelResearcher.load(g.MODELS_GENSIM_PATH + model_name, model_name,  model_type="gensim")
        for model_name in g.ALLOWED_MODELS_TRANSFORMER:
            modelResearcher.load(g.MODELS_TRANSFORMER_PATH + model_name, model_name, model_type="transformer")

        res = modelResearcher.get_roc_auc(df, image_path=g.IMAGE_PATH, field_1="text_rp", field_2="text_proj")
        return res

    @app.route("/api/docs/get-list-of-allowed-models", methods=['GET'])
    def get_list_models():
        return jsonify(g.ALLOWED_MODELS_GENSIM + g.ALLOWED_MODELS_TRANSFORMER)
