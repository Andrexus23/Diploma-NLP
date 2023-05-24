import configparser
import logging
import sys
import time
import traceback
import pymorphy2
import sklearn.utils
from flask import Flask, jsonify, request
import pandas as pd
from flask_restful import Api, Resource, reqparse
import json
from flask_swagger_ui import get_swaggerui_blueprint
from nltk.corpus import stopwords

config = configparser.ConfigParser()
config.read(sys.argv[-1])
sys.path.append(config['path']['SERVICE_PATH'])

import Common
import ModelResearcher as MR


SWAGGER_URL = config['path']['SWAGGER_URL']  # URL для размещения SWAGGER_UI
API_URL = config['path']['API_URL']
TRAIN_PATH = config['path']['TRAIN_PATH']
PREPROCESSED_PATH = config['path']['PREPROCESSED_PATH']
MODELS_GENSIM_PATH = config['path']['MODELS_GENSIM_PATH']
MODELS_TRANSFORMER_PATH = config['path']['MODELS_TRANSFORMER_PATH']
UPLOADED = config['path']['UPLOADED']
ALLOWED_MODELS_GENSIM = json.loads(config['models']['gensim'])
ALLOWED_MODELS_TRANSFORMER = json.loads(config['models']['transformer'])
MR.model_types = json.loads(config['models']['model_types'])
MR.redis_host = config['hosts']['redis_host']
MR.redis_port = int(config['ports']['redis_port'])
server_host = config['hosts']['server_host']
server_port = int(config['ports']['server_port'])

print(API_URL)

punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
app = Flask(__name__)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
)
app.register_blueprint(swaggerui_blueprint)
api = Api(app)


@app.route("/api/docs/train/uploadDataset", methods=['POST'])
def upload_train_data():
    if len(request.files):
        dataset = request.files.get("file")
        dataset.save(TRAIN_PATH)
        dataset.close()
        return {"Success": "File's been successfully uploaded"}
    return {"Error": "Couldn't load file"}


@app.route("/api/docs/train/<string:name>", methods=['GET'])
def train_model(name):
    res = None
    modelResearcher = MR.ModelResearcher()
    preprocessed_texts = None
    try:
        preprocessed_texts = pd.read_json(PREPROCESSED_PATH)
    except FileNotFoundError:
        start = time.perf_counter()
        texts = pd.read_json(TRAIN_PATH)
        preprocessed_texts = Common.preprocess_and_save(texts, PREPROCESSED_PATH)
        end = time.perf_counter()
        res = f'Preprocessing time: {end - start:0.4f} secs'
        print(res)

    if name not in ALLOWED_MODELS_GENSIM:
        return jsonify({"Error": "Incorrect model name"})
    try:
        start = time.perf_counter()
        modelResearcher.train(preprocessed_texts, model=name, model_path=MODELS_GENSIM_PATH)
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
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})
    modelResearcher = MR.ModelResearcher()
    type_ = None
    if name in ALLOWED_MODELS_GENSIM:
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    elif name in ALLOWED_MODELS_TRANSFORMER:
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    else:
        return jsonify({"Error": "Something went wrong"})
    try:
        values = request.values.values()
        first = next(values)
        second = next(values)
        exists = modelResearcher.load(path, type_)
        if not exists:
            return {"Error": "No model: you should train or download it"}
        sim = None
        if type_ == "gensim":
            first = Common.preprocess(first, stop_words, punctuation_marks, morph)
            second = Common.preprocess(second, stop_words, punctuation_marks, morph)
            sim = round(
                modelResearcher.predict_sentences_similarity(pd.Series([first]), pd.Series([second]), model_name=name)[
                    0], 4)
        elif type_ == "transformer":
            sim = modelResearcher.predict_transfomer_two_texts(first, second, path, 4)

        if sim < 0:
            sim = 0.0
        print(sim)
        return jsonify({"Texts' similarity": str(sim)})
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"Error": "Something went wrong"})


@app.route("/api/docs/maximize-f1-score/<string:name>", methods=['POST'])
def maximize_f1_score(name):
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})

    if name in ALLOWED_MODELS_GENSIM:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    else:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    exists = modelResearcher.load(path, model_type=type_)

    if not exists:
        return {"Error": "No model: you should train or download it"}
    dataset = request.files['file']
    filename = UPLOADED + "/" + dataset.filename
    dataset.save(filename)
    dataset.close()
    try:
        start = time.perf_counter()
        df = pd.read_json(filename)
        if name in ALLOWED_MODELS_GENSIM:
            df = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
            res = modelResearcher.maximize_f1_score(df["preprocessed_text_rp"], df["preprocessed_text_proj"], df,
                                                    model_name=name,
                                                    model_type="gensim",
                                                    step=0.02)
        else:
            res = modelResearcher.maximize_f1_score(df["text_rp"], df["text_proj"], df,
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
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})

    if name in ALLOWED_MODELS_GENSIM:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    else:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    exists = modelResearcher.load(path, model_type=type_)

    if not exists:
        return {"Error": "No model: you should train or download it"}
    dataset = request.files['file']
    filename = UPLOADED + "/" + dataset.filename
    dataset.save(filename)
    dataset.close()
    try:
        df = pd.read_json(filename)
        df = sklearn.utils.shuffle(df)
        df = df.reset_index().drop(labels='index', axis=1)
        if name in ALLOWED_MODELS_GENSIM:
            df = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
            res = modelResearcher.maximize_f1_score_loo(df["preprocessed_text_rp"], df["preprocessed_text_proj"], df,
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
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})

    if name in ALLOWED_MODELS_GENSIM:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    else:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    exists = modelResearcher.load(path, model_type=type_)

    if not exists:
        return {"Error": "No model: you should train or download it"}
    dataset = request.files['file']
    filename = UPLOADED + "/" + dataset.filename
    dataset.save(filename)
    dataset.close()
    res = None

    try:
        df = pd.read_json(filename)
        df_train_f1, df_test_f1 = modelResearcher.get_train_test_dfs_for_f1(df)
        if name in ALLOWED_MODELS_GENSIM:
            df_train_f1 = modelResearcher.preprocess_and_save_pairs(df_train_f1, 'text_rp', 'text_proj')
            df_test_f1 = modelResearcher.preprocess_and_save_pairs(df_test_f1, 'text_rp', 'text_proj')
            res = modelResearcher.maximize_f1_score_train_test(df_train_f1, df_test_f1,
                                                               model_name=name,
                                                               model_type="gensim",
                                                               field_1="preprocessed_text_rp",
                                                               field_2="preprocessed_text_proj",
                                                               step=0.02)
        elif name in ALLOWED_MODELS_TRANSFORMER:
            res = modelResearcher.maximize_f1_score_train_test(df_train_f1, df_test_f1,
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
    if (name not in ALLOWED_MODELS_GENSIM) and \
            (name not in ALLOWED_MODELS_TRANSFORMER):
        return jsonify({"Error": "No such model in service"})

    if name in ALLOWED_MODELS_GENSIM:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_GENSIM_PATH + name
        type_ = "gensim"
    else:
        modelResearcher = MR.ModelResearcher()
        path = MODELS_TRANSFORMER_PATH + name
        type_ = "transformer"
    exists = modelResearcher.load(path, model_type=type_)

    if not exists:
        return {"Error": "No model: you should train or download it"}
    dataset = request.files['file']
    filename = UPLOADED + "/" + dataset.filename
    dataset.save(filename)
    dataset.close()
    df = pd.read_json(filename)
    res = None
    if name in ALLOWED_MODELS_GENSIM:
        df_preprocessed = modelResearcher.preprocess_and_save_pairs(df, 'text_rp', 'text_proj')
        res = modelResearcher.match_texts_from_corpus(df_preprocessed,
                                                      model_name=name,
                                                      model_type="gensim",
                                                      field_1="preprocessed_text_rp",
                                                      field_2="preprocessed_text_proj")

    elif name in ALLOWED_MODELS_TRANSFORMER:
        res = modelResearcher.match_texts_from_corpus(df,
                                                      model_name=name,
                                                      model_type="transformer",
                                                      field_1="text_rp",
                                                      field_2="text_proj")
    df.insert(loc=4, column='score', value=res)
    return df.to_json(orient="records", force_ascii=False)


@app.route("/api/docs/get-list-of-allowed-models", methods=['GET'])
def get_list_models():
    return jsonify(ALLOWED_MODELS_GENSIM + ALLOWED_MODELS_TRANSFORMER)


if __name__ == '__main__':
    app.run(port=server_port, host=server_host, debug=True)
