import base64
import json
import pandas as pd
import pymorphy2
import nltk
import sentence_transformers
from flask import make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import gensim
import gensim.models as models
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.utils import shuffle
import zipfile
import sys
import requests, io
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from src.nlp import Common

nltk.download('punkt')
nltk.download('stopwords')

punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
model_types = ["transformer", "gensim"]


class ModelResearcher:
    def __init__(self):
        self.models = {}
        self.model = None

    def load(self, path, model_type):
        try:
            if model_type == "gensim":
                self.models.setdefault(path, models.ldamodel.LdaModel.load(path))
            elif model_type == "transformer":
                self.models.setdefault(path, SentenceTransformer(path))
            self.model = self.models[path]
            return True
        except Exception as e:
            print(e)
            return False

    def predict_gensim_two_texts(self, text1, text2, model_name, round_number=4):
        self.model = self.models[model_name]
        first = Common.preprocess(text1, punctuation_marks, stop_words, morph)
        second = Common.preprocess(text2, punctuation_marks, stop_words, morph)
        print(first, second)
        sim = self.model.wv.n_similarity(first, second)
        return round(sim, round_number)

    def predict_transfomer_two_texts(self, text_1, text_2, model_name, round_number=4):
        sentences_rp = Common.sent_preprocess(text_1)
        sentences_proj = Common.sent_preprocess(text_2)
        self.model = self.models[model_name]

        def comp(e):
            return e['cos_sim']

        sentences_proj_embeddings = []
        for sentence_proj in sentences_proj:
            sentences_proj_embeddings += [self.model.encode(sentence_proj, convert_to_tensor=True)]

        max_sims = []
        for sentence_rp in sentences_rp:
            sentence_rp_embedding = self.model.encode(sentence_rp, convert_to_tensor=True)
            sim = []
            # print(f'RP sentence: {sentence_rp[0]}')
            for i in range(len(sentences_proj_embeddings)):
                sim += [{'proj': sentences_proj[i],
                         'cos_sim': float(sentence_transformers.util.cos_sim(sentence_rp_embedding,
                                                                             sentences_proj_embeddings[i]))
                         }]

            # sim.sort(key=comp, reverse=True)
            max_sims.append(max(sim, key=comp)['cos_sim'])
            # print(np.round(np.mean(max_sims), round_number))
        return np.round(np.mean(max_sims), round_number)

    def predict_sim_two_texts(self, text1, text2, model_name, model_type, round_number=4):
        if model_type == "transformer":
            return self.predict_transfomer_two_texts(text1, text2, model_name)
        elif model_type == "gensim":
            return self.predict_gensim_two_texts(text1, text2, model_name)

    def preprocess_and_save_pairs(self, data_df: pd.DataFrame, text_field_1, text_field_2, path=None):
        data_df['preprocessed_' + text_field_1] = data_df.apply(
            lambda row: Common.preprocess(row[text_field_1], punctuation_marks, stop_words, morph), axis=1)
        data_df['preprocessed_' + text_field_2] = data_df.apply(
            lambda row: Common.preprocess(row[text_field_2], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field_1, text_field_2], axis=1)
        data_df_preprocessed.reset_index(drop=True, inplace=True)
        if path is not None:
            data_df_preprocessed.to_json(path)
        return data_df_preprocessed

    def train(self, data_df: pd.DataFrame, model_path, model="w2v"):
        if model == "w2v":
            train_part = data_df['preprocessed_texts']
            self.model = gensim.models.Word2Vec(sentences=train_part, min_count=5, vector_size=50, epochs=10)
            self.model.save(model_path + model)
        elif model == "fast_text":
            train_part = data_df['preprocessed_texts']
            self.model = gensim.models.FastText(sentences=train_part, min_count=5, vector_size=50, epochs=10)
            # self.model.build_vocab(corpus_iterable=train_part)
            # self.model.train(corpus_iterable=train_part, total_examples=len(train_part), epochs=10)
            self.model.save(model_path + model)
        return

    def predict_sentences_similarity(self, sentences_1: pd.Series, sentences_2: pd.Series):
        if sentences_1.size != sentences_2.size:
            return None
        else:
            if self.model is not None:
                sentences_sim = np.zeros(sentences_1.size)
                sz = sentences_1.size
                for i in range(sz):
                    sentences_1_words = [w for w in sentences_1[i] if w in self.model.wv.index_to_key]
                    sentences_2_words = [w for w in sentences_2[i] if w in self.model.wv.index_to_key]
                    sim = self.model.wv.n_similarity(sentences_1_words, sentences_2_words)
                    sentences_sim[i] = sim

                return list(sentences_sim)
            else:
                return None

    def get_train_test_dfs_for_f1(self, df_match, df_unmatch):
        df_match = shuffle(df_match)
        df_unmatch = shuffle(df_unmatch)

        df_train_f1 = pd.concat(
            [pd.DataFrame(df_match[0:len(df_match) // 2]), pd.DataFrame(df_unmatch[0:len(df_unmatch) // 2])], axis=0)
        df_test_f1 = pd.concat(
            [pd.DataFrame(df_match[len(df_match) // 2:]), pd.DataFrame(df_unmatch[len(df_unmatch) // 2:])], axis=0)
        return df_train_f1, df_test_f1

    def maximize_f1_score(self, sentences_1: pd.Series, sentences_2: pd.Series, df, model_name, model_type, LOO=False,
                          step=0.02):
        print(model_name, LOO)
        if sentences_1.size != sentences_2.size:
            return None
        else:
            threshold = 0
            thresholds = []
            f1_score = 0
            h = step
            steps = np.linspace(0, 1, num=int(1 / h))
            steps = np.round(steps, 2)
            sim = []
            if model_type == "gensim":
                sim = self.predict_sentences_similarity(sentences_1, sentences_2)
            else:
                for i in range(len(sentences_1)):
                    sim += [self.predict_sim_two_texts(sentences_1[i], sentences_2[i], model_name=model_name,
                                                       model_type="transformer")]

            steps, thresholds, f1_score, cutoff = Common.max_f1_score(sim, df, step=0.02)

            print(f'usual f1-score: {f1_score}')

            fig = plt.figure(figsize=(10, 8))
            plt.grid(True)
            plt.xlabel("Cutoff")
            plt.ylabel("F1-score")
            plt.plot(steps, thresholds)
            plt.plot(cutoff, f1_score, 'r*')
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            encoded_img = base64.b64encode(buf.getbuffer()).decode("ascii")
            image_url = f"data:image/png;base64,{encoded_img}"

            res = {}

            if LOO:
                predictions = []
                for i in range(len(sim)):
                    current_df = df.drop(i).reset_index().drop(labels='index', axis=1)
                    current_sim = sim[:i] + sim[i + 1:]
                    steps, thresholds, f1_temp, cutoff_temp = Common.max_f1_score(current_sim, current_df)
                    print(f1_temp, cutoff_temp)
                    if sim[i] >= cutoff_temp:
                        predictions.append(True)
                    else:
                        predictions.append(False)
                f1_score_loo = Common.calc_f1_score_loo(lambda: Common.get_states_loo(predictions, df))
                print(f1_score, f1_score_loo)
                res.setdefault("f1-score_loo", f1_score_loo)

            res.setdefault("cutoff", cutoff)
            res.setdefault("f1-score", f1_score)
            res.setdefault("image", image_url)
            return res