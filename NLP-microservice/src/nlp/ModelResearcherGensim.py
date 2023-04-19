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


class ModelResearcherGensim:
    def __init__(self):
        self.model = None

    def load(self, path):
        try:
            self.model = models.ldamodel.LdaModel.load(path)
            return True
        except FileNotFoundError:
            return False

    def predict_sim_two_texts(self, text1, text2, round_number=4):
        if self.model:
            first = Common.preprocess(text1, punctuation_marks, stop_words, morph)
            second = Common.preprocess(text2, punctuation_marks, stop_words, morph)
            print(first, second)
            sim = self.model.wv.n_similarity(first, second)
            return round(sim, round_number)
        return None

    def preprocess_and_save_pairs(self, data_df: pd.DataFrame,  text_field_1, text_field_2, path=None) -> pd.DataFrame:
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

                return sentences_sim
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

    def maximize_f1_score(self, sentences_1: pd.Series, sentences_2: pd.Series, df, step=0.02):

        if sentences_1.size != sentences_2.size or self.model is None:
            return None
        else:
            threshold = 0
            thresholds = []
            max_ = 0
            h = step
            steps = np.linspace(0, 1, num=int(1 / h))
            steps = np.round(steps, 2)
            h_max = 0
            sim = self.predict_sentences_similarity(sentences_1, sentences_2)
            for i in steps:
                threshold = Common.calc_f1_score(sim, df, h)
                thresholds.append(threshold)
                if threshold > max_:
                    max_ = threshold
                    h_max = h
                h += step

            fig = plt.figure(figsize=(10, 8))
            plt.grid(True)
            plt.xlabel("Cutoff")
            plt.ylabel("F1-score")
            plt.plot(steps, thresholds)
            plt.plot(h_max, max_, 'r*')

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            encoded_img = base64.b64encode(buf.getbuffer()).decode("ascii")
            image_url = f"data:image/png;base64,{encoded_img}"
            return {
                "cutoff": h_max,
                "f1-score": max_,
                "image": image_url
            }
