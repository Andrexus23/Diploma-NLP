import json
import pandas as pd
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
import gensim.models as models
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.utils import shuffle
import zipfile
import sys
import requests, io
import re
import random
import numpy as np
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')



def preprocess(text: str, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if re.match(r'(\d.|\d)', lemma) is None:
                if lemma not in stop_words:
                    preprocessed_text.append(lemma)
    return preprocessed_text


def read_json(path: str):
    file = open(path)
    data = json.load(file)
    return pd.DataFrame(data)


def get_states(sim, df, match_threshold):
    (TP, FP, FN, TN) = (0, 0, 0, 0)
    for i in range(sim.size):
        if df['need_match'][i]:
            if sim[i] >= match_threshold:
                TP += 1
            else:
                FN += 1
        else:
            if sim[i] >= match_threshold:
                FP += 1
            else:
                TN += 1

    return (TP, FP, FN, TN)


def calc_f1_score(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    #     print(TP, FP, FN, TN)
    return round(float(2 * TP / (2 * TP + FP + FN)), 3)


def calc_accuracy(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    return round(float((TP + TN) / (TP + TN + FP + FN)), 3)


punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()


class ModelResearcher:
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
            first = preprocess(text1, punctuation_marks, stop_words, morph)
            second = preprocess(text2, punctuation_marks, stop_words, morph)
            sim = self.model.wv.n_similarity(first, second)
            return round(sim, round_number)
        return None

    def preprocess_and_save(self, data_df: pd.DataFrame, path, text_field='text') -> pd.DataFrame:
        # for preprocessing dataset. Use it only in critical cases cause it's too slow on big datasets
        data_df['preprocessed_' + text_field] = data_df.apply(
            lambda row: preprocess(row[text_field], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field], axis=1)
        data_df_preprocessed.reset_index(drop=True, inplace=True)
        if path is not None:
            data_df_preprocessed.to_json(path)
        return data_df_preprocessed

    def preprocess_and_save_pairs(self, data_df: pd.DataFrame, path, text_field_1, text_field_2) -> pd.DataFrame:
        data_df['preprocessed_' + text_field_1] = data_df.apply(
            lambda row: preprocess(row[text_field_1], punctuation_marks, stop_words, morph), axis=1)
        data_df['preprocessed_' + text_field_2] = data_df.apply(
            lambda row: preprocess(row[text_field_2], punctuation_marks, stop_words, morph), axis=1)
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
            train_part = data_df['preprocessed_texts'].tolist()
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
        return (df_train_f1, df_test_f1)

    def get_optimal_threshold(self, sentences_1: pd.Series, sentences_2: pd.Series, df, step=0.02):
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
                threshold = calc_f1_score(sim, df, h)
                print(h, threshold)
                thresholds.append(threshold)
                if threshold > max_:
                    max_ = threshold
                    h_max = h
                h += step
            plt.plot(steps, thresholds)
            plt.figure(figsize=(10, 3))
            plt.show()

            return h_max