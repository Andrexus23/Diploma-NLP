import json
import pickle

import pandas as pd
import pymorphy2
import nltk
import redis
import sys
import time
from nltk.corpus import stopwords
import gensim
import gensim.models as models
from sentence_transformers import SentenceTransformer
from sklearn.utils import shuffle
import numpy as np
import matplotlib
from sklearn.metrics import auc
import matplotlib.pyplot as plt
matplotlib.use('agg')
import matplotlib.pyplot as plt
import Common
import hashlib
import sentence_transformers

fontsize = 18
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['figure.dpi'] = 300
punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
model_types = []
redis_port = 0
redis_host = '-'
line_thickness = 3

class ModelResearcher:
    def __init__(self):
        self.models = {}
        self.model = None
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True, db=0)

    def load(self, path, model_name, model_type):
        try:
            if model_type == "gensim":
                self.models.setdefault(path, [models.ldamodel.LdaModel.load(path), model_name, model_type])
            elif model_type == "transformer":
                self.models.setdefault(path, [SentenceTransformer(path), model_name, model_type])
            self.model = self.models[path][0]
            return True
        except Exception as e:
            print(e)
            return False

    def predict_transfomer_two_texts(self, text_1, text_2, model_name, round_number=4):
        text = text_1 + text_2 + model_name + str(round_number)
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached_value = self.redis.hget(cache_key, "value")
        print(cache_key, cached_value)
        if cached_value:
            return float(cached_value)

        sentences_rp = Common.sent_preprocess(text_1)
        sentences_proj = Common.sent_preprocess(text_2)

        def comp(e):
            return e['cos_sim']

        sentences_proj_embeddings = []
        for sentence_proj in sentences_proj:
            sentences_proj_embeddings += [self.model.encode(sentence_proj, convert_to_tensor=True)]

        max_sims = []
        for sentence_rp in sentences_rp:
            sentence_rp_embedding = self.model.encode(sentence_rp, convert_to_tensor=True)
            sim = []
            for i in range(len(sentences_proj_embeddings)):
                sim += [{'proj': sentences_proj[i],
                         'cos_sim': float(sentence_transformers.util.cos_sim(sentence_rp_embedding,
                                                                             sentences_proj_embeddings[i]))
                         }]

            max_sims.append(max(sim, key=comp)['cos_sim'])

        value = float(np.round(np.mean(max_sims), round_number))
        self.redis.hmset(cache_key, {"value": value})
        self.redis.expire(cache_key, 60 * 60 * 24 * 7)
        return value

    def preprocess_and_save_pairs(self, data_df: pd.DataFrame, text_field_1, text_field_2, path=None):
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed['preprocessed_' + text_field_1] = data_df_preprocessed.apply(
            lambda row: Common.preprocess(row[text_field_1], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed['preprocessed_' + text_field_2] = data_df_preprocessed.apply(
            lambda row: Common.preprocess(row[text_field_2], punctuation_marks, stop_words, morph), axis=1)
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
            self.model.save(model_path + model)
        return

    def predict_sentences_similarity(self, sentences_1: pd.Series, sentences_2: pd.Series, model_name="w2v"):
        if sentences_1.size != sentences_2.size:
            return None
        else:
            if self.model is not None:
                sentences_sim = np.zeros(sentences_1.size)
                sz = sentences_1.size
                sentences_1_words = None
                sentences_2_words = None
                for i in range(sz):
                    sentences_1_words = [w for w in sentences_1[i] if w in self.model.wv.index_to_key]
                    sentences_2_words = [w for w in sentences_2[i] if w in self.model.wv.index_to_key]

                    sim = self.model.wv.n_similarity(sentences_1_words, sentences_2_words)
                    sentences_sim[i] = sim

                return list(sentences_sim)
            else:
                return None

    def get_train_test_dfs_for_f1(self, df):
        df_match = shuffle(df[df["need_match"] == True])
        df_unmatch = shuffle(df[df["need_match"] == False])
        df_train_f1 = pd.concat(
            [pd.DataFrame(df_match[0:(len(df_match) // 2) + (len(df_match) % 2)]),
             pd.DataFrame(df_unmatch[0:(len(df_unmatch) // 2) + (len(df_unmatch) % 2)])], axis=0).reset_index(drop=True)
        df_test_f1 = pd.concat(
            [pd.DataFrame(df_match[len(df_match) // 2 + (len(df_match) % 2):]),
             pd.DataFrame(df_unmatch[len(df_unmatch) // 2 + (len(df_unmatch) % 2):])], axis=0).reset_index(drop=True)
        return df_train_f1, df_test_f1

    def maximize_f1_score(self, sentences_1, sentences_2, df, image_path, model_name, model_type,
                          step=0.02):
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
                sim = self.predict_sentences_similarity(sentences_1, sentences_2, model_name=model_name)
            else:
                for i in range(len(sentences_1)):
                    sim += [self.predict_transfomer_two_texts(sentences_1[i], sentences_2[i], model_name=model_name)]

            steps, thresholds, f1_score, cutoff = Common.max_f1_score(sim, df, step=0.02)
            fig = plt.figure(figsize=(7, 6))
            plt.grid(True)
            if model_name == "paraphrase-multilingual-MiniLM-L12-v2":
                model_name = "multilingual"
            plt.title(f"Basic maximization: {model_name}")
            plt.xlabel("Cutoff", fontsize=fontsize)
            plt.ylabel("F1-score", fontsize=fontsize)
            plt.plot(steps, thresholds, label="F1-score(cutoff)", linewidth=line_thickness)
            plt.plot(cutoff, f1_score, "r*", label="Max F1-score" )
            plt.legend(loc="best")
            imname = "Maximization-F1-score-" + model_name + ".png"
            plt.savefig(image_path + imname)
            print(image_path + imname)
            res = {}
            preds = [sim[i] >= cutoff for i in range(len(df))]
            metrics = Common.calc_all(sim, df, cutoff)
            res.setdefault("cutoff", cutoff)
            res.setdefault("f1-score", metrics["f1-score"])
            res.setdefault("precision", metrics["precision"])
            res.setdefault("recall", metrics["recall"])
            res.setdefault("sim", str(preds))
            res.setdefault("image", imname)
            return res

    def maximize_f1_score_loo(self, sentences_1: pd.Series, sentences_2: pd.Series, df, model_name, model_type,
                              step=0.02):
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
                sim = self.predict_sentences_similarity(sentences_1, sentences_2, model_name=model_name)
            else:
                for i in range(len(sentences_1)):
                    sim += [self.predict_transfomer_two_texts(sentences_1[i], sentences_2[i], model_name=model_name)]

            predictions = []
            cutoffs = []

            for i in range(len(sim)):
                current_df = df.drop(i).reset_index().drop(labels='index', axis=1)
                current_sim = sim[:i] + sim[i + 1:]
                steps, thresholds, f1_temp, cutoff_temp = Common.max_f1_score(current_sim, current_df)
                cutoffs.append(cutoff_temp)
                if sim[i] >= cutoff_temp:
                    predictions.append(True)
                else:
                    predictions.append(False)

            cutoff_mean = round(np.mean(cutoffs), 3)
            f1_score_loo = Common.calc_f1_score_loo(lambda: Common.get_states_loo(predictions, df))
            res = {}
            metrics = Common.calc_all_loo(lambda: Common.get_states_loo(predictions, df))
            metrics_from_cutoff = Common.calc_all(sim, df, cutoff_mean)
            preds = [sim[i] >= cutoff_mean for i in range(len(df))]
            res.setdefault("1_cutoff_mean", cutoff_mean)
            res.setdefault("1_f1-score_cutoff", metrics_from_cutoff["f1-score"])
            res.setdefault("1_precision_cutoff", metrics_from_cutoff["precision"])
            res.setdefault("1_recall_cutoff", metrics_from_cutoff["recall"])
            res.setdefault("1_sim (cutoff)", str(preds))
            res.setdefault("2_f1-score_loo", metrics["f1-score"])
            res.setdefault("2_precision_loo", metrics["precision"])
            res.setdefault("2_recall_loo", metrics["recall"])

            return res

    def maximize_f1_score_train_test(self, df_train, df_test, image_path, model_name, model_type, field_1, field_2,
                                     step=0.02):
        steps = np.linspace(0, 1, num=int(1 / step))
        steps = np.round(steps, 2)
        sim_train = []
        sim_test = []
        sentences_1_train = df_train[field_1]
        sentences_2_train = df_train[field_2]
        sentences_1_test = df_test[field_1]
        sentences_2_test = df_test[field_2]
        if sentences_1_train.size != sentences_2_train.size or \
                sentences_1_test.size != sentences_2_test.size:
            raise ValueError("Error length")

        if model_type == "gensim":
            sim_train = self.predict_sentences_similarity(sentences_1_train, sentences_2_train, model_name=model_name)
            sim_test = self.predict_sentences_similarity(sentences_1_test, sentences_2_test, model_name=model_name)
        elif model_type == "transformer":
            for i in range(len(sentences_1_train)):
                sim_train += [
                    self.predict_transfomer_two_texts(sentences_1_train[i], sentences_2_train[i],
                                                      model_name=model_name)]
            for i in range(len(sentences_1_test)):
                sim_test += [
                    self.predict_transfomer_two_texts(sentences_1_test[i], sentences_2_test[i], model_name=model_name)]

        steps, thresholds, f1_score_train, cutoff = Common.max_f1_score(sim_train, df_train, step=0.02)
        metrics_train = Common.calc_all(sim_train, df_train, cutoff)
        metrics_test = Common.calc_all(sim_test, df_test, cutoff)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.title("Train dataset: " + model_name, fontsize=fontsize)
        plt.xlabel("Cutoff", fontsize=fontsize)
        plt.ylabel("F1-score", fontsize=fontsize)
        plt.plot(steps, thresholds, linewidth=line_thickness, label="F1-score(cutoff)")
        plt.plot(cutoff, f1_score_train, 'r*', label="Max F1-score")
        plt.annotate(f'({cutoff}, {f1_score_train})', (cutoff - 0.06, f1_score_train + 0.01), fontsize=fontsize - 6)
        plt.legend(loc="best")
        imname = "Maximization-F1-score-train-" + model_name + ".png"
        plt.savefig(image_path + imname)
        res = {}
        res.setdefault("1_cutoff_train", cutoff)
        res.setdefault("1_f1-score_train", metrics_train["f1-score"])
        res.setdefault("1_precision_train", metrics_train["precision"])
        res.setdefault("1_recall_train", metrics_train["recall"])
        res.setdefault("2_f1-score_test", metrics_test["f1-score"])
        res.setdefault("2_precision_test", metrics_test["precision"])
        res.setdefault("2_recall_test", metrics_test["recall"])
        res.setdefault("2_image", imname)
        return res

    def get_roc_auc(self, df, field_1, field_2, image_path, step=0.02):
        colors = {'darkorange', 'dodgerblue', 'brown', 'olivedrab'}
        df_preprocessed = self.preprocess_and_save_pairs( df, field_1, field_2)
        fig = plt.figure(figsize=(10, 8))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC')
        res = {}

        for model in self.models.items():
            self.model = model[1][0]
            sim = []
            if model[1][2] == "gensim":
                sim = self.match_texts_from_corpus(df_preprocessed, model_name=model[1][1], model_type=model[1][2], field_1='preprocessed_' + field_1,
                                                   field_2='preprocessed_' + field_2)
            else:
                sim = self.match_texts_from_corpus(df, model_name=model[1][1], model_type=model[1][2],
                                                   field_1=field_1,
                                                   field_2=field_2)
            steps, tprs, fprs, cutoff = Common.max_diff_tpr_fpr(sim, df)
            roc_auc = auc(fprs, tprs)
            preds = [sim[i] >= cutoff for i in range(len(df))]
            res.setdefault(str(model[1][1]), {"AUC": round(roc_auc, 3), "cutoff": cutoff, "sim": str(preds)})
            if model[1][1] == "paraphrase-multilingual-MiniLM-L12-v2":
                model[1][1] = "multilingual"

            plt.plot(fprs, tprs, color=colors.pop(), linewidth=line_thickness,
                     label=f'ROC {model[1][1]} (area = {round(roc_auc, 3)}, cutoff = {cutoff})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=line_thickness)
        plt.legend(loc="best")
        imname = "ROC-AUC" + ".png"
        plt.savefig(image_path + imname)
        res.setdefault("image", imname)
        return res

    def match_texts_from_corpus(self, df, model_name, model_type, field_1, field_2):
        sim = []
        sentences_1 = df[field_1]
        sentences_2 = df[field_2]
        start = time.time()
        if model_type == "gensim":
            sim = self.predict_sentences_similarity(sentences_1, sentences_2, model_name=model_name)
        else:
            for i in range(len(sentences_1)):
                sim += [self.predict_transfomer_two_texts(sentences_1[i], sentences_2[i], model_name=model_name)]
        sim = [round(i, 3) for i in sim]
        res = time.time() - start
        print(f"Затраченное время: {round(res, 6)} секунд")
        print("Затраченное время: {}".format(time.strftime("%H:%M:%S", time.gmtime(res))))
        return sim
