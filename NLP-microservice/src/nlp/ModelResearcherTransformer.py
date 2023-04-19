import base64
import io
import time

import numpy as np
import pandas as pd
import pymorphy2
import sentence_transformers
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

from src.nlp import Common
from sentence_transformers import SentenceTransformer

ALLOWED_MODELS_TRANSFORMER = [
    "rubert-base-cased",
    "paraphrase-multilingual-MiniLM-L12-v2"]

punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()


class ModelResearcherTransformer:
    def __init__(self):
        self.model = None

    def load(self, name):
        if self.model is None:
            try:
                self.model = SentenceTransformer(name)
                return True
            except ValueError as e:
                print(e)
                return False
            except OSError as e:
                print(e)
                return False

    def predict_sim_two_texts(self, text_1, text_2):
        if self.model is not None:
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
                # print(f'RP sentence: {sentence_rp[0]}')
                for i in range(len(sentences_proj_embeddings)):
                    sim += [{'proj': sentences_proj[i],
                             'cos_sim': float(sentence_transformers.util.cos_sim(sentence_rp_embedding,
                                                                                 sentences_proj_embeddings[i]))
                             }]

                # sim.sort(key=comp, reverse=True)
                max_sims.append(max(sim, key=comp)['cos_sim'])

            return np.mean(max_sims)

    def maximize_f1_score(self, texts_rp, texts_proj, df_f1, step=0.02):
        threshold = 0
        thresholds = []
        max_ = 0
        h_max = 0
        h = step
        steps = np.linspace(0, 1, num=int(1 / h))
        steps = np.round(steps, 2)
        sim = []
        for i in range(len(texts_rp)):
            sim += [self.predict_sim_two_texts(texts_rp[i], texts_proj[i])]
        for i in steps:
            threshold = Common.calc_f1_score(sim, df_f1, h)
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

