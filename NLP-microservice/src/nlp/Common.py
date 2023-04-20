import json
import re

import nltk
import numpy as np
import pandas as pd
import pymorphy2
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')

punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()


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


def preprocess_and_save(data_df: pd.DataFrame, path, text_field='text') -> pd.DataFrame:
    # for preprocessing dataset. Use it only in critical cases cause it's too slow on big datasets
    data_df['preprocessed_' + text_field] = data_df.apply(
        lambda row: preprocess(row[text_field], punctuation_marks, stop_words, morph), axis=1)
    data_df_preprocessed = data_df.copy()
    data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field], axis=1)
    data_df_preprocessed.reset_index(drop=True, inplace=True)
    if path is not None:
        data_df_preprocessed.to_json(path)
    return data_df_preprocessed


def sent_preprocess(text: str):
    preprocessed_text = sent_tokenize(text)
    for i in range(len(preprocessed_text)):
        res = re.sub(r'([^\w\s])|([0-9]+)', '', preprocessed_text[i])
        # res = re.sub(r'', '' ,  res)
        preprocessed_text[i] = res
    preprocessed_text = list(filter(lambda sentence: sentence != '', preprocessed_text))
    return preprocessed_text


def read_json(path: str):
    file = open(path)
    data = json.load(file)
    return pd.DataFrame(data)


def get_states(sim, df, match_threshold):
    (TP, FP, FN, TN) = (0, 0, 0, 0)
    print(sim)
    for i in range(len(sim)):
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

    return TP, FP, FN, TN


def get_states_loo(predictions, df):
    (TP, FP, FN, TN) = (0, 0, 0, 0)
    print(len(df), len(predictions))
    for i in range(len(df)):
        if df['need_match'][i]:
            if predictions[i]:
                TP += 1
            else:
                FN += 1
        else:
            if predictions[i]:
                FP += 1
            else:
                TN += 1

    return TP, FP, FN, TN


def max_f1_score_loo(sim, df, step=0.02):
    threshold = 0
    thresholds = []
    max_ = 0
    step_max_ = 0
    h = step
    steps = np.linspace(0, 1, num=int(1 / h))
    steps = np.round(steps, 2)

    for i in steps:
        threshold = calc_f1_score(sim, df, h)
        thresholds.append(threshold)
        if threshold > max_:
            max_ = threshold
            step_max_ = h
        h += step
    print(max_, step_max_)
    return (steps, thresholds, max_, step_max_)


def calc_f1_score_loo(calc_states):
    (TP, FP, FN, TN) = calc_states()
    return round(float(2 * TP / (2 * TP + FP + FN)), 3)


def calc_f1_score(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    #     print(TP, FP, FN, TN)
    return round(float(2 * TP / (2 * TP + FP + FN)), 3)


def calc_accuracy(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    return round(float((TP + TN) / (TP + TN + FP + FN)), 3)
