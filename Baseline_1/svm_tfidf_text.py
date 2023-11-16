import numpy as np
import pandas as pd
import re
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from transformers import get_linear_schedule_with_warmup, pipeline, CamembertTokenizer, CamembertForSequenceClassification, MarianMTModel, MarianTokenizer

from collections import Counter
from math import floor, ceil
from sklearn.model_selection import train_test_split
import pickle
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score
from torchtext.legacy import data
import os
import random
from ast import literal_eval
import torch.optim as optim
import statistics
import shap
import matplotlib.pyplot as pl
import seaborn as sn
from tinytag import TinyTag

from nltk.corpus import wordnet
import nltk
import scipy.stats
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import spacy

stemming = spacy.load('fr_core_news_md')

root_dir = '/Home/Users/mabi_kanaan/WORK/eop_clean/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


french_stopwords = ['ça', 'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 
                    'en', 'et', 'eux', 'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 
                    'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos', 'notre', 'nous', 'on',
                    'ou', 'par', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta',
                    'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd',
                    'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant',
                    'étante', 'étants', 'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont',
                    'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions',
                    'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 
                    'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 
                    'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes', 'ayants',
                    'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura',
                    'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 
                    'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 
                    'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 
                    'eussent', 'ça']


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_data(my_seed=42):

    X = []
    y = []

    '''
        This block of code is supposed to fill X and y.
        X -> emergency call texts
        y -> severity labels

        Currently unavailable due to it being sensitive code.
        TODO: re-implement it in a non-sensitive way
    '''

    X = preprocess_text(X)

    processed_text = remove_non_frequent_words(X)

    train_text, test_text, train_labels, test_labels = train_test_split(processed_text, y, test_size=0.2, random_state=my_seed)
    vocab = flatten_words(processed_text, get_unique=True)
    return train_text, test_text, train_labels, test_labels, vocab


def train_svm(vocab, train_labels, train_text, test_labels, test_text):
    tfidf = TfidfVectorizer(stop_words=french_stopwords, vocabulary=vocab)
    training_matrix = tfidf.fit_transform(train_text)
    test_matrix = tfidf.fit_transform(test_text)
    svm = SVC(kernel='rbf')

    svm.fit(training_matrix, train_labels)

    dev_predicted = svm.predict(test_matrix)
    print("\nAccuracy ", accuracy_score(test_labels, dev_predicted))

    return accuracy_score(test_labels, dev_predicted), precision_score(test_labels, dev_predicted, average='macro'),recall_score(test_labels, dev_predicted, average='macro'), f1_score(test_labels, dev_predicted, average='macro')


def preprocess_text(X):
    for i in range(0, len(X)):
        lowercased_text = X[i].lower() # lowercase
        stemming_doc = stemming(lowercased_text) # stemming
        stemmed_text = ''
        for token in stemming_doc:
            stemmed_text += token.lemma_ + " "
        X[i] = [stemmed_text]
    return X

def remove_non_frequent_words(X):
    all_text = []

    # create dict of words with count of each word
    for i in range(0, len(X)):
        t = X[i]
        splitted = t[0].split(" ")
        dict_words = {}
        for word in splitted:
            if word not in dict_words:
                dict_words[word] = 1
            else:
                dict_words[word] = dict_words[word] + 1

    # remove infrequent and very frequent words
    for i in range(0, len(X)):    
        for word in dict_words.keys():
            if dict_words[word] == 1 or dict_words[word] >= 108:
                t[0].replace(word, '')

        X[i] = t
        all_text.append(X[i][0])

    return all_text

def flatten_words(list1d, get_unique=False):
    qa = [s.split() for s in list1d]
    if get_unique:
        return sorted(list(set([w for sent in qa for w in sent])))
    else:
        return [w for sent in qa for w in sent]


if __name__ == '__main__':

    seeds = [777, 123, 42, 555, 666, 888, 111, 222, 100, 10]
    accuracies = []
    recalls = []
    precisions = []
    f1s = []

    for my_seed in seeds:
        seed_everything(my_seed)

        train_text, test_text, train_labels, test_labels, vocab = get_data(my_seed=my_seed)
        acc,rec,pre,f1 = train_svm(vocab, train_labels, train_text, test_labels, test_text)
        accuracies.append(acc)
        recalls.append(rec)
        precisions.append(pre)
        f1s.append(f1)

print("Mean accuracy ", str(statistics.fmean(accuracies)))
print("Mean recall ", str(statistics.fmean(recalls)))
print("Mean precision ", str(statistics.fmean(precisions)))
print("Mean f1 ", str(statistics.fmean(f1s)))
