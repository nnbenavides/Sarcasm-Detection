# Author: Nicholas Benavides, Ray Thai, & Crystal Zheng
# Code liberally inspired by and lifted from:
# https://github.com/kolchinski/reddit-sarc
# https://github.com/cgpotts/cs224u

# Imports
import os
import csv
import json
from itertools import islice, chain
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier
from sklearn.linear_model import LogisticRegression
import random
import torch.nn as nn
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import utils

# Read in Data
pol_dir = '../SARC/2.0/main'
comments_file = os.path.join(pol_dir, 'comments.json')
train_file = os.path.join(pol_dir, 'train-balanced.csv')

with open(comments_file, 'r') as f:
    comments = json.load(f)

# Format data
train_ancestors = []
train_responses = []
train_labels = []
lower = True
with open(train_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        ancestors = row[0].split(' ')
        responses = row[1].split(' ')
        labels = row[2].split(' ')
        if lower:
            train_ancestors.append([comments[r]['text'].lower() for r in ancestors])
            train_responses.append([comments[r]['text'].lower() for r in responses])
        else:
            train_ancestors.append([comments[r]['text'] for r in ancestors])
            train_responses.append([comments[r]['text'] for r in responses])
        train_labels.append(labels)

def unigrams_phi_c(comment):
    return Counter(nltk.word_tokenize(comment))

from collections import defaultdict
train_vocab = defaultdict(int)
for pair in train_responses:
    for comment in pair:
        for w in nltk.word_tokenize(comment):
            train_vocab[w] += 1

responses = train_responses
phi_c = unigrams_phi_c
N = len(responses)
feat_dicts = [[],[]]
for i in range(N):
	assert len(responses[i]) == 2
	feat_dicts[0].append(phi_c(responses[i][0]))
	feat_dicts[1].append(phi_c(responses[i][1]))

def fit_basic_rnn(X, y, hidden_dim, max_iter, hidden_activation, eta):
	if hidden_dim is None:
		hidden_dim = 50
	if max_iter is None:
		max_iter = 100
	if hidden_activation is None:
		hidden_activation = nn.Tanh()
	if eta is None:
		eta = 0.01

	mod = TorchShallowNeuralClassifier(hidden_dim = hidden_dim, max_iter = max_iter, hidden_activation = hidden_activation, eta = eta)
	mod.fit(X, y)
	return mod


elmo_X = np.load('main-balanced-elmo-X.npy')
elmo_y = np.load('main-balanced-elmo-y.npy')
n = len(elmo_X)

glove_X = np.load('main-balanced-glove-X.npy')
glove_y = np.load('main-balanced-glove-y.npy')
np.random.seed(224)

train_indices = np.random.choice(range(n), round(0.95*n), replace = False)
train_indices_subset = np.random.choice(train_indices, round(0.95*n), replace = False)
elmo_X_train = elmo_X[train_indices_subset]
elmo_y_train = elmo_y[train_indices_subset]
glove_X_train = glove_X[train_indices_subset]
glove_y_train = glove_y[train_indices_subset]
print('size of training set:', str(len(elmo_X_train)))

other_indices = list(set(range(n)) - set(train_indices))
print(len(other_indices))
dev_indices = np.random.choice(other_indices, round(0.5*len(other_indices)), replace = False)
elmo_X_dev = elmo_X[dev_indices]
elmo_y_dev = elmo_y[dev_indices]
glove_X_dev = glove_X[dev_indices]
glove_y_dev = glove_y[dev_indices]
print('size of dev set:', str(len(elmo_X_dev)))

test_indices = list(set(other_indices) - set(dev_indices))
elmo_X_test = elmo_X[test_indices]
elmo_y_test = elmo_y[test_indices]
glove_X_test = glove_X[test_indices]
glove_y_test = glove_y[test_indices]
print('size of test set:', str(len(elmo_X_test)))


# Fit Shallow RNN w/ ELMo embeddings
elmo_dim = 50
elmo_activation = nn.ReLU()
elmo_iters = 250
elmo_eta = 0.01
elmo_model = fit_basic_rnn(elmo_X_train, elmo_y_train, elmo_dim, elmo_iters, elmo_activation, elmo_eta)

elmo_predictions = elmo_model.predict(elmo_X_dev)
elmo_report = classification_report(elmo_y_dev, elmo_predictions, output_dict = True)
elmo_macro_f1 = elmo_report['macro avg']['f1-score']
print(classification_report(elmo_y_dev, elmo_predictions))

cm=confusion_matrix(elmo_y_dev,elmo_predictions)
df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Sarcastic', 'Sarcastic']],
                  columns = [i for i in ['Non-Sarcastic', 'Sarcastic']])
plt.figure(figsize = (10,7))
ax = sn.heatmap(df_cm, annot=True)
plt.savefig("shallow_elmo_cm.png")

# Fit Shallow RNN w/ GloVe embeddings
glove_dim = 50
glove_activation = nn.ReLU()
glove_iters = 100
glove_eta = 0.01
glove_model = fit_basic_rnn(glove_X_train, glove_y_train, glove_dim, glove_iters, glove_activation, glove_eta)

glove_predictions = glove_model.predict(glove_X_dev)
glove_report = classification_report(glove_y_dev, glove_predictions, output_dict = True)
glove_macro_f1 = glove_report['macro avg']['f1-score']
print(classification_report(glove_y_dev, glove_predictions))

cm=confusion_matrix(glove_y_dev,glove_predictions)
df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Sarcastic', 'Sarcastic']],
                  columns = [i for i in ['Non-Sarcastic', 'Sarcastic']])
plt.figure(figsize = (10,7))
ax = sn.heatmap(df_cm, annot=True)
plt.savefig("shallow_glove_cm.png")

# McNemar's Test
m = utils.mcnemar(glove_y_dev, glove_predictions, elmo_predictions)
p = "p < 0.0001" if m[1] < 0.0001 else m[1]
print("McNemar's test: {0:0.02f} ({1:})".format(m[0], p))

# Error Analysis
error_analysis_examples = []
elmo_correct = 0
glove_correct = 0
overlap = 0
for i, og_ind in enumerate(dev_indices):
    elmo_pred = elmo_predictions[i]
    glove_pred = glove_predictions[i]
    actual = elmo_y_dev[i]
    winner = ''
    if elmo_pred == actual and glove_pred == actual:
        overlap += 1
        elmo_correct += 1
        glove_correct += 1
    elif elmo_pred != actual and glove_pred != actual:
        continue
    else:
        if elmo_pred == actual and glove_pred != actual: # ELMo wins
            winner = 'ELMo'
            elmo_correct += 1
        elif elmo_pred != actual and glove_pred == actual:
            winner = 'GloVe'
            glove_correct += 1
        
        if actual == 0:
            error_analysis_examples.append((responses[og_ind][0], responses[og_ind][1], winner))
        else:
            error_analysis_examples.append((responses[og_ind][1], responses[og_ind][0], winner))
print('Of the examples ELMo got right, GLoVe got ', str(round(100*overlap/elmo_correct, 1)), '% right.')
print('Of the examples GloVe got right, ELMo got ', str(round(100*overlap/glove_correct, 1)), '% right.')
overlap /= len(dev_indices)
print('Overall overlap: ', round(overlap*100, 1))
num_examples_to_analyze = 500
error_indices = np.random.choice(range(len(error_analysis_examples)), num_examples_to_analyze, replace = False)
error_examples = [error_analysis_examples[i] for i in error_indices]

for i, tup in enumerate(error_analysis_examples):#enumerate(error_examples):
    print('\nExample #', str(i+1), ' of ', str(num_examples_to_analyze))
    print('Winner: ', tup[2])
    print('Actual non-sarcastic comment: ', tup[0])
    print('Word count: ', len(tup[0].split()))
    print('Actual sarcastic comment: ', tup[1])
    print('Word count: ', len(tup[1].split()))
