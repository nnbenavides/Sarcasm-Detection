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
import scipy.stats

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

def fit_maxent_classifier(X, y):
	mod = LogisticRegression(fit_intercept = True)
	mod.fit(X,y)
	return mod

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return float(m), float(m-h), float(m+h)


#TorchShallowNeural Classifier w/ ELMo Embeddings
elmo_X = np.load('main-balanced-elmo-X.npy')
elmo_y = np.load('main-balanced-elmo-y.npy')
n = len(elmo_X)

# Model hyperparameters
dim = 50
iters = 250
activation = nn.ReLU()
eta = 0.01

f1s = []
for i in range(10):
	train_indices = np.random.choice(range(n), round(0.95*n), replace = False)
	train_indices_subset = np.random.choice(train_indices, round(0.95*n), replace = False)
	elmo_X_train = elmo_X[train_indices_subset]
	elmo_y_train = elmo_y[train_indices_subset]
	index_map_train = {}
	for i in range(len(train_indices_subset)):
		index_map_train[i] = train_indices_subset[i]
	print('size of training set:', str(len(index_map_train)))

	other_indices = list(set(range(n)) - set(train_indices))
	print(len(other_indices))
	dev_indices = np.random.choice(other_indices, round(0.5*len(other_indices)), replace = False)
	elmo_X_dev = elmo_X[dev_indices]
	elmo_y_dev = elmo_y[dev_indices]
	index_map_dev = {}
	for i in range(len(dev_indices)):
		index_map_dev[i] = dev_indices[i]
	print('size of dev set:', str(len(index_map_dev)))

	test_indices = list(set(other_indices) - set(dev_indices))
	elmo_X_test = elmo_X[test_indices]
	elmo_y_test = elmo_y[test_indices]
	index_map_test = {}
	for i in range(len(test_indices)):
		index_map_test[i] = test_indices[i]
	print('size of test set:', str(len(index_map_test)))

	model = fit_basic_rnn(elmo_X_train, elmo_y_train, dim, iters, activation, eta)
	predictions = model.predict(elmo_X_dev)
	report = classification_report(elmo_y_dev, predictions, output_dict = True)
	macro_f1 = report['macro avg']['f1-score']
	f1s.append(macro_f1)
	print(f1s)
	print('\n ran iteration ', str(i+1), ' out of 10')

mean, low, high = mean_confidence_interval(f1s)
print('Mean: ', str(mean))
print('Low: ', str(low))
print('High: ', str(high))