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

from collections import defaultdict
train_vocab = defaultdict(int)
for pair in train_responses:
	for comment in pair:
		for w in nltk.word_tokenize(comment):
			train_vocab[w] += 1
train_vocab = Counter(train_vocab)
print(len(train_vocab))

def unigrams_phi_c(comment):
	return Counter(nltk.word_tokenize(comment))

def concat_phi_r(response_features_pair):
	assert len(response_features_pair) == 2
	cat = np.concatenate((response_features_pair[0], response_features_pair[1]))
	return cat

def embed_phi_c(comment, embeddings):
	words = nltk.word_tokenize(comment)
	unk = np.zeros(next(iter(embeddings.values())).shape)
	return np.sum([embeddings[w] if w in embeddings else unk for w in words], axis=0)

def fasttext_phi_c(comment):
	return embed_phi_c(comment, fasttext_lookup)

responses = train_responses
phi_c = unigrams_phi_c
N = len(responses)
feat_dicts = [[],[]]
for i in range(N):
	assert len(responses[i]) == 2
	feat_dicts[0].append(phi_c(responses[i][0]))
	feat_dicts[1].append(phi_c(responses[i][1]))

'''
# GloVe Embeddings
i=0
glove_lookup = {}
with open('../../static/glove/glove.6B.300d.txt') as f:
#with open('../../static/') as f:
	while True:
		try:
			x = next(f)
		except:
			break
		try:
			fields = x.strip().split()
			idx = fields[0]
			if idx not in train_vocab: continue
			if idx in glove_lookup:
				print("Duplicate! ", idx)
			vec = np.array(fields[1:], dtype=np.float32)
			glove_lookup[idx] = vec
			i += 1
			#if i%500 == 0: print(i)
		except:
			pass


#print(len(glove_lookup))
#print(type(glove_lookup['the']), glove_lookup['the'].shape, sum(glove_lookup['the']))
'''

def build_dataset(ancestors, responses, labels, phi_c, phi_a, phi_r, vectorizer=None, vectorize = True):
	X = []
	Y = []
	feat_dicts = [[],[]]
	N = len(ancestors)
	assert N == len(responses) == len(labels)
	print(N)
	for i in range(N):
		if i % 1000 == 0 and i > 0:
			print(i)
		assert len(responses[i]) == 2
		feat_dicts[0].append(phi_c(responses[i][0]))
		feat_dicts[1].append(phi_c(responses[i][1]))
	
		#We only care about the first of the two labels since in the balanced setting
		#they're either 0 1 or 1 0
		Y.append(int(labels[i][0]))
			
	if vectorize:
		# In training, we want a new vectorizer:
		if vectorizer == None:
			vectorizer = DictVectorizer(sparse=False)
			#print(feat_dicts[0][:10], feat_dicts[1][:10])
			feat_matrix = vectorizer.fit_transform(feat_dicts[0] + feat_dicts[1])
		# In assessment, we featurize using the existing vectorizer:
		else:
			feat_matrix = vectorizer.transform(chain(feat_dicts[0], feat_dicts[1]))
		
		response_pair_feats = [feat_matrix[:N], feat_matrix[N:]]
	else:
		response_pair_feats = feat_dicts
	
	X = [phi_r((response_pair_feats[0][i], response_pair_feats[1][i])) for i in range(N)]
	
	return {'X': np.array(X),
			'y': np.array(Y),
			'vectorizer': vectorizer,
			'raw_examples': (ancestors, responses)}

def xval_model(model_fit_fn, X, y, folds):
	kf = KFold(folds)
	macro_f1_avg = 0
	for train, test in kf.split(X, y):
		model = model_fit_fn(X[train], y[train])
		predictions = model.predict(X[test])
		report = classification_report(y[test], predictions, output_dict = True)
		macro_f1_avg += report['macro avg']['f1-score']
		print(classification_report(y[test], predictions, digits=3))
	macro_f1_avg /= folds
	output = 'Average Macro F1 Score across folds = ' + str(macro_f1_avg)
	print(output)

'''
unigram_dataset = build_dataset(train_ancestors, train_responses, train_labels, unigrams_phi_c, None, concat_phi_r)

unigram_dataset['X'].shape
np.save('main-balanced-unigram-X.npy', unigram_dataset['X'])
np.save('main-balanced-unigram-y.npy', unigram_dataset['y'])
'''

def glove_phi_c(comment):
	return embed_phi_c(comment, glove_lookup)

'''
glove_dataset = build_dataset(
	train_ancestors, train_responses, train_labels, glove_phi_c, None, concat_phi_r, None, False)

fasttext_dataset['X'].shape
np.save('main-balanced-glove-X.npy', glove_dataset['X'])
np.save('main-balanced-glove-y.npy', glove_dataset['y'])
'''

# ELMo Embeddings
from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()

def elmo_phi_c(comment):
	vecs = elmo.embed_sentence(nltk.word_tokenize(comment))
	elmo_avg_vec = vecs.mean(axis = 0)
	return elmo_avg_vec[0]

'''
elmo_dataset = build_dataset(
	train_ancestors, train_responses, train_labels, elmo_phi_c, None, concat_phi_r, None, False)
np.save('pol-balanced-elmo-X.npy', elmo_dataset['X'])
np.save('pol-balanced-elmo-y.npy', elmo_dataset['y'])
'''

# Fit TorchShallowNeuralClassifier
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

# Fit logistic regression model
def fit_maxent_classifier(X, y):
	mod = LogisticRegression(fit_intercept = True)
	mod.fit(X,y)
	return mod

#TorchShallowNeural Classifier w/ ELMo Embeddings
elmo_X = np.load('main-balanced-elmo-X.npy')
elmo_y = np.load('main-balanced-elmo-y.npy')
n = len(elmo_X)
np.random.seed(224)

train_indices = np.random.choice(range(n), round(0.95*n), replace = False)
train_indices_subset = np.random.choice(train_indices, 50000, replace = False)
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

hidden_dims = [50, 100, 250, 500]
hidden_activations = [nn.Tanh(), nn.ReLU()]
max_iters = [100, 250, 500]
etas = [0.1, 0.01, 0.001]
n_models = len(hidden_dims)*len(hidden_activations)*len(max_iters)*len(etas)

best_dim = 0
best_activation = None
best_iters = 0
best_eta = 0
best_f1 = 0
i = 0
np.random.seed(1738)
for i in range(20):
    dim = np.random.choice(hidden_dims)
    iters = np.random.choice(max_iters)
    activation = np.random.choice(hidden_activations)
    eta = np.random.choice(etas)
    model = fit_basic_rnn(elmo_X_train, elmo_y_train, dim, iters, activation, eta)
    predictions = model.predict(elmo_X_dev)
    report = classification_report(elmo_y_dev, predictions, output_dict = True)
    macro_f1 = report['macro avg']['f1-score']
    i += 1
    print('\nfit model ', str(i), ' out of 20')
    if macro_f1 > best_f1:
        best_dim = dim
        best_activation = activation
        best_iters = iters
        best_eta = eta
        best_f1 = macro_f1
print('Best F1: ',str(best_f1))
print('Best dim: ',str(best_dim))
print('Best activation: ', str(best_activation))
print('Best eta: ', str(best_eta))
print('Best iters: ', str(best_iters))

# Error Analysis
missed_dev_indices = []
missed_preds = []
for i, pred in enumerate(predictions):
    if pred != elmo_y_dev[i]:
        missed_dev_indices.append(i)
        missed_preds.append(pred)

num_examples_to_analyze = 200
indices_to_analyze = np.random.choice(range(len(missed_preds)), num_examples_to_analyze, replace = False)
for i, ind in enumerate(indices_to_analyze):
    missed_pred = missed_preds[ind]
    missed_og_index = index_map_dev[ind]
    print('\nMissed Example #', str(i+1), ' of ', str(num_examples_to_analyze))
    if missed_pred == 0: #originally predicted the first comment to be non-sarcastic
        print('Actual non-sarcastic comment: ', responses[missed_og_index][1])
        print('Word count: ', len((responses[missed_og_index][1]).split()))
        print('Actual sarcastic comment: ', responses[missed_og_index][0])
        print('Word count: ', len((responses[missed_og_index][0]).split()))

    else:
        print('Actual non-sarcastic comment: ', responses[missed_og_index][0])
        print('Word count: ', len((responses[missed_og_index][0]).split()))
        print('Actual sarcastic comment: ', responses[missed_og_index][1])
        print('Word count: ', len((responses[missed_og_index][1]).split()))