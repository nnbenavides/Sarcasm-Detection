#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Author: Nicholas Benavides, Ray Thai, & Crystal Zheng
# Code liberally inspired by and lifted from:
# https://github.com/kolchinski/reddit-sarc
# https://github.com/cgpotts/cs224u


# In[3]:


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

# In[4]:


pol_dir = '../SARC/2.0/main'
comments_file = os.path.join(pol_dir, 'comments.json')
train_file = os.path.join(pol_dir, 'train-balanced.csv')


# In[5]:


with open(comments_file, 'r') as f:
	comments = json.load(f)


# In[6]:


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


# In[7]:


from collections import defaultdict
train_vocab = defaultdict(int)
for pair in train_responses:
	for comment in pair:
		for w in nltk.word_tokenize(comment):
			train_vocab[w] += 1
train_vocab = Counter(train_vocab)
print(len(train_vocab))


# In[8]:


def unigrams_phi_c(comment):
	return Counter(nltk.word_tokenize(comment))


# In[9]:


def concat_phi_r(response_features_pair):
	assert len(response_features_pair) == 2
	cat = np.concatenate((response_features_pair[0], response_features_pair[1]))
	return cat
	


# In[10]:


def embed_phi_c(comment, embeddings):
	words = nltk.word_tokenize(comment)
	unk = np.zeros(next(iter(embeddings.values())).shape)
	return np.sum([embeddings[w] if w in embeddings else unk for w in words], axis=0)


# In[11]:


def fasttext_phi_c(comment):
	return embed_phi_c(comment, fasttext_lookup)


# In[13]:

'''
# FastText Embeddings
i=0
fasttext_lookup = {}
with open('../../static/wiki-news-300d-1M-subword.vec') as f:
	while True:
		try:
			x = next(f)
		except:
			break
		try:
			fields = x.strip().split()
			idx = fields[0]
			if idx not in train_vocab: continue
			if idx in fasttext_lookup:
				print("Duplicate! ", idx)
			vec = np.array(fields[1:], dtype=np.float32)
			fasttext_lookup[idx] = vec
			i += 1
			#if i%500 == 0: print(i)
		except:
			pass


#print(len(fasttext_lookup))
#print(type(fasttext_lookup['the']), fasttext_lookup['the'].shape, sum(fasttext_lookup['the']))
'''

# In[14]:


responses = train_responses
phi_c = unigrams_phi_c
N = len(responses)
feat_dicts = [[],[]]
for i in range(N):
	assert len(responses[i]) == 2
	feat_dicts[0].append(phi_c(responses[i][0]))
	feat_dicts[1].append(phi_c(responses[i][1]))


# In[15]:

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

# In[16]:


#phi_c turns comments into features
#phi_a combines ancestor features into summary
#phi_r combines response features into summary
#Note that this is for the "balanced" framing!
#TODO: Initially ignoring ancestors, include them as another vector later
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
		#print(response_pair_feats[0])

	#assert len(feat_matrix == 2*N) 
	#print((feat_matrix[0]), len(feat_matrix[1]))
	
	X = [phi_r((response_pair_feats[0][i], response_pair_feats[1][i])) for i in range(N)]
	#X = list(map(phi_r, response_pair_feats))
	
	return {'X': np.array(X),
			'y': np.array(Y),
			'vectorizer': vectorizer,
			'raw_examples': (ancestors, responses)}

		


# In[32]:


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


# In[25]:
'''
unigram_dataset = build_dataset(train_ancestors, train_responses, train_labels, unigrams_phi_c, None, concat_phi_r)

unigram_dataset['X'].shape
np.save('main-balanced-unigram-X.npy', unigram_dataset['X'])
np.save('main-balanced-unigram-y.npy', unigram_dataset['y'])
'''

# In[21]:

'''
fasttext_dataset = build_dataset(
	train_ancestors, train_responses, train_labels, fasttext_phi_c, None, concat_phi_r, None, False)

fasttext_dataset['X'].shape
np.save('main-balanced-fasttext-X.npy', fasttext_dataset['X'])
np.save('main-balanced-fasttext-y.npy', fasttext_dataset['y'])
'''


# In[26]:


def glove_phi_c(comment):
	return embed_phi_c(comment, glove_lookup)


# In[27]:

'''
glove_dataset = build_dataset(
	train_ancestors, train_responses, train_labels, glove_phi_c, None, concat_phi_r, None, False)

fasttext_dataset['X'].shape
np.save('main-balanced-glove-X.npy', glove_dataset['X'])
np.save('main-balanced-glove-y.npy', glove_dataset['y'])
'''


# In[28]:


# ELMo Embeddings
from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()


# In[29]:


def elmo_phi_c(comment):
	vecs = elmo.embed_sentence(nltk.word_tokenize(comment))
	elmo_avg_vec = vecs.mean(axis = 0)
	return elmo_avg_vec[0]


# In[30]:

'''
elmo_dataset = build_dataset(
	train_ancestors, train_responses, train_labels, elmo_phi_c, None, concat_phi_r, None, False)
np.save('pol-balanced-elmo-X.npy', elmo_dataset['X'])
np.save('pol-balanced-elmo-y.npy', elmo_dataset['y'])
'''


# In[37]:


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

# In[35]:


#TorchShallowNeural Classifier w/ ELMo Embeddings
elmo_X = np.load('main-balanced-elmo-X.npy')
elmo_y = np.load('main-balanced-elmo-y.npy')
n = len(elmo_X)
np.random.seed(224)

train_indices = np.random.choice(range(n), round(0.95*n), replace = False)
elmo_X_train = elmo_X[train_indices]
elmo_y_train = elmo_y[train_indices]
index_map_train = {}
for i in range(len(train_indices)):
	index_map_train[i] = train_indices[i]
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
#xval_model(fit_maxent_classifier, elmo_X, elmo_y, 5)

'''
#TorchShallowNeural Classifier w/ Unigram Features
#unigram_X = np.load('pol-balanced-unigram-X.npy')
#unigram_y = np.load('pol-balanced-unigram-y.npy')
xval_model(fit_maxent_classifier, unigram_dataset['X'], unigram_dataset['y'], 5)
'''

#TorchShallowNeural Classifier w/ FastText Embeddings
fasttext_X = np.load('pol-balanced-fasttext-X.npy')
fasttext_y = np.load('pol-balanced-fasttext-y.npy')
xval_model(fit_maxent_classifier, fasttext_X, fasttext_y, 5)


# In[41]:


#TorchShallowNeural Classifier w/ GloVe Embeddings
glove_X = np.load('pol-balanced-glove-X.npy')
glove_y = np.load('pol-balanced-glove-y.npy')
xval_model(fit_maxent_classifier, glove_X, glove_y, 5)
