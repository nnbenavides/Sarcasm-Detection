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


# In[4]:


pol_dir = '../SARC/2.0/pol'
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


#unigram_dataset = build_dataset(train_ancestors, train_responses, train_labels, unigrams_phi_c, None, concat_phi_r)
#unigram_dataset['X'].shape
#np.save('pol-balanced-unigram-X.npy', unigram_dataset['X'])
#np.save('pol-balanced-unigram-y.npy', unigram_dataset['y'])


# In[21]:


'''fasttext_dataset = build_dataset(
    train_ancestors, train_responses, train_labels, fasttext_phi_c, None, concat_phi_r, None, False)

fasttext_dataset['X'].shape
np.save('pol-balanced-fasttext-X.npy', fasttext_dataset['X'])
np.save('pol-balanced-fasttext-y.npy', fasttext_dataset['y'])
'''


# In[26]:


def glove_phi_c(comment):
    return embed_phi_c(comment, glove_lookup)


# In[27]:


'''glove_dataset = build_dataset(
    train_ancestors, train_responses, train_labels, glove_phi_c, None, concat_phi_r, None, False)

fasttext_dataset['X'].shape
np.save('pol-balanced-glove-X.npy', glove_dataset['X'])
np.save('pol-balanced-glove-y.npy', glove_dataset['y'])
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


'''elmo_dataset = build_dataset(
    train_ancestors, train_responses, train_labels, elmo_phi_c, None, concat_phi_r, None, False)
np.save('pol-balanced-elmo-X.npy', elmo_dataset['X'])
np.save('pol-balanced-elmo-y.npy', elmo_dataset['y'])
'''


# In[37]:


def fit_basic_rnn(X, y):  
    mod = TorchShallowNeuralClassifier(hidden_dim = 200, max_iter = 100)
    mod.fit(X, y)
    return mod


# In[35]:


#TorchShallowNeural Classifier w/ ELMo Embeddings
elmo_X = np.load('pol-balanced-elmo-X.npy')
elmo_y = np.load('pol-balanced-elmo-y.npy')
xval_model(fit_basic_rnn, elmo_X, elmo_y, 5)


# In[38]:


#TorchShallowNeural Classifier w/ Unigram Features
unigram_X = np.load('pol-balanced-unigram-X.npy')
unigram_y = np.load('pol-balanced-unigram-y.npy')
xval_model(fit_basic_rnn, unigram_X, unigram_y, 5)


# In[40]:


#TorchShallowNeural Classifier w/ FastText Embeddings
fasttext_X = np.load('pol-balanced-fasttext-X.npy')
fasttext_y = np.load('pol-balanced-fasttext-y.npy')
xval_model(fit_basic_rnn, fasttext_X, fasttext_y, 5)


# In[41]:


#TorchShallowNeural Classifier w/ GloVe Embeddings
glove_X = np.load('pol-balanced-glove-X.npy')
glove_y = np.load('pol-balanced-glove-y.npy')
xval_model(fit_basic_rnn, glove_X, glove_y, 5)


# In[ ]:




