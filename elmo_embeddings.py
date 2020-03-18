# Imports
import allennlp
import os
import csv
import json
import nltk
import torch
from collections import Counter
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from collections import defaultdict
import math
from utils import *

# Read in Data
print("starting elmo embedding script")
responses = get_textual_examples('../SARC/2.0/main')
print(len(responses))
print("done formatting data")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize ElmoEmbedder
elmo = ElmoEmbedder()

# Run batches of sentences through the ElmoEmbedder
print("starting to run batches of examples through the ElmoEmbedder")
batch_size = 64
num_batches = math.ceil(len(responses)/batch_size)
all_example_embeddings = []
print('num batches', num_batches)
for i in range(num_batches):
    batch_examples = []
    if i % 100 == 0:
        print('on batch', i)
    if i == num_batches - 1:
        batch_examples = responses[batch_size*i:]
    else:
        batch_examples = responses[batch_size*i:batch_size*(i+1)]
    
    first_sentence_embeddings = []
    second_sentence_embeddings = []
    
    first_sentences = [nltk.word_tokenize(x[0]) for x in batch_examples]
    second_sentences = [nltk.word_tokenize(x[1]) for x in batch_examples]
    
    first_batch = elmo.embed_batch(first_sentences)
    second_batch = elmo.embed_batch(second_sentences)
    for j, sent in enumerate(first_batch):
        first_sent_embedding = np.mean(sent[2], axis = 0)
        first_sentence_embeddings.append(list(first_sent_embedding))
        second_sent_embedding = np.mean(second_batch[j][2], axis = 0)
        second_sentence_embeddings.append(list(second_sent_embedding))
    example_embeddings = [first_sentence_embeddings[k] + second_sentence_embeddings[k] for k in range(len(first_sentence_embeddings))]
    all_example_embeddings += example_embeddings
print("starting to save elmo-X file")
np.save('balanced-elmo-X.npy', all_example_embeddings)
print("saved elmo-X file")
labels = [int(x[0]) for x in train_labels]
print("starting to save elmo-Y file")
np.save('balanced-elmo-Y.npy', labels)
print("saved elmo-Y file")

