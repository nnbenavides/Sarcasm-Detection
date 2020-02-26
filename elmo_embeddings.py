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

# Read in Data
data_dir = '../SARC/2.0/main'
comments_file = os.path.join(data_dir, 'comments.json')
train_file = os.path.join(data_dir, 'train-balanced.csv')

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

train_vocab = defaultdict(int)
for pair in train_responses:
    for comment in pair:
        for w in nltk.word_tokenize(comment):
            train_vocab[w] += 1
train_vocab = Counter(train_vocab)
print(len(train_vocab))
responses = train_responses
print(len(responses))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

elmo = ElmoEmbedder()

batch_size = 64
num_batches = math.ceil(len(responses)/batch_size)
all_example_embeddings = []
for i in range(num_batches):
    batch_examples = []
    if i % 10 == 0:
        print('on batch', i)
    if i == num_batches - 1:
        batch_examples = responses[batch_size*i:]
    else:
        batch_examples = responses[batch_size*i:batch_size*(i+1)]
    
    first_sentence_embeddings = []
    second_sentence_embeddings = []
    
    first_sentences = [nltk.word_tokenize(x[0]) for x in batch_examples]
    second_sentences = [nltk.word_tokenize(x[1]) for x in batch_examples]
    
    first_batch = elmo.embed_batch(first_sentences[:64])
    second_batch = elmo.embed_batch(second_sentences[:64])
    for j, sent in enumerate(first_batch):
        first_sent_embedding = np.mean(sent[2], axis = 0)
        first_sentence_embeddings.append(list(first_sent_embedding))
        second_sent_embedding = np.mean(second_batch[j][2], axis = 0)
        second_sentence_embeddings.append(list(second_sent_embedding))
    example_embeddings = [first_sentence_embeddings[k] + second_sentence_embeddings[k] for k in range(len(first_sentence_embeddings))]
    all_example_embeddings += example_embeddings

np.save('balanced-elmo-X.npy', all_example_embeddings)
labels = [int(x[0]) for x in train_labels]
np.save('balanced-elmo-Y.npy', labels)

