# Author: Nicholas Benavides
# Code liberally inspired by and lifted from:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py

# Imports
import torch 
import torch.nn as nn
from torch.utils import data
import numpy as np
from sklearn.metrics import classification_report
from torch.autograd import Variable
import os
import csv
import json
from itertools import islice, chain
import nltk
from collections import Counter

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
responses = train_responses

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
elmo_X = np.load('main-balanced-elmo-X.npy')
elmo_y = np.load('main-balanced-elmo-y.npy')

glove_X = np.load('main-balanced-glove-X.npy')
glove_y = np.load('main-balanced-glove-y.npy')

# Concatenate X and y matrices
elmo_data = []
for i in range(len(elmo_X)):
   elmo_data.append([elmo_X[i], elmo_y[i]])
print(len(elmo_data))
print(len(elmo_data[0]))
print(len(elmo_data[0][0]))

glove_data = []
for i in range(len(glove_X)):
   glove_data.append([glove_X[i], glove_y[i]])

# Randomly select training examples, map indicies for error analysis
n = len(elmo_X)
np.random.seed(224)
train_indices = np.random.choice(range(n), round(0.95*n), replace = False)
train_indices_subset = np.random.choice(train_indices, round(0.95*n), replace = False)
elmo_train = []
glove_train = []
for i in train_indices:
    if i in train_indices_subset:
        elmo_train.append(elmo_data[i])
        glove_train.append(glove_data[i])
print('length of training list: ', len(elmo_train))

# Randomly select dev examples, map indicies for error analysis
other_indices = list(set(range(n)) - set(train_indices))
print(len(other_indices))
dev_indices = np.random.choice(other_indices, round(0.5*len(other_indices)), replace = False)
elmo_dev = []
glove_dev = []
for i in other_indices:
    if i in dev_indices:
        elmo_dev.append(elmo_data[i])
        glove_dev.append(glove_data[i])
print('length of dev list: ', len(elmo_dev))

# Randomly select test examples, map indicies for error analysis
test_indices = list(set(other_indices) - set(dev_indices))
elmo_test = [elmo_data[i] for i in test_indices]
glove_test = [glove_data[i] for i in test_indices]
print('length of test list: ', len(elmo_test))

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(device)
        # Forward propagate LSTM
        x = x.unsqueeze(0).float()
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        return out

# Model parameters
sequence_length = 1
input_size = 2048
num_classes = 2

# Fixed Hyperparameters
batch_size = 32

# Hyperparameters
hidden_size = 250
layers = 5
epochs = 5
learning_rate = 0.001 

# Initialize data loaders
train_loader = torch.utils.data.DataLoader(dataset=elmo_train,
                                           batch_size=batch_size, 
                                           shuffle=True)

dev_loader = torch.utils.data.DataLoader(dataset=elmo_dev,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Initialize model
model = BiRNN(input_size, hidden_size, layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(epochs):
    for j, (sequences, labels) in enumerate(train_loader):
        
        sequences = sequences.view(sequences.shape[0], input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

# Format predictions so we can compute macro avg F1 score
with torch.no_grad():
    correct = 0
    total = 0
    preds = []
    actuals = []
    for sequences, labels in dev_loader:
        sequences = sequences.view(sequences.shape[0], input_size).to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        preds.append(predicted.tolist())
        actuals.append(labels.tolist())

elmo_preds = [item for sublist in preds for item in sublist]
dev_actuals = [item for sublist in actuals for item in sublist]

# Model parameters
sequence_length = 1
input_size = 600
num_classes = 2

# Fixed Hyperparameters
batch_size = 32

# Hyperparameters
hidden_size = 100
layers = 1
epochs = 10
learning_rate = 0.001

# Initialize data loaders
train_loader = torch.utils.data.DataLoader(dataset=glove_train,
                                           batch_size=batch_size, 
                                           shuffle=True)

dev_loader = torch.utils.data.DataLoader(dataset=glove_dev,
                                          batch_size=batch_size, 
                                          shuffle=False)
# Initialize model
model = BiRNN(input_size, hidden_size, layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(epochs):
    for j, (sequences, labels) in enumerate(train_loader):
        
        sequences = sequences.view(sequences.shape[0], input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

# Format predictions so we can compute macro avg F1 score
with torch.no_grad():
    correct = 0
    total = 0
    preds = []
    actuals = []
    for sequences, labels in dev_loader:
        sequences = sequences.view(sequences.shape[0], input_size).to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        preds.append(predicted.tolist())
        actuals.append(labels.tolist())

glove_preds = [item for sublist in preds for item in sublist]

# Error Analysis
error_analysis_examples = []
elmo_correct = 0
glove_correct = 0
overlap = 0
for i, og_ind in enumerate(dev_indices):
    elmo_pred = elmo_preds[i]
    glove_pred = glove_preds[i]
    actual = dev_actuals[i]
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

for i, tup in enumerate(error_examples):
    print('\nExample #', str(i+1), ' of ', str(num_examples_to_analyze))
    print('Winner: ', tup[2])
    print('Actual non-sarcastic comment: ', tup[0])
    print('Word count: ', len(tup[0].split()))
    print('Actual sarcastic comment: ', tup[1])
    print('Word count: ', len(tup[1].split()))

