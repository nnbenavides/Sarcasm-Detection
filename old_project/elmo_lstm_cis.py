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

# Function to compute 95% confidence interval based on a normal distribution
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return float(m), float(m-h), float(m+h)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
elmo_X = np.load('main-balanced-elmo-X.npy')
elmo_y = np.load('main-balanced-elmo-y.npy')

# Concatenate X and y matrices
data = []
for i in range(len(elmo_X)):
   data.append([elmo_X[i], elmo_y[i]])
print(len(data))
print(len(data[0]))
print(len(data[0][0]))


# Model parameters
sequence_length = 1
input_size = 2048
num_classes = 2

# Fixed Hyperparameters
batch_size = 32
hidden_size = 500
layers = 1
epochs = 5
learning_rate = 0.001

# Train 10 models, use the resulting F1 scores to generate a 95% confidence interval
f1s = []
for k in range(10):
    # Randomly select training examples, map indicies for error analysis
    n = len(elmo_X)
    train_indices = np.random.choice(range(n), round(0.95*n), replace = False)
    train_indices_subset = np.random.choice(train_indices, round(0.95*n), replace = False)
    elmo_train = []
    for i in train_indices:
        if i in train_indices_subset:
            elmo_train.append(data[i])
    print('length of training list: ', len(elmo_train))
    index_map_train = {}
    for i in range(len(train_indices_subset)):
        index_map_train[i] = train_indices_subset[i]
    print('size of training set:', str(len(index_map_train)))

    # Randomly select dev examples, map indicies for error analysis
    other_indices = list(set(range(n)) - set(train_indices))
    print(len(other_indices))
    dev_indices = np.random.choice(other_indices, round(0.5*len(other_indices)), replace = False)
    elmo_dev = []
    for i in other_indices:
        if i in dev_indices:
            elmo_dev.append(data[i])
    print('length of dev list: ', len(elmo_dev))
    index_map_dev = {}
    for i in range(len(dev_indices)):
        index_map_dev[i] = dev_indices[i]
    print('size of dev set:', str(len(index_map_dev)))

    # Randomly select test examples, map indicies for error analysis
    test_indices = list(set(other_indices) - set(dev_indices))
    elmo_test = [data[i] for i in test_indices]
    print('length of test list: ', len(elmo_test))
    index_map_test = {}
    for i in range(len(test_indices)):
        index_map_test[i] = test_indices[i]
    print('size of test set:', str(len(index_map_test)))

    # Initialize data loaders
    train_loader = torch.utils.data.DataLoader(dataset=elmo_train,
                                               batch_size=batch_size, 
                                               shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dataset=elmo_dev,
                                              batch_size=batch_size, 
                                              shuffle=True)

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

    preds = [item for sublist in preds for item in sublist]
    actuals = [item for sublist in actuals for item in sublist]
    report = classification_report(actuals, preds, digits=3, output_dict = True)
    macro_f1 = report['macro avg']['f1-score']
    f1s.append(macro_f1)
    print(f1s)
    print('\n ran iteration ', str(i+1), ' out of 10')

mean, low, high = mean_confidence_interval(f1s)
print('Low: ', str(low))
print('Mean: ', str(mean))
print('High: ', str(high))
