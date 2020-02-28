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
import models
from utils import *

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
elmo_X = np.load('balanced-elmo-X.npy')
elmo_y = np.load('balanced-elmo-y.npy')

# Concatenate X and y matrices
data = []
for i in range(len(elmo_X)):
   data.append([elmo_X[i], elmo_y[i]])
print(len(data))
print(len(data[0]))
print(len(data[0][0]))

# Randomly select training examples, map indicies for error analysis
n = len(elmo_X)
elmo_train, index_map_train, train_indices = get_training_indices(data, n, 0.95, False)

# Randomly select dev examples, map indicies for error analysis
elmo_dev, index_map_dev, dev_indices, other_indices = get_dev_indices(data, train_indices, n, 0.5)

# Randomly select test examples, map indicies for error analysis
elmo_test, index_map_test = get_test_indices(data, other_indices, dev_indices)

# Model parameters
sequence_length = 1
input_size = 2048
num_classes = 2

# Fixed Hyperparameters
batch_size = 32

# Hyperparameters to experiment with
hidden_sizes = [64, 128, 256, 512]
num_layers = [1]
num_epochs = [10, 20, 30, 50]
learning_rates = [0.1, 0.01, 0.001]

# Initialize data loaders
train_loader = torch.utils.data.DataLoader(dataset=elmo_train,
                                           batch_size=batch_size, 
                                           shuffle=True)

dev_loader = torch.utils.data.DataLoader(dataset=elmo_dev,
                                          batch_size=batch_size, 
                                          shuffle=False)

# initialize variables to track best model
best_size = 0
best_layers = 0
best_epochs = 0
best_eta = 0
best_f1 = 0
best_model = None

# Hyperparameter search
for i in range(25):
    # Randomly select parameters
    hidden_size_ind = np.random.randint(0, len(hidden_sizes))
    hidden_size = hidden_sizes[hidden_size_ind]
    hidden_layer_ind = np.random.randint(0, len(num_layers))
    layers = num_layers[hidden_layer_ind]
    epochs_ind = np.random.randint(0, len(num_epochs))
    epochs = num_epochs[epochs_ind]
    lr_ind = np.random.randint(0, len(learning_rates))
    learning_rate = learning_rates[lr_ind]

    # Initialize model
    model = models.BiLSTM(input_size, hidden_size, layers, num_classes, device).to(device)
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
    print('\nfit model ', str(i+1), ' out of 25')

    # Update best model parameters if we achieve a new best F1
    if macro_f1 > best_f1:
        best_size = hidden_size
        best_layers = layers
        best_epochs = epochs
        best_eta = learning_rate
        best_f1 = macro_f1
        best_model = model
        print('\nupdated best parameters')

print('Best F1: ',str(best_f1))
print('Best hidden_size: ',str(best_size))
print('Best num_layers: ', str(best_layers))
print('Best num_epochs: ', str(best_epochs))
print('Best learning_rate: ', str(best_eta))
#torch.save(best_model.state_dict(), 'model.ckpt')

# Error Analysis
write_errors('errors.txt', preds, actuals, index_map_dev)