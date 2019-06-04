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

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
elmo_X = np.load('pol-balanced-elmo-X.npy')
elmo_y = np.load('pol-balanced-elmo-y.npy')

# Concatenate X and y matrices
data = []
for i in range(len(elmo_X)):
   data.append([elmo_X[i], elmo_y[i]])
print(len(data))
print(len(data[0]))
print(len(data[0][0]))

# Randomly select training examples, map indicies for error analysis
n = len(elmo_X)
np.random.seed(224)
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
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float()#.to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float()#.to(device)
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

# Hyperparameters to experiment with
hidden_sizes = [100, 250, 500, 1000]
num_layers = [1, 5, 10]
num_epochs = [5, 10, 20]
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

np.random.seed(1738)
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
    model = BiRNN(input_size, hidden_size, layers, num_classes)#.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        for j, (sequences, labels) in enumerate(train_loader):
            
            sequences = sequences.view(sequences.shape[0], input_size)#.to(device)
            labels = labels#.to(device)
            
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
            sequences = sequences.view(sequences.shape[0], input_size)#.to(device)
            labels = labels#.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted.tolist())
            actuals.append(labels.tolist())

    preds = [item for sublist in preds for item in sublist]
    actuals = [item for sublist in actuals for item in sublist]
    report = classification_report(actuals, preds, digits=3, output_dict = True)
    macro_f1 = report['macro avg']['f1-score']
    print('\nfit model ', str(i), ' out of 25')

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
torch.save(best_model.state_dict(), 'model.ckpt')