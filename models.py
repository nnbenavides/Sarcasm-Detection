import torch 
import torch.nn as nn

# Bidirectional recurrent neural network (many-to-one)
class BiLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
		super(BiLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
	
	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device) # 2 for bidirection 
		c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device)
		
		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		# Decode the hidden state of the last time step
		out = self.fc(out[-1, :, :])
		return out

class BiGRU(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
		super(BiGRU, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
	
	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device) # 2 for bidirection
		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		# Decode the hidden state of the last time step
		out = self.fc(out[-1, :, :])
		return out

class BiLSTMLin(nn.Module):
	def __init__(self, input_size, hidden_sizes, num_layers, num_classes, device):
		super(BiLSTMLin, self).__init__()
		self.lstm_hidden_size = hidden_sizes[0]
		self.linear_hidden_size = hidden_sizes[1]
		self.num_layers = num_layers
		self.device = device
		self.lstm = nn.LSTM(input_size, self.lstm_hidden_size, num_layers, batch_first=False, bidirectional=True)
		self.fc1 = nn.Linear(self.lstm_hidden_size*2, self.linear_hidden_size)
		self.fc2 = nn.Linear(self.linear_hidden_size, num_classes)  # 2 for bidirection
	
	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_hidden_size).float().to(self.device) # 2 for bidirection 
		c0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_hidden_size).float().to(self.device)
		
		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out = self.fc1(out)
		# Decode the hidden state of the last time step
		out = self.fc2(out[-1, :, :])
		return out

class BiGRULin(nn.Module):
	def __init__(self, input_size, hidden_sizes, num_layers, num_classes, device):
		super(BiGRULin, self).__init__()
		self.lstm_hidden_size = hidden_sizes[0]
		self.linear_hidden_size = hidden_sizes[1]
		self.num_layers = num_layers
		self.device = device
		self.gru = nn.GRU(input_size, self.lstm_hidden_size, num_layers, batch_first=False, bidirectional=True)
		self.fc1 = nn.Linear(self.lstm_hidden_size*2, self.linear_hidden_size)
		self.fc2 = nn.Linear(self.linear_hidden_size, num_classes)  # 2 for bidirection
	
	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_hidden_size).float().to(self.device) # 2 for bidirection 

		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out = self.fc1(out)
		# Decode the hidden state of the last time step
		out = self.fc2(out[-1, :, :])
		return out

