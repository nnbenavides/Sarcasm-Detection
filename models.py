import torch
import torch.nn as nn

# Bidirectional recurrent neural network (many-to-one)
class BiLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device, lstm_dropout = 0.0, other_dropout = 0.0):
		super(BiLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		self.p_dropout = other_dropout
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True, dropout = lstm_dropout)
		self.dropout = nn.Dropout(other_dropout)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device) # 2 for bidirection
		c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device)

		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		if self.p_dropout > 0.0:
			out = self.dropout(out)
		# Decode the hidden state of the last time step
		out = self.fc(out[-1, :, :])
		return out

class BiGRU(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device, gru_dropout = 0.0, other_dropout = 0.0):
		super(BiGRU, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		self.p_dropout = other_dropout
		self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True, dropout = gru_dropout)
		self.dropout = nn.Dropout(other_dropout)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device) # 2 for bidirection
		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		if self.p_dropout > 0.0:
			out = self.dropout(out)
		# Decode the hidden state of the last time step
		out = self.fc(out[-1, :, :])
		return out

class BiLSTMLin(nn.Module):
	def __init__(self, input_size, hidden_sizes, num_layers, num_classes, device, lstm_dropout = 0.0, other_dropout = 0.0):
		super(BiLSTMLin, self).__init__()
		self.lstm_hidden_size = hidden_sizes[0]
		self.linear_hidden_size = hidden_sizes[1]
		self.num_layers = num_layers
		self.device = device
		self.lstm = nn.LSTM(input_size, self.lstm_hidden_size, num_layers, batch_first=False, bidirectional=True, dropout = lstm_dropout)
		self.fc1 = nn.Linear(self.lstm_hidden_size*2, self.linear_hidden_size)
		self.dropout = nn.Dropout(other_dropout)
		self.p_dropout = other_dropout
		self.fc2 = nn.Linear(self.linear_hidden_size, num_classes)  # 2 for bidirection

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_hidden_size).float().to(self.device) # 2 for bidirection
		c0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_hidden_size).float().to(self.device)

		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out = self.fc1(out)
		if self.p_dropout > 0.0:
			out = self.dropout(out)
		# Decode the hidden state of the last time step
		out = self.fc2(out[-1, :, :])
		return out

class BiGRULin(nn.Module):
	def __init__(self, input_size, hidden_sizes, num_layers, num_classes, device, gru_dropout = 0.0, other_dropout = 0.0):
		super(BiGRULin, self).__init__()
		self.lstm_hidden_size = hidden_sizes[0]
		self.linear_hidden_size = hidden_sizes[1]
		self.num_layers = num_layers
		self.device = device
		self.gru = nn.GRU(input_size, self.lstm_hidden_size, num_layers, batch_first=False, bidirectional=True, dropout = gru_dropout)
		self.fc1 = nn.Linear(self.lstm_hidden_size*2, self.linear_hidden_size)
		self.dropout = nn.Dropout(other_dropout)
		self.p_dropout = other_dropout
		self.fc2 = nn.Linear(self.linear_hidden_size, num_classes)  # 2 for bidirection

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.lstm_hidden_size).float().to(self.device) # 2 for bidirection

		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out = self.fc1(out)
		if self.p_dropout > 0.0:
			out = self.dropout(out)
		# Decode the hidden state of the last time step
		out = self.fc2(out[-1, :, :])
		return out

class BiLSTMAttn(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device, lstm_dropout = 0.0, other_dropout = 0.0):
		super(BiLSTMAttn, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		self.p_dropout = other_dropout
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True, dropout = lstm_dropout)
		self.attention_layer = Attention(hidden_size*2)
		self.dropout = nn.Dropout(other_dropout)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device) # 2 for bidirection
		c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device)

		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out = out.permute(1, 0, 2) # batch first

		out = self.attention_layer(out)

		if self.p_dropout > 0.0:
			out = self.dropout(out)

		# Decode the hidden state of the last time step
		out = self.fc(out)

		return out

class BiGRUAttn(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device, gru_dropout = 0.0, other_dropout = 0.0):
		super(BiGRUAttn, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device
		self.p_dropout = other_dropout
		self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True, dropout = gru_dropout)
		self.attention_layer = Attention(hidden_size*2)
		self.dropout = nn.Dropout(other_dropout)
		self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).float().to(self.device) # 2 for bidirection
		# Forward propagate LSTM
		x = x.unsqueeze(0).float()
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		out = out.permute(1, 0, 2) # batch first

		out = self.attention_layer(out)

		if self.p_dropout > 0.0:
			out = self.dropout(out)
		# Decode the hidden state of the last time step
		out = self.fc(out)
		return out

# Code for attention layer inspired by https://gist.githubusercontent.com/MLWhiz/1ac0841f0333a97396d300b8f4c247c9/raw/aa352c54d00f801ea1579790652ff8ebb160b01b/pytorch_attention.py
class Attention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.feature_dim = feature_dim

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        feature_dim = self.feature_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, x.size(0))

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
