import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    LSTM Neural Network for stock price prediction.
    
    Architecture:
    - Input: (batch, seq_length, n_features)
    - LSTM layers with dropout
    - Fully connected output layer
    - Output: (batch, 1) predicted price
    """
    
    def __init__(self, input_size, hidden_size=None, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features (e.g., 356 from unified array)
            hidden_size (int): Number of hidden units in LSTM (default: input_size*2)
            num_layers (int): Number of LSTM layers (default: 2)
            dropout (float): Dropout rate between LSTM layers (default: 0.2)
            output_size (int): Number of output values (default: 1 for single price prediction)
        """
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size * 2
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch, seq_length, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        # c_n shape: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last time step
        # lstm_out[:, -1, :] shape: (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(last_output)
        
        return output
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden and cell states for LSTM.
        
        Args:
            batch_size (int): Size of the batch
            device (str): Device to create tensors on ('cpu' or 'cuda')
        
        Returns:
            tuple: (h_0, c_0) initial hidden and cell states
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for stock price prediction.
    Processes sequences in both forward and backward directions.
    """
    
    def __init__(self, input_size, hidden_size=None, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the Bidirectional LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM (default: input_size*2)
            num_layers (int): Number of LSTM layers (default: 2)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(BidirectionalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size * 2
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer (input size doubled due to bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last time step
        # For bidirectional, concatenates forward and backward outputs
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(last_output)
        
        return output


class LSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism for stock price prediction.
    Attention helps the model focus on important time steps.
    """
    
    def __init__(self, input_size, hidden_size=None, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the LSTM with Attention model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM (default: input_size*2)
            num_layers (int): Number of LSTM layers (default: 2)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(LSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size * 2
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network with attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch, seq_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention weights
        # attention_weights shape: (batch, seq_length, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention weights
        # context_vector shape: (batch, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Pass through output layer
        output = self.fc(context_vector)
        
        return output
