import torch
import torch.nn as nn


class StockCNN(nn.Module):
    """
    1D CNN for stock price prediction.
    Uses temporal convolutions to extract patterns from time series data.
    
    Architecture:
    - Input: (batch, seq_length, n_features) or (batch, n_features, seq_length)
    - Multiple 1D convolutional layers
    - Max pooling
    - Fully connected layers
    - Output: (batch, 1) predicted price
    """
    
    def __init__(self, input_size, seq_length, num_filters=None, kernel_size=3, 
                 num_conv_layers=3, dropout=0.2, output_size=1):
        """
        Initialize the CNN model.
        
        Args:
            input_size (int): Number of input features (e.g., 356 from unified array)
            seq_length (int): Length of input sequences
            num_filters (int): Number of filters in conv layers (default: input_size//4)
            kernel_size (int): Size of convolutional kernel (default: 3)
            num_conv_layers (int): Number of convolutional layers (default: 3)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(StockCNN, self).__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_filters = num_filters if num_filters is not None else max(input_size // 4, 32)
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.output_size = output_size
        
        # Convolutional layers
        conv_layers = []
        in_channels = input_size
        
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters * (2 ** i),
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])
            in_channels = num_filters * (2 ** i)
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolutions
        self.flat_size = self._get_flat_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def _get_flat_size(self):
        """Calculate the flattened size after conv layers."""
        # Create dummy input: (1, n_features, seq_length)
        dummy_input = torch.zeros(1, self.input_size, self.seq_length)
        dummy_output = self.conv_layers(dummy_input)
        return dummy_output.view(1, -1).size(1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # Reshape input from (batch, seq_length, n_features) to (batch, n_features, seq_length)
        x = x.transpose(1, 2)
        
        # Pass through convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        output = self.fc_layers(x)
        
        return output


class ResidualCNN(nn.Module):
    """
    CNN with residual connections for stock price prediction.
    Residual connections help with gradient flow in deeper networks.
    """
    
    def __init__(self, input_size, seq_length, num_filters=None, kernel_size=3, 
                 num_blocks=3, dropout=0.2, output_size=1):
        """
        Initialize the Residual CNN model.
        
        Args:
            input_size (int): Number of input features
            seq_length (int): Length of input sequences
            num_filters (int): Number of filters in conv layers (default: input_size//4)
            kernel_size (int): Size of convolutional kernel (default: 3)
            num_blocks (int): Number of residual blocks (default: 3)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(ResidualCNN, self).__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_filters = num_filters if num_filters is not None else max(input_size // 4, 32)
        self.output_size = output_size
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters, num_filters, kernel_size, dropout)
            for _ in range(num_blocks)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.fc = nn.Linear(num_filters, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # Reshape: (batch, seq_length, n_features) -> (batch, n_features, seq_length)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Output
        output = self.fc(x)
        
        return output


class ResidualBlock(nn.Module):
    """
    Residual block for CNN.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class CNN_LSTM_Hybrid(nn.Module):
    """
    Hybrid model combining CNN and LSTM.
    CNN extracts spatial features, LSTM captures temporal dependencies.
    """
    
    def __init__(self, input_size, seq_length, num_filters=None, kernel_size=3,
                 lstm_hidden=None, lstm_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the CNN-LSTM hybrid model.
        
        Args:
            input_size (int): Number of input features
            seq_length (int): Length of input sequences
            num_filters (int): Number of CNN filters (default: input_size//4)
            kernel_size (int): CNN kernel size (default: 3)
            lstm_hidden (int): LSTM hidden size (default: input_size//2)
            lstm_layers (int): Number of LSTM layers (default: 2)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(CNN_LSTM_Hybrid, self).__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        num_filters = num_filters if num_filters is not None else max(input_size // 4, 32)
        lstm_hidden = lstm_hidden if lstm_hidden is not None else max(input_size // 2, 64)
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, 
                     padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, 
                     padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=num_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(lstm_hidden, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # CNN feature extraction
        # Reshape: (batch, seq_length, n_features) -> (batch, n_features, seq_length)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        
        # Reshape back for LSTM: (batch, seq_length, conv_features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Output
        output = self.fc(last_output)
        
        return output
