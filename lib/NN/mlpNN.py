import torch
import torch.nn as nn


class StockMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (Feedforward Neural Network) for stock price prediction.
    
    Architecture:
    - Input: (batch, n_features)
    - Multiple fully connected hidden layers with ReLU
    - Output: (batch, 1) predicted price
    
    This is a simple baseline model without temporal modeling.
    Works directly on feature vectors (no sequence required).
    """
    
    def __init__(self, input_size, hidden_sizes=None, dropout=0.2, output_size=1):
        """
        Initialize the MLP model.
        
        Args:
            input_size (int): Number of input features (e.g., 356 from unified array)
            hidden_sizes (list): List of hidden layer sizes (default: [input_size*2, input_size*2])
            dropout (float): Dropout rate between layers (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(StockMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [input_size * 2, input_size * 2]
        self.output_size = output_size
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        return self.network(x)


class DeepMLP(nn.Module):
    """
    Deeper MLP with batch normalization for stock price prediction.
    Batch normalization helps stabilize training in deeper networks.
    """
    
    def __init__(self, input_size, hidden_sizes=None, 
                 dropout=0.3, use_batchnorm=True, output_size=1):
        """
        Initialize the Deep MLP model.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (list): List of hidden layer sizes (default: [input_size*2, input_size*2, input_size, input_size//2])
            dropout (float): Dropout rate (default: 0.3)
            use_batchnorm (bool): Whether to use batch normalization (default: True)
            output_size (int): Number of output values (default: 1)
        """
        super(DeepMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [input_size * 2, input_size * 2, input_size, input_size // 2]
        self.output_size = output_size
        self.use_batchnorm = use_batchnorm
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections for better gradient flow.
    Good for very deep networks.
    """
    
    def __init__(self, input_size, hidden_size=None, num_blocks=4, 
                 dropout=0.2, output_size=1):
        """
        Initialize the Residual MLP model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layers (default: input_size*2)
            num_blocks (int): Number of residual blocks (default: 4)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(ResidualMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size * 2
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_size, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        output = self.output_layer(x)
        
        return output


class MLPResidualBlock(nn.Module):
    """
    Residual block for MLP.
    """
    
    def __init__(self, hidden_size, dropout=0.2):
        super(MLPResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input of shape (batch, hidden_size)
        
        Returns:
            torch.Tensor: Output of shape (batch, hidden_size)
        """
        residual = x
        out = self.block(x)
        out = out + residual  # Residual connection
        out = self.relu(out)
        return out


class SimpleMLP(nn.Module):
    """
    Very simple 2-layer MLP: input → (input*2) → ReLU → (input*2) → ReLU → output
    
    Good as a baseline or for quick experiments.
    """
    
    def __init__(self, input_size, hidden_size=None, output_size=1):
        """
        Initialize the Simple MLP.
        
        Args:
            input_size (int): Number of input features (e.g., 2 or 356)
            hidden_size (int): Size of hidden layers (default: input_size*2)
            output_size (int): Number of output values (default: 1)
        """
        super(SimpleMLP, self).__init__()
        
        if hidden_size is None:
            hidden_size = input_size * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        return self.network(x)


class AttentionMLP(nn.Module):
    """
    MLP with self-attention mechanism.
    Learns which features are most important for prediction.
    """
    
    def __init__(self, input_size, hidden_size=None, num_heads=8, 
                 dropout=0.2, output_size=1):
        """
        Initialize the Attention MLP model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layers (default: input_size*2)
            num_heads (int): Number of attention heads (default: 8)
            dropout (float): Dropout rate (default: 0.2)
            output_size (int): Number of output values (default: 1)
        """
        super(AttentionMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size * 2
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention (treating features as sequence)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add dummy sequence dimension for attention
        x = x.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.layer_norm(x + ffn_out)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Output
        output = self.output_layer(x)
        
        return output
