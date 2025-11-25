import torch
import torch.nn as nn
import math


class StateSpaceModel(nn.Module):
    """
    State Space Model (SSM) for stock price prediction.
    Based on structured state space models like S4.
    
    SSMs are effective for long-range dependencies in time series.
    """
    
    def __init__(self, input_size, state_size=None, output_size=1, num_layers=4, dropout=0.1):
        """
        Initialize the State Space Model.
        
        Args:
            input_size (int): Number of input features
            state_size (int): Dimension of hidden state (default: input_size//4)
            output_size (int): Number of output values (default: 1)
            num_layers (int): Number of SSM layers (default: 4)
            dropout (float): Dropout rate (default: 0.1)
        """
        super(StateSpaceModel, self).__init__()
        
        self.input_size = input_size
        self.state_size = state_size if state_size is not None else max(input_size // 4, 32)
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, state_size)
        
        # Stack of SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMLayer(state_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(state_size)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the SSM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Pass through SSM layers with residual connections
        for ssm_layer, layer_norm in zip(self.ssm_layers, self.layer_norms):
            residual = x
            x = ssm_layer(x)
            x = layer_norm(x + residual)
        
        # Take the last time step
        x = x[:, -1, :]
        
        # Output projection
        output = self.output_proj(x)
        
        return output


class SSMLayer(nn.Module):
    """
    Single State Space Model layer.
    Implements a simplified version of structured state space.
    """
    
    def __init__(self, state_size, dropout=0.1):
        super(SSMLayer, self).__init__()
        
        self.state_size = state_size
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.B = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.C = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.D = nn.Parameter(torch.randn(state_size) * 0.01)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through SSM layer.
        
        Args:
            x (torch.Tensor): Input of shape (batch, seq_length, state_size)
        
        Returns:
            torch.Tensor: Output of shape (batch, seq_length, state_size)
        """
        batch_size, seq_length, _ = x.size()
        
        # Initialize state
        state = torch.zeros(batch_size, self.state_size, device=x.device)
        outputs = []
        
        # Iterate over sequence
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            # State update: s_t = A @ s_{t-1} + B @ x_t
            state = torch.matmul(state, self.A.t()) + torch.matmul(x_t, self.B.t())
            
            # Output: y_t = C @ s_t + D * x_t
            output = torch.matmul(state, self.C.t()) + self.D * x_t
            outputs.append(output)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        output = self.dropout(output)
        
        return output


class MambaModel(nn.Module):
    """
    Mamba-inspired model for stock prediction.
    Simplified implementation inspired by the Mamba architecture.
    Uses selective state space modeling.
    """
    
    def __init__(self, input_size, hidden_size=None, state_size=None, 
                 num_layers=4, dropout=0.1, output_size=1):
        """
        Initialize the Mamba-inspired model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Hidden dimension size (default: input_size*2)
            state_size (int): State space dimension (default: input_size//16)
            num_layers (int): Number of Mamba layers (default: 4)
            dropout (float): Dropout rate (default: 0.1)
            output_size (int): Number of output values (default: 1)
        """
        super(MambaModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size * 2
        self.state_size = state_size if state_size is not None else max(input_size // 16, 16)
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_size, state_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through the Mamba model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, output_size)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Pass through Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Output projection
        output = self.output_head(x)
        
        return output


class MambaBlock(nn.Module):
    """
    Mamba block with selective state space modeling.
    """
    
    def __init__(self, hidden_size, state_size, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Projections for selective SSM
        self.x_proj = nn.Linear(hidden_size, state_size)
        self.delta_proj = nn.Linear(hidden_size, state_size)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.B = nn.Parameter(torch.randn(state_size, hidden_size) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_size, state_size) * 0.01)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Gating
        self.gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        Forward pass through Mamba block.
        
        Args:
            x (torch.Tensor): Input of shape (batch, seq_length, hidden_size)
        
        Returns:
            torch.Tensor: Output of shape (batch, seq_length, hidden_size)
        """
        residual = x
        x = self.norm(x)
        
        batch_size, seq_length, _ = x.size()
        
        # Selective mechanism
        x_proj = self.x_proj(x)
        delta = torch.sigmoid(self.delta_proj(x))
        
        # Initialize state
        state = torch.zeros(batch_size, self.state_size, device=x.device)
        outputs = []
        
        # Selective state space update
        for t in range(seq_length):
            x_t = x_proj[:, t, :]
            delta_t = delta[:, t, :]
            
            # Selective state update
            state = (1 - delta_t) * state + delta_t * (torch.matmul(state, self.A.t()) + x_t)
            
            # Output
            out_t = torch.matmul(state, self.C.t())
            outputs.append(out_t)
        
        output = torch.stack(outputs, dim=1)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        output = gate * output
        
        # Output projection
        output = self.output_proj(output)
        output = self.dropout(output)
        
        # Residual connection
        output = output + residual
        
        return output


class LinearSSM(nn.Module):
    """
    Simplified linear state space model for efficient computation.
    Good for very long sequences.
    """
    
    def __init__(self, input_size, state_size=None, output_size=1, dropout=0.1):
        """
        Initialize Linear SSM.
        
        Args:
            input_size (int): Number of input features
            state_size (int): State dimension (default: input_size//4)
            output_size (int): Number of outputs (default: 1)
            dropout (float): Dropout rate (default: 0.1)
        """
        super(LinearSSM, self).__init__()
        
        self.input_size = input_size
        self.state_size = state_size if state_size is not None else max(input_size // 4, 32)
        
        # Input and output projections
        self.input_proj = nn.Linear(input_size, state_size)
        
        # State space parameters (simplified)
        self.A = nn.Parameter(torch.randn(state_size) * 0.01)
        self.B = nn.Linear(state_size, state_size, bias=False)
        self.C = nn.Linear(state_size, state_size, bias=False)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, state_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (batch, seq_length, n_features)
        
        Returns:
            torch.Tensor: Output of shape (batch, output_size)
        """
        # Project input
        x = self.input_proj(x)
        
        batch_size, seq_length, _ = x.size()
        
        # Diagonal A matrix for efficient computation
        A_diag = torch.diag(self.A)
        
        # Initialize state
        state = torch.zeros(batch_size, self.state_size, device=x.device)
        
        # Process sequence
        for t in range(seq_length):
            x_t = x[:, t, :]
            state = torch.matmul(state, A_diag) + self.B(x_t)
        
        # Final output transformation
        output = self.C(state)
        output = self.output_proj(output)
        
        return output
