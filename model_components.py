import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any


class FocalLoss(nn.Module):
    """Focal Loss for binary classification"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure inputs and targets have the same shape
        inputs = inputs.squeeze()  # Remove extra dimension
        targets = targets.float()  # Ensure targets are float

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class LSTMExpert(nn.Module):
    """LSTM model for short-term patterns"""
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32,
                 num_layers: int = 2, dropout: float = 0.2, fc_dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, num_layers=num_layers,
                             dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, num_layers=num_layers,
                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(fc_dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim2)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim2 // 2)
        self.fc2 = nn.Linear(hidden_dim2 // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class WaveNetExpert(nn.Module):
    """WaveNet-inspired architecture for medium-term patterns"""
    def __init__(self, input_dim: int, filters: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(filters, filters, kernel_size=3, padding=4, dilation=4)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(filters)
        self.fc1 = nn.Linear(filters, filters // 2)
        self.fc2 = nn.Linear(filters // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=2)  # Global average pooling
        x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class TransformerExpert(nn.Module):
    """Transformer model for capturing long-term dependencies"""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super(TransformerExpert, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, 100, d_model)  # Max sequence length of 100
        )

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, features]
        seq_len = x.size(1)

        # Embed input
        x = self.embedding(x)

        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take the output for the last token
        x = x[:, -1, :]

        # Output layer
        return self.output_layer(x)


class GatingNetwork(nn.Module):
    """Gating network to dynamically weight expert predictions based on market conditions"""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super(GatingNetwork, self).__init__()

        # First process the market conditions with a feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # Then use these features to determine expert weights
        self.expert_weighting = nn.Linear(hidden_dim, 3)  # 3 experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For sequence data, take the mean across the sequence dimension
        if len(x.shape) == 3:  # [batch_size, seq_len, features]
            x = x.mean(dim=1)  # Average across sequence dimension

        # Extract features
        features = self.feature_extractor(x)

        # Output weights for each expert
        logits = self.expert_weighting(features)
        return torch.softmax(logits, dim=1)


class MixtureOfExperts(nn.Module):
    """Mixture of experts model combining predictions from multiple timeframes"""
    def __init__(self, input_dim: int, seq_lengths: list, config: Dict = None):
        super(MixtureOfExperts, self).__init__()

        # Use configuration if provided, otherwise use defaults
        if config is None:
            config = {
                'lstm': {'hidden_dim1': 64, 'hidden_dim2': 32, 'num_layers': 2,
                         'dropout': 0.2, 'fc_dropout': 0.3},
                'wavenet': {'filters': 32, 'dropout': 0.3},
                'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1},
                'gating': {'hidden_dim': 32}
            }

        # Initialize experts with correct input dimension
        self.short_term_expert = LSTMExpert(
            input_dim,
            hidden_dim1=config['lstm']['hidden_dim1'],
            hidden_dim2=config['lstm']['hidden_dim2'],
            num_layers=config['lstm']['num_layers'],
            dropout=config['lstm']['dropout'],
            fc_dropout=config['lstm']['fc_dropout']
        )

        self.medium_term_expert = WaveNetExpert(
            input_dim,
            filters=config['wavenet']['filters'],
            dropout=config['wavenet']['dropout']
        )

        self.long_term_expert = TransformerExpert(
            input_dim,
            d_model=config['transformer']['d_model'],
            nhead=config['transformer']['nhead'],
            num_layers=config['transformer']['num_layers'],
            dropout=config['transformer']['dropout']
        )

        # Initialize gating network with the same input dimension as the experts
        self.gating_network = GatingNetwork(
            input_dim,
            hidden_dim=config['gating']['hidden_dim']
        )

        # Store sequence lengths
        self.seq_lengths = seq_lengths
        self.input_dim = input_dim

    def forward(self, expert_inputs: Dict[str, torch.Tensor],
                market_conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get predictions from each expert
        expert1_pred = self.short_term_expert(expert_inputs['expert1_input'])
        expert2_pred = self.medium_term_expert(expert_inputs['expert2_input'])
        expert3_pred = self.long_term_expert(expert_inputs['expert3_input'])

        # Stack expert predictions
        expert_preds = torch.stack([
            expert1_pred,
            expert2_pred,
            expert3_pred
        ], dim=1)  # Shape: [batch_size, num_experts, 1]

        # Get gating weights
        # The gating network now handles the sequence dimension internally
        gating_weights = self.gating_network(market_conditions)
        gating_weights = gating_weights.unsqueeze(-1)  # Add dimension for broadcasting

        # Combine predictions
        combined_pred = (expert_preds * gating_weights).sum(dim=1)  # Shape: [batch_size, 1]

        return combined_pred, gating_weights.squeeze(-1), expert_preds.squeeze(-1)
