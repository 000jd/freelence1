import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Any, Tuple, Optional

from model_components import MixtureOfExperts, FocalLoss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def should_stop(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """Train model for one epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        expert_inputs = batch['expert_inputs']
        market_conditions = batch['market_conditions']
        targets = batch['target']

        predictions, _, _ = model(expert_inputs, market_conditions)
        predictions = predictions.squeeze()  # Remove extra dimension
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    """Validate model performance"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            expert_inputs = batch['expert_inputs']
            market_conditions = batch['market_conditions']
            targets = batch['target']

            predictions, _, _ = model(expert_inputs, market_conditions)
            predictions = predictions.squeeze()  # Remove extra dimension
            loss = criterion(predictions, targets)

            total_loss += loss.item()

    return total_loss / len(dataloader)


class TradingEnv(gym.Env):
    """Custom trading environment for reinforcement learning gating network"""

    def __init__(self, dataset, model, window_size=20):
        super(TradingEnv, self).__init__()

        self.dataset = dataset
        self.model = model
        self.window_size = window_size

        # Observation space: market conditions
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )

        # Action space: weights for each expert (0-1 for each)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        # Start at a random position in dataset
        self.position = self.window_size + np.random.randint(0, len(self.dataset) - self.window_size - 50)
        self.done = False
        self.portfolio_value = 1000.0  # Initial portfolio value
        self.position_type = 0  # 0: no position, 1: long, -1: short

        return self._get_observation()

    def _get_observation(self):
        # Get market regime features
        sample = self.dataset[self.position]
        return sample['market_conditions'].numpy()

    def step(self, action):
        # Normalize action to sum to 1
        action = np.array(action)
        action_sum = action.sum()
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.array([1/3, 1/3, 1/3])  # Equal weights if all actions are 0

        # Get current sample
        sample = self.dataset[self.position]

        # Forward pass through model components
        with torch.no_grad():
            expert_inputs = sample['expert_inputs']
            market_conditions = sample['market_conditions']

            # Get expert predictions
            short_term_pred = self.model.short_term_expert(expert_inputs['expert1_input'].unsqueeze(0))
            medium_term_pred = self.model.medium_term_expert(expert_inputs['expert2_input'].unsqueeze(0))
            long_term_pred = self.model.long_term_expert(expert_inputs['expert3_input'].unsqueeze(0))

            # Combine predictions using action weights
            preds = torch.stack([short_term_pred, medium_term_pred, long_term_pred], dim=1)
            combined_pred = torch.sum(preds * torch.tensor(action).unsqueeze(0).unsqueeze(2).float(), dim=1)

        # Trading decision
        trading_signal = 1 if combined_pred.item() > 0.5 else -1

        # Check if prediction was correct
        next_sample = self.dataset[self.position + 1]
        price_change = next_sample['target'].item() * 2 - 1  # Convert to [-1, 1]

        # Calculate reward
        if self.position_type == 0:
            # Enter position
            self.position_type = trading_signal
            reward = 0  # No reward for entering position
        else:
            # Calculate returns
            returns = self.position_type * price_change

            # Update portfolio value
            self.portfolio_value *= (1 + returns * 0.01)  # 1% position size

            # Reward is log return
            reward = np.log(1 + returns * 0.01)

            # Update position
            self.position_type = trading_signal

        # Move to next day
        self.position += 1

        # Check if done
        if self.position >= len(self.dataset) - 2:
            self.done = True

        # Get next observation
        next_observation = self._get_observation()

        info = {
            'portfolio_value': self.portfolio_value,
            'prediction': combined_pred.item(),
            'actual': price_change,
            'action_weights': action
        }

        return next_observation, reward, self.done, info

    def render(self):
        pass


def train_mixture_of_experts(dataset, config) -> MixtureOfExperts:
    """Enhanced training with better regularization and loss function"""
    # Extract configuration parameters
    batch_size = config.get('training', 'batch_size', 32)

    # Create dataloaders
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Get input dimension from a sample batch
    sample_batch = next(iter(train_dataloader))
    input_dim = sample_batch['expert_inputs']['expert1_input'].shape[-1]

    # Get sequence lengths from dataset
    sequence_lengths = dataset.sequence_lengths

    # Create model config
    model_config = {
        'lstm': {
            'hidden_dim1': config.get('model', 'lstm_hidden_dim1', 64),
            'hidden_dim2': config.get('model', 'lstm_hidden_dim2', 32),
            'num_layers': config.get('model', 'lstm_num_layers', 2),
            'dropout': config.get('model', 'lstm_dropout', 0.2),
            'fc_dropout': config.get('model', 'lstm_fc_dropout', 0.3)
        },
        'wavenet': {  # Added missing wavenet configuration
            'filters': config.get('model', 'wavenet_filters', 32),
            'dropout': config.get('model', 'wavenet_dropout', 0.3)
        },
        'transformer': {
            'd_model': config.get('model', 'transformer_d_model', 64),
            'nhead': config.get('model', 'transformer_nhead', 4),
            'num_layers': config.get('model', 'transformer_num_layers', 2),
            'dropout': config.get('model', 'transformer_dropout', 0.1)
        },
        'gating': {
            'hidden_dim': config.get('model', 'gating_hidden_dim', 32)
        }
    }

    # Create model
    model = MixtureOfExperts(input_dim, sequence_lengths, model_config)

    # Create loss function, optimizer, and scheduler
    criterion = FocalLoss(
        alpha=config.get('training', 'focal_loss_alpha', 0.25),
        gamma=config.get('training', 'focal_loss_gamma', 2)
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training', 'learning_rate', 0.001),
        weight_decay=config.get('training', 'weight_decay', 0.01)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get('training', 'scheduler_T_0', 10),
        T_mult=config.get('training', 'scheduler_T_mult', 2),
        eta_min=config.get('training', 'learning_rate', 0.001) * config.get('training', 'scheduler_eta_min_factor', 0.0001)
    )

    early_stopper = EarlyStopping(
        patience=config.get('training', 'early_stopping_patience', 10),
        min_delta=config.get('training', 'early_stopping_min_delta', 0.001)
    )

    # Training loop
    epochs = config.get('training', 'epochs', 50)
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer)

        model.eval()
        val_loss = validate(model, val_dataloader, criterion)

        scheduler.step()

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        if early_stopper.should_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    return model


def train_gating_with_rl(model: MixtureOfExperts, dataset, config) -> MixtureOfExperts:
    """Train the gating network using reinforcement learning"""
    # Extract RL configuration parameters
    total_timesteps = config.get('rl', 'total_timesteps', 5000)
    learning_rate = config.get('rl', 'learning_rate', 0.0003)
    n_steps = config.get('rl', 'n_steps', 64)
    batch_size = config.get('rl', 'batch_size', 32)
    n_epochs = config.get('rl', 'n_epochs', 10)
    gamma = config.get('rl', 'gamma', 0.99)
    gae_lambda = config.get('rl', 'gae_lambda', 0.95)
    clip_range = config.get('rl', 'clip_range', 0.2)

    # Skip RL training if dataset is too small
    if len(dataset) < 100:
        print("Dataset too small for RL training. Skipping.")
        return model

    print("Starting reinforcement learning training for gating network...")

    # Create custom environment
    env = TradingEnv(dataset, model)

    # Wrap environment in DummyVecEnv for stable-baselines
    vec_env = DummyVecEnv([lambda: env])

    try:
        # Create PPO agent
        agent = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=0
        )

        # Train agent
        agent.learn(total_timesteps=total_timesteps)

        print("Reinforcement learning for gating network completed")

        # Note: In a full implementation, we would extract the policy network weights
        # and update the gating network. For simplicity, we're keeping the separately
        # trained gating network as is.
    except Exception as e:
        print(f"Error in RL training: {e}")
        print("Continuing with supervised-trained gating network")

    return model
