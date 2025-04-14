import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any, Optional


class MultiAssetDataset(Dataset):
    def __init__(self,
                 csv_file: str,
                 sequence_lengths: List[int] = [10, 20, 60],
                 target_asset: str = 'S&P_500_return',
                 target_horizon: int = 20,
                 test_size: float = 0.2,
                 val_size: float = 0.1):
        """
        Dataset for multi-asset financial data following best practices for stock prediction:

        1. Uses log returns instead of raw prices to make data stationary
        2. Implements a binary classification approach (up/down) instead of regression
        3. Uses multiple timeframes to capture different patterns (short/medium/long term)
        4. Predicts further into the future (e.g., 20 days ≈ 1 month ahead)
        5. Uses sufficient historical data (e.g., 60 days ≈ 3 months) for prediction

        Args:
            csv_file: Path to CSV file with financial data
            sequence_lengths: List of sequence lengths for different timeframes (default: [10, 20, 60])
            target_asset: Target asset to predict (default: 'S&P_500_return')
            target_horizon: Number of days ahead to predict (default: 20 days ≈ 1 month)
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
        """
        # Load data from CSV
        self.df = pd.read_csv(csv_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
        self.df = self.df.sort_values('Date')

        # Store parameters
        self.sequence_lengths = sequence_lengths
        self.max_seq_len = max(sequence_lengths)
        self.target_horizon = target_horizon
        self.target_asset = target_asset

        # Preprocess data
        self.preprocess_data()

        # Calculate target_idx once
        self.target_idx = list(self.df.columns).index(self.target_asset) - 1  # -1 because we dropped 'Date'

        # Create train/val/test splits
        self.train_indices, self.val_indices, self.test_indices = self.create_train_val_test_split(
            test_size=test_size,
            val_size=val_size
        )

        # Scale data using only training data
        self.scale_data()

        # Convert to tensor
        self.data = torch.FloatTensor(self.normalized_data)

    def preprocess_data(self) -> None:
        """Preprocess the dataset following best practices for stock prediction:
        1. Convert prices to log returns to remove trend and make stationary
        2. Handle missing values and outliers
        3. Prepare data for classification (up/down) instead of regression
        """
        # Convert string numbers with commas to float
        for col in self.df.columns:
            if col != 'Date':
                try:
                    self.df[col] = self.df[col].astype(str).str.replace(',', '').astype(float)
                except:
                    continue

        # Handle missing values
        self.df = self.df.ffill()  # Forward fill (use previous day's value)
        self.df = self.df.bfill()  # Backward fill for any remaining NaNs

        # Replace NaNs in volume columns with 0
        volume_columns = [col for col in self.df.columns if 'Vol' in col]
        for col in volume_columns:
            self.df[col] = self.df[col].fillna(0)

        # Handle outliers in crypto volumes which can be extreme
        crypto_volumes = ["Bitcoin_Vol.", "Ethereum_Vol."]
        self.df[crypto_volumes] = self.df[crypto_volumes].apply(self.clip_outliers_iqr)

        # Convert prices to log returns
        # This is a best practice for financial time series because:
        # 1. It makes the series more stationary (removes trend)
        # 2. Log returns can be summed over time to get cumulative returns
        # 3. It normalizes the scale across different assets
        price_columns = [col for col in self.df.columns if 'Price' in col]
        for col in price_columns:
            return_col = col.replace("_Price", "_return")
            self.df[col] = np.log(self.df[col] / self.df[col].shift(1))
            self.df.rename(columns={col: return_col}, inplace=True)

            # Update target asset name if it's being transformed
            if col == self.target_asset:
                self.target_asset = return_col

        # Drop rows with NaN values (first row due to returns calculation)
        self.df = self.df.dropna()

    def clip_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Clip outliers based on IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)

    def create_train_val_test_split(self, test_size: float, val_size: float) -> Tuple[List[int], List[int], List[int]]:
        """Create train/validation/test split indices preserving time order"""
        total_samples = len(self.df) - self.max_seq_len - self.target_horizon
        test_split_idx = int(total_samples * (1 - test_size))
        val_split_idx = int(test_split_idx * (1 - val_size))

        return (
            list(range(val_split_idx)),
            list(range(val_split_idx, test_split_idx)),
            list(range(test_split_idx, total_samples))
        )

    def scale_data(self) -> None:
        """Scale the data using StandardScaler"""
        # Get features to scale (all columns except Date)
        features_to_scale = self.df.columns.drop('Date')

        # Initialize scaler
        self.scaler = StandardScaler()

        # Fit scaler only on training data
        train_data = self.df.iloc[self.train_indices]
        self.scaler.fit(train_data[features_to_scale])

        # Transform all data
        self.normalized_data = self.scaler.transform(self.df[features_to_scale])

    def __len__(self) -> int:
        """Return the length of usable data"""
        return len(self.df) - self.max_seq_len - self.target_horizon

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample for model training/evaluation

        This method creates a sample with:
        1. Multiple timeframe inputs for each expert model
        2. Market conditions data
        3. Binary classification target (1.0 if price goes up, 0.0 if down)
        """
        # Create dictionary for expert inputs
        expert_inputs = {}

        # Get data for each expert (timeframe)
        # Each expert gets a different historical window length
        # This allows the model to capture patterns at different time scales
        for i, seq_len in enumerate(self.sequence_lengths):
            seq_start = idx + self.max_seq_len - seq_len
            seq_end = idx + self.max_seq_len
            sequence = self.data[seq_start:seq_end]
            expert_inputs[f'expert{i+1}_input'] = sequence

        # Get market conditions (using the longest sequence)
        # This will be used by the gating network to determine which expert to trust
        market_conditions = self.data[idx + self.max_seq_len - self.max_seq_len:idx + self.max_seq_len]

        # Ensure market_conditions has the right shape for the gating network
        # The gating network expects [batch_size, seq_len, features]

        # Get target - binary classification (up/down) instead of regression
        # This simplifies the problem and often works better for financial forecasting
        # target_idx is already calculated in __init__
        current_value = self.data[idx + self.max_seq_len, self.target_idx]
        future_value = self.data[idx + self.max_seq_len + self.target_horizon, self.target_idx]

        # For returns, positive value means price went up, negative means it went down
        # We're predicting if the cumulative return over target_horizon days is positive
        target = 1.0 if future_value > current_value else 0.0

        return {
            'expert_inputs': expert_inputs,
            'market_conditions': market_conditions,
            'target': torch.tensor(target, dtype=torch.float)
        }

    def get_train_dataset(self) -> Subset:
        """Return training dataset"""
        subset = Subset(self, self.train_indices)
        subset.target_idx = self.target_idx  # Pass target_idx to subset
        return subset

    def get_val_dataset(self) -> Subset:
        """Return validation dataset"""
        subset = Subset(self, self.val_indices)
        subset.target_idx = self.target_idx  # Pass target_idx to subset
        return subset

    def get_test_dataset(self) -> Subset:
        """Return test dataset"""
        subset = Subset(self, self.test_indices)
        subset.target_idx = self.target_idx  # Pass target_idx to subset
        return subset


class CustomSubset(Dataset):
    """Subset of a dataset at specified indices"""

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

        # Add target_idx attribute for backtesting
        if hasattr(dataset, 'target_idx'):
            self.target_idx = dataset.target_idx

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)
