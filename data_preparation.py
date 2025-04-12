import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Any, Optional


class MultiAssetDataset(Dataset):
    def __init__(self, 
                 csv_file: str, 
                 sequence_lengths: List[int] = [5, 20, 60], 
                 target_asset: str = 'S&P_500_Price', 
                 target_horizon: int = 1, 
                 test_size: float = 0.2, 
                 val_size: float = 0.1):
        """
        Dataset for multi-asset financial data.
        
        Args:
            csv_file: Path to CSV file with financial data
            sequence_lengths: List of sequence lengths for different timeframes
            target_asset: Target asset to predict (column name)
            target_horizon: Number of days ahead to predict
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
        """
        # Load data from CSV
        self.df = pd.read_csv(csv_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
        self.df = self.df.sort_values('Date')  # Ensure chronological order
        
        # Handle missing values
        self.df = self.preprocess_missing_values(self.df)
        
        # Separate price and volume columns
        self.price_cols = [col for col in self.df.columns if 'Price' in col]
        self.volume_cols = [col for col in self.df.columns if 'Vol' in col]
        
        # Store target asset
        self.target_asset = target_asset
        
        # Calculate additional features
        self.calculate_technical_indicators()
        
        # Calculate market regimes
        self.calculate_market_regime()
        
        # Normalize data
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.technical_scaler = MinMaxScaler()
        
        # Normalize prices, volumes, and technical indicators separately
        normalized_prices = self.price_scaler.fit_transform(self.df[self.price_cols])
        normalized_volumes = self.volume_scaler.fit_transform(self.df[self.volume_cols].fillna(0))
        normalized_technical = self.technical_scaler.fit_transform(
            self.df[[col for col in self.df.columns if col.startswith('tech_')]].fillna(0)
        )
        
        # Combine normalized data
        all_features = np.hstack([normalized_prices, normalized_volumes, normalized_technical])
        
        # Add market regime features
        regime_features = self.df[['regime_bull', 'regime_bear', 'regime_sideways', 'regime_volatile']].values
        
        # Combine all features
        self.normalized_data = np.hstack([all_features, regime_features])
        
        # Convert to tensor
        self.data = torch.FloatTensor(self.normalized_data)
        
        # Store parameters
        self.sequence_lengths = sequence_lengths
        self.max_seq_len = max(sequence_lengths)
        self.target_horizon = target_horizon
        self.target_idx = self.price_cols.index(target_asset)
        
        # Create train/validation/test split
        self.train_indices, self.val_indices, self.test_indices = self.create_train_val_test_split(test_size, val_size)
    
    def preprocess_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and convert string numbers to float in the dataset"""
        # First, convert any string numbers with commas to float
        for col in df.columns:
            if col != 'Date':  # Skip the Date column
                try:
                    # Remove commas and convert to float
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                except:
                    continue
        
        # Handle missing values
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # For any remaining NaNs in volume columns, replace with 0
        for col in df.columns:
            if 'Vol' in col:
                df[col] = df[col].fillna(0)
        
        return df
        
    def calculate_technical_indicators(self) -> None:
        """Calculate technical indicators for each price series"""
        for col in self.price_cols:
            price_series = self.df[col]
            
            # Moving averages
            self.df[f'tech_{col}_MA5'] = price_series.rolling(window=5).mean()
            self.df[f'tech_{col}_MA20'] = price_series.rolling(window=20).mean()
            
            # Price momentum
            self.df[f'tech_{col}_mom5'] = price_series.pct_change(periods=5)
            self.df[f'tech_{col}_mom10'] = price_series.pct_change(periods=10)
            
            # Volatility
            self.df[f'tech_{col}_volatility'] = price_series.pct_change().rolling(window=20).std()
            
            # Relative Strength Index (RSI)
            delta = price_series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            self.df[f'tech_{col}_RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate cross-asset correlations for the target asset
        target_series = self.df[self.target_asset]
        for col in self.price_cols:
            if col != self.target_asset:
                # 10-day rolling correlation
                self.df[f'tech_corr_{col}'] = target_series.rolling(10).corr(self.df[col])
        
        # Fill NaN values
        tech_cols = [col for col in self.df.columns if col.startswith('tech_')]
        self.df[tech_cols] = self.df[tech_cols].fillna(method='bfill').fillna(0)
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: pandas Series of prices
            period: RSI calculation period (default: 14)
            
        Returns:
            pandas Series containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_market_regime(self, window_size: int = 20) -> None:
        """Enhanced market regime detection"""
        # Initialize all regime columns at once
        regime_columns = {
            'regime_bull': 0.0,
            'regime_bear': 0.0,
            'regime_sideways': 0.0,
            'regime_volatile': 0.0
        }
        
        # Create new DataFrame with regime columns
        regime_df = pd.DataFrame(regime_columns, index=self.df.index)
        
        # Calculate technical indicators
        returns = self.df[self.target_asset].pct_change()
        sma_20 = self.df[self.target_asset].rolling(window=20).mean()
        sma_50 = self.df[self.target_asset].rolling(window=50).mean()
        volatility = returns.rolling(window=window_size).std()
        rsi = self.calculate_rsi(self.df[self.target_asset], period=14)
        
        # Set regime values
        regime_df.loc[
            (sma_20 > sma_50) & 
            (returns > 0) & 
            (rsi > 50) & 
            (volatility < volatility.quantile(0.7)),
            'regime_bull'
        ] = 1.0
        
        regime_df.loc[
            (sma_20 < sma_50) & 
            (returns < 0) & 
            (rsi < 50) & 
            (volatility < volatility.quantile(0.8)),
            'regime_bear'
        ] = 1.0
        
        regime_df.loc[
            (abs(sma_20 - sma_50) / sma_50 < 0.02) & 
            (volatility < volatility.quantile(0.3)) &
            (rsi.between(40, 60)),
            'regime_sideways'
        ] = 1.0
        
        regime_df.loc[
            (volatility > volatility.quantile(0.7)) |
            (abs(returns) > returns.abs().quantile(0.9)),
            'regime_volatile'
        ] = 1.0
        
        # Join regime columns to main DataFrame
        self.df = pd.concat([self.df, regime_df], axis=1)
    
    def create_train_val_test_split(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
        """Create train/validation/test split indices while preserving time order"""
        total_samples = len(self.data) - self.max_seq_len - self.target_horizon
        test_split_idx = int(total_samples * (1 - test_size))
        val_split_idx = int(test_split_idx * (1 - val_size))
        
        return (
            list(range(val_split_idx)),  # train
            list(range(val_split_idx, test_split_idx)),  # validation
            list(range(test_split_idx, total_samples))  # test
        )
    
    def __len__(self) -> int:
        return len(self.data) - self.max_seq_len - self.target_horizon
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get samples for each expert with different sequence lengths"""
        # Create samples for each expert with different sequence lengths
        samples = {}
        market_conditions = self.data[idx + self.max_seq_len, -4:]  # Last 4 features are market regimes
        
        for i, seq_len in enumerate(self.sequence_lengths):
            start_idx = idx + (self.max_seq_len - seq_len)
            end_idx = idx + self.max_seq_len
            samples[f'expert{i+1}_input'] = self.data[start_idx:end_idx]
        
        # Target is binary: 1 if price goes up, 0 if down
        # Use the specified target asset index
        current_price = self.data[idx + self.max_seq_len, self.target_idx]
        future_price = self.data[idx + self.max_seq_len + self.target_horizon, self.target_idx]
        target = 1 if future_price > current_price else 0
        
        return {
            'expert_inputs': samples,
            'market_conditions': market_conditions,
            'target': torch.tensor(target, dtype=torch.float)
        }
    
    def get_train_dataset(self) -> 'CustomSubset':
        """Return a subset dataset for training"""
        return CustomSubset(self, self.train_indices)
    
    def get_val_dataset(self) -> 'CustomSubset':
        """Return a subset dataset for validation"""
        return CustomSubset(self, self.val_indices)
    
    def get_test_dataset(self) -> 'CustomSubset':
        """Return a subset dataset for testing"""
        return CustomSubset(self, self.test_indices)


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
