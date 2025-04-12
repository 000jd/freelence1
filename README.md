# Financial Market Prediction Model

This project implements a multi-expert ensemble model for financial market prediction. The model combines predictions from different neural network architectures (LSTM, WaveNet, Transformer) using a dynamic gating mechanism that adapts to different market regimes.

## Project Structure

The project has been modularized for better readability and maintainability:

- `config.yaml`: Configuration file with all hyperparameters
- `config.py`: Configuration loader
- `data_preparation.py`: Dataset and data processing classes
- `model_components.py`: Neural network models and components
- `training.py`: Training functions and utilities
- `evaluation.py`: Evaluation and backtesting functions
- `main.py`: Entry point that ties everything together

## Features

- Multi-timeframe analysis with different sequence lengths
- Market regime detection
- Mixture of experts architecture with dynamic weighting
- Reinforcement learning for optimizing the gating network
- Comprehensive backtesting with risk management
- Visualization of model performance

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Gym
- Stable-Baselines3
- PyYAML

## Usage

1. Configure the model parameters in `config.yaml`
2. Run the model:

```bash
python main.py --csv path/to/your/data.csv --target "S&P_500_Price" --epochs 50
```

### Command Line Arguments

- `--config`: Path to configuration file (default: config.yaml)
- `--csv`: Path to CSV file with financial data
- `--target`: Target asset to predict
- `--epochs`: Number of training epochs

## Configuration

The `config.yaml` file contains all the hyperparameters for the model:

- Dataset parameters (sequence lengths, target horizon, etc.)
- Model parameters (hidden dimensions, dropout rates, etc.)
- Training parameters (learning rate, batch size, epochs, etc.)
- Evaluation parameters (position size, risk thresholds, etc.)

## Evaluation

The model generates comprehensive visualization files for in-depth analysis:

### Model Evaluation Visualizations
- `model_evaluation_basic.png`: Basic model performance metrics
  - Expert weights over time
  - Prediction distribution
  - Expert accuracy comparison
  - Confusion matrix

- `model_evaluation_advanced.png`: Advanced model analysis
  - Predicted vs. actual values over time
  - Residual error distribution
  - Residual error over time
  - Calibration curve
  - Expert predictions distribution
  - Gating weights by market regime

### Backtesting Visualizations
- `backtest_results_basic.png`: Basic backtesting results
  - Equity curve
  - Returns distribution

- `backtest_results_advanced.png`: Advanced backtesting analysis
  - Equity curve with trade markers (entries, exits, peak)
  - Drawdown plot
  - Daily returns vs. prediction confidence
  - Position distribution
  - Cumulative returns by confidence level
  - Monthly returns heatmap

## License

MIT
