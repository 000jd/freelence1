import argparse
import os
from typing import Dict, Any, Tuple

from config import get_config
from data_preparation import MultiAssetDataset
from training import train_mixture_of_experts, train_gating_with_rl
from evaluation import evaluate_model, backtest_trading_strategy, save_model


def main(config_path: str = 'config.yaml', csv_path: str = None) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Main function to train and evaluate the model

    Args:
        config_path: Path to configuration file
        csv_path: Path to CSV file (overrides environment variable)

    Returns:
        Tuple of (model, evaluation metrics, backtest results)
    """
    # Load configuration
    config = get_config(config_path)

    # Extract dataset parameters
    dataset_config = config.get('dataset')
    csv_file = csv_path if csv_path else os.environ.get('CSV_FILE', 'stock_data.csv')
    target_asset = dataset_config.get('target_asset', 'S&P_500_Price')
    sequence_lengths = dataset_config.get('sequence_lengths', [5, 20, 60])
    target_horizon = dataset_config.get('target_horizon', 1)
    test_size = dataset_config.get('test_size', 0.2)
    val_size = dataset_config.get('val_size', 0.1)

    # Configuration is passed directly to training functions

    print("Loading and preparing dataset...")

    # Create dataset
    dataset = MultiAssetDataset(
        csv_file=csv_file,
        sequence_lengths=sequence_lengths,
        target_asset=target_asset,
        target_horizon=target_horizon,
        test_size=test_size,
        val_size=val_size
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Split dataset
    train_dataset = dataset.get_train_dataset()
    test_dataset = dataset.get_test_dataset()

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Train model
    print("Training model...")
    model = train_mixture_of_experts(dataset, config)

    # Train gating network with reinforcement learning (optional)
    use_rl = config.get('rl', 'enabled', False)
    if use_rl:
        print("Training gating network with reinforcement learning...")
        model = train_gating_with_rl(model, train_dataset, config)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_dataset)

    # Backtest trading strategy
    print("Backtesting trading strategy...")
    backtest_results = backtest_trading_strategy(model, test_dataset, config)

    # Save model
    save_model(model)

    return model, metrics, backtest_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate multi-expert stock prediction model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--csv', type=str, help='Path to CSV file with financial data')
    parser.add_argument('--target', type=str, help='Target asset to predict')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')

    args = parser.parse_args()

    # Update configuration from command line arguments
    config = get_config(args.config)
    if args.target:
        config.update('dataset', 'target_asset', args.target)
    if args.epochs:
        config.update('training', 'epochs', args.epochs)

    # Save updated configuration
    config.save()

    # Run main function with CSV path
    main(args.config, args.csv)
