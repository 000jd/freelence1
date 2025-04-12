import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from typing import Dict, Any, List, Optional

from model_components import MixtureOfExperts


def evaluate_model(model: MixtureOfExperts, test_dataset) -> Dict[str, Any]:
    """Evaluate model performance"""
    model.eval()

    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Track metrics
    correct_predictions = 0
    all_predictions = []
    all_targets = []
    all_gating_weights = []
    expert_predictions = {
        'short_term': [],
        'medium_term': [],
        'long_term': []
    }

    with torch.no_grad():
        for batch in dataloader:
            expert_inputs = batch['expert_inputs']
            market_conditions = batch['market_conditions']
            targets = batch['target']

            predictions, gating_weights, expert_preds = model(expert_inputs, market_conditions)
            predictions = predictions.squeeze()

            # Track predictions
            predicted_classes = (predictions > 0.5).float()
            correct_predictions += (predicted_classes == targets).sum().item()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_gating_weights.append(gating_weights.cpu().numpy())

            # Store individual expert predictions
            for i, name in enumerate(['short_term', 'medium_term', 'long_term']):
                expert_predictions[name].extend(expert_preds[:, i].cpu().numpy())

    # Calculate metrics
    accuracy = correct_predictions / len(test_dataset)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_gating_weights = np.concatenate(all_gating_weights, axis=0)

    binary_predictions = (all_predictions > 0.5).astype(int)
    precision = precision_score(all_targets, binary_predictions)
    recall = recall_score(all_targets, binary_predictions)
    f1 = f1_score(all_targets, binary_predictions)
    conf_matrix = confusion_matrix(all_targets, binary_predictions)

    # Print metrics
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)

    # Calculate and print expert accuracies
    expert_accuracies = {}
    for expert_name, preds in expert_predictions.items():
        expert_preds = np.array(preds)
        binary_expert_preds = (expert_preds > 0.5).astype(int)
        expert_acc = np.mean((binary_expert_preds == all_targets).astype(int))
        expert_accuracies[expert_name] = expert_acc
        print(f"\n{expert_name} Accuracy: {expert_acc:.4f}")

    # Create visualization plots - Page 1: Basic Metrics
    plt.figure(figsize=(15, 12))

    # Plot 1: Expert Weights Over Time
    plt.subplot(2, 2, 1)
    for i, name in enumerate(['Short-term', 'Medium-term', 'Long-term']):
        plt.plot(all_gating_weights[:, i], label=name)
    plt.title('Expert Weights Over Time')
    plt.xlabel('Sample')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)

    # Plot 2: Prediction Distribution
    plt.subplot(2, 2, 2)
    plt.hist(all_predictions, bins=50, alpha=0.5, label='Predictions')
    plt.axvline(0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.title('Prediction Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    plt.legend()

    # Plot 3: Expert Accuracy Comparison
    plt.subplot(2, 2, 3)
    expert_names = list(expert_accuracies.keys())
    accuracies = [expert_accuracies[name] for name in expert_names]
    plt.bar(expert_names, accuracies)
    plt.title('Expert Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    # Plot 4: Confusion Matrix Heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig('model_evaluation_basic.png')

    # Create visualization plots - Page 2: Advanced Metrics
    plt.figure(figsize=(15, 15))

    # Plot 1: Predicted vs. Actual Values Over Time
    plt.subplot(3, 2, 1)
    plt.plot(all_targets, label='Actual', alpha=0.7)
    plt.plot(all_predictions, label='Predicted', alpha=0.7)
    plt.title('Predicted vs. Actual Values Over Time')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot 2: Residual Error Distribution
    plt.subplot(3, 2, 2)
    residuals = all_predictions - all_targets
    sns.histplot(residuals, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Residual Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')

    # Plot 3: Residual Error Over Time
    plt.subplot(3, 2, 3)
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residual Error Over Time')
    plt.xlabel('Sample')
    plt.ylabel('Error')
    plt.grid(True)

    # Plot 4: Calibration Curve
    plt.subplot(3, 2, 4)
    # Group predictions into bins and calculate actual positive rate
    bin_edges = np.linspace(0, 1, 11)  # 10 bins
    bin_indices = np.digitize(all_predictions, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_actual = np.array([np.mean(all_targets[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan for i in range(len(bin_centers))])

    # Plot calibration curve
    plt.plot(bin_centers, bin_centers, 'r--', label='Perfect Calibration')
    plt.plot(bin_centers, bin_actual, 'bo-', label='Model Calibration')
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.legend()
    plt.grid(True)

    # Plot 5: Expert Predictions Distribution
    plt.subplot(3, 2, 5)
    for name, preds in expert_predictions.items():
        sns.kdeplot(np.array(preds), label=name)
    plt.axvline(0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.title('Expert Predictions Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Density')
    plt.legend()

    # Plot 6: Gating Weights by Market Regime
    plt.subplot(3, 2, 6)
    # Assuming market regimes are the last 4 features in the dataset
    # We'll use the dominant regime for each sample
    market_regimes = np.zeros(len(all_gating_weights))
    for i, batch in enumerate(dataloader):
        if i * dataloader.batch_size >= len(market_regimes):
            break
        market_conditions = batch['market_conditions'].cpu().numpy()
        for j in range(min(len(market_conditions), len(market_regimes) - i * dataloader.batch_size)):
            market_regimes[i * dataloader.batch_size + j] = np.argmax(market_conditions[j])

    # Create a scatter plot with color based on market regime
    for regime in range(4):  # Assuming 4 regimes: bull, bear, sideways, volatile
        regime_indices = np.where(market_regimes == regime)[0]
        if len(regime_indices) > 0:
            regime_name = ['Bull', 'Bear', 'Sideways', 'Volatile'][regime]
            plt.scatter(regime_indices, all_gating_weights[regime_indices, 0],
                       alpha=0.5, label=f'{regime_name} - Short-term')

    plt.title('Short-term Expert Weight by Market Regime')
    plt.xlabel('Sample')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_evaluation_advanced.png')
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'expert_accuracies': expert_accuracies,
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'all_gating_weights': all_gating_weights
    }


def backtest_trading_strategy(model: MixtureOfExperts, test_dataset, config) -> Dict[str, Any]:
    """Enhanced backtesting with position sizing and risk management"""
    # Extract backtesting configuration
    initial_capital = config.get('backtest', 'initial_capital', 10000)
    position_size = config.get('backtest', 'position_size', 0.1)
    max_loss_per_trade = config.get('backtest', 'max_loss_per_trade', 0.02)
    trailing_stop = config.get('backtest', 'trailing_stop', 0.015)
    long_threshold = config.get('backtest', 'long_threshold', 0.6)
    short_threshold = config.get('backtest', 'short_threshold', 0.4)

    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    capital = initial_capital
    position = 0
    trade_history = []
    equity_curve = [capital]

    # Risk management parameters
    trailing_stop_price = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= len(test_dataset) - 1:
                break

            expert_inputs = batch['expert_inputs']
            market_conditions = batch['market_conditions']

            # Get prediction and confidence
            prediction, gating_weights, expert_preds = model(expert_inputs, market_conditions)
            confidence = abs(prediction.item() - 0.5) * 2  # Scale confidence to [0, 1]

            # Dynamic position sizing based on confidence
            position_size_adj = position_size * confidence

            # Generate signal with minimum confidence threshold
            signal = 0
            if prediction.item() > long_threshold:  # Require stronger conviction for long
                signal = 1
            elif prediction.item() < short_threshold:  # Require stronger conviction for short
                signal = -1

            # Apply risk management
            if position != 0:
                # Check stop loss
                current_price = test_dataset[i]['expert_inputs']['expert1_input'][-1, test_dataset.target_idx]
                trade_value = (current_price - entry_price) * position * position_size_adj * capital

                if trade_value < -max_loss_per_trade * capital:
                    position = 0
                    capital += trade_value

                # Check trailing stop
                if trailing_stop_price is not None:
                    if position == 1 and current_price < trailing_stop_price:
                        position = 0
                        capital += trade_value
                    elif position == -1 and current_price > trailing_stop_price:
                        position = 0
                        capital += trade_value

            # Update position and calculate returns
            if signal != position:
                if position != 0:  # Close existing position
                    capital += trade_value
                position = signal
                if position != 0:  # Open new position
                    entry_price = current_price
                    trailing_stop_price = entry_price * (1 - trailing_stop * position)

            # Update equity curve and trade history
            equity_curve.append(capital)
            trade_history.append({
                'signal': signal,
                'position': position,
                'confidence': confidence,
                'capital': capital
            })

    # Calculate returns and metrics
    returns = np.diff(equity_curve)
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    std_return = np.std(returns) if len(returns) > 0 else 1

    # Calculate Sharpe ratio (avoid division by zero)
    sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return != 0 else 0

    # Create visualization plots - Page 1: Basic Backtest Results
    plt.figure(figsize=(15, 10))

    # Plot 1: Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()

    # Plot 2: Returns Distribution
    plt.subplot(2, 1, 2)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    plt.hist(returns, bins=50, alpha=0.75)
    plt.title('Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('backtest_results_basic.png')

    # Create visualization plots - Page 2: Advanced Backtest Analysis
    plt.figure(figsize=(15, 15))

    # Plot 1: Equity Curve with Markers
    plt.subplot(3, 2, 1)
    plt.plot(equity_curve, label='Portfolio Value')

    # Add markers for trades
    entry_points = []
    exit_points = []
    entry_values = []
    exit_values = []

    for i in range(1, len(trade_history)):
        # Entry point (position changed from 0 to non-zero)
        if trade_history[i-1]['position'] == 0 and trade_history[i]['position'] != 0:
            entry_points.append(i)
            entry_values.append(equity_curve[i])
        # Exit point (position changed from non-zero to 0)
        elif trade_history[i-1]['position'] != 0 and trade_history[i]['position'] == 0:
            exit_points.append(i)
            exit_values.append(equity_curve[i])

    plt.scatter(entry_points, entry_values, color='g', marker='^', label='Entry')
    plt.scatter(exit_points, exit_values, color='r', marker='v', label='Exit')

    # Mark peak value
    peak_idx = np.argmax(equity_curve)
    plt.scatter(peak_idx, equity_curve[peak_idx], color='gold', marker='*', s=200, label='Peak')

    plt.title('Equity Curve with Trade Markers')
    plt.xlabel('Trading Day')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()

    # Plot 2: Drawdown Plot
    plt.subplot(3, 2, 2)
    drawdowns = []
    peak = equity_curve[0]

    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)

    plt.plot(drawdowns, color='r')
    plt.title('Drawdown Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)

    # Plot 3: Daily Returns vs. Confidence
    plt.subplot(3, 2, 3)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    confidences = [trade['confidence'] for trade in trade_history[:-1]]  # Exclude last point

    # Make sure arrays have the same length
    min_len = min(len(daily_returns), len(confidences))
    if min_len > 0:
        plt.scatter(confidences[:min_len], daily_returns[:min_len], alpha=0.5)
        plt.title('Daily Returns vs. Prediction Confidence')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Daily Return')
        plt.grid(True)

        # Calculate correlation
        if min_len > 1:
            correlation = np.corrcoef(confidences[:min_len], daily_returns[:min_len])[0, 1]
            plt.annotate(f'Correlation: {correlation:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
    else:
        plt.text(0.5, 0.5, 'Not enough data for scatter plot',
                ha='center', va='center', fontsize=12)
        plt.axis('off')

    # Plot 4: Position Distribution
    plt.subplot(3, 2, 4)
    positions = [trade['position'] for trade in trade_history]
    position_counts = {
        -1: positions.count(-1),  # Short
        0: positions.count(0),    # No position
        1: positions.count(1)     # Long
    }

    plt.bar(['Short', 'No Position', 'Long'], [position_counts[-1], position_counts[0], position_counts[1]])
    plt.title('Position Distribution')
    plt.ylabel('Count')

    # Plot 5: Cumulative Returns by Market Regime
    plt.subplot(3, 2, 5)

    # Collect market regime data (assuming we can infer from the confidence)
    # This is a simplification - in a real implementation, you'd use actual market regime data
    regimes = []
    for trade in trade_history:
        if trade['confidence'] > 0.8:
            regimes.append('High Confidence')
        elif trade['confidence'] > 0.5:
            regimes.append('Medium Confidence')
        else:
            regimes.append('Low Confidence')

    # Make sure we have enough data
    if len(regimes) > 1 and len(equity_curve) > 1:
        # Calculate cumulative returns by regime
        regime_returns = {'High Confidence': [], 'Medium Confidence': [], 'Low Confidence': []}

        # Make sure we don't go out of bounds
        min_len = min(len(equity_curve) - 1, len(regimes))

        for i in range(min_len):
            daily_return = (equity_curve[i+1] - equity_curve[i]) / equity_curve[i]
            regime = regimes[i]
            regime_returns[regime].append(daily_return)

        # Plot cumulative returns
        for regime, returns_list in regime_returns.items():
            if returns_list:  # Only plot if we have data
                cumulative_returns = np.cumprod(1 + np.array(returns_list)) - 1
                plt.plot(cumulative_returns, label=regime)

        plt.title('Cumulative Returns by Confidence Level')
        plt.xlabel('Trading Day')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Not enough data for cumulative returns',
                ha='center', va='center', fontsize=12)
        plt.axis('off')

    # Plot 6: Monthly Returns Heatmap
    plt.subplot(3, 2, 6)

    # Create a simple monthly returns simulation (in a real implementation, use actual dates)
    # Here we'll just divide the returns into 12 equal segments to simulate months
    if len(daily_returns) >= 12:
        segment_size = len(daily_returns) // 12
        monthly_returns = [np.mean(daily_returns[i:i+segment_size]) for i in range(0, len(daily_returns), segment_size)][:12]

        # Reshape for heatmap (3x4 grid for 12 months)
        monthly_returns_grid = np.array(monthly_returns).reshape(3, 4)

        # Create heatmap
        sns.heatmap(monthly_returns_grid, annot=True, fmt='.2%', cmap='RdYlGn',
                   xticklabels=['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec'],
                   yticklabels=['Year 1', 'Year 2', 'Year 3'])
        plt.title('Monthly Returns Heatmap (Simulated)')
    else:
        plt.text(0.5, 0.5, 'Not enough data for monthly returns',
                ha='center', va='center', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('backtest_results_advanced.png')
    plt.close()

    return {
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'trade_history': trade_history,
        'equity_curve': equity_curve
    }


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate the maximum drawdown from peak to trough"""
    peak = equity_curve[0]
    max_drawdown = 0

    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def save_model(model: MixtureOfExperts, filename: str = 'mixture_of_experts_model.pth') -> None:
    """Save model to file"""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(filename: str = 'mixture_of_experts_model.pth',
               dataset = None, config = None) -> MixtureOfExperts:
    """Load model from file"""
    if dataset is None:
        raise ValueError("Dataset is required to determine input dimensions")

    # Get input dimension from dataset
    sample_batch = next(iter(DataLoader(dataset, batch_size=1)))
    input_dim = sample_batch['expert_inputs']['expert1_input'].shape[-1]
    seq_lengths = dataset.sequence_lengths

    # Create model with configuration
    model_config = None
    if config is not None:
        model_config = {
            'lstm': {
                'hidden_dim1': config.get('model', 'lstm', {}).get('hidden_dim1', 64),
                'hidden_dim2': config.get('model', 'lstm', {}).get('hidden_dim2', 32),
                'num_layers': config.get('model', 'lstm', {}).get('num_layers', 2),
                'dropout': config.get('model', 'lstm', {}).get('dropout', 0.2),
                'fc_dropout': config.get('model', 'lstm', {}).get('fc_dropout', 0.3)
            },
            'wavenet': {
                'filters': config.get('model', 'wavenet', {}).get('filters', 32),
                'dropout': config.get('model', 'wavenet', {}).get('dropout', 0.3)
            },
            'transformer': {
                'd_model': config.get('model', 'transformer', {}).get('d_model', 64),
                'nhead': config.get('model', 'transformer', {}).get('nhead', 4),
                'num_layers': config.get('model', 'transformer', {}).get('num_layers', 2),
                'dropout': config.get('model', 'transformer', {}).get('dropout', 0.1)
            },
            'gating': {
                'hidden_dim': config.get('model', 'gating', {}).get('hidden_dim', 32)
            }
        }

    model = MixtureOfExperts(input_dim, seq_lengths, model_config)
    model.load_state_dict(torch.load(filename))
    model.eval()

    print(f"Model loaded from {filename}")
    return model
