#!/usr/bin/env python
# coding: utf-8

# # Module 08: Strategy Logic
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Strategy Framework Overview](#strategy-framework-overview)
# 3. [Data Preparation](#data-preparation)
# 4. [Model Training](#model-training)
# 5. [Entry and Exit Signals](#entry-and-exit-signals)
# 6. [Trade Sizing](#trade-sizing)
# 7. [Leverage](#leverage)
# 8. [Strategy Performance Analysis](#strategy-performance-analysis)
# 9. [Practical Exercises](#practical-exercises)
# 10. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Understand the complete trading strategy pipeline
# - Implement time-based and predicate-based entry/exit signals
# - Apply static and dynamic (compounding) trade sizing
# - Understand leverage mechanics and their impact on returns
# - Calculate gross P&L and equity curves
# - Evaluate strategy performance with realistic parameters

# ---
#
# ## Strategy Framework Overview
#
# A complete trading strategy consists of three core components:
#
# ```
# signal = model(features)    # Generate predictions
# orders = strategy(signal)   # Convert to trading decisions
# results = execute(orders)   # Execute and track P&L
# ```
#
# ### Key Strategic Decisions
#
# 1. **Entry/Exit Signals**: When to enter and exit positions
# 2. **Trade Sizing**: How much capital to allocate per trade
# 3. **Leverage**: How much borrowed capital to use
#
# ### Strategy Types
#
# | Type | Description | Fee Structure |
# |------|-------------|---------------|
# | **Maker** | Provides liquidity (limit orders) | Lower fees, rebates possible |
# | **Taker** | Consumes liquidity (market orders) | Higher fees, guaranteed execution |

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ---
#
# ## Data Preparation
#
# ### Loading Market Data
#
# We'll use BTCUSDT perpetual futures data to build and test our strategy.

# In[ ]:


url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

print(f"Data shape: {btcusdt.shape}")
print(f"Date range: {btcusdt.index.min()} to {btcusdt.index.max()}")
btcusdt.head()


# ### Computing Log Returns
#
# Log returns are preferred for their mathematical properties:
#
# $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
#
# **Key advantage**: Log returns are additive over time:
#
# $$r_{t_1 \to t_n} = \sum_{i=1}^{n} r_{t_i}$$

# In[ ]:


btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())
btcusdt[['close', 'close_log_return']].head()


# ### Creating Lagged Features
#
# For our autoregressive model, we use past returns to predict future direction:

# In[ ]:


btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)

# Remove rows with NaN values
btcusdt = btcusdt.dropna()

btcusdt[['close_log_return', 'close_log_return_lag_1',
         'close_log_return_lag_2', 'close_log_return_lag_3']].head()


# ### Binary Classification Target
#
# We convert continuous returns into a binary classification problem:
#
# $$y_t = \begin{cases} 1 & \text{if } r_t > 0 \text{ (Long)} \\ 0 & \text{if } r_t \leq 0 \text{ (Short)} \end{cases}$$

# In[ ]:


btcusdt['close_log_return_dir'] = (btcusdt['close_log_return'] > 0).astype(int)

# Check class balance
print("Direction distribution:")
print(btcusdt['close_log_return_dir'].value_counts())
print(f"\nUp ratio: {btcusdt['close_log_return_dir'].mean():.2%}")


# ### Train/Test Split
#
# For time series data, we use a temporal split to prevent look-ahead bias:
#
# ```
# Time:   t0 -------- t_split -------- t_end
# Train:  [============]
# Test:                 [================]
# ```

# In[ ]:


def time_split(df, train_size=0.75):
    """
    Split time series data into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    train_size : float
        Proportion of data for training (0 to 1)

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    split_idx = int(len(df) * train_size)
    return df[:split_idx].copy(), df[split_idx:].copy()


btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print(f"Training set: {len(btcusdt_train):,} samples")
print(f"Test set: {len(btcusdt_test):,} samples")
print(f"\nTrain period: {btcusdt_train.index.min()} to {btcusdt_train.index.max()}")
print(f"Test period: {btcusdt_test.index.min()} to {btcusdt_test.index.max()}")


# In[ ]:


# Verify class balance in both sets
print("Training set direction distribution:")
print(btcusdt_train['close_log_return_dir'].value_counts())
print(f"\nTest set direction distribution:")
print(btcusdt_test['close_log_return_dir'].value_counts())


# ---
#
# ## Model Training
#
# ### Reproducibility Setup
#
# Setting random seeds ensures consistent results across runs.

# In[ ]:


import random
import os
from sklearn.preprocessing import StandardScaler

# Reproducibility settings
SEED = 99
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ### Feature Standardization
#
# Standardization improves gradient descent convergence:
#
# $$x_{scaled} = \frac{x - \mu}{\sigma}$$

# In[ ]:


features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
target = 'close_log_return_dir'

# Fit scaler on training data only (prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(btcusdt_train[features].values)
X_test_scaled = scaler.transform(btcusdt_test[features].values)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ### Logistic Regression Model
#
# The logistic regression model outputs probability of upward movement:
#
# $$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

# In[ ]:


class LogisticRegression(nn.Module):
    """
    Binary logistic regression classifier.

    Outputs raw logits (pre-sigmoid values).
    Use BCEWithLogitsLoss for numerical stability.
    """
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize model
n_features = len(features)
model = LogisticRegression(n_features)

# Loss function: Binary Cross-Entropy with Logits
criterion = nn.BCEWithLogitsLoss()

# Optimizer: Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.001)


# ### Training Loop
#
# Full-batch gradient descent for simplicity:

# In[ ]:


n_epochs = 15000
print_every = 3000

print("Training Logistic Regression Model")
print("=" * 40)

for epoch in range(n_epochs):
    # Forward pass
    optimizer.zero_grad()
    y_pred_logits = model(X_train)
    loss = criterion(y_pred_logits, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print("=" * 40)
print("Training complete!")
print(f"\nFinal weights: {model.linear.weight.data.numpy().flatten()}")
print(f"Final bias: {model.linear.bias.data.item():.6f}")


# ### Model Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

# Generate predictions on test set
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_proba = torch.sigmoid(y_pred_logits)
    y_pred_binary = (y_pred_proba >= 0.5).float()

# Convert to numpy
y_test_np = y_test.squeeze().numpy()
y_pred_binary_np = y_pred_binary.squeeze().numpy()
y_pred_proba_np = y_pred_proba.squeeze().numpy()

# Calculate metrics
accuracy = np.mean(y_pred_binary_np == y_test_np)

print("Model Evaluation on Test Set")
print("=" * 40)
print(f"Total samples: {len(X_test):,}")
print(f"Directional accuracy: {accuracy:.4f}")
print(f"\nPrediction distribution:")
print(f"  Long (1):  {int(np.sum(y_pred_binary_np == 1)):,}")
print(f"  Short (0): {int(np.sum(y_pred_binary_np == 0)):,}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_np, y_pred_binary_np)
print(cm)


# ---
#
# ## Entry and Exit Signals
#
# ### Adding Predictions to Test Data

# In[ ]:


btcusdt_test['y_pred_binary'] = y_pred_binary_np
btcusdt_test['y_pred_proba'] = y_pred_proba_np

btcusdt_test[['close_log_return', 'y_pred_binary', 'y_pred_proba']].head(10)


# ### Converting Predictions to Trading Signals
#
# Map binary predictions to directional signals:
#
# $$\text{signal}_t = \begin{cases} +1 & \text{if prediction} = 1 \text{ (Long)} \\ -1 & \text{if prediction} = 0 \text{ (Short)} \end{cases}$$

# In[ ]:


btcusdt_test['dir_signal'] = np.where(btcusdt_test['y_pred_binary'] == 1, 1, -1)

print("Signal distribution:")
print(btcusdt_test['dir_signal'].value_counts())


# ### Types of Entry/Exit Signals
#
# #### 1. Time-Based Signals
#
# Trade at every time step based on the model's prediction.
# This is the default approach we've been using.

# In[ ]:


# Time-based: Trade every period
print("Time-based signals: Trading every period")
print(f"Total trades: {len(btcusdt_test):,}")


# #### 2. Predicate-Based Signals
#
# Only trade when model confidence exceeds a threshold:
#
# $$\text{trade if } P(y=1|x) \geq \theta_{high} \text{ OR } P(y=1|x) \leq \theta_{low}$$

# In[ ]:


# Only trade when probability >= 55% or <= 45%
theta_high = 0.55
theta_low = 0.45

high_confidence_mask = (btcusdt_test['y_pred_proba'] >= theta_high) | \
                       (btcusdt_test['y_pred_proba'] <= theta_low)

print(f"Predicate-based signals (confidence threshold: {theta_low}-{theta_high})")
print(f"Total high-confidence trades: {high_confidence_mask.sum():,}")
print(f"Filtered out: {(~high_confidence_mask).sum():,} low-confidence signals")


# ### Calculating Trade Returns
#
# Trade return = Signal × Market Return
#
# $$r_{trade,t} = \text{signal}_t \times r_{market,t}$$

# In[ ]:


btcusdt_test['trade_log_return'] = btcusdt_test['dir_signal'] * btcusdt_test['close_log_return']

# Compare time-based vs predicate-based
print("Strategy comparison:")
print(f"Time-based cumulative return: {btcusdt_test['trade_log_return'].sum():.4f}")
print(f"Predicate-based cumulative return: {btcusdt_test.loc[high_confidence_mask, 'trade_log_return'].sum():.4f}")


# In[ ]:


# Visualize cumulative returns
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Time-based strategy
btcusdt_test['trade_log_return'].cumsum().plot(ax=axes[0])
axes[0].set_title('Time-Based Strategy: Cumulative Log Returns')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Cumulative Log Return')
axes[0].grid(True, alpha=0.3)

# Predicate-based strategy
btcusdt_test.loc[high_confidence_mask, 'trade_log_return'].cumsum().plot(ax=axes[1])
axes[1].set_title(f'Predicate-Based Strategy (θ={theta_low}-{theta_high}): Cumulative Log Returns')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cumulative Log Return')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ---
#
# ## Trade Sizing
#
# ### 1. Static Trade Sizing
#
# Fixed position size for every trade:
#
# $$\text{Position Value} = \text{Fixed Size}$$

# In[ ]:


# Static position size
INITIAL_CAPITAL = 50.0

btcusdt_test['pre_trade_value'] = INITIAL_CAPITAL
btcusdt_test['post_trade_value'] = np.exp(btcusdt_test['trade_log_return']) * INITIAL_CAPITAL

btcusdt_test[['pre_trade_value', 'post_trade_value', 'trade_log_return']].head(10)


# ### Calculating Gross P&L
#
# $$\text{Gross P\&L}_t = \text{Post Trade Value}_t - \text{Pre Trade Value}_t$$

# In[ ]:


btcusdt_test['trade_gross_pnl'] = btcusdt_test['post_trade_value'] - btcusdt_test['pre_trade_value']

print("Static Sizing Performance Metrics")
print("=" * 40)
print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
print(f"Mean P&L per trade: ${btcusdt_test['trade_gross_pnl'].mean():.4f}")
print(f"Total P&L: ${btcusdt_test['trade_gross_pnl'].sum():.2f}")
print(f"Final equity: ${INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum():.2f}")
print(f"Total return: {(INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum()) / INITIAL_CAPITAL:.2%}")


# In[ ]:


# Plot equity curve (static sizing)
btcusdt_test['equity_curve_static'] = INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].cumsum()

btcusdt_test['equity_curve_static'].plot(figsize=(15, 6))
plt.title('Equity Curve - Static Position Sizing')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ### 2. Dynamic Trade Sizing (Compounding)
#
# Position size grows with accumulated profits:
#
# $$\text{Position Value}_t = \text{Initial Capital} \times e^{\sum_{i=1}^{t-1} r_{trade,i}}$$
#
# This leverages the additive property of log returns.

# In[ ]:


# Cumulative log return for compounding
btcusdt_test['cum_trade_log_return'] = btcusdt_test['trade_log_return'].cumsum()

# Post-trade value with compounding
btcusdt_test['post_trade_value_compound'] = np.exp(btcusdt_test['cum_trade_log_return']) * INITIAL_CAPITAL

# Pre-trade value is previous period's post-trade value
btcusdt_test['pre_trade_value_compound'] = btcusdt_test['post_trade_value_compound'].shift().fillna(INITIAL_CAPITAL)

# Gross P&L with compounding
btcusdt_test['trade_gross_pnl_compound'] = btcusdt_test['post_trade_value_compound'] - btcusdt_test['pre_trade_value_compound']

btcusdt_test[['trade_log_return', 'cum_trade_log_return',
              'pre_trade_value_compound', 'post_trade_value_compound']].head(10)


# In[ ]:


print("Dynamic (Compounding) Sizing Performance Metrics")
print("=" * 40)
print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
print(f"Mean P&L per trade: ${btcusdt_test['trade_gross_pnl_compound'].mean():.4f}")
print(f"Final equity: ${btcusdt_test['post_trade_value_compound'].iloc[-1]:.2f}")
print(f"Total return: {btcusdt_test['post_trade_value_compound'].iloc[-1] / INITIAL_CAPITAL:.2%}")

# Compound return calculation
compound_return = np.exp(btcusdt_test['trade_log_return'].sum())
print(f"\nCompound multiplier: {compound_return:.4f}x")


# In[ ]:


# Compare static vs dynamic sizing
fig, ax = plt.subplots(figsize=(15, 6))

# Static sizing equity curve
btcusdt_test['equity_curve_static'].plot(ax=ax, label='Static Sizing')

# Dynamic sizing equity curve
btcusdt_test['post_trade_value_compound'].plot(ax=ax, label='Dynamic (Compounding)')

plt.title('Equity Curves: Static vs Dynamic Position Sizing')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ---
#
# ## Leverage
#
# ### Understanding Leverage
#
# Leverage amplifies both gains and losses:
#
# $$\text{Leveraged Return} = \text{Leverage} \times \text{Unleveraged Return}$$
#
# $$\text{Leveraged P\&L} = \text{Leverage} \times \text{Unleveraged P\&L}$$

# In[ ]:


# Example: Leverage effect
equity = 50.0
leverage = 2.0

# Positive trade
trade_pnl_positive = 10.0
print(f"Positive trade P&L: ${trade_pnl_positive:.2f}")
print(f"With {leverage}x leverage: ${trade_pnl_positive * leverage:.2f}")

# Negative trade
trade_pnl_negative = -20.0
print(f"\nNegative trade P&L: ${trade_pnl_negative:.2f}")
print(f"With {leverage}x leverage: ${trade_pnl_negative * leverage:.2f}")


# ### Applying Leverage to Strategy

# In[ ]:


LEVERAGE = 2.0

# Leveraged position values
btcusdt_test['post_trade_value_leveraged'] = np.exp(btcusdt_test['cum_trade_log_return']) * INITIAL_CAPITAL * LEVERAGE
btcusdt_test['pre_trade_value_leveraged'] = btcusdt_test['post_trade_value_leveraged'].shift().fillna(INITIAL_CAPITAL * LEVERAGE)
btcusdt_test['trade_gross_pnl_leveraged'] = btcusdt_test['post_trade_value_leveraged'] - btcusdt_test['pre_trade_value_leveraged']

print(f"Leveraged Strategy Performance ({LEVERAGE}x)")
print("=" * 40)
print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
print(f"Initial position (leveraged): ${INITIAL_CAPITAL * LEVERAGE:.2f}")
print(f"Final equity (leveraged): ${btcusdt_test['post_trade_value_leveraged'].iloc[-1]:.2f}")
print(f"Total return: {btcusdt_test['post_trade_value_leveraged'].iloc[-1] / INITIAL_CAPITAL:.2%}")


# In[ ]:


# Compare unleveraged vs leveraged
fig, ax = plt.subplots(figsize=(15, 6))

# Unleveraged (1x)
(np.exp(btcusdt_test['trade_log_return'].cumsum()) * INITIAL_CAPITAL).plot(ax=ax, label='1x (No Leverage)')

# Leveraged (2x)
(np.exp(btcusdt_test['trade_log_return'].cumsum()) * INITIAL_CAPITAL * LEVERAGE).plot(ax=ax, label=f'{LEVERAGE}x Leverage')

plt.title('Equity Curves: Unleveraged vs Leveraged')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ### Leverage Risk Warning
#
# **Important**: Higher leverage increases both potential returns AND potential losses.
#
# | Leverage | 10% Gain | 10% Loss |
# |----------|----------|----------|
# | 1x | +10% | -10% |
# | 2x | +20% | -20% |
# | 5x | +50% | -50% |
# | 10x | +100% | -100% (liquidation) |

# ---
#
# ## Strategy Performance Analysis
#
# ### Final Performance Summary

# In[ ]:


# Calculate Sharpe ratio
returns = btcusdt_test['trade_log_return']
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 24)  # Annualized for hourly data

print("=" * 50)
print("STRATEGY PERFORMANCE SUMMARY")
print("=" * 50)
print(f"\nTime Period: {btcusdt_test.index.min()} to {btcusdt_test.index.max()}")
print(f"Total Trades: {len(btcusdt_test):,}")

print("\n--- Returns ---")
print(f"Cumulative Log Return: {returns.sum():.4f}")
print(f"Compound Multiplier: {np.exp(returns.sum()):.4f}x")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")

print("\n--- Static Sizing ($50 initial) ---")
print(f"Final Equity: ${INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum():.2f}")
print(f"Total Return: {(INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum()) / INITIAL_CAPITAL:.2%}")

print("\n--- Dynamic Sizing with Compounding ($50 initial) ---")
print(f"Final Equity: ${btcusdt_test['post_trade_value_compound'].iloc[-1]:.2f}")
print(f"Total Return: {btcusdt_test['post_trade_value_compound'].iloc[-1] / INITIAL_CAPITAL:.2%}")

print(f"\n--- With {LEVERAGE}x Leverage ($50 initial) ---")
print(f"Final Equity: ${btcusdt_test['post_trade_value_leveraged'].iloc[-1]:.2f}")
print(f"Total Return: {btcusdt_test['post_trade_value_leveraged'].iloc[-1] / INITIAL_CAPITAL:.2%}")


# ### The Alpha Formula
#
# The complete formula for trading alpha:
#
# $$\text{Alpha} = \text{Statistical Edge} \times \text{Execution}$$
#
# Breaking down execution:
#
# $$\text{Alpha} = \text{Statistical Edge} \times \text{Compounding} \times \text{Leverage}$$
#
# Where:
# - **Statistical Edge**: Model's ability to predict direction (from Module 05-07)
# - **Compounding**: Reinvesting profits to grow position size
# - **Leverage**: Borrowing to amplify returns (and risks)

# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Adding Transaction Costs (Taker Strategy)
#
# Add taker transaction fees to calculate net return and P&L.
# Remember to account for **round-trip fees** (entry + exit).
#
# Typical taker fee: 0.04% - 0.075% per side

# In[ ]:


# TODO: Implement taker fee calculation
# taker_fee = 0.0005  # 0.05% per side
# round_trip_fee = taker_fee * 2
#
# btcusdt_test['trade_net_return'] = btcusdt_test['trade_log_return'] - round_trip_fee
# Compare net vs gross returns


# ### Exercise 2: Maker Strategy Analysis
#
# Implement a maker strategy with limit orders.
# Maker fees are typically lower (or even negative with rebates).
#
# Consider:
# - Fill probability (not all limit orders get filled)
# - Queue position
# - Adverse selection

# In[ ]:


# TODO: Implement maker strategy analysis
# maker_fee = -0.0002  # -0.02% (rebate)
# fill_rate = 0.7  # Assume 70% fill rate


# ### Exercise 3: Timeframe Optimization
#
# Resample the data to different timeframes (4H, 1D) and compare strategy performance.
# Find the optimal timeframe for a taker strategy that overcomes fees.

# In[ ]:


# TODO: Implement timeframe resampling
# btcusdt_4h = btcusdt.resample('4H').agg({
#     'open': 'first',
#     'high': 'max',
#     'low': 'min',
#     'close': 'last',
#     'volume': 'sum'
# })


# ### Exercise 4: Risk Management
#
# Implement a maximum drawdown stop:
# - Stop trading if drawdown exceeds 20%
# - Reduce position size during drawdown periods

# In[ ]:


# TODO: Implement drawdown calculation and risk management
# rolling_max = equity_curve.cummax()
# drawdown = (equity_curve - rolling_max) / rolling_max


# ---
#
# ## Key Takeaways
#
# 1. **Strategy Pipeline**: Signal → Order → Execution
#    - Each step affects final performance
#
# 2. **Entry/Exit Signals**:
#    - Time-based: Trade every period
#    - Predicate-based: Trade only on high-confidence signals
#
# 3. **Trade Sizing**:
#    - Static: Fixed position size, linear growth
#    - Dynamic (Compounding): Position grows with profits, exponential growth
#
# 4. **Leverage**:
#    - Amplifies both gains and losses
#    - Higher leverage = higher risk of liquidation
#
# 5. **The Alpha Formula**:
#    $$\text{Alpha} = \text{Statistical Edge} \times \text{Compounding} \times \text{Leverage}$$
#
# 6. **Key Metrics**:
#    - Cumulative log return: $\sum r_t$
#    - Compound multiplier: $e^{\sum r_t}$
#    - Sharpe ratio: $\frac{\bar{r}}{\sigma_r}$
#
# 7. **Transaction Costs**:
#    - Maker vs Taker fees significantly impact profitability
#    - Must be factored into any realistic backtesting
#
# ---
#
# **Congratulations!** You have completed the Quant Trading Accelerator.
#
# You now have the foundational knowledge to:
# - Build quantitative trading models
# - Implement proper backtesting with cross-validation
# - Design and evaluate trading strategies
# - Understand risk and position management
#
# **Next Steps**: Apply these concepts to real trading with proper risk management!

