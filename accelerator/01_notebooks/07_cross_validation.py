#!/usr/bin/env python
# coding: utf-8

# # Module 07: Cross-Validation
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Why Cross-Validation](#why-cross-validation)
# 3. [Time Series Split](#time-series-split)
# 4. [Expanding Window](#expanding-window)
# 5. [Rolling Window](#rolling-window)
# 6. [Comparing CV Methods](#comparing-cv-methods)
# 7. [Practical Exercises](#practical-exercises)
# 8. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Understand why cross-validation is crucial for model evaluation
# - Implement time series split validation
# - Implement expanding window cross-validation
# - Implement rolling window cross-validation
# - Compare model performance across different CV methods
# - Choose the appropriate CV strategy for your trading system

# In[ ]:


# Core libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
import random
import os


# ---
#
# ## Why Cross-Validation?
#
# ### The Problem with Single Train/Test Split
#
# A single train/test split can be:
# - **Sensitive to the split point**: Different splits may give very different results
# - **Non-representative**: The test period might be unusually easy or hard
# - **Overfit to one period**: Model might work only for that specific time range
#
# ### Cross-Validation Benefits
#
# 1. **More robust estimates**: Average performance across multiple periods
# 2. **Variance estimation**: See how performance varies
# 3. **Better generalization**: Tests model on different market conditions
# 4. **Detect overfitting**: Inconsistent performance signals problems

# ---
#
# ## Cross-Validation Methods for Time Series
#
# **Important**: Standard k-fold CV violates temporal order! We need specialized methods:
#
# ### 1. Time Series Split
# ```
# Time:  t0 ---- t1 ---- t2 ---- t3 ---- t4 ---- t5 ---- t6 ---- t7
# Train: [===============================]
# Test:                                  [=========================]
# ```
#
# ### 2. Expanding Window
# ```
# Fold 1: Train [###]     Test [--]
# Fold 2: Train [######]  Test    [--]
# Fold 3: Train [#########] Test      [--]
# ```
#
# ### 3. Rolling Window
# ```
# Fold 1: Train [###]     Test [--]
# Fold 2:       Train [###]    Test [--]
# Fold 3:            Train [###]    Test [--]
# ```

# ---
#
# ## Load and Prepare Data

# In[ ]:


url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

# Feature engineering
btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())
btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)
btcusdt = btcusdt.dropna()

print(f"Total samples: {len(btcusdt)}")
btcusdt.head()


# ---
#
# ## Helper Functions

# In[ ]:


def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, no_epochs, verbose=True):
    """
    Train a PyTorch model with reproducible settings.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    criterion : loss function
        Loss function (e.g., nn.HuberLoss)
    optimizer : optimizer
        PyTorch optimizer
    X_train, y_train : tensors
        Training data
    X_test, y_test : tensors
        Test data (unused during training, for reference)
    no_epochs : int
        Number of training epochs
    verbose : bool
        Print training progress

    Returns
    -------
    model : trained model
    """
    # Reproducibility
    SEED = 99
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Training loop
    for epoch in range(no_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    if verbose:
        print(f"Trained weights: {model.weight.data}")
        print(f"Trained bias: {model.bias.data}")

    return model


# In[ ]:


def test_model_predictions(model, X_test):
    """Get model predictions on test data."""
    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
    return y_hat.squeeze(1)


# In[ ]:


def eval_profitability(model, df_test, X_test, target='close_log_return'):
    """
    Evaluate model profitability on test set.

    Returns expected trade log return.
    """
    y_hat = test_model_predictions(model, X_test).numpy()
    df_test = df_test.copy()
    df_test['y_hat'] = y_hat
    df_test['dir_signal'] = np.sign(y_hat)
    df_test['trade_log_return'] = df_test['dir_signal'] * df_test[target]
    df_test['is_won'] = df_test['trade_log_return'] > 0

    return df_test['trade_log_return'].mean()


# In[ ]:


def eval_model_profitability(df_train, df_test, features, target):
    """
    Train model and evaluate profitability.

    Returns expected trade log return on test set.
    """
    no_features = len(features)

    # Create model
    model = nn.Linear(no_features, 1)
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Convert to tensors
    X_train = torch.tensor(df_train[features].values, dtype=torch.float32)
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32)
    y_train = torch.tensor(df_train[target].values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(df_test[target].values, dtype=torch.float32).unsqueeze(1)

    # Train
    train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test,
                no_epochs=5000, verbose=False)

    # Evaluate
    return eval_profitability(model, df_test, X_test, target)


# ---
#
# ## Time Series Split
#
# Simple chronological train/test split at different ratios.

# In[ ]:


def timesplit(df, train_size=0.75):
    """Split data chronologically."""
    i = int(len(df) * train_size)
    return df[:i].copy(), df[i:].copy()


# In[ ]:


# Test different split ratios
features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
target = 'close_log_return'

print("Time Series Split at Different Train Sizes:")
print("-" * 50)

for train_ratio in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    btcusdt_train, btcusdt_test = timesplit(btcusdt, train_size=train_ratio)
    ev = eval_model_profitability(btcusdt_train, btcusdt_test, features, target)
    print(f"Train {train_ratio:.0%} | Test {1-train_ratio:.0%} | E[Trade Return]: {ev:.6f}")


# ### Aggregate Across Multiple Splits

# In[ ]:


train_sizes = []
evs = []

for train_split in [0.4 + 0.1 * i for i in range(6)]:
    btcusdt_train, btcusdt_test = timesplit(btcusdt, train_size=train_split)
    ev = eval_model_profitability(btcusdt_train, btcusdt_test, features, target)
    train_sizes.append(train_split)
    evs.append(ev)

cv_results = pd.DataFrame({'train_size': train_sizes, 'ev': evs})
cv_results


# In[ ]:


print(f"Mean Expected Value: {cv_results['ev'].mean():.6f}")
print(f"Std of Expected Value: {cv_results['ev'].std():.6f}")


# ---
#
# ## Rolling Window Cross-Validation
#
# Fixed window size that rolls forward through time.
#
# ```
# Time:     [=========================================]
#            t1  t2  t3  t4  t5  t6  t7  t8  t9  t10
#
# Fold 1:   [####]  [--]
# Fold 2:       [####]  [--]
# Fold 3:           [####]  [--]
# Fold 4:               [####]  [--]
# ```

# In[ ]:


# Calculate window size (approximately 1 month of hourly data)
hours_in_month = 24 * 30
print(f"Hours in a month: {hours_in_month}")
print(f"Total rows: {len(btcusdt)}")
print(f"Number of possible monthly windows: {len(btcusdt) / hours_in_month:.1f}")


# In[ ]:


def eval_rolling_window_cv(df, features, target, window_size, no_iterations):
    """
    Evaluate model using rolling window cross-validation.

    Parameters
    ----------
    df : DataFrame
        Full dataset
    features : list
        Feature column names
    target : str
        Target column name
    window_size : int
        Size of each window (train and test)
    no_iterations : int
        Number of rolling iterations

    Returns
    -------
    DataFrame with results for each fold
    """
    window_no = []
    ev = []

    for i in range(no_iterations):
        # Calculate indices
        train_start = window_size * i
        train_end = window_size * (i + 1)
        test_start = train_end
        test_end = test_start + window_size

        # Check bounds
        if test_end > len(df):
            print(f"Warning: Fold {i} exceeds data bounds, stopping.")
            break

        # Split data
        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # Evaluate
        window_no.append(i)
        ev.append(eval_model_profitability(df_train, df_test, features, target))

    return pd.DataFrame({'window_no': window_no, 'ev': ev})


# In[ ]:


# Run rolling window CV
window_size = 724  # ~1 month of hourly data
rw_results = eval_rolling_window_cv(btcusdt, features, target, window_size, 6)
rw_results


# In[ ]:


print(f"Rolling Window CV - Mean E[Return]: {rw_results['ev'].mean():.6f}")
print(f"Rolling Window CV - Std E[Return]: {rw_results['ev'].std():.6f}")


# In[ ]:


# Visualize results
plt.figure(figsize=(12, 6))
plt.bar(rw_results['window_no'], rw_results['ev'])
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=rw_results['ev'].mean(), color='g', linestyle='--', alpha=0.5, label='Mean')
plt.xlabel('Window Number')
plt.ylabel('Expected Trade Return')
plt.title('Rolling Window Cross-Validation Results')
plt.legend()
plt.show()


# ---
#
# ## Expanding Window Cross-Validation
#
# Training window expands over time while test window stays fixed.
#
# ```
# Time:     [=========================================]
#            t1  t2  t3  t4  t5  t6  t7  t8  t9  t10
#
# Fold 1:   [####]  [--]
# Fold 2:   [########]  [--]
# Fold 3:   [############]  [--]
# Fold 4:   [################]  [--]
# ```

# In[ ]:


def eval_expanding_window_cv(df, features, target, window_size, no_iterations):
    """
    Evaluate model using expanding window cross-validation.

    Parameters
    ----------
    df : DataFrame
        Full dataset
    features : list
        Feature column names
    target : str
        Target column name
    window_size : int
        Size of each test window and initial train window
    no_iterations : int
        Number of expanding iterations

    Returns
    -------
    DataFrame with results for each fold
    """
    iteration_no = []
    ev = []

    for i in range(no_iterations):
        # Train window expands, test window is fixed size
        train_start = 0
        train_end = window_size + i * window_size
        test_start = train_end
        test_end = test_start + window_size

        # Check bounds
        if test_end > len(df):
            print(f"Warning: Fold {i} exceeds data bounds, stopping.")
            break

        # Split data
        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # Evaluate
        iteration_no.append(i + 1)
        ev.append(eval_model_profitability(df_train, df_test, features, target))

    return pd.DataFrame({'iteration_no': iteration_no, 'ev': ev})


# In[ ]:


# Show expanding window indices
window_size = 724
print("Expanding Window Indices:")
print("-" * 50)
for i in range(6):
    train_start = 0
    train_end = window_size + i * window_size
    test_start = train_end
    test_end = test_start + window_size
    print(f"Fold {i+1}: Train[{train_start}:{train_end}] ({train_end} samples) | Test[{test_start}:{test_end}]")


# In[ ]:


# Run expanding window CV
ew_results = eval_expanding_window_cv(btcusdt, features, target, window_size, 6)
ew_results


# In[ ]:


print(f"Expanding Window CV - Mean E[Return]: {ew_results['ev'].mean():.6f}")
print(f"Expanding Window CV - Std E[Return]: {ew_results['ev'].std():.6f}")


# In[ ]:


# Visualize results
plt.figure(figsize=(12, 6))
plt.bar(ew_results['iteration_no'], ew_results['ev'])
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=ew_results['ev'].mean(), color='g', linestyle='--', alpha=0.5, label='Mean')
plt.xlabel('Iteration Number')
plt.ylabel('Expected Trade Return')
plt.title('Expanding Window Cross-Validation Results')
plt.legend()
plt.show()


# ---
#
# ## Comparing CV Methods

# In[ ]:


# Summary comparison
comparison = pd.DataFrame({
    'Method': ['Time Series Split', 'Rolling Window', 'Expanding Window'],
    'Mean E[Return]': [cv_results['ev'].mean(), rw_results['ev'].mean(), ew_results['ev'].mean()],
    'Std E[Return]': [cv_results['ev'].std(), rw_results['ev'].std(), ew_results['ev'].std()]
})
comparison


# In[ ]:


# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(range(len(cv_results)), cv_results['ev'])
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title(f"Time Series Split\nMean: {cv_results['ev'].mean():.6f}")
axes[0].set_xlabel('Split')
axes[0].set_ylabel('E[Trade Return]')

axes[1].bar(range(len(rw_results)), rw_results['ev'])
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title(f"Rolling Window\nMean: {rw_results['ev'].mean():.6f}")
axes[1].set_xlabel('Window')

axes[2].bar(range(len(ew_results)), ew_results['ev'])
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_title(f"Expanding Window\nMean: {ew_results['ev'].mean():.6f}")
axes[2].set_xlabel('Iteration')

plt.tight_layout()
plt.show()


# ---
#
# ## Choosing a CV Strategy
#
# | Method | Pros | Cons | Best For |
# |--------|------|------|----------|
# | **Time Series Split** | Simple, fast | Single estimate | Quick checks |
# | **Rolling Window** | Consistent train size, adapts to regime changes | Discards old data | Non-stationary markets |
# | **Expanding Window** | Uses all available data, stable | Train size varies | Stable markets |

# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Optimize Rolling Window Size
#
# Try different window sizes to find the optimal value.

# In[ ]:


# TODO: Test window sizes from 500 to 1000 hours
# Find the window size with best mean E[Return]


# ### Exercise 2: Optimize Expanding Window Initial Size
#
# Experiment with different starting window sizes.

# In[ ]:


# TODO: Test different initial train sizes
# Compare results


# ### Exercise 3: Walk-Forward Optimization
#
# Implement walk-forward optimization where model hyperparameters
# are tuned on each fold's training data.

# In[ ]:


# TODO: For each fold:
# 1. Split train into train/validation
# 2. Tune hyperparameters on validation
# 3. Retrain on full train with best hyperparameters
# 4. Evaluate on test


# ---
#
# ## Key Takeaways
#
# 1. **Single train/test split is unreliable**
#    - Results depend heavily on where you split
#    - Use multiple evaluations for robust estimates
#
# 2. **Time series requires special CV methods**
#    - Never shuffle time series data
#    - Preserve temporal order in all splits
#
# 3. **Rolling Window CV**:
#    - Fixed train window slides through time
#    - Good for non-stationary markets
#    - Adapts to regime changes
#
# 4. **Expanding Window CV**:
#    - Train window grows over time
#    - Uses all historical data
#    - Better for stable relationships
#
# 5. **Evaluation metrics**:
#    - Look at **mean** and **std** of results
#    - High variance suggests model is unstable
#    - Consistent positive E[Return] indicates robust edge
#
# ---
#
# **Next Module**: Strategy Logic - Building a complete trading strategy
