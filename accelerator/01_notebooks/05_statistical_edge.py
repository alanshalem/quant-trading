#!/usr/bin/env python
# coding: utf-8

# # Module 05: Statistical Edge
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Matrix Algebra Fundamentals](#matrix-algebra-fundamentals)
# 3. [Matrix-Vector Multiplication](#matrix-vector-multiplication)
# 4. [Building a Statistical Edge](#building-a-statistical-edge)
# 5. [Training a Linear Model](#training-a-linear-model)
# 6. [Evaluating Predictability](#evaluating-predictability)
# 7. [Measuring Statistical Edge](#measuring-statistical-edge)
# 8. [Practical Exercises](#practical-exercises)
# 9. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Perform essential matrix algebra operations
# - Understand the relationship: **model = statistical edge**, **strategy = execution**
# - Train a linear regression model using PyTorch
# - Evaluate model predictability using directional accuracy
# - Measure statistical edge through expected trade returns
# - Calculate and interpret the Sharpe ratio

# In[ ]:


# Core libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


# ---
#
# ## Matrix Algebra Fundamentals
#
# Matrices are 2D arrays of numbers. Understanding matrix operations is essential
# for machine learning and quantitative finance.
#
# ### What is a Matrix?
#
# A matrix is a rectangular array of numbers arranged in rows and columns:
#
# $$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

# In[ ]:


# Create a 3x3 matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


# In[ ]:


# Access row 0 (first row)
matrix[0]


# In[ ]:


# Access element at row 0, column 0
matrix[0][0]


# In[ ]:


# Access element at row 1, column 2
matrix[1][2]


# ### Matrix-Scalar Operations
#
# When we add a scalar to a matrix, it's added to every element:

# In[ ]:


# Using nested loops (slow way)
no_rows = len(matrix)
no_cols = len(matrix[0])

matrix_copy = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for i in range(no_rows):
    for j in range(no_cols):
        matrix_copy[i][j] += 1

matrix_copy


# In[ ]:


# Using NumPy (fast, vectorized way)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A + 1


# In[ ]:


# Scalar operations are commutative
1 + A


# In[ ]:


# Scalar multiplication
A * 2


# ---
#
# ## Matrix-Vector Multiplication
#
# This is the core operation in machine learning! It maps features to predictions.
#
# ### The Linear Model
#
# $$\hat{y} = X \cdot w + b$$
#
# Where:
# - $X$ = Feature matrix (n samples x m features)
# - $w$ = Weight vector (m features)
# - $b$ = Bias scalar
# - $\hat{y}$ = Predictions (n samples)

# In[ ]:


# Feature matrix: 3 samples, 2 features each
X = np.array([
    [-0.1, -0.2],   # Sample 1: [lag_1, lag_2]
    [-0.2, -0.4],   # Sample 2
    [-0.4, -0.8]    # Sample 3
])
X


# In[ ]:


# Weight vector: one weight per feature
w = np.array([-0.5, -0.1])
w


# In[ ]:


# Matrix-vector multiplication: X @ w
y_hat = np.dot(X, w)
y_hat


# ### Understanding the Calculation
#
# For each sample, we compute the dot product of features with weights:
#
# $$\hat{y}_1 = x_{11} \cdot w_1 + x_{12} \cdot w_2 = (-0.1)(-0.5) + (-0.2)(-0.1) = 0.07$$

# In[ ]:


# Manual verification
w1, w2 = w[0], w[1]
np.array([
    -0.1 * w1 + -0.2 * w2,  # Sample 1
    -0.2 * w1 + -0.4 * w2,  # Sample 2
    -0.4 * w1 + -0.8 * w2   # Sample 3
])


# ### Adding Bias

# In[ ]:


bias = 0.0001
y_hat_with_bias = np.dot(X, w) + bias
y_hat_with_bias


# ### Broadcasting vs Matrix Multiplication
#
# **Important distinction**:
# - `X * w` = Element-wise multiplication (broadcasting)
# - `X @ w` or `np.dot(X, w)` = Matrix multiplication

# In[ ]:


# Broadcasting (element-wise) - NOT what we want for linear models
X * w


# In[ ]:


# Matrix multiplication - this is what we want
np.dot(X, w)


# ---
#
# ## Building a Statistical Edge
#
# ### The Key Insight
#
# - **Model** = Statistical edge (ability to predict)
# - **Strategy** = Execution of the statistical edge
#
# A good model finds patterns; a good strategy exploits them profitably.
#
# ### Statistical Edge = Good Forecast
#
# If we can predict the direction of price movement better than random chance,
# we have a statistical edge.

# ### Load OHLC Data

# In[ ]:


url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

print(f"Data shape: {btcusdt.shape}")
btcusdt.head()


# ### Feature Engineering: Log Returns and Lags

# In[ ]:


# Calculate log returns
btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())

# Create lagged features
btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)

# Remove NaN rows
btcusdt = btcusdt.dropna()
btcusdt[['close_log_return', 'close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']].head()


# ### Check Serial Correlation

# In[ ]:


btcusdt[['close_log_return', 'close_log_return_lag_1',
         'close_log_return_lag_2', 'close_log_return_lag_3']].corr()


# ### Visualize Feature Relationships

# In[ ]:


sns.pairplot(btcusdt[['close_log_return', 'close_log_return_lag_1',
                      'close_log_return_lag_2', 'close_log_return_lag_3']],
             diag_kind='kde')
plt.suptitle('Feature Relationships', y=1.02)
plt.show()


# ### Prepare Features and Target

# In[ ]:


# Feature matrix X
X = btcusdt[['close_log_return_lag_1', 'close_log_return_lag_2',
             'close_log_return_lag_3']].values
print(f"X shape: {X.shape}")


# In[ ]:


# Target vector y
y = btcusdt['close_log_return'].values
print(f"y shape: {y.shape}")


# ---
#
# ## Training a Linear Model
#
# ### Time Series Train/Test Split
#
# **Critical**: For time series, we must split chronologically to avoid look-ahead bias!
#
# ```
# Time:  t0 ---- t1 ---- t2 ---- t3 ---- t4 ---- t5 ---- t6 ---- t7
# Train: [===============================]
# Test:                                  [=========================]
# ```

# In[ ]:


def time_split(x, train_size=0.75):
    """Split data chronologically for time series."""
    i = int(len(x) * train_size)
    return x[:i].copy(), x[i:].copy()

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print(f"Train: {len(btcusdt_train)} samples ({btcusdt_train.index.min()} to {btcusdt_train.index.max()})")
print(f"Test: {len(btcusdt_test)} samples ({btcusdt_test.index.min()} to {btcusdt_test.index.max()})")


# ### PyTorch Model Training

# In[ ]:


import random
import os

# -------------------------------------------------------
# REPRODUCIBILITY SETTINGS
# -------------------------------------------------------
SEED = 99
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------------
# PREPARE DATA
# -------------------------------------------------------
features = ['close_log_return_lag_3']  # AR(1) model with lag-3
target = 'close_log_return'

# Convert to PyTorch tensors
X_train = torch.tensor(btcusdt_train[features].values, dtype=torch.float32)
X_test = torch.tensor(btcusdt_test[features].values, dtype=torch.float32)
y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# In[ ]:


# -------------------------------------------------------
# DEFINE MODEL
# -------------------------------------------------------
no_features = len(features)

# Simple linear regression: y = Wx + b
model = nn.Linear(no_features, 1)

# Huber loss (robust to outliers)
criterion = nn.HuberLoss()

# SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[ ]:


# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------
for epoch in range(5000):
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# Final parameters
print(f"\nTrained weight: {model.weight.data}")
print(f"Trained bias: {model.bias.data}")


# In[ ]:


# Save model
torch.save(model.state_dict(), "model.pth")


# ---
#
# ## Evaluating Predictability
#
# ### Generate Predictions

# In[ ]:


# Get predictions on test set
y_hat = model(X_test)
y_hat_np = y_hat.detach().squeeze().numpy()

btcusdt_test['y_hat'] = y_hat_np
btcusdt_test[['close_log_return', 'y_hat']].head(10)


# ### Add Directional Signal
#
# Convert continuous predictions to trading signals:
# - **+1** = Go Long (betting price goes up)
# - **-1** = Go Short (betting price goes down)

# In[ ]:


btcusdt_test['dir_signal'] = np.sign(btcusdt_test['y_hat'])
btcusdt_test[['close_log_return', 'y_hat', 'dir_signal']].head(10)


# ### Directional Accuracy
#
# How often do we predict the correct direction?

# In[ ]:


btcusdt_test['is_won'] = btcusdt_test['dir_signal'] == np.sign(btcusdt_test[target])
da = btcusdt_test['is_won'].mean()
print(f"Directional Accuracy: {da:.2%}")


# **Interpretation**: If DA > 50%, we have some predictive power over random guessing.

# ---
#
# ## Measuring Statistical Edge
#
# ### Trade Returns
#
# When we're correct on direction, we profit. When wrong, we lose.
#
# $$\text{Trade Return} = \text{Signal} \times \text{Actual Return}$$

# In[ ]:


btcusdt_test['trade_log_return'] = btcusdt_test['dir_signal'] * btcusdt_test[target]
btcusdt_test[['dir_signal', 'close_log_return', 'is_won', 'trade_log_return']].head(10)


# ### Statistical Edge = Positive Expected Value

# In[ ]:


expected_trade_return = btcusdt_test['trade_log_return'].mean()
print(f"Expected Trade Return: {expected_trade_return:.6f}")

has_statistical_edge = expected_trade_return > 0
print(f"Has Statistical Edge: {has_statistical_edge}")


# **Key Insight**: If E[trade return] > 0, we have a statistical edge!

# ### Total Return

# In[ ]:


total_log_return = btcusdt_test['trade_log_return'].sum()
print(f"Total Log Return: {total_log_return:.4f}")

# Convert to simple return
total_return = np.exp(total_log_return)
print(f"Total Return: {total_return:.2%}")


# In[ ]:


# Final portfolio value
initial_capital = 100
final_value = np.exp(total_log_return) * initial_capital
print(f"${initial_capital:.2f} -> ${final_value:.2f}")


# ### Equity Curve

# In[ ]:


cum_trade_log_returns = btcusdt_test['trade_log_return'].cumsum()

plt.figure(figsize=(15, 6))
cum_trade_log_returns.plot()
plt.title('Cumulative Log Returns')
plt.ylabel('Cumulative Log Return')
plt.xlabel('Time')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()


# In[ ]:


# Gross equity curve
gross_equity_curve = np.exp(cum_trade_log_returns) * initial_capital

plt.figure(figsize=(15, 6))
gross_equity_curve.plot()
plt.title(f'Equity Curve (Starting Capital: ${initial_capital})')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Time')
plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
plt.show()


# ### Sharpe Ratio

# In[ ]:


# Raw Sharpe ratio (per period)
sharpe_raw = btcusdt_test['trade_log_return'].mean() / btcusdt_test['trade_log_return'].std()
print(f"Raw Sharpe Ratio: {sharpe_raw:.4f}")


# In[ ]:


# Annualized Sharpe ratio
trading_days_per_year = 365
hours_per_day = 24
periods_per_year = trading_days_per_year * hours_per_day

sharpe_annual = sharpe_raw * np.sqrt(periods_per_year)
print(f"Annualized Sharpe Ratio: {sharpe_annual:.2f}")


# **Interpretation**:
# - Sharpe < 0: Losing money on average
# - Sharpe 0-1: Below average risk-adjusted returns
# - Sharpe 1-2: Good risk-adjusted returns
# - Sharpe > 2: Excellent risk-adjusted returns

# ---
#
# ## Transaction Costs Consideration
#
# ### Taker vs Maker
#
# - **Taker**: Takes liquidity (market orders), pays higher fees
# - **Maker**: Adds liquidity (limit orders), pays lower fees (sometimes negative!)
#
# Transaction costs can turn a positive edge into a negative one:
# - Taker fees reduce win amounts and amplify loss amounts
# - A small positive EV can become negative after fees
#
# **Key Insight**: The viability of a strategy depends heavily on execution costs.

# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Dot Product Implementation
#
# Implement matrix-vector multiplication manually.

# In[ ]:


X = [
    [-0.1, -0.01],
    [0.2, 0.5]
]
w = [-0.5, -0.25]
y_hat = []

# TODO: Write a loop to compute y_hat = X @ w


# In[ ]:


# Verify
expected = [-0.1 * -0.5 + -0.01 * -0.25, 0.2 * -0.5 + 0.5 * -0.25]
print(f"Expected: {expected}")
# print(f"Your result: {y_hat}")


# ### Exercise 2: Matrix Transpose
#
# Transpose flips rows and columns: $A^T_{ij} = A_{ji}$

# In[ ]:


X = [
    [-0.1, -0.01, -0.2],
    [0.2, 0.5, 0.1]
]
X_transpose = []

# TODO: Transpose X from shape (2,3) to (3,2)


# In[ ]:


# Verify
expected = [[-0.1, 0.2], [-0.01, 0.5], [-0.2, 0.1]]
# X_transpose == expected


# ### Exercise 3: Hadamard Product (Element-wise)
#
# The Hadamard product multiplies corresponding elements.

# In[ ]:


y_true = [[0.01, -0.02], [-0.01, -0.03]]
y_hat = [[0.02, -0.03], [0.01, -0.01]]
error = []

# TODO: Calculate error = y_true - y_hat (element-wise)


# In[ ]:


# Verify
expected = [[0.01 - 0.02, -0.02 - (-0.03)], [-0.01 - 0.01, -0.03 - (-0.01)]]
# error == expected


# ---
#
# ## Key Takeaways
#
# 1. **Matrix multiplication** is the core operation in ML models:
#    $$\hat{y} = X \cdot w + b$$
#
# 2. **Statistical edge** = positive expected trade return
#    - Model finds patterns, strategy exploits them
#
# 3. **Directional accuracy** measures how often we predict correctly
#    - DA > 50% suggests predictive power
#
# 4. **Sharpe ratio** measures risk-adjusted returns:
#    $$SR = \frac{\bar{r}}{\sigma_r} \times \sqrt{T}$$
#
# 5. **Transaction costs** can eliminate a statistical edge
#    - Consider taker vs maker execution
#
# 6. **Key formulas**:
#    - Trade return: $r_{trade} = \text{signal} \times r_{actual}$
#    - Expected value: $E[r_{trade}]$ > 0 means statistical edge
#    - Total return: $R_{total} = e^{\sum r_t}$
#
# ---
#
# **Next Module**: Classification - Binary prediction and model evaluation metrics
