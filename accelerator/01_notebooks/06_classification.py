#!/usr/bin/env python
# coding: utf-8

# # Module 06: Classification
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Classification vs Regression](#classification-vs-regression)
# 3. [Binary Classification Target](#binary-classification-target)
# 4. [Logistic Regression](#logistic-regression)
# 5. [Confusion Matrix](#confusion-matrix)
# 6. [ROC AUC](#roc-auc)
# 7. [Profitability Analysis](#profitability-analysis)
# 8. [Excess Predictability](#excess-predictability)
# 9. [Practical Exercises](#practical-exercises)
# 10. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Understand when to use classification vs regression for trading
# - Train a logistic regression model in PyTorch
# - Evaluate model performance using confusion matrix metrics
# - Understand and calculate ROC AUC score
# - Analyze model profitability beyond just accuracy
# - Calculate Excess Predictability (Gerko Statistic)

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


# ---
#
# ## Classification vs Regression
#
# ### Regression Output
# - Continuous value: `0.0012` (predicted log return)
# - Useful for: Position sizing, exact predictions
#
# ### Classification Output
# - Discrete class: `UP` (75% confidence) or `DOWN` (25% confidence)
# - Useful for: Clear trading signals, probability calibration
#
# ### When to Use Which?
#
# | Use Case | Approach |
# |----------|----------|
# | Predict exact return | Regression |
# | Predict direction only | Classification |
# | Position sizing based on confidence | Classification (with probabilities) |
# | Simple up/down signals | Classification |

# ---
#
# ## Binary Classification Target
#
# We convert continuous returns into binary classes:
# - **1** = Long (price goes up)
# - **0** = Short (price goes down)

# In[ ]:


# Load data
url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')
btcusdt.head()


# In[ ]:


# Calculate log returns
btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())

# Create lagged features
btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)

# Drop NaN rows
btcusdt = btcusdt.dropna()


# In[ ]:


# Create binary classification target
# 1 = positive return (go long)
# 0 = negative return (go short)
btcusdt['close_log_return_dir'] = btcusdt['close_log_return'].map(lambda x: 1 if x > 0 else 0)
btcusdt[['close_log_return', 'close_log_return_dir']].head(10)


# ### Check Target Balance
#
# Important: Check if classes are balanced (roughly 50/50).

# In[ ]:


btcusdt['close_log_return_dir'].value_counts()


# In[ ]:


# Calculate balance ratio
balance_ratio = btcusdt['close_log_return_dir'].value_counts(normalize=True)
print(f"Up ratio: {balance_ratio[1]:.2%}")
print(f"Down ratio: {balance_ratio[0]:.2%}")


# ---
#
# ## Train/Test Split

# In[ ]:


def time_split(x, train_size=0.75):
    """Split data chronologically for time series."""
    i = int(len(x) * train_size)
    return x[:i].copy(), x[i:].copy()

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print(f"Train samples: {len(btcusdt_train)}")
print(f"Test samples: {len(btcusdt_test)}")


# In[ ]:


# Check class balance in both splits
print("\nTrain class distribution:")
print(btcusdt_train['close_log_return_dir'].value_counts())

print("\nTest class distribution:")
print(btcusdt_test['close_log_return_dir'].value_counts())


# ---
#
# ## Logistic Regression
#
# ### The Sigmoid Function
#
# Logistic regression uses the **sigmoid function** to map outputs to probabilities:
#
# $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
#
# Where $z = w \cdot x + b$ (linear combination)
#
# The sigmoid maps any real number to the range (0, 1), which we interpret as probability.

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


# In[ ]:


# -------------------------------------------------------
# FEATURE STANDARDIZATION
# -------------------------------------------------------
features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
target = 'close_log_return_dir'

# Fit scaler on training set only (prevent data leakage!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(btcusdt_train[features].values)
X_test_scaled = scaler.transform(btcusdt_test[features].values)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)


# In[ ]:


# -------------------------------------------------------
# LOGISTIC REGRESSION MODEL
# -------------------------------------------------------
class LogisticRegression(nn.Module):
    """
    Binary classification using logistic regression.

    Output: raw logits (use sigmoid for probabilities)
    """
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)  # Returns logits

no_features = len(features)
model = LogisticRegression(no_features)

# Binary Cross-Entropy with Logits (numerically stable)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# In[ ]:


# -------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------
for epoch in range(15000):
    optimizer.zero_grad()

    # Forward pass: get logits
    y_pred_logits = model(X_train)

    # Compute loss
    loss = criterion(y_pred_logits, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 3000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print(f"\nFinal weight: {model.linear.weight.data}")
print(f"Final bias: {model.linear.bias.data}")


# In[ ]:


# -------------------------------------------------------
# GENERATE PREDICTIONS
# -------------------------------------------------------
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_proba = torch.sigmoid(y_pred_logits)  # Convert to probabilities
    y_pred_binary = (y_pred_proba >= 0.5).float()  # Threshold at 0.5

y_test_np = y_test.squeeze().numpy()
y_pred_binary_np = y_pred_binary.squeeze().numpy()
y_pred_proba_np = y_pred_proba.squeeze().numpy()


# ---
#
# ## Confusion Matrix
#
# The confusion matrix shows prediction vs actual outcomes:
#
# |  | Predicted DOWN | Predicted UP |
# |--|----------------|--------------|
# | **Actual DOWN** | TN (True Negative) | FP (False Positive) |
# | **Actual UP** | FN (False Negative) | TP (True Positive) |

# In[ ]:


cm = confusion_matrix(y_test_np, y_pred_binary_np)
print("Confusion Matrix:")
print(cm)


# In[ ]:


# Extract values
TN = cm[0][0]  # True Negative (Correctly predicted DOWN)
FN = cm[1][0]  # False Negative (Predicted DOWN, was UP)
FP = cm[0][1]  # False Positive (Predicted UP, was DOWN)
TP = cm[1][1]  # True Positive (Correctly predicted UP)

print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"False Positives (FP): {FP}")
print(f"True Positives (TP): {TP}")


# ### Key Metrics

# In[ ]:


# Overall Accuracy (Win Rate)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy (Win Rate): {accuracy:.2%}")


# In[ ]:


# Precision: When we predict UP, how often are we correct?
precision_up = TP / (TP + FP) if (TP + FP) > 0 else 0
print(f"Precision (UP): {precision_up:.2%}")


# In[ ]:


# Recall (Sensitivity): Of all actual UPs, how many did we catch?
recall_up = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f"Recall (UP): {recall_up:.2%}")


# In[ ]:


# Precision for DOWN
precision_down = TN / (TN + FN) if (TN + FN) > 0 else 0
print(f"Precision (DOWN): {precision_down:.2%}")


# In[ ]:


# Signal distribution
long_ratio = (FP + TP) / (FN + TN + TP + FP)
short_ratio = (FN + TN) / (FN + TN + TP + FP)
print(f"Long signal ratio: {long_ratio:.2%}")
print(f"Short signal ratio: {short_ratio:.2%}")
print(f"Long/Short imbalance: {long_ratio/short_ratio:.2f}")


# ---
#
# ## ROC AUC
#
# **ROC** (Receiver Operating Characteristic) curve shows the trade-off between
# True Positive Rate and False Positive Rate at different thresholds.
#
# **AUC** (Area Under Curve) summarizes the curve:
# - AUC = 0.5: Random guessing
# - AUC = 1.0: Perfect discrimination
# - AUC < 0.5: Worse than random (inverted predictions)

# In[ ]:


auc = roc_auc_score(y_test, y_pred_proba_np)
print(f"ROC AUC Score: {auc:.4f}")


# In[ ]:


# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_proba_np)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ---
#
# ## Profitability Analysis
#
# **Important**: High accuracy doesn't guarantee profits!
#
# We need to analyze actual trading performance.

# In[ ]:


# Add predictions to test data
btcusdt_test['y_pred_binary'] = y_pred_binary_np
btcusdt_test['y_pred_proba'] = y_pred_proba_np

# Create directional signal: 1 for long, -1 for short
btcusdt_test['dir_signal'] = np.where(btcusdt_test['y_pred_binary'] == 1, 1, -1)


# In[ ]:


# Check signal distribution
btcusdt_test['dir_signal'].value_counts()


# In[ ]:


# Calculate trade returns
btcusdt_test['trade_log_return'] = btcusdt_test['dir_signal'] * btcusdt_test['close_log_return']


# In[ ]:


# Cumulative returns
btcusdt_test['cum_trade_log_return'] = btcusdt_test['trade_log_return'].cumsum()

plt.figure(figsize=(15, 8))
btcusdt_test['cum_trade_log_return'].plot()
plt.title('Cumulative Trade Log Returns')
plt.ylabel('Cumulative Log Return')
plt.xlabel('Time')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()


# In[ ]:


# Compare with underlying asset
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

axes[0].set_title('Model Cumulative Returns')
btcusdt_test['cum_trade_log_return'].plot(ax=axes[0])
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

axes[1].set_title('BTC/USDT Price')
btcusdt_test['close'].plot(ax=axes[1])

plt.tight_layout()
plt.show()


# ### Equity Curve and Performance Metrics

# In[ ]:


initial_capital = 100

# Gross P&L
btcusdt_test['trade_gross_pnl'] = np.exp(btcusdt_test['cum_trade_log_return']) * initial_capital

plt.figure(figsize=(15, 8))
btcusdt_test['trade_gross_pnl'].plot()
plt.title(f'Equity Curve (Starting Capital: ${initial_capital})')
plt.ylabel('Portfolio Value ($)')
plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
plt.show()


# In[ ]:


# Sharpe Ratio
sharpe_raw = btcusdt_test['trade_log_return'].mean() / btcusdt_test['trade_log_return'].std()
sharpe_annual = sharpe_raw * np.sqrt(365 * 24)

print(f"Raw Sharpe: {sharpe_raw:.4f}")
print(f"Annualized Sharpe: {sharpe_annual:.2f}")


# In[ ]:


# Total compound return
total_compound_return = np.exp(btcusdt_test['trade_log_return'].sum())
print(f"Total Compound Return: {total_compound_return:.2%}")
print(f"Final Portfolio Value: ${total_compound_return * initial_capital:.2f}")


# ---
#
# ## Excess Predictability (Gerko Statistic)
#
# How much better is our model compared to random guessing?
#
# ### Create Random Benchmark

# In[ ]:


# Generate random binary predictions
rng = np.random.default_rng(SEED)
random_binary = rng.integers(low=0, high=2, size=len(btcusdt_test))

# Convert to signals (-1, +1)
btcusdt_test['random_dir_signal'] = random_binary * 2 - 1
btcusdt_test['random_dir_signal'].value_counts()


# In[ ]:


# Random trade returns
btcusdt_test['random_trade_log_return'] = btcusdt_test['random_dir_signal'] * btcusdt_test['close_log_return']
btcusdt_test['cum_random_trade_log_return'] = btcusdt_test['random_trade_log_return'].cumsum()


# In[ ]:


# Plot comparison
plt.figure(figsize=(15, 8))
btcusdt_test['cum_trade_log_return'].plot(label='Model')
btcusdt_test['cum_random_trade_log_return'].plot(label='Random')
plt.title('Model vs Random Benchmark')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()


# In[ ]:


# Excess Predictability
excess_predictability = btcusdt_test['cum_trade_log_return'] - btcusdt_test['cum_random_trade_log_return']

plt.figure(figsize=(15, 8))
excess_predictability.plot()
plt.title('Excess Predictability (Model - Random)')
plt.ylabel('Excess Log Return')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()


# In[ ]:


# Total excess predictability
total_excess = btcusdt_test['trade_log_return'].sum() - btcusdt_test['random_trade_log_return'].sum()
print(f"Total Excess Predictability: {total_excess:.4f}")


# ---
#
# ## Return Decomposition (Anatolyev)
#
# Returns can be decomposed into direction and magnitude:
#
# $$y_t = \text{sign}(y_t) \times |y_t|$$
#
# - $\text{sign}(y_t)$ = Direction (+1 or -1)
# - $|y_t|$ = Magnitude (absolute size)
#
# This suggests we could build separate models for direction and magnitude!

# In[ ]:


# Example
returns = np.array([-0.05, 0.03, -0.02, 0.04])

directions = np.sign(returns)
magnitudes = np.abs(returns)
reconstructed = directions * magnitudes

print(f"Returns: {returns}")
print(f"Directions: {directions}")
print(f"Magnitudes: {magnitudes}")
print(f"Reconstructed: {reconstructed}")


# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Excess Profitability
#
# Create a test statistic comparing model profitability to buy & hold.

# In[ ]:


# TODO: Compare model's profitability over buy & hold (HODL)
# Hint: Calculate buy & hold returns and subtract from model returns


# ### Exercise 2: Regression Model for Magnitude
#
# Train a regression model to predict the absolute size of future log returns.

# In[ ]:


# TODO: Create target = |close_log_return|
# Train a regression model


# ### Exercise 3: Combined Direction + Magnitude Model
#
# Use both classification (direction) and regression (magnitude) to predict
# the decomposed log return.
#
# $$\hat{y}_t = \hat{\text{sign}}(y_t) \times \hat{|y_t|}$$

# In[ ]:


# TODO: Combine direction prediction from classification model
# with magnitude prediction from regression model


# ---
#
# ## Key Takeaways
#
# 1. **Classification vs Regression**:
#    - Classification predicts discrete classes (UP/DOWN)
#    - Outputs probability scores for confidence-based trading
#
# 2. **Logistic Regression**:
#    - Uses sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
#    - Output is probability of UP class
#
# 3. **Confusion Matrix** metrics:
#    - Accuracy: Overall correct predictions
#    - Precision: Accuracy of positive predictions
#    - Recall: Coverage of actual positives
#
# 4. **ROC AUC**:
#    - AUC = 0.5: Random guessing
#    - AUC > 0.5: Better than random
#    - AUC = 1.0: Perfect discrimination
#
# 5. **Excess Predictability**:
#    - Measures improvement over random baseline
#    - Critical for assessing true model value
#
# 6. **Return Decomposition**: $y_t = \text{sign}(y_t) \times |y_t|$
#    - Can model direction and magnitude separately
#
# ---
#
# **Next Module**: Cross-Validation - Robust model evaluation techniques
