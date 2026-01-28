#!/usr/bin/env python
# coding: utf-8

# # Module 04: Time Series Analysis
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Statistical Foundations](#statistical-foundations)
# 3. [Central Tendency](#central-tendency)
# 4. [Spread and Volatility](#spread-and-volatility)
# 5. [Risk-Adjusted Returns](#risk-adjusted-returns)
# 6. [Correlation](#correlation)
# 7. [Time Series Fundamentals](#time-series-fundamentals)
# 8. [Differencing and Stationarity](#differencing-and-stationarity)
# 9. [Autoregressive Models](#autoregressive-models)
# 10. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Apply key statistical concepts to financial time series
# - Understand and calculate risk-adjusted returns (Sharpe Ratio)
# - Identify and interpret correlation patterns
# - Work with real OHLC cryptocurrency data
# - Understand stationarity and why differencing matters
# - Create lagged features for autoregressive models
# - Understand mean reversion and momentum dynamics

# ---
#
# ## Statistical Foundations
#
# Time series analysis builds on fundamental statistical concepts.
# Let's review the key measures we'll use throughout our trading models.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---
#
# ## Central Tendency
#
# ### Mean (Average)
#
# The **arithmetic mean** is the sum of values divided by the count:
#
# $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
#
# **In trading**: The mean P&L tells us our expected return per trade.

# In[ ]:


trade_pnl = [10.0, 11.0, 9.0, -400.0, 2.0, 10.0, 11.0, 11.0, 8, 10.0, 10.2]
mean_pnl = np.mean(trade_pnl)
print(f"Mean P&L: ${mean_pnl:.2f}")


# **Problem**: The mean is heavily influenced by outliers (the -400 loss).

# ### Median
#
# The **median** is the middle value when data is sorted. It's robust to outliers:
#
# $$\text{Median} = x_{(n+1)/2} \text{ for odd } n$$

# In[ ]:


median_pnl = np.median(trade_pnl)
print(f"Median P&L: ${median_pnl:.2f}")


# **Key Insight**: The median ($10.0) is much more representative of typical trades
# than the mean (-$28.00), which is skewed by one large loss.

# ---
#
# ## Spread and Volatility
#
# ### Standard Deviation
#
# The **standard deviation** measures the typical distance from the mean:
#
# $$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
#
# **In trading**: Standard deviation measures **volatility** - the risk of returns.

# In[ ]:


# Portfolio A: Consistent small gains
portfolio_a = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
print(f"Portfolio A - Total P&L: ${np.sum(portfolio_a):.2f}")
print(f"Portfolio A - Std Dev: ${np.std(portfolio_a):.4f}")


# In[ ]:


# Portfolio B: Large swings, same total
portfolio_b = [-10.0, 10.0, -10.0, 15.0, -10.0, 16.0, -18.0, 13.0, -10.0, 10.0]
print(f"Portfolio B - Total P&L: ${np.sum(portfolio_b):.2f}")
print(f"Portfolio B - Std Dev: ${np.std(portfolio_b):.4f}")


# **Key Insight**: Both portfolios have similar total P&L (~$6), but Portfolio B
# is much riskier with 20x higher volatility!

# ---
#
# ## Risk-Adjusted Returns
#
# ### Comparing Portfolios
#
# Raw returns don't tell the whole story. We need **risk-adjusted** metrics.

# In[ ]:


# Create comparison table
benchmark = pd.DataFrame([
    ['A', np.sum(portfolio_a), np.mean(portfolio_a), np.std(portfolio_a)],
    ['B', np.sum(portfolio_b), np.mean(portfolio_b), np.std(portfolio_b)]
], columns=['portfolio', 'total_pnl', 'mean_pnl', 'std_pnl'])

benchmark


# ### Sharpe Ratio
#
# The **Sharpe Ratio** measures return per unit of risk:
#
# $$\text{Sharpe Ratio} = \frac{\mathbb{E}[R] - R_f}{\sigma_R}$$
#
# Where:
# - $\mathbb{E}[R]$ = Expected return (mean)
# - $R_f$ = Risk-free rate (often 0 for simplicity)
# - $\sigma_R$ = Standard deviation of returns

# In[ ]:


benchmark['sharpe'] = benchmark['mean_pnl'] / benchmark['std_pnl']
benchmark


# **Interpretation**:
# - Portfolio A: Sharpe = 1.03 (excellent risk-adjusted returns)
# - Portfolio B: Sharpe = 0.05 (poor risk-adjusted returns)
#
# **Even though both have similar total P&L, Portfolio A is vastly superior!**

# ---
#
# ## Correlation
#
# **Correlation** measures the linear relationship between two variables:
#
# $$\rho_{xy} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
#
# Correlation ranges from -1 to +1:
# - **+1**: Perfect positive correlation
# - **0**: No linear relationship
# - **-1**: Perfect negative correlation
#
# ### Positive Correlation

# In[ ]:


x = [1, 2, 3, 4]
y = [1, 3, 2, 5]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Strong positive)")


# In[ ]:


# Perfect positive correlation
x = [1, 2, 3, 4]
y = [2, 3, 4, 5]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Perfect positive)")


# ### Negative Correlation

# In[ ]:


x = [1, 2, 3, 4]
y = [-1, -3, -2, -6]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Strong negative)")


# In[ ]:


# Perfect negative correlation
x = [1, 2, 3, 4]
y = [-1, -2, -3, -4]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Perfect negative)")


# **In trading**: We look for correlations between lagged returns and future returns
# to find predictable patterns.

# ---
#
# ## Time Series Fundamentals
#
# ### Loading Real Market Data
#
# Let's work with real BTCUSDT perpetual futures data.

# In[ ]:


# Load hourly OHLC data
url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

print(f"Data shape: {btcusdt.shape}")
print(f"Date range: {btcusdt.index.min()} to {btcusdt.index.max()}")
btcusdt.head()


# ### Visualizing the Price Series

# In[ ]:


btcusdt['close'].plot(figsize=(15, 6), title='BTC/USDT Close Price')
plt.ylabel('Price (USDT)')
plt.show()


# ### Price Distribution
#
# Raw prices are **not normally distributed** - they have a long right tail:

# In[ ]:


btcusdt['close'].hist(bins=200, figsize=(15, 6))
plt.title("Distribution of Close Prices")
plt.xlabel("Close Price (USDT)")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


print(f"Mean price: ${btcusdt['close'].mean():,.2f}")
print(f"Std dev: ${btcusdt['close'].std():,.2f}")


# ---
#
# ## Differencing and Stationarity
#
# ### The Problem with Raw Prices
#
# Raw prices are **non-stationary** - their statistical properties change over time.
# This makes them difficult to model and predict.
#
# ### Differencing
#
# **Differencing** converts prices to price changes:
#
# $$\Delta P_t = P_t - P_{t-1}$$

# In[ ]:


btcusdt['close_delta'] = btcusdt['close'] - btcusdt['close'].shift()
btcusdt[['close', 'close_delta']].head(10)


# ### Distribution of Price Changes
#
# Differenced data is much closer to normally distributed:

# In[ ]:


btcusdt['close_delta'].hist(bins=80, figsize=(15, 6))
plt.title("Distribution of Close Price Delta (More Normal!)")
plt.xlabel("Price Change (USDT)")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


print(f"Mean price change: ${btcusdt['close_delta'].mean():.2f}")
print(f"Std dev: ${btcusdt['close_delta'].std():.2f}")


# ### Stationarity Check: Yearly Statistics
#
# True stationarity requires that statistical properties remain constant over time.
# Let's check if our differenced series is truly stationary:

# In[ ]:


yearly_stats = btcusdt['close_delta'].resample('YE').agg(['mean', 'std'])
yearly_stats


# **Warning**: The standard deviation varies significantly by year!
# This is **heteroskedasticity** (changing volatility), common in financial data.
# Differencing alone doesn't guarantee stationarity.

# ---
#
# ## Creating Lagged Features
#
# ### Lag Operations
#
# **Lags** shift data by a specified number of periods, creating features
# for autoregressive models:
#
# $$x_{t-k} = \text{lag}(x_t, k)$$

# In[ ]:


btcusdt['close_delta_lag_1'] = btcusdt['close_delta'].shift(1)
btcusdt['close_delta_lag_2'] = btcusdt['close_delta'].shift(2)
btcusdt['close_delta_lag_3'] = btcusdt['close_delta'].shift(3)
btcusdt['close_delta_lag_4'] = btcusdt['close_delta'].shift(4)


# In[ ]:


btcusdt[['close_delta', 'close_delta_lag_1', 'close_delta_lag_2',
         'close_delta_lag_3', 'close_delta_lag_4']].head(10)


# In[ ]:


# Remove NaN rows for analysis
btcusdt[['close_delta', 'close_delta_lag_1', 'close_delta_lag_2',
         'close_delta_lag_3', 'close_delta_lag_4']].dropna().head(10)


# ---
#
# ## Serial Correlation
#
# **Serial correlation** (autocorrelation) measures how correlated a time series
# is with its own past values. This is crucial for finding predictability!
#
# $$\rho_k = \text{Corr}(x_t, x_{t-k})$$

# In[ ]:


# Correlation matrix of deltas with their lags
corr_matrix = btcusdt[['close_delta', 'close_delta_lag_1', 'close_delta_lag_2',
                       'close_delta_lag_3', 'close_delta_lag_4']].dropna().corr()
corr_matrix


# **Interpretation**:
# - Look at the first row (close_delta correlations)
# - Small negative correlations with lags suggest weak **mean reversion**
# - If we found positive correlations, it would suggest **momentum**

# ---
#
# ## Autoregressive Models
#
# ### AR(1): First-Order Autoregressive Model
#
# The **AR(1) model** predicts the next value based on the previous value:
#
# $$y_t = w \cdot y_{t-1} + b + \epsilon_t$$
#
# Where:
# - $w$ = Weight (coefficient)
# - $b$ = Bias (intercept)
# - $\epsilon_t$ = Random error term

# In[ ]:


# Create simple time series
ts = pd.DataFrame({'log_return': [-0.1, 0.2, -0.2, 0.1, -0.3, 0.3]})

# Add date index
n = len(ts)
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq='D')
ts.index = dates

ts


# In[ ]:


# Create lag feature
ts['log_return_lag_1'] = ts['log_return'].shift()
ts


# ### AR(1) Model Definition

# In[ ]:


# AR(1) model: y = w * y_lag_1 + b
y = ts['log_return']
y_lag_1 = ts['log_return_lag_1']

# Initial parameters (untrained)
weight, bias = 0.0, 0.0

# Prediction (would be 0 with these parameters)
y_hat = weight * y_lag_1 + bias


# ---
#
# ## Fundamental Trading Dynamics
#
# The AR(1) model can capture two fundamental market behaviors:
#
# 1. **Mean Reversion**: Prices tend to return to their average ($w < 0$)
# 2. **Momentum**: Prices continue in their current direction ($w > 0$)
#
# ### Mean Reversion
#
# In mean-reverting markets, a positive return is followed by a negative return,
# and vice versa. The weight $w$ is **negative**.

# In[ ]:


weight = -0.5  # Negative weight = mean reversion
bias = 0.00001

ts['y_hat_reversion'] = weight * ts['log_return_lag_1'] + bias
ts[['log_return', 'log_return_lag_1', 'y_hat_reversion']]


# **Interpretation**: When yesterday's return was positive (+0.2), our model
# predicts a negative return today (-0.1). This is mean reversion behavior.

# ### Momentum
#
# In trending markets, a positive return is followed by another positive return.
# The weight $w$ is **positive**.

# In[ ]:


# Create trending series
ts_momentum = pd.DataFrame({
    'log_return': [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
})

n = len(ts_momentum)
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq='D')
ts_momentum.index = dates
ts_momentum['log_return_lag_1'] = ts_momentum['log_return'].shift()
ts_momentum


# In[ ]:


weight = 0.5  # Positive weight = momentum
bias = 0.0001

ts_momentum['y_hat_momentum'] = weight * ts_momentum['log_return_lag_1'] + bias
ts_momentum[['log_return', 'log_return_lag_1', 'y_hat_momentum']]


# **Interpretation**: When yesterday's return was positive (+0.1), our model
# predicts another positive return (+0.05). This is momentum behavior.

# ---
#
# ## Summary: Mean Reversion vs Momentum
#
# | Behavior | AR(1) Weight | Pattern | Strategy |
# |----------|-------------|---------|----------|
# | Mean Reversion | $w < 0$ | +/- oscillation | Fade moves |
# | Random Walk | $w = 0$ | No pattern | Cannot predict |
# | Momentum | $w > 0$ | Trends continue | Follow trends |

# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Calculate Serial Correlation
#
# Calculate the correlation between log returns and their lag-1 values.

# In[ ]:


# TODO: Add log returns to btcusdt
# btcusdt['log_return'] = ...


# In[ ]:


# TODO: Calculate correlation between log_return and log_return_lag_1


# ### Exercise 2: Identify Market Regime
#
# Based on the correlation you found, is BTC showing mean reversion or momentum?

# In[ ]:


# TODO: Analyze and interpret the correlation


# ### Exercise 3: Compare Different Lag Periods
#
# Create lag-1, lag-2, and lag-3 features for log returns.
# Which lag has the strongest correlation?

# In[ ]:


# TODO: Create multiple lags and compare correlations


# ---
#
# ## Key Takeaways
#
# 1. **Central tendency**: Mean is sensitive to outliers; median is robust
#
# 2. **Risk-adjusted returns**: Sharpe ratio = return / risk
#    - High Sharpe = good risk-adjusted performance
#
# 3. **Correlation**: Measures linear relationship (-1 to +1)
#    - Serial correlation reveals predictable patterns
#
# 4. **Stationarity**: Financial data is often non-stationary
#    - Differencing helps but doesn't guarantee stationarity
#    - Log returns are preferred over price levels
#
# 5. **AR(1) Model**: $y_t = w \cdot y_{t-1} + b$
#    - $w < 0$: Mean reversion
#    - $w > 0$: Momentum
#
# 6. **Key formulas**:
#    - Sharpe Ratio: $SR = \frac{\bar{r}}{\sigma_r}$
#    - Correlation: $\rho_{xy} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$
#    - AR(1): $y_t = w \cdot y_{t-1} + b$
#
# ---
#
# **Next Module**: Statistical Edge - Matrix algebra and building ML models for trading
