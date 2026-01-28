#!/usr/bin/env python
# coding: utf-8

# # Module 02: Arrays and Data Structures
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Introduction to Arrays](#introduction-to-arrays)
# 3. [Python Lists](#python-lists)
# 4. [List Operations](#list-operations)
# 5. [NumPy Arrays](#numpy-arrays)
# 6. [Logarithms in Finance](#logarithms-in-finance)
# 7. [Log Returns](#log-returns)
# 8. [Practical Exercises](#practical-exercises)
# 9. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Understand arrays as the fundamental building block of quantitative trading
# - Work with Python lists and NumPy arrays efficiently
# - Understand computational complexity of list operations
# - Apply logarithms to financial calculations
# - Calculate and interpret log returns
# - Understand why log returns are preferred in quantitative finance

# ---
#
# ## Introduction to Arrays
#
# Arrays are **the most fundamental data structure** in quantitative trading, machine learning, and AI.
# Every price series, every feature vector, and every model parameter is stored as an array.
#
# In trading, we constantly work with sequences of data:
# - Price time series: `[100.5, 101.2, 99.8, 102.1, ...]`
# - Trade P&L history: `[150.0, -75.0, 200.0, -50.0, ...]`
# - Model features: `[lag_1, lag_2, volatility, momentum, ...]`

# ---
#
# ## Python Lists
#
# Python lists are the simplest array-like structure. They are flexible but not optimized for numerical computations.

# ### Creating a List
#
# Let's create a simple price series:

# In[ ]:


prices = [10.2, 9.4, 9.9, 10.5]
prices


# In[ ]:


# Check the type
type(prices)


# In[ ]:


# Get the length (number of elements)
len(prices)


# ### Accessing Elements
#
# Python uses **zero-based indexing** (first element is at index 0):
#
# ```
# Index:    0      1      2      3
# Prices: [10.2,  9.4,   9.9,  10.5]
# Neg:     -4     -3     -2     -1
# ```

# In[ ]:


# Forward indexing (0, 1, 2, ...)
print(f"prices[0] = {prices[0]}")  # First element
print(f"prices[1] = {prices[1]}")  # Second element
print(f"prices[2] = {prices[2]}")  # Third element
print(f"prices[3] = {prices[3]}")  # Fourth element


# In[ ]:


# Negative indexing (-1, -2, -3, ...)
print(f"prices[-1] = {prices[-1]}")  # Last element
print(f"prices[-2] = {prices[-2]}")  # Second to last
print(f"prices[-3] = {prices[-3]}")  # Third to last
print(f"prices[-4] = {prices[-4]}")  # Fourth to last (same as first)


# In[ ]:


# Accessing out of bounds raises an IndexError
# Uncomment to see the error:
# prices[5]


# ---
#
# ## List Operations
#
# ### Updating Elements

# In[ ]:


prices = [10.2, 9.4, 9.9, 10.5]
prices[0] = None  # Replace first element
prices


# In[ ]:


prices[-1] = None  # Replace last element
prices


# ### Removing Elements
#
# There are multiple ways to remove elements from a list:

# In[ ]:


prices = [10.2, 9.4, 9.9, 10.5]

# pop() removes and returns the last element
last_price = prices.pop()
print(f"Removed: {last_price}, Remaining: {prices}")


# In[ ]:


prices = [10.2, 9.4, 9.9, 10.5]

# pop(0) removes and returns the first element
first_price = prices.pop(0)
print(f"Removed: {first_price}, Remaining: {prices}")


# In[ ]:


prices = [10.2, 9.4, 9.9]

# del removes by index without returning
del prices[0]
prices


# ### Performance Consideration: O(1) vs O(n)
#
# **Critical for trading systems**: List operations have different computational complexities.
#
# - `pop()` (remove last): **O(1)** - Constant time
# - `pop(0)` (remove first): **O(n)** - Linear time (must shift all elements)
#
# For large datasets, this difference is significant:

# In[ ]:


import time

n = 200_000_000
prices_ts = [1.0 for _ in range(n)]


# In[ ]:


# Removing from the front - O(n) - SLOW!
start = time.time()
prices_ts.pop(0)
elapsed = time.time() - start
print(f"pop(0) time: {elapsed*1000:.1f} ms")


# In[ ]:


prices_ts = [1.0 for _ in range(n)]

# Removing from the back - O(1) - FAST!
start = time.time()
prices_ts.pop()
elapsed = time.time() - start
print(f"pop() time: {elapsed*1000:.4f} ms")


# **Key Insight**: When building trading systems that process millions of data points,
# always consider the computational complexity of your operations.

# ### Adding Elements

# In[ ]:


prices = []

# append() adds a single element to the end
prices.append(10.5)
prices


# In[ ]:


# extend() adds multiple elements
prices.extend([11.4, 9.5, 12.3])
prices


# ### Homogeneous vs Inhomogeneous Arrays
#
# Python lists can hold mixed types (inhomogeneous):

# In[ ]:


# Inhomogeneous - different types (avoid in numerical computing)
mixed = [1.0, "a", True, 2]
mixed


# In[ ]:


# Homogeneous - same type (preferred for numerical computing)
floats = [1.0, 2.0, 3.0, 4.0]
integers = [1, 2, 3, 4]
print(f"Floats: {floats}")
print(f"Integers: {integers}")


# **Best Practice**: Always use homogeneous arrays for numerical data to avoid type conversion overhead.

# ---
#
# ## Loops
#
# Loops allow us to iterate over arrays and perform calculations:

# In[ ]:


# Basic range loop
for i in range(5):
    print(i)


# In[ ]:


# Iterating over a list
prices = [10.2, 9.5, 11.5, 9.4]
for price in prices:
    print(price)


# ### Calculating Total P&L with a Loop
#
# A common trading task - summing up individual trade P&Ls:

# In[ ]:


trade_pnls = [1.2, -2.0, -1.0, 4.1]

total_pnl = 0.0
for trade_pnl in trade_pnls:
    total_pnl += trade_pnl

print(f"Total P&L: ${total_pnl:.2f}")


# In[ ]:


# Verify: manual calculation
1.2 + (-2.0) + (-1.0) + 4.1


# ---
#
# ## NumPy Arrays
#
# **NumPy** (Numerical Python) is the foundation of scientific computing in Python.
# NumPy arrays are:
# - Faster than Python lists (implemented in C)
# - Support vectorized operations (no explicit loops needed)
# - Memory efficient (contiguous memory layout)

# In[ ]:


import numpy as np

# Create an array of 100 million ones
n = 100_000_000
a = np.ones(n)
a


# In[ ]:


len(a)


# ### Vectorized Operations vs Loops
#
# NumPy's vectorized operations are **orders of magnitude faster** than Python loops:

# In[ ]:


# NumPy sum - vectorized (uses SIMD instructions)
import time

start = time.time()
result = np.sum(a)
elapsed = time.time() - start
print(f"NumPy sum: {result:.0f}")
print(f"Time: {elapsed*1000:.1f} ms")


# In[ ]:


# Python loop - sequential (interpreted)
start = time.time()
total = 0.0
for val in a:
    total += val
elapsed = time.time() - start
print(f"Loop sum: {total:.0f}")
print(f"Time: {elapsed*1000:.1f} ms")


# **Why is NumPy faster?**
#
# 1. **SIMD (Single Instruction, Multiple Data)**: Processes multiple values in a single CPU instruction
# 2. **No type checking**: All elements are the same type
# 3. **Memory locality**: Contiguous memory access
# 4. **Compiled C code**: No Python interpreter overhead

# ---
#
# ## Logarithms in Finance
#
# Logarithms are fundamental in quantitative finance. The natural logarithm (ln) is the inverse of the exponential function:
#
# $$e^x = y \iff \ln(y) = x$$
#
# Where $e \approx 2.71828$ is Euler's number.

# In[ ]:


# Exponential function
np.exp(2)


# In[ ]:


# Log is the inverse of exp
np.log(np.exp(2))


# ### Compound Growth
#
# Consider investing $1,000 at a 5% annual rate:
#
# $$V_t = V_0 \times (1 + r)^t$$
#
# Where:
# - $V_t$ = Value at time $t$
# - $V_0$ = Initial value
# - $r$ = Growth rate (0.05 = 5%)
# - $t$ = Number of periods

# In[ ]:


# Year 1
print(f"Year 1: ${1000 * 1.05:.2f}")

# Year 2
print(f"Year 2: ${1000 * 1.05 * 1.05:.2f}")

# Year 3
print(f"Year 3: ${1000 * 1.05 * 1.05 * 1.05:.2f}")


# In[ ]:


# Using a loop
capital = 1000
for year in range(1, 21):
    capital *= 1.05
    if year % 5 == 0:
        print(f"Year {year}: ${capital:.2f}")


# In[ ]:


# Using the formula directly
1000 * 1.05 ** 20


# ### How Long to Double Your Investment?
#
# We want to find $t$ such that:
#
# $$2V_0 = V_0 \times (1 + r)^t$$
#
# Simplifying:
#
# $$2 = (1 + r)^t$$
#
# Taking the natural log of both sides:
#
# $$\ln(2) = t \times \ln(1 + r)$$
#
# Therefore:
#
# $$t = \frac{\ln(2)}{\ln(1 + r)}$$

# In[ ]:


r = 0.05  # 5% annual return
t = np.log(2) / np.log(1 + r)
print(f"Time to double at 5% annual return: {t:.2f} years")


# In[ ]:


# Verify
1000 * 1.05 ** t


# **Rule of 72**: A quick approximation is $t \approx 72 / (r \times 100)$. At 5%, this gives $72/5 = 14.4$ years.

# ---
#
# ## Log Returns
#
# ### Why Returns Instead of Prices?
#
# Raw prices are not directly comparable across different assets or time periods.
# A $100 gain means different things depending on the capital:

# In[ ]:


pnl = 100

# $100 gain on $50 investment = 200% return
print(f"$100 on $50: {pnl / 50:.0%} return")

# $100 gain on $99 investment = ~101% return
print(f"$100 on $99: {pnl / 99:.1%} return")


# ### Simple Returns vs Log Returns
#
# **Simple Return**:
#
# $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$
#
# **Log Return** (Continuously Compounded Return):
#
# $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$
#
# ### Why Log Returns are Preferred in Quant Finance
#
# 1. **Additivity**: Log returns sum across time: $r_{total} = r_1 + r_2 + ... + r_n$
# 2. **Symmetry**: A +10% log return followed by -10% returns to the original value
# 3. **Statistical Properties**: More likely to be normally distributed
# 4. **Numerical Stability**: No issues with compounding over long periods

# In[ ]:


# Example: Portfolio goes $100 -> $120 -> $100
portfolio = [100, 120, 100]

# Simple returns
simple_returns = [(120-100)/100, (100-120)/120]
print(f"Simple returns: {simple_returns}")
print(f"Sum of simple returns: {sum(simple_returns):.4f}")  # Not zero!


# In[ ]:


# Log returns
log_returns = [np.log(120/100), np.log(100/120)]
print(f"Log returns: {log_returns}")
print(f"Sum of log returns: {sum(log_returns):.6f}")  # Zero! (within floating point precision)


# **Key Insight**: Log returns are additive, which makes them ideal for:
# - Calculating cumulative returns
# - Portfolio aggregation
# - Statistical analysis
# - Model training

# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Calculate the Average Log Return
#
# Given a series of log returns, calculate the average (mean) log return using a loop.

# In[ ]:


log_returns = [-0.1, 0.22, 0.15, 0.344, -0.2]
avg_log_return = 0.0

# TODO: Write a loop to calculate the average log return
# Hint: sum all returns, then divide by the count


# In[ ]:


# Your loop goes here


# In[ ]:


# Verify your answer
expected = 0.0828
print(f"Your answer: {avg_log_return}")
print(f"Expected: {expected}")
print(f"Correct: {abs(avg_log_return - expected) < 0.0001}")


# ### Exercise 2: Calculate Total Log Returns
#
# Given a portfolio value series, calculate the log returns and verify that the sum of log returns
# equals the total log return from start to end.

# In[ ]:


portfolio = [100, 120, 100, 80, 155]
log_returns = []

# TODO: Calculate log returns between consecutive portfolio values
# Hint: log_return = np.log(P_t / P_{t-1})


# In[ ]:


# Your loop goes here


# In[ ]:


# Verify: starting value * exp(sum of log returns) should equal ending value
total_log_return = np.sum(log_returns) if log_returns else 0
final_value = 100 * np.exp(total_log_return)
print(f"Calculated final value: {final_value:.1f}")
print(f"Actual final value: 155.0")
print(f"Correct: {abs(final_value - 155.0) < 0.01}")


# ### Exercise 3: Calculate Cumulative Log Returns
#
# Calculate the cumulative log returns at each time step.

# In[ ]:


portfolio = [100, 120, 100, 80, 155]
cum_log_returns = []

# TODO: Calculate cumulative log returns
# Each entry should be the sum of all log returns up to that point


# In[ ]:


# Your loop goes here


# In[ ]:


# Verify
expected = [
    np.log(120/100),
    np.log(120/100) + np.log(100/120),
    np.log(120/100) + np.log(100/120) + np.log(80/100),
    np.log(120/100) + np.log(100/120) + np.log(80/100) + np.log(155/80)
]
print(f"Your answer: {cum_log_returns}")
print(f"Expected: {expected}")


# ### Challenge Exercise: Portfolio Tracking
#
# Implement a function that takes a list of prices and returns:
# 1. Log returns
# 2. Cumulative log returns
# 3. Running portfolio value (starting with $100)

# In[ ]:


def analyze_portfolio(prices):
    """
    Analyze a price series and return portfolio metrics.

    Parameters:
    -----------
    prices : list
        List of prices

    Returns:
    --------
    dict with keys:
        - 'log_returns': list of log returns
        - 'cum_log_returns': list of cumulative log returns
        - 'portfolio_value': list of portfolio values starting at $100
    """
    # TODO: Implement this function
    pass


# In[ ]:


# Test your function
test_prices = [100, 105, 103, 110, 108]
result = analyze_portfolio(test_prices)
if result:
    print(f"Log Returns: {result['log_returns']}")
    print(f"Cumulative Returns: {result['cum_log_returns']}")
    print(f"Portfolio Value: {result['portfolio_value']}")


# ---
#
# ## Key Takeaways
#
# 1. **Arrays are fundamental**: Every aspect of quantitative trading relies on array operations
#
# 2. **Computational complexity matters**: Use `pop()` instead of `pop(0)` for O(1) performance
#
# 3. **NumPy is essential**: Vectorized operations are orders of magnitude faster than loops
#
# 4. **Log returns are preferred** because they are:
#    - Additive across time
#    - Symmetric (gains and losses are comparable)
#    - More likely to be normally distributed
#
# 5. **Key formulas**:
#    - Log return: $r_t = \ln(P_t / P_{t-1})$
#    - Total return: $r_{total} = \sum_{i=1}^{n} r_i$
#    - Final value: $V_n = V_0 \times e^{r_{total}}$
#
# ---
#
# **Next Module**: Vectorization - Building efficient data analysis tools using vectorized operations
