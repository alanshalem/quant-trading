#!/usr/bin/env python
# coding: utf-8

# # Module 03: Vectorization
#
# **Quant Trading Accelerator**
#
# ---

# ## Table of Contents
#
# 1. [Learning Objectives](#learning-objectives)
# 2. [Introduction to Vectorization](#introduction-to-vectorization)
# 3. [Building a Vector Class](#building-a-vector-class)
# 4. [Vector-Scalar Operations](#vector-scalar-operations)
# 5. [Vector-Vector Operations](#vector-vector-operations)
# 6. [Vectorized Statistics](#vectorized-statistics)
# 7. [Building a DataFrame Library](#building-a-dataframe-library)
# 8. [Matrices and Feature Engineering](#matrices-and-feature-engineering)
# 9. [Practical Exercises](#practical-exercises)
# 10. [Key Takeaways](#key-takeaways)

# ---
#
# ## Learning Objectives
#
# By the end of this module, you will be able to:
#
# - Understand vectorization and why it's essential for quantitative finance
# - Build custom Vector and DataFrame classes for financial data analysis
# - Perform vectorized arithmetic and statistical operations
# - Calculate the Sharpe ratio using vectorized operations
# - Create features and targets for machine learning models
# - Understand the performance benefits of SIMD operations

# ---
#
# ## Introduction to Vectorization
#
# **Vectorization** is the process of converting operations that work on single values (scalars)
# into operations that work on entire arrays simultaneously.
#
# ### Why Vectorization Matters in Quant Trading
#
# 1. **Speed**: Vectorized operations are 10-100x faster than Python loops
# 2. **Code Clarity**: Vectorized code is more readable and maintainable
# 3. **Memory Efficiency**: Better cache utilization with contiguous arrays
# 4. **SIMD**: Modern CPUs can process multiple values in a single instruction
#
# **SIMD** (Single Instruction, Multiple Data) allows the CPU to perform the same
# operation on multiple data points simultaneously.

# In[ ]:


import numpy as np

# ---
#
# ## Building a Vector Class
#
# Let's build a `Vector` class that wraps NumPy arrays and provides
# financial computing functionality.

# In[ ]:


class Vector:
    """
    A lightweight vector wrapper around NumPy arrays for financial computing.

    Provides elementwise arithmetic operations, statistical methods,
    and operator overloads for clean, readable code.

    Parameters
    ----------
    data : array_like
        Input data (list, tuple, or NumPy array)

    Attributes
    ----------
    data : np.ndarray
        The underlying NumPy array

    Examples
    --------
    >>> returns = Vector([0.01, -0.02, 0.015, 0.008])
    >>> returns.mean()
    0.00325
    >>> returns.std()
    0.0138...
    """

    data: np.ndarray

    def __init__(self, data) -> None:
        """Initialize the vector with the given data."""
        self.data = np.array(data)

    # -----------------------------------------------------------------
    # Basic Arithmetic Methods
    # -----------------------------------------------------------------
    def add(self, y) -> np.ndarray:
        """Add a scalar or array to the vector elementwise."""
        return self.data + y

    def sub(self, y) -> np.ndarray:
        """Subtract a scalar or array from the vector elementwise."""
        return self.data - y

    def mul(self, y) -> np.ndarray:
        """Multiply the vector by a scalar or array elementwise."""
        return self.data * y

    def div(self, y) -> np.ndarray:
        """Divide the vector by a scalar or array elementwise."""
        return self.data / y

    # -----------------------------------------------------------------
    # Statistical Methods
    # -----------------------------------------------------------------
    def sum(self):
        """Return the sum of all elements."""
        return np.sum(self.data)

    def mean(self):
        """
        Return the arithmetic mean.

        $$\\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i$$
        """
        return np.mean(self.data)

    def var(self) -> np.ndarray:
        """
        Return the population variance.

        $$\\sigma^2 = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\bar{x})^2$$
        """
        mu = self.mean()
        return np.mean((self.data - mu) ** 2)

    def std(self):
        """
        Return the population standard deviation.

        $$\\sigma = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\bar{x})^2}$$
        """
        return np.sqrt(self.var())

    def len(self):
        """Return the number of elements."""
        return len(self.data)

    # -----------------------------------------------------------------
    # Operator Overloads
    # -----------------------------------------------------------------
    def __add__(self, other):
        """Implements self + other."""
        return Vector(self.data + self._to_array(other))

    def __sub__(self, other):
        """Implements self - other."""
        return Vector(self.data - self._to_array(other))

    def __mul__(self, other):
        """Implements self * other."""
        return Vector(self.data * self._to_array(other))

    def __truediv__(self, other):
        """Implements self / other."""
        return Vector(self.data / self._to_array(other))

    def __radd__(self, other):
        """Implements other + self."""
        return self.__add__(other)

    def __rsub__(self, other):
        """Implements other - self."""
        return Vector(self._to_array(other) - self.data)

    def __rmul__(self, other):
        """Implements other * self."""
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Implements other / self."""
        return Vector(self._to_array(other) / self.data)

    def __pow__(self, power):
        """Implements self ** power."""
        return Vector(self.data ** power)

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------
    def __getitem__(self, index):
        """Allow element or slice access via v[index]."""
        result = self.data[index]
        if isinstance(result, np.ndarray):
            return Vector(result)
        return result

    def __len__(self):
        """Return length for len(v)."""
        return len(self.data)

    # -----------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------
    def _to_array(self, x):
        """Convert input to NumPy array for arithmetic operations."""
        if isinstance(x, Vector):
            return x.data
        return np.array(x)

    def __repr__(self):
        """Return string representation."""
        return f"Vector({self.data})"


# ---
#
# ## Vector-Scalar Operations
#
# When we perform arithmetic between a vector and a scalar, the operation
# is applied to every element. This is called **broadcasting**.

# In[ ]:


vec = Vector([1, 2, 3])
vec


# ### Addition

# In[ ]:


# Add 1 to every element
vec + 1


# In[ ]:


# Equivalent manual calculation
[1 + 1, 2 + 1, 3 + 1]


# ### Subtraction

# In[ ]:


vec - 2


# In[ ]:


[1 - 2, 2 - 2, 3 - 2]


# ### Multiplication and Division

# In[ ]:


vec * 2


# In[ ]:


vec / 2


# ### The Slow Way: Python Loops
#
# Without vectorization, we would need explicit loops:

# In[ ]:


# This is slow and verbose - DON'T do this in production
v = []
for e in [1, 2, 3]:
    v.append(e + 1)
v


# ---
#
# ## The Power of Vectorization: Performance Comparison
#
# Let's see the dramatic performance difference between loops and vectorized operations.

# In[ ]:


import time

n = 100_000_000
v1 = [1 for _ in range(n)]  # Python list
v2 = Vector(v1)              # Vectorized


# In[ ]:


# Python loop - SLOW
start = time.time()
y = []
for x in v1:
    y.append(x + 1)
elapsed_loop = time.time() - start
print(f"Python loop: {elapsed_loop*1000:.1f} ms")
print(f"Last 10 elements: {y[-10:]}")


# In[ ]:


# Vectorized operation - FAST
start = time.time()
y = v2 + 1
elapsed_vec = time.time() - start
print(f"Vectorized: {elapsed_vec*1000:.1f} ms")
print(f"Last 10 elements: {y.data[-10:]}")


# In[ ]:


# Speedup factor
print(f"Speedup: {elapsed_loop / elapsed_vec:.1f}x faster")


# **Key Insight**: Vectorized operations leverage SIMD (Single Instruction, Multiple Data)
# instructions on modern CPUs, processing multiple values in parallel.

# ---
#
# ## Vector-Vector Operations
#
# When performing operations between two vectors of the same length,
# the operation is applied elementwise.

# In[ ]:


x = Vector([1, 2, 3, 4])
y = Vector([1, -1, 2, -2])


# In[ ]:


# Elementwise addition
x + y


# In[ ]:


# Verify manually
[1 + 1, 2 + (-1), 3 + 2, 4 + (-2)]


# In[ ]:


# Elementwise subtraction
x - y


# In[ ]:


# Elementwise multiplication (Hadamard product)
x * y


# ---
#
# ## Vectorized Statistics
#
# Statistical calculations are fundamental in quantitative finance.
# Let's see how to compute them efficiently with vectorized operations.
#
# ### Variance Calculation
#
# The **variance** measures the spread of returns around the mean:
#
# $$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

# In[ ]:


log_returns = Vector([0.01, 0.015, 0.02, -0.01])
mu = log_returns.mean()
print(f"Mean return: {mu:.4f}")


# In[ ]:


# Step 1: Subtract mean from each return (deviations)
deviations = log_returns - mu
deviations


# In[ ]:


# Step 2: Square each deviation
squared_deviations = (log_returns - mu) ** 2
squared_deviations


# In[ ]:


# Step 3: Take the mean of squared deviations = variance
variance = ((log_returns - mu) ** 2).mean()
print(f"Variance: {variance:.8f}")


# In[ ]:


# Verify with our Vector method
log_returns.var()


# ### Standard Deviation
#
# The **standard deviation** is the square root of variance:
#
# $$\sigma = \sqrt{\sigma^2}$$

# In[ ]:


# Manual calculation
np.sqrt(log_returns.var())


# In[ ]:


# Using our Vector method
log_returns.std()


# ---
#
# ## Vectorized Sharpe Ratio
#
# The **Sharpe Ratio** is the most important risk-adjusted performance metric in finance:
#
# $$\text{Sharpe Ratio} = \frac{\mathbb{E}[R] - R_f}{\sigma_R} \approx \frac{\bar{r}}{\sigma_r}$$
#
# Where:
# - $\mathbb{E}[R]$ = Expected return
# - $R_f$ = Risk-free rate (often assumed to be 0 for simplicity)
# - $\sigma_R$ = Standard deviation of returns
#
# The Sharpe ratio tells us **how much return we get per unit of risk**.

# In[ ]:


# Portfolio A: Consistent small gains
portfolio_a = Vector([0.01, 0.01, 0.02, -0.01])

print(f"Total return: {portfolio_a.sum():.4f}")
print(f"Mean return: {portfolio_a.mean():.4f}")
print(f"Std deviation: {portfolio_a.std():.4f}")
print(f"Sharpe Ratio: {portfolio_a.mean() / portfolio_a.std():.4f}")


# In[ ]:


# Portfolio B: Large swings, same total
portfolio_b = Vector([-0.01, -0.01, -0.01, 0.06])

print(f"Total return: {portfolio_b.sum():.4f}")
print(f"Mean return: {portfolio_b.mean():.4f}")
print(f"Std deviation: {portfolio_b.std():.4f}")
print(f"Sharpe Ratio: {portfolio_b.mean() / portfolio_b.std():.4f}")


# **Key Insight**: Both portfolios have the same total return (0.03), but Portfolio A
# has a higher Sharpe ratio because it achieves this return with lower volatility.

# ---
#
# ## Building a DataFrame Library
#
# Now let's build a simple DataFrame class for tabular financial data.

# ### The Column Class

# In[ ]:


class Column:
    """
    Represents a single column of data in a tabular dataset.

    Each Column has a name and a vector of data. Supports operations
    like shift (for creating lags), division, and logarithm.

    Parameters
    ----------
    name : str
        The column name
    x : array_like
        The column data
    """

    vec: 'Vector'

    def __init__(self, name, x):
        """Initialize a Column with name and data."""
        self.vec = Vector(x)
        self.name = name

    def len(self):
        """Return the number of elements."""
        return len(self.vec)

    def sum(self):
        """Return the sum of all elements."""
        return np.sum(self.vec)

    def shift(self, n=1):
        """
        Shift data by n positions (lag operation).

        Creates a lagged version of the column, essential for
        time series analysis and autoregressive models.

        Parameters
        ----------
        n : int
            Number of positions to shift (default=1)

        Returns
        -------
        np.ndarray
            Shifted data with NaN for missing values
        """
        return np.concatenate(([np.nan] * n, self.vec[:-n]))

    def div(self, y) -> np.ndarray:
        """Divide by another Column or array elementwise."""
        if isinstance(y, Column):
            y = y.vec
        return self.vec / y

    def log(self):
        """Compute natural logarithm elementwise."""
        return np.log(self.vec)

    def __truediv__(self, other) -> np.ndarray:
        """Enable '/' operator for division."""
        return self.div(other)

    def __repr__(self):
        """Return string representation."""
        preview = ", ".join(map(str, self.vec[:10]))
        if len(self.vec) > 10:
            preview += ", ..."
        return f"Column(name='{self.name}', data=[{preview}], len={len(self.vec)})"


# In[ ]:


# Create a column of trade P&L
col = Column('trade_pnl', [2.0, -1.0, 3.0, 1.5])
col


# ### The DataFrame Class

# In[ ]:


class DataFrame:
    """
    A simple tabular data structure for financial data analysis.

    Maintains a list of Column objects and supports basic operations
    like adding columns, selecting columns, and pretty printing.

    Parameters
    ----------
    cols : list
        List of Column objects
    """

    def __init__(self, cols):
        """Initialize with a list of columns."""
        self.cols = cols

    def __len__(self):
        """Return the number of rows."""
        return self.cols[0].len()

    def append(self, col):
        """
        Append or update a column in the DataFrame.

        If a column with the same name exists, it will be replaced.
        """
        for i, c in enumerate(self.cols):
            if col.name == c.name:
                self.cols[i] = col
                return
        self.cols.append(col)

    def add_col(self, name, col):
        """Create and append a new column."""
        self.cols.append(Column(name, col))

    def __getitem__(self, keys):
        """
        Select columns by name.

        Parameters
        ----------
        keys : str or list of str
            Column name(s) to select

        Returns
        -------
        np.ndarray
            1D array for single column, 2D array for multiple columns
        """
        if isinstance(keys, str):
            for col in self.cols:
                if col.name == keys:
                    return col.vec.data
            raise KeyError(f"Column '{keys}' not found.")
        elif isinstance(keys, list):
            selected_cols = []
            for key in keys:
                for col in self.cols:
                    if col.name == key:
                        selected_cols.append(col.vec)
                        break
                else:
                    raise KeyError(f"Column '{key}' not found.")
            return np.column_stack(selected_cols)
        else:
            raise TypeError("Key must be a string or list of strings.")

    def __repr__(self):
        """Return a formatted table representation."""
        col_names = [col.name for col in self.cols]

        # Determine column widths
        col_widths = []
        preview_rows = min(len(self), 10)
        for col in self.cols:
            data_preview = [str(x) for x in col.vec[:preview_rows]]
            max_data_width = max(len(x) for x in data_preview) if data_preview else 0
            width = max(len(col.name), max_data_width)
            col_widths.append(width)

        # Format header
        header = " | ".join(
            name.ljust(width) for name, width in zip(col_names, col_widths)
        )
        separator = "-+-".join("-" * width for width in col_widths)

        # Format rows
        rows = []
        for i in range(preview_rows):
            row = " | ".join(
                str(col.vec[i]).ljust(width) for col, width in zip(self.cols, col_widths)
            )
            rows.append(row)

        table = "\n".join([header, separator] + rows)
        if len(self) > 10:
            table += "\n..."
        return table


# ---
#
# ## Working with Time Series Data
#
# Let's create a simple price time series and compute log returns.

# In[ ]:


from datetime import datetime, timedelta

# Create sample data
time = Column('time', [datetime(2025, 10, 1) + timedelta(days=i+1) for i in range(7)])
price = Column('price', [10.0, 11.0, 12.0, 10.0, 13.0, 14.0, 15.0])

table = DataFrame([time, price])
table


# ### Creating Lagged Features
#
# The `shift()` operation creates lagged versions of a column, essential for
# time series analysis:

# In[ ]:


# Create price lag (yesterday's price)
price_lag_1 = price.shift()
price_lag_1


# In[ ]:


table.append(Column('price_lag_1', price_lag_1))
table


# ### Computing Price Ratios

# In[ ]:


# Price ratio: P_t / P_{t-1}
ratio = price / price_lag_1
table.append(Column('ratio', ratio))
table


# ### Computing Log Returns
#
# Log returns are calculated as:
#
# $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$

# In[ ]:


# Log returns from price ratio
ratio_col = Column('ratio', ratio)
log_return = ratio_col.log()
log_return


# In[ ]:


log_return_col = Column('log_return', log_return)
table.append(log_return_col)
table


# ### Creating Autoregressive Features
#
# For AR(1) models, we need lagged log returns:

# In[ ]:


log_return_lag_1_col = Column('log_return_lag_1', log_return_col.shift())
table.append(log_return_lag_1_col)
table


# ---
#
# ## Matrices and Feature Engineering
#
# ### Column-Major vs Row-Major Order
#
# Understanding matrix layout is important for ML feature engineering.
#
# **Column-Major** (each row is a feature vector):

# In[ ]:


x = [1, 2, 3, 4]
y = [1, 1, 1, 1]
matrix_col = np.array([x, y])
matrix_col


# In[ ]:


# Accessing rows
print(f"Row 0: {matrix_col[0]}")
print(f"Row 1: {matrix_col[1]}")


# **Row-Major** (each row is an observation):

# In[ ]:


# Row-major format (more common in ML)
matrix_row = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
matrix_row


# In[ ]:


# Each row is an observation
print(f"Observation 0: {matrix_row[0]}")
print(f"Observation 1: {matrix_row[1]}")


# ### Creating Features (X) and Target (y)
#
# For machine learning, we need to separate features from the target:

# In[ ]:


# Features: lagged log returns (input to our model)
X = table[['log_return_lag_1']]
X


# In[ ]:


# Target: current log return (what we want to predict)
y = table['log_return']
y


# ---
#
# ## Practical Exercises
#
# ### Exercise 1: Create Log Returns
#
# Given a price series, create a log return column using vectorized operations.

# In[ ]:


from datetime import datetime, timedelta

cols = [
    Column('date', [datetime(2025, 10, 1) + timedelta(days=1+i) for i in range(10)]),
    Column('price', [10.0, 8.0, 11.0, 7.0, 9.0, 12.0, 8.0, 9.0, 7.0, 10.0])
]
df = DataFrame(cols)
df


# In[ ]:


# TODO: Add log return column to the DataFrame as a vectorized one-liner
# Hint: log_return = ln(price / price_lag_1)


# In[ ]:


# Verification: checks the last column (log return) is calculated correctly
expected = [np.nan, np.log(8.0/10.0), np.log(11.0/8.0), np.log(7.0/11.0),
     np.log(9.0/7.0), np.log(12.0/9.0), np.log(8.0/12.0),
     np.log(9.0/8.0), np.log(7.0/9.0), np.log(10.0/7.0)]
# np.allclose(df.cols[-1].vec.data, expected, equal_nan=True)


# ### Exercise 2: Add Log Return Lag
#
# Add a lagged log return column for AR(1) modeling.

# In[ ]:


# TODO: Add log_return_lag_1 column


# In[ ]:


# Verification
expected_lag = [np.nan, np.nan, np.log(8.0/10.0), np.log(11.0/8.0), np.log(7.0/11.0),
     np.log(9.0/7.0), np.log(12.0/9.0), np.log(8.0/12.0),
     np.log(9.0/8.0), np.log(7.0/9.0)]
# np.allclose(df.cols[-1].vec.data, expected_lag, equal_nan=True)


# ### Exercise 3: Implement a Sharpe Ratio Function
#
# Create a function that calculates the Sharpe ratio from a Vector of returns.

# In[ ]:


def sharpe_ratio(returns: Vector, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio.

    Parameters
    ----------
    returns : Vector
        Vector of returns
    risk_free_rate : float
        The risk-free rate (default 0)

    Returns
    -------
    float
        The Sharpe ratio
    """
    # TODO: Implement this function
    pass


# In[ ]:


# Test your function
test_returns = Vector([0.01, 0.02, -0.01, 0.015, 0.005])
# sharpe_ratio(test_returns)


# ---
#
# ## Key Takeaways
#
# 1. **Vectorization is essential**: Operations on entire arrays are 10-100x faster than Python loops
#
# 2. **SIMD operations**: Modern CPUs process multiple values in a single instruction
#
# 3. **The Vector class** provides:
#    - Elementwise arithmetic (+, -, *, /)
#    - Statistical methods (mean, var, std)
#    - Operator overloading for clean code
#
# 4. **The Column/DataFrame classes** enable:
#    - Lagged features via `shift()`
#    - Log return calculations
#    - Feature matrix construction for ML
#
# 5. **Key formulas**:
#    - Mean: $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$
#    - Variance: $\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$
#    - Sharpe Ratio: $SR = \frac{\bar{r}}{\sigma_r}$
#
# ---
#
# **Next Module**: Time Series Analysis - Statistics, stationarity, and autoregression
