#!/usr/bin/env python
# coding: utf-8

# # Module 01: Python Fundamentals for Quantitative Trading
#
# **Quant Trading Accelerator** | Internal Training Program
#
# ---

# ## Table of Contents
#
# 1. [Introduction & Learning Objectives](#introduction)
# 2. [Variables: The Building Blocks](#variables)
# 3. [Numeric Types: Precision Matters](#numeric-types)
# 4. [Arithmetic Operations: Financial Calculations](#arithmetic)
# 5. [Strings: Working with Symbols & Identifiers](#strings)
# 6. [Practical Exercises](#exercises)
# 7. [Key Takeaways](#takeaways)

# ---
#
# ## 1. Introduction & Learning Objectives <a name="introduction"></a>

# ### What is Quantitative Trading?
#
# Quantitative trading uses mathematical models and computational algorithms to identify and execute trading opportunities. The typical workflow follows this pipeline:
#
# ```
# Market Data → Feature Engineering → Model → Signal Generation → Risk Management → Execution
# ```
#
# As a quant, you'll work across this entire pipeline. Python is the lingua franca of quantitative finance due to its:
#
# - **Rapid prototyping** capabilities
# - **Rich ecosystem** (NumPy, Pandas, PyTorch, Polars)
# - **Production readiness** for live trading systems

# ### Learning Objectives
#
# By the end of this module, you will be able to:
#
# 1. **Declare and manipulate variables** for storing market data
# 2. **Understand numeric precision** and its implications for financial calculations
# 3. **Perform arithmetic operations** common in trading (returns, PnL, position sizing)
# 4. **Parse and manipulate strings** for ticker symbols and data identifiers
# 5. **Apply these concepts** to real trading scenarios

# ### Prerequisites
#
# - Basic computer literacy
# - Intellectual curiosity
# - No prior programming experience required

# ---
#
# ## 2. Variables: The Building Blocks <a name="variables"></a>

# ### What is a Variable?
#
# A **variable** is a named container that stores data in memory. In trading systems, variables hold critical information like:
#
# - Current asset prices
# - Position sizes
# - Risk parameters
# - Order identifiers
#
# Think of variables as labeled boxes where you store values for later use.

# ### Basic Variable Assignment
#
# The assignment operator `=` stores a value in a variable. The variable name goes on the left, the value on the right.

# In[ ]:


# Store a price value in a variable called 'price'
# This could represent the current bid price of an asset
price = 10.0


# In[ ]:


# Access the stored value by referencing the variable name
price


# ### Variable Reassignment
#
# Variables can be updated. This is essential for tracking changing market conditions.

# In[ ]:


# Simulate a price tick: the price increases by 0.50
price = price + 0.5


# In[ ]:


# The variable now holds the updated value
price


# ### Creating New Variables from Existing Ones
#
# You can derive new variables from calculations involving existing variables.

# In[ ]:


# Calculate what the price was before the tick
# This is a common operation when computing price changes
old_price = price - 0.5


# In[ ]:


old_price


# ### Important: Value vs. Reference
#
# When you assign one variable to another, Python copies the **value**, not the reference (for primitive types like numbers).

# In[ ]:


# Demonstration of value copying behavior
price = 10.0
new_price = price    # new_price gets a COPY of price's value (10.0)
price = 11.0         # Changing price does NOT affect new_price


# In[ ]:


# new_price still holds the original copied value
new_price


# In[ ]:


# price has the new value
price


# > **Trading Implication**: This behavior is crucial when storing historical prices. Copying a variable creates an independent snapshot, not a linked reference.

# ---
#
# ## 3. Numeric Types: Precision Matters <a name="numeric-types"></a>

# ### Why Numeric Types Matter in Trading
#
# Financial calculations require careful attention to numeric precision. A rounding error of $0.01 per trade, compounded over millions of trades, can result in significant losses or regulatory issues.
#
# Python has two primary numeric types:
#
# | Type | Description | Use Case |
# |------|-------------|----------|
# | `int` | Whole numbers | Trade counts, order IDs |
# | `float` | Decimal numbers | Prices, returns, quantities |

# ### Checking Variable Types
#
# Use the `type()` function to inspect a variable's data type.

# In[ ]:


# price is a float (decimal number)
type(price)


# In[ ]:


# Store the number of trades executed today
no_trades = 100


# In[ ]:


no_trades


# In[ ]:


# no_trades is an integer (whole number)
type(no_trades)


# ### Type Conversion (Casting)
#
# Sometimes you need to convert between types. This is called **casting**.

# In[ ]:


# Convert integer to float
no_trades = float(no_trades)


# In[ ]:


type(no_trades)


# In[ ]:


# Notice the decimal point indicating it's now a float
no_trades


# In[ ]:


# Convert float to integer (truncates decimal part - does NOT round!)
int(100.12)


# In[ ]:


# Convert integer to float
float(100)


# In[ ]:


# Convert string to float (common when parsing market data feeds)
float("262.82")


# > **Warning**: `int()` truncates towards zero, it does not round. `int(2.9)` returns `2`, not `3`.

# ### Floating-Point Precision Caveat
#
# Floating-point numbers have limited precision. This can cause unexpected behavior:

# In[ ]:


# This should equal 0.3, but due to floating-point representation...
0.1 + 0.1 + 0.1


# > **Best Practice**: For financial calculations requiring exact precision, use the `decimal` module or integer arithmetic (e.g., store prices in cents/pips).

# ---
#
# ## 4. Arithmetic Operations: Financial Calculations <a name="arithmetic"></a>

# ### Operator Precedence
#
# Python follows standard mathematical order of operations (PEMDAS/BODMAS):
#
# 1. **P**arentheses
# 2. **E**xponents
# 3. **M**ultiplication / **D**ivision
# 4. **A**ddition / **S**ubtraction

# In[ ]:


# Multiplication happens before addition
3 + 3 * 3  # = 3 + 9 = 12


# In[ ]:


# Equivalent explicit calculation
3 + 9


# In[ ]:


# Use parentheses to change order of operations
(3 + 3) * 3  # = 6 * 3 = 18


# In[ ]:


# Equivalent explicit calculation
6 * 3


# ### Common Financial Calculations

# #### Calculating Future Value with Growth Rate
#
# If price grows by 5%, the new price is:
#
# $$P_{new} = P_{old} \times (1 + r)$$
#
# where $r$ is the growth rate.

# In[ ]:


# Calculate price after 5% increase
price * 1.05


# #### Applying a Discount
#
# $$P_{discounted} = P_{original} \times (1 - d)$$
#
# where $d$ is the discount rate.

# In[ ]:


# Calculate price after 10% discount
price * (1 - 0.1)


# #### Simple Return Calculation
#
# The simple return between two prices is:
#
# $$R = \frac{P_{t} - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

# In[ ]:


# Calculate simple return
price_t = 105.0
price_t_minus_1 = 100.0

simple_return = (price_t - price_t_minus_1) / price_t_minus_1
simple_return


# #### Log Return Calculation
#
# Log returns are preferred in quantitative finance because they are:
# - **Time-additive**: $r_{t_1 \to t_3} = r_{t_1 \to t_2} + r_{t_2 \to t_3}$
# - **Symmetric**: A +10% gain and -10% loss don't net to zero with simple returns
#
# $$r_{log} = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

# In[ ]:


import math

log_return = math.log(price_t / price_t_minus_1)
log_return


# ---
#
# ## 5. Strings: Working with Symbols & Identifiers <a name="strings"></a>

# ### What are Strings?
#
# **Strings** are sequences of characters, used for:
#
# - Ticker symbols (AAPL, BTCUSDT, ES=F)
# - Order IDs
# - Exchange identifiers
# - Log messages

# In[ ]:


# Define a ticker symbol
symbol = 'GOOG'


# In[ ]:


symbol


# In[ ]:


type(symbol)


# ### String Concatenation
#
# Use `+` to combine strings. This is useful for building column names or identifiers.

# In[ ]:


# Build a column name for a price series
col = symbol + "_price"


# In[ ]:


col


# In[ ]:


# Alternative: prefix notation
col = "price_" + symbol


# In[ ]:


col


# ### String Methods
#
# Strings have built-in methods for manipulation.

# #### The `.replace()` Method
#
# Replace occurrences of a substring with another string.

# In[ ]:


# Replace underscores with hyphens
col.replace("_", "-")


# > **Important**: String methods return a NEW string. The original is unchanged.

# In[ ]:


# The original string is unmodified
col


# In[ ]:


# To persist the change, reassign the variable
col = col.replace("_", "-")


# In[ ]:


col


# #### The `.split()` Method
#
# Split a string into a list of substrings based on a delimiter.

# In[ ]:


# Split by hyphen
col.split('-')


# In[ ]:


# Store the result in a variable
tokens = col.split('-')


# In[ ]:


tokens


# #### Indexing Lists
#
# Access individual elements using square bracket notation. **Indices start at 0.**

# In[ ]:


# First element (index 0)
tokens[0]


# In[ ]:


# Second element (index 1)
tokens[1]


# #### The `.find()` Method and String Slicing
#
# Find the position of a substring and extract portions of the string.

# In[ ]:


# Find the position of the hyphen
i = col.find('-')


# In[ ]:


i


# In[ ]:


# Extract everything before the hyphen (string slicing)
# Syntax: string[start:end] - end is exclusive
col[0:i]


# In[ ]:


# Extract everything after the hyphen
col[i+1:]


# ### String Formatting
#
# Combining strings with numbers requires explicit conversion or f-strings.

# In[ ]:


price


# In[ ]:


# This would cause a TypeError:
# symbol + " price is " + price  # Cannot concatenate str and float


# In[ ]:


# Option 1: Explicit conversion with str()
symbol + " price is " + str(price)


# In[ ]:


# Option 2: f-strings (recommended - cleaner and more flexible)
f"{symbol} price is {price}"


# #### Advanced f-string Formatting
#
# f-strings support format specifiers for controlling output.

# In[ ]:


# Format with 2 decimal places
f"{symbol} price is ${price:.2f}"


# In[ ]:


# Format large numbers with thousands separator
volume = 1500000
f"{symbol} volume: {volume:,}"


# In[ ]:


# Format percentages
pct_change = 0.0523
f"{symbol} change: {pct_change:.2%}"


# ---
#
# ## 6. Practical Exercises <a name="exercises"></a>

# ### Exercise 1: Calculate Price Delta
#
# **Scenario**: You're tracking daily price movements. Calculate the absolute price change between yesterday and today.
#
# $$\Delta P = P_{today} - P_{yesterday}$$

# In[ ]:


price_today = 100.0
price_yesterday = 90.0

# TODO: Calculate the price delta
price_delta = 0.0  # Replace this with your calculation


# In[ ]:


# YOUR CODE HERE



# In[ ]:


# Validation (should return True)
price_delta == 10


# ### Exercise 2: Calculate Total Profit & Loss (PnL)
#
# **Scenario**: You've executed 4 trades today with individual PnL values. Calculate your total PnL.
#
# $$\text{Total PnL} = \sum_{i=1}^{n} \text{PnL}_i$$

# In[ ]:


trade1_pnl = 1.2    # Winning trade
trade2_pnl = -2.0   # Losing trade
trade3_pnl = 3.0    # Winning trade
trade4_pnl = 8.5    # Winning trade

# TODO: Calculate total PnL
total_pnl = 0.0  # Replace this with your calculation


# In[ ]:


# YOUR CODE HERE



# In[ ]:


# Validation (should return True)
total_pnl == 10.7


# ### Exercise 3: Parse Market Data
#
# **Scenario**: You receive market data as a string in the format `"SYMBOL:PRICE"`. Parse this string to extract the symbol and price as separate variables.

# In[ ]:


s = 'AAPL:262.82'

# TODO: Extract symbol and price from the string
symbol = ''   # Should be 'AAPL'
price = 0.0   # Should be 262.82


# In[ ]:


# YOUR CODE HERE



# In[ ]:


# Validation (should return True)
symbol == 'AAPL'


# In[ ]:


# Validation (should return True)
price == 262.82


# ### Exercise 4 (Challenge): Calculate Position Value
#
# **Scenario**: You hold a position in TSLA. Calculate the total position value and the unrealized PnL.
#
# $$\text{Position Value} = \text{Quantity} \times \text{Current Price}$$
#
# $$\text{Unrealized PnL} = \text{Quantity} \times (\text{Current Price} - \text{Entry Price})$$

# In[ ]:


ticker = "TSLA"
quantity = 150           # Number of shares held
entry_price = 180.50     # Price at which position was opened
current_price = 195.75   # Current market price

# TODO: Calculate position value and unrealized PnL
position_value = 0.0     # Replace with your calculation
unrealized_pnl = 0.0     # Replace with your calculation


# In[ ]:


# YOUR CODE HERE



# In[ ]:


# Display results using f-strings with proper formatting
f"{ticker} | Position Value: ${position_value:,.2f} | Unrealized PnL: ${unrealized_pnl:,.2f}"


# ---
#
# ## 7. Key Takeaways <a name="takeaways"></a>

# ### Summary
#
# | Concept | Key Points |
# |---------|------------|
# | **Variables** | Named containers for data; use descriptive names |
# | **Integers** | Whole numbers; use for counts and IDs |
# | **Floats** | Decimal numbers; use for prices and returns |
# | **Type Conversion** | `int()`, `float()`, `str()` for explicit casting |
# | **Arithmetic** | Follows PEMDAS; use parentheses for clarity |
# | **Strings** | Immutable; methods return new strings |
# | **f-strings** | Preferred method for string formatting |

# ### Best Practices for Quant Code
#
# 1. **Use descriptive variable names**: `entry_price` not `p1`
# 2. **Be explicit about types**: Convert strings to floats when parsing data
# 3. **Use f-strings for formatting**: Cleaner and more maintainable
# 4. **Comment your financial formulas**: Future you will thank present you
# 5. **Test edge cases**: What happens when price is 0? Negative?

# ### What's Next?
#
# In **Module 02**, we'll cover:
#
# - **Lists and Collections**: Storing multiple prices, time series data
# - **Loops**: Iterating over trade data
# - **Conditionals**: Implementing trading logic (if signal > threshold, buy)

# ---
#
# **End of Module 01**
#
# *Questions? Contact the Quant Research team.*
