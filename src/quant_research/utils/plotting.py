"""
plotting.py - Visualization Utilities
======================================
Functions for creating charts and plots using Altair and Matplotlib.
"""

from typing import List, Optional
import numpy as np
import polars as pl
import altair
import matplotlib.pyplot as plt


def plot(df: pl.DataFrame, col: str, title: str = "") -> altair.Chart:
    """
    Create a smooth density plot for analyzing feature distributions.

    Useful for:
    - Understanding data distributions before modeling
    - Detecting outliers and skewness
    - Comparing feature distributions across different time periods
    - Validating data preprocessing steps

    Args:
        df: DataFrame containing the column to plot
        col: Name of the column to visualize
        title: Optional chart title (defaults to None)

    Returns:
        Altair Chart object (displays automatically in Jupyter)

    Example:
        >>> # Plot distribution of returns
        >>> plot(df, 'returns', title='Return Distribution')
        >>>
        >>> # Plot price changes
        >>> plot(df, 'price_change', title='Price Change Distribution')

    Note:
        The density estimation uses kernel density estimation (KDE)
        with basis interpolation for smooth curves.
    """
    return altair.Chart(df).mark_area(
        opacity=0.7,             # Semi-transparent fill
        interpolate='basis'      # Smooth curve interpolation
    ).transform_density(
        col,                     # Column to compute density for
        as_=[col, 'density']     # Output column names
    ).encode(
        x=altair.X(f'{col}:Q', title=col),           # X-axis: feature values
        y=altair.Y('density:Q', title='Density')     # Y-axis: probability density
    ).properties(
        width=600,
        height=400,
        title=title if title else f'Distribution of {col}'
    )


def plot_distribution(data: pl.DataFrame, col: str, label: Optional[str] = None, n_bins: int = 100) -> altair.Chart:
    """
    Create an interactive histogram to visualize the distribution of a column.

    This function generates a binned histogram with interactive zoom capabilities,
    useful for analyzing the distribution of returns, prices, or other metrics.

    Args:
        data: DataFrame containing the column to plot
        col: Name of the column to visualize
        label: Optional custom label for the plot title. If None, uses column name
        n_bins: Maximum number of bins for the histogram (default: 100)

    Returns:
        Interactive Altair Chart object with zoom/pan capabilities

    Example:
        >>> # Plot return distribution
        >>> plot_distribution(df, 'log_return', label='Daily Returns', n_bins=50)
        >>>
        >>> # Plot price distribution
        >>> plot_distribution(df, 'close', n_bins=100)

    Note:
        The chart includes interactive selection intervals for zooming.
        Use mouse drag to zoom into specific regions of the distribution.
    """
    return altair.Chart(data).mark_bar().encode(
        altair.X(f'{col}:Q', bin=altair.Bin(maxbins=n_bins)),
        y='count()'
    ).properties(
        width=600,
        height=400,
        title=f'Distribution of {label if label else col}'
    ).configure_scale(zero=False).add_params(
        altair.selection_interval(bind='scales')
    )


def plot_static_timeseries(ts: pl.DataFrame, sym: str, col: str, interval_size: str):
    """
    Create a static matplotlib line plot of time series data.

    This function generates a traditional line chart for visualizing how a metric
    evolves over time. Useful for creating publication-ready static charts.

    Args:
        ts: DataFrame with 'datetime' column and the column to plot
        sym: Symbol name (e.g., 'BTCUSDT') for the plot title
        col: Name of the column to plot on Y-axis
        interval_size: Time interval label (e.g., '1h', '12h') for the title

    Returns:
        None (displays the plot)

    Example:
        >>> # Plot closing prices
        >>> plot_static_timeseries(df, 'BTCUSDT', 'close', '1h')
        >>>
        >>> # Plot volume over time
        >>> plot_static_timeseries(df, 'ETHUSDT', 'volume', '4h')

    Note:
        This creates a static matplotlib figure. For interactive plots,
        use plot_dyn_timeseries() instead.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(ts['datetime'], ts[col], label=col)
    plt.title(f'{sym} {interval_size} Bars')
    plt.xlabel('time')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_multiple_lines(
    df: pl.DataFrame,
    cols_to_plot: List[str],
    sym: str,
    width: int = 15,
    height: int = 6,
    xlabel_unit: str = "Time Step"
):
    """
    Plots multiple columns from a Polars DataFrame on the same axes using Matplotlib.
    The x-axis uses a simple numerical index (since no datetime column is present).

    Parameters:
    -----------
    df : polars.DataFrame
        The Polars DataFrame containing the columns to plot.
    cols_to_plot : list[str]
        A list of column names to plot (e.g., ['log_return', 'mean']).
    sym : str
        A symbol or identifier for the series (used in the title).
    width : int, default 15
        Width of the plot in inches.
    height : int, default 6
        Height of the plot in inches.
    xlabel_unit : str, default 'Time Step'
        Label for the X-axis (the numerical index).
    """
    # 1. Create the numerical index for the x-axis
    x_index = np.arange(len(df))

    # 2. Set the figure size (controls the width/height)
    plt.figure(figsize=(width, height))

    # 3. Loop through the list of columns and plot each one
    for col in cols_to_plot:
        if col in df.columns:
            # Extract column data as a NumPy array (efficient)
            y_values = df[col].to_numpy()

            # Plot the line, using the column name for the label
            plt.plot(x_index, y_values, label=col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # 4. Finalize the plot

    # Dynamically generate the title based on the symbol and columns
    title_cols = ', '.join(cols_to_plot)
    plt.title(f'{sym} Series: {title_cols}')

    plt.xlabel(xlabel_unit)
    plt.ylabel('Value')  # Generic Y-label since multiple series are plotted
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    plt.show()


def plot_dyn_timeseries(ts: pl.DataFrame, sym: str, col: str, time_interval: str) -> altair.Chart:
    """
    Create an interactive time series line plot with zoom and tooltip capabilities.

    This function generates a dynamic Altair chart that allows users to explore
    time series data interactively with mouse-driven zoom on both axes.

    Args:
        ts: DataFrame with 'datetime' column and the column to plot
        sym: Symbol name (e.g., 'BTCUSDT') for the plot title
        col: Name of the column to plot on Y-axis
        time_interval: Time interval label (e.g., '1h', '12h') for the title

    Returns:
        Interactive Altair Chart with zoom and tooltip features

    Example:
        >>> # Plot interactive price chart
        >>> chart = plot_dyn_timeseries(df, 'BTCUSDT', 'close', '1h')
        >>> chart  # Display in Jupyter
        >>>
        >>> # Plot returns with zoom
        >>> plot_dyn_timeseries(df, 'BTCUSDT', 'log_return', '12h')

    Note:
        - Drag on the chart to zoom into specific regions
        - Hover over data points to see exact values
        - Independent zoom controls for X and Y axes
    """
    return altair.Chart(ts).mark_line(tooltip=True).encode(
        x="datetime",
        y=col
    ).properties(
        width=800,
        height=400,
        title=f"{sym} {time_interval} {col}"
    ).configure_scale(zero=False).add_selection(
        altair.selection_interval(bind='scales', encodings=['x']),  # Only zoom x-axis
        altair.selection_interval(bind='scales', encodings=['y'])   # Only zoom y-axis
    )


def plot_column(df, col_name, figsize=(15, 6), title=None, xlabel='Index'):
    """
    Plot a column from a Polars DataFrame using matplotlib.

    Parameters:
    -----------
    df : polars.DataFrame
        The Polars DataFrame
    column_name : str
        Name of the column to plot
    figsize : tuple, default (15, 6)
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None, uses column name
    xlabel : str, default 'Index'
        X-axis label
    ylabel : str, optional
        Y-axis label. If None, uses column name
    """

    if title is None:
        title = col_name

    chart = df[col_name].plot.line()
    return chart.properties(
        width=800,
        height=400,
        title=title
    )
