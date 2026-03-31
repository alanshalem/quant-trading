#!/usr/bin/env python
# coding: utf-8

# # Módulo 03: Vectorización
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [Introducción a la Vectorización](#introduction-to-vectorization)
# 3. [Construyendo una Clase Vector](#building-a-vector-class)
# 4. [Operaciones Vector-Escalar](#vector-scalar-operations)
# 5. [Operaciones Vector-Vector](#vector-vector-operations)
# 6. [Estadísticas Vectorizadas](#vectorized-statistics)
# 7. [Construyendo una Librería DataFrame](#building-a-dataframe-library)
# 8. [Matrices e Ingeniería de Features](#matrices-and-feature-engineering)
# 9. [Ejercicios Prácticos](#practical-exercises)
# 10. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Entender la vectorización y por qué es esencial para finanzas cuantitativas
# - Construir clases Vector y DataFrame personalizadas para análisis de datos financieros
# - Realizar operaciones aritméticas y estadísticas vectorizadas
# - Calcular el Sharpe Ratio usando operaciones vectorizadas
# - Crear features y targets para modelos de machine learning
# - Entender los beneficios de rendimiento de las operaciones SIMD

# ---
#
# ## Introducción a la Vectorización
#
# **Vectorización** es el proceso de convertir operaciones que trabajan sobre valores individuales (escalares)
# en operaciones que trabajan sobre arrays completos simultáneamente.
#
# ### Por Qué Importa la Vectorización en Trading Cuantitativo
#
# 1. **Velocidad**: Las operaciones vectorizadas son 10-100x más rápidas que los bucles de Python
# 2. **Claridad de Código**: El código vectorizado es más legible y mantenible
# 3. **Eficiencia de Memoria**: Mejor utilización de caché con arrays contiguos
# 4. **SIMD**: Las CPUs modernas pueden procesar múltiples valores en una sola instrucción
#
# **SIMD** (Single Instruction, Multiple Data) permite a la CPU realizar la misma
# operación en múltiples puntos de datos simultáneamente.

import numpy as np

# ---
#
# ## Construyendo una Clase Vector
#
# Construyamos una clase `Vector` que envuelve arrays de NumPy y proporciona
# funcionalidad de computación financiera.

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

# ## Operaciones Vector-Escalar
#
# Cuando realizamos aritmética entre un vector y un escalar, la operación
# se aplica a cada elemento. Esto se llama **broadcasting**.

vec = Vector([1, 2, 3])
vec

# ### Adición

# Sumar 1 a cada elemento
vec + 1

# Cálculo manual equivalente
[1 + 1, 2 + 1, 3 + 1]

# ### Sustracción

vec - 2

[1 - 2, 2 - 2, 3 - 2]

# ### Multiplicación y División

vec * 2

vec / 2

# ### La Forma Lenta: Bucles de Python
#
# Sin vectorización, necesitaríamos bucles explícitos:

# Esto es lento y verboso - NO hagas esto en producción
v = []
for e in [1, 2, 3]:
    v.append(e + 1)
v

# ---
#
# ## El Poder de la Vectorización: Comparación de Rendimiento
#
# Veamos la diferencia dramática de rendimiento entre bucles y operaciones vectorizadas.

import time

n = 100_000_000
v1 = [1 for _ in range(n)]  # Lista de Python
v2 = Vector(v1)              # Vectorizado

# Bucle de Python - LENTO
start = time.time()
y = []
for x in v1:
    y.append(x + 1)
elapsed_loop = time.time() - start
print(f"Python loop: {elapsed_loop*1000:.1f} ms")
print(f"Last 10 elements: {y[-10:]}")

# Operación vectorizada - RÁPIDA
start = time.time()
y = v2 + 1
elapsed_vec = time.time() - start
print(f"Vectorized: {elapsed_vec*1000:.1f} ms")
print(f"Last 10 elements: {y.data[-10:]}")

# Factor de aceleración
print(f"Speedup: {elapsed_loop / elapsed_vec:.1f}x faster")

# **Punto Clave**: Las operaciones vectorizadas aprovechan las instrucciones SIMD (Single Instruction, Multiple Data)
# en CPUs modernas, procesando múltiples valores en paralelo.
#
# ---
#
# ## Operaciones Vector-Vector
#
# Al realizar operaciones entre dos vectores de la misma longitud,
# la operación se aplica elemento a elemento.

x = Vector([1, 2, 3, 4])
y = Vector([1, -1, 2, -2])

# Adición elemento a elemento
x + y

# Verificar manualmente
[1 + 1, 2 + (-1), 3 + 2, 4 + (-2)]

# Sustracción elemento a elemento
x - y

# Multiplicación elemento a elemento (producto de Hadamard)
x * y

# ---
#
# ## Estadísticas Vectorizadas
#
# Los cálculos estadísticos son fundamentales en finanzas cuantitativas.
# Veamos cómo calcularlos eficientemente con operaciones vectorizadas.
#
# ### Cálculo de Varianza
#
# La **varianza** mide la dispersión de los retornos alrededor de la media:
#
# $$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

log_returns = Vector([0.01, 0.015, 0.02, -0.01])
mu = log_returns.mean()
print(f"Mean return: {mu:.4f}")

# Paso 1: Restar la media de cada retorno (desviaciones)
deviations = log_returns - mu
deviations

# Paso 2: Elevar al cuadrado cada desviación
squared_deviations = (log_returns - mu) ** 2
squared_deviations

# Paso 3: Tomar la media de las desviaciones al cuadrado = varianza
variance = ((log_returns - mu) ** 2).mean()
print(f"Variance: {variance:.8f}")

# Verificar con nuestro método de Vector
log_returns.var()

# ### Desviación Estándar
#
# La **desviación estándar** es la raíz cuadrada de la varianza:
#
# $$\sigma = \sqrt{\sigma^2}$$

# Cálculo manual
np.sqrt(log_returns.var())

# Usando nuestro método de Vector
log_returns.std()

# ---
#
# ## Sharpe Ratio Vectorizado
#
# El **Sharpe Ratio** es la métrica de rendimiento ajustado por riesgo más importante en finanzas:
#
# $$\text{Sharpe Ratio} = \frac{\mathbb{E}[R] - R_f}{\sigma_R} \approx \frac{\bar{r}}{\sigma_r}$$
#
# Donde:
# - $\mathbb{E}[R]$ = Retorno esperado
# - $R_f$ = Tasa libre de riesgo (frecuentemente asumida como 0 por simplicidad)
# - $\sigma_R$ = Desviación estándar de los retornos
#
# El Sharpe Ratio nos dice **cuánto retorno obtenemos por unidad de riesgo**.

# Portafolio A: Ganancias pequeñas consistentes
portfolio_a = Vector([0.01, 0.01, 0.02, -0.01])

print(f"Total return: {portfolio_a.sum():.4f}")
print(f"Mean return: {portfolio_a.mean():.4f}")
print(f"Std deviation: {portfolio_a.std():.4f}")
print(f"Sharpe Ratio: {portfolio_a.mean() / portfolio_a.std():.4f}")

# Portafolio B: Grandes oscilaciones, mismo total
portfolio_b = Vector([-0.01, -0.01, -0.01, 0.06])

print(f"Total return: {portfolio_b.sum():.4f}")
print(f"Mean return: {portfolio_b.mean():.4f}")
print(f"Std deviation: {portfolio_b.std():.4f}")
print(f"Sharpe Ratio: {portfolio_b.mean() / portfolio_b.std():.4f}")

# **Punto Clave**: Ambos portafolios tienen el mismo retorno total (0.03), pero el Portafolio A
# tiene un Sharpe Ratio más alto porque logra este retorno con menor volatilidad.
#
# ---
#
# ## Construyendo una Librería DataFrame
#
# Ahora construyamos una clase DataFrame simple para datos financieros tabulares.
#
# ### La Clase Column

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

# Crear una columna de PnL de operaciones
col = Column('trade_pnl', [2.0, -1.0, 3.0, 1.5])
col

# ### La Clase DataFrame

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

        # Determinar anchos de columna
        col_widths = []
        preview_rows = min(len(self), 10)
        for col in self.cols:
            data_preview = [str(x) for x in col.vec[:preview_rows]]
            max_data_width = max(len(x) for x in data_preview) if data_preview else 0
            width = max(len(col.name), max_data_width)
            col_widths.append(width)

        # Formatear encabezado
        header = " | ".join(
            name.ljust(width) for name, width in zip(col_names, col_widths)
        )
        separator = "-+-".join("-" * width for width in col_widths)

        # Formatear filas
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
# ## Trabajando con Datos de Series Temporales
#
# Creemos una serie temporal de precios simple y calculemos log returns.

from datetime import datetime, timedelta

# Crear datos de ejemplo
time = Column('time', [datetime(2025, 10, 1) + timedelta(days=i+1) for i in range(7)])
price = Column('price', [10.0, 11.0, 12.0, 10.0, 13.0, 14.0, 15.0])

table = DataFrame([time, price])
table

# ### Creando Features con Lag
#
# La operación `shift()` crea versiones con lag de una columna, esencial para
# análisis de series temporales:

# Crear lag de precio (precio de ayer)
price_lag_1 = price.shift()
price_lag_1

table.append(Column('price_lag_1', price_lag_1))
table

# ### Calculando Ratios de Precio

# Ratio de precio: P_t / P_{t-1}
ratio = price / price_lag_1
table.append(Column('ratio', ratio))
table

# ### Calculando Log Returns
#
# Los log returns se calculan como:
#
# $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$

# Log returns desde el ratio de precio
ratio_col = Column('ratio', ratio)
log_return = ratio_col.log()
log_return

log_return_col = Column('log_return', log_return)
table.append(log_return_col)
table

# ### Creando Features Autorregresivas
#
# Para modelos AR(1), necesitamos log returns con lag:

log_return_lag_1_col = Column('log_return_lag_1', log_return_col.shift())
table.append(log_return_lag_1_col)
table

# ---
#
# ## Matrices e Ingeniería de Features
#
# ### Orden Column-Major vs Row-Major
#
# Entender la disposición de matrices es importante para la ingeniería de features en ML.
#
# **Column-Major** (cada fila es un vector de features):

x = [1, 2, 3, 4]
y = [1, 1, 1, 1]
matrix_col = np.array([x, y])
matrix_col

# Accediendo a filas
print(f"Row 0: {matrix_col[0]}")
print(f"Row 1: {matrix_col[1]}")

# **Row-Major** (cada fila es una observación):

# Formato row-major (más común en ML)
matrix_row = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
matrix_row

# Cada fila es una observación
print(f"Observation 0: {matrix_row[0]}")
print(f"Observation 1: {matrix_row[1]}")

# ### Creando Features (X) y Target (y)
#
# Para machine learning, necesitamos separar features del target:

# Features: log returns con lag (entrada a nuestro modelo)
X = table[['log_return_lag_1']]
X

# Target: log return actual (lo que queremos predecir)
y = table['log_return']
y

# ---
#
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Crear Log Returns
#
# Dada una serie de precios, crea una columna de log return usando operaciones vectorizadas.

from datetime import datetime, timedelta

cols = [
    Column('date', [datetime(2025, 10, 1) + timedelta(days=1+i) for i in range(10)]),
    Column('price', [10.0, 8.0, 11.0, 7.0, 9.0, 12.0, 8.0, 9.0, 7.0, 10.0])
]
df = DataFrame(cols)
df

# TODO: Agrega la columna de log return al DataFrame como operación vectorizada de una línea
# Pista: log_return = ln(price / price_lag_1)

# Verificación: comprueba que la última columna (log return) se calculó correctamente
expected = [np.nan, np.log(8.0/10.0), np.log(11.0/8.0), np.log(7.0/11.0),
     np.log(9.0/7.0), np.log(12.0/9.0), np.log(8.0/12.0),
     np.log(9.0/8.0), np.log(7.0/9.0), np.log(10.0/7.0)]
# np.allclose(df.cols[-1].vec.data, expected, equal_nan=True)

# ### Ejercicio 2: Agregar Log Return con Lag
#
# Agrega una columna de log return con lag para modelado AR(1).

# TODO: Agrega la columna log_return_lag_1

# Verificación
expected_lag = [np.nan, np.nan, np.log(8.0/10.0), np.log(11.0/8.0), np.log(7.0/11.0),
     np.log(9.0/7.0), np.log(12.0/9.0), np.log(8.0/12.0),
     np.log(9.0/8.0), np.log(7.0/9.0)]
# np.allclose(df.cols[-1].vec.data, expected_lag, equal_nan=True)

# ### Ejercicio 3: Implementar una Función de Sharpe Ratio
#
# Crea una función que calcule el Sharpe Ratio a partir de un Vector de retornos.

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
    # TODO: Implementa esta función
    pass

# Prueba tu función
test_returns = Vector([0.01, 0.02, -0.01, 0.015, 0.005])
# sharpe_ratio(test_returns)

# ---
#
# ## Puntos Clave
#
# 1. **La vectorización es esencial**: Las operaciones sobre arrays completos son 10-100x más rápidas que los bucles de Python
#
# 2. **Operaciones SIMD**: Las CPUs modernas procesan múltiples valores en una sola instrucción
#
# 3. **La clase Vector** proporciona:
#    - Aritmética elemento a elemento (+, -, *, /)
#    - Métodos estadísticos (mean, var, std)
#    - Sobrecarga de operadores para código limpio
#
# 4. **Las clases Column/DataFrame** permiten:
#    - Features con lag mediante `shift()`
#    - Cálculos de log return
#    - Construcción de matrices de features para ML
#
# 5. **Fórmulas clave**:
#    - Media: $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$
#    - Varianza: $\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$
#    - Sharpe Ratio: $SR = \frac{\bar{r}}{\sigma_r}$
#
# ---
#
# **Siguiente Módulo**: Análisis de Series Temporales - Estadística, estacionariedad y autoregresión
