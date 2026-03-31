#!/usr/bin/env python
# coding: utf-8

# # Módulo 05: Statistical Edge
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [Fundamentos de Álgebra Matricial](#matrix-algebra-fundamentals)
# 3. [Multiplicación Matriz-Vector](#matrix-vector-multiplication)
# 4. [Construyendo un Statistical Edge](#building-a-statistical-edge)
# 5. [Entrenando un Modelo Lineal](#training-a-linear-model)
# 6. [Evaluando la Predictibilidad](#evaluating-predictability)
# 7. [Midiendo el Statistical Edge](#measuring-statistical-edge)
# 8. [Ejercicios Prácticos](#practical-exercises)
# 9. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Realizar operaciones esenciales de álgebra matricial
# - Entender la relación: **modelo = statistical edge**, **estrategia = ejecución**
# - Entrenar un modelo de regresión lineal usando PyTorch
# - Evaluar la predictibilidad del modelo usando precisión direccional
# - Medir el statistical edge a través de retornos esperados por operación
# - Calcular e interpretar el Sharpe Ratio

# Librerías principales
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning con PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# ---
#
# ## Fundamentos de Álgebra Matricial
#
# Las matrices son arrays 2D de números. Entender las operaciones matriciales es esencial
# para machine learning y finanzas cuantitativas.
#
# ### ¿Qué es una Matriz?
#
# Una matriz es un arreglo rectangular de números organizados en filas y columnas:
#
# $$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

# Crear una matriz 3x3
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Acceder a la fila 0 (primera fila)
matrix[0]

# Acceder al elemento en fila 0, columna 0
matrix[0][0]

# Acceder al elemento en fila 1, columna 2
matrix[1][2]

# ### Operaciones Matriz-Escalar
#
# Cuando sumamos un escalar a una matriz, se suma a cada elemento:

# Usando bucles anidados (forma lenta)
no_rows = len(matrix)
no_cols = len(matrix[0])

matrix_copy = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for i in range(no_rows):
    for j in range(no_cols):
        matrix_copy[i][j] += 1

matrix_copy

# Usando NumPy (forma rápida, vectorizada)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A + 1

# Las operaciones escalares son conmutativas
1 + A

# Multiplicación escalar
A * 2

# ---
#
# ## Multiplicación Matriz-Vector
#
# ¡Esta es la operación central en machine learning! Mapea features a predicciones.
#
# ### El Modelo Lineal
#
# $$\hat{y} = X \cdot w + b$$
#
# Donde:
# - $X$ = Matriz de features (n muestras x m features)
# - $w$ = Vector de pesos (m features)
# - $b$ = Escalar de sesgo
# - $\hat{y}$ = Predicciones (n muestras)

# Matriz de features: 3 muestras, 2 features cada una
X = np.array([
    [-0.1, -0.2],   # Muestra 1: [lag_1, lag_2]
    [-0.2, -0.4],   # Muestra 2
    [-0.4, -0.8]    # Muestra 3
])
X

# Vector de pesos: un peso por feature
w = np.array([-0.5, -0.1])
w

# Multiplicación matriz-vector: X @ w
y_hat = np.dot(X, w)
y_hat

# ### Entendiendo el Cálculo
#
# Para cada muestra, calculamos el producto punto de features con pesos:
#
# $$\hat{y}_1 = x_{11} \cdot w_1 + x_{12} \cdot w_2 = (-0.1)(-0.5) + (-0.2)(-0.1) = 0.07$$

# Verificación manual
w1, w2 = w[0], w[1]
np.array([
    -0.1 * w1 + -0.2 * w2,  # Muestra 1
    -0.2 * w1 + -0.4 * w2,  # Muestra 2
    -0.4 * w1 + -0.8 * w2   # Muestra 3
])

# ### Agregando Sesgo

bias = 0.0001
y_hat_with_bias = np.dot(X, w) + bias
y_hat_with_bias

# ### Broadcasting vs Multiplicación Matricial
#
# **Distinción importante**:
# - `X * w` = Multiplicación elemento a elemento (broadcasting)
# - `X @ w` o `np.dot(X, w)` = Multiplicación matricial

# Broadcasting (elemento a elemento) - NO es lo que queremos para modelos lineales
X * w

# Multiplicación matricial - esto es lo que queremos
np.dot(X, w)

# ---
#
# ## Construyendo un Statistical Edge
#
# ### La Idea Clave
#
# - **Modelo** = Statistical edge (capacidad de predecir)
# - **Estrategia** = Ejecución del statistical edge
#
# Un buen modelo encuentra patrones; una buena estrategia los explota de forma rentable.
#
# ### Statistical Edge = Buen Pronóstico
#
# Si podemos predecir la dirección del movimiento de precio mejor que el azar,
# tenemos un statistical edge.
#
# ### Cargar Datos OHLC

url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

print(f"Data shape: {btcusdt.shape}")
btcusdt.head()

# ### Ingeniería de Features: Log Returns y Lags

# Calcular log returns
btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())

# Crear features con lag
btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)

# Eliminar filas NaN
btcusdt = btcusdt.dropna()
btcusdt[['close_log_return', 'close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']].head()

# ### Verificar Correlación Serial

btcusdt[['close_log_return', 'close_log_return_lag_1',
         'close_log_return_lag_2', 'close_log_return_lag_3']].corr()

# ### Visualizar Relaciones entre Features

sns.pairplot(btcusdt[['close_log_return', 'close_log_return_lag_1',
                      'close_log_return_lag_2', 'close_log_return_lag_3']],
             diag_kind='kde')
plt.suptitle('Feature Relationships', y=1.02)
plt.show()

# ### Preparar Features y Target

# Matriz de features X
X = btcusdt[['close_log_return_lag_1', 'close_log_return_lag_2',
             'close_log_return_lag_3']].values
print(f"X shape: {X.shape}")

# Vector target y
y = btcusdt['close_log_return'].values
print(f"y shape: {y.shape}")

# ---
#
# ## Entrenando un Modelo Lineal
#
# ### Split Train/Test para Series Temporales
#
# **Crítico**: Para series temporales, ¡debemos dividir cronológicamente para evitar look-ahead bias!
#
# ```
# Tiempo: t0 ---- t1 ---- t2 ---- t3 ---- t4 ---- t5 ---- t6 ---- t7
# Train:  [===============================]
# Test:                                   [=========================]
# ```

def time_split(x, train_size=0.75):
    """Split data chronologically for time series."""
    i = int(len(x) * train_size)
    return x[:i].copy(), x[i:].copy()

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print(f"Train: {len(btcusdt_train)} samples ({btcusdt_train.index.min()} to {btcusdt_train.index.max()})")
print(f"Test: {len(btcusdt_test)} samples ({btcusdt_test.index.min()} to {btcusdt_test.index.max()})")

# ### Entrenamiento del Modelo con PyTorch

import random
import os

# ---
# CONFIGURACIÓN DE REPRODUCIBILIDAD
# ---
SEED = 99
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---
# PREPARAR DATOS
# ---
features = ['close_log_return_lag_3']  # Modelo AR(1) con lag-3
target = 'close_log_return'

# Convertir a tensores de PyTorch
X_train = torch.tensor(btcusdt_train[features].values, dtype=torch.float32)
X_test = torch.tensor(btcusdt_test[features].values, dtype=torch.float32)
y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ---
# DEFINIR MODELO
# ---
no_features = len(features)

# Regresión lineal simple: y = Wx + b
model = nn.Linear(no_features, 1)

# Huber loss (robusta ante outliers)
criterion = nn.HuberLoss()

# Optimizador SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---
# BUCLE DE ENTRENAMIENTO
# ---
for epoch in range(5000):
    # Resetear gradientes
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train)

    # Calcular loss
    loss = criterion(y_pred, y_train)

    # Backward pass
    loss.backward()

    # Actualizar pesos
    optimizer.step()

    # Imprimir progreso
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# Parámetros finales
print(f"\nTrained weight: {model.weight.data}")
print(f"Trained bias: {model.bias.data}")

# Guardar modelo
torch.save(model.state_dict(), "model.pth")

# ---
#
# ## Evaluando la Predictibilidad
#
# ### Generar Predicciones

# Obtener predicciones en el set de test
y_hat = model(X_test)
y_hat_np = y_hat.detach().squeeze().numpy()

btcusdt_test['y_hat'] = y_hat_np
btcusdt_test[['close_log_return', 'y_hat']].head(10)

# ### Agregar Señal Direccional
#
# Convertir predicciones continuas a señales de trading:
# - **+1** = Ir Long (apostar a que el precio sube)
# - **-1** = Ir Short (apostar a que el precio baja)

btcusdt_test['dir_signal'] = np.sign(btcusdt_test['y_hat'])
btcusdt_test[['close_log_return', 'y_hat', 'dir_signal']].head(10)

# ### Precisión Direccional
#
# ¿Con qué frecuencia predecimos la dirección correcta?

btcusdt_test['is_won'] = btcusdt_test['dir_signal'] == np.sign(btcusdt_test[target])
da = btcusdt_test['is_won'].mean()
print(f"Directional Accuracy: {da:.2%}")

# **Interpretación**: Si DA > 50%, tenemos algo de poder predictivo sobre el azar.
#
# ---
#
# ## Midiendo el Statistical Edge
#
# ### Retornos por Operación
#
# Cuando acertamos la dirección, ganamos. Cuando nos equivocamos, perdemos.
#
# $$\text{Trade Return} = \text{Signal} \times \text{Actual Return}$$

btcusdt_test['trade_log_return'] = btcusdt_test['dir_signal'] * btcusdt_test[target]
btcusdt_test[['dir_signal', 'close_log_return', 'is_won', 'trade_log_return']].head(10)

# ### Statistical Edge = Valor Esperado Positivo

expected_trade_return = btcusdt_test['trade_log_return'].mean()
print(f"Expected Trade Return: {expected_trade_return:.6f}")

has_statistical_edge = expected_trade_return > 0
print(f"Has Statistical Edge: {has_statistical_edge}")

# **Punto Clave**: ¡Si E[trade return] > 0, tenemos un statistical edge!
#
# ### Retorno Total

total_log_return = btcusdt_test['trade_log_return'].sum()
print(f"Total Log Return: {total_log_return:.4f}")

# Convertir a retorno simple
total_return = np.exp(total_log_return)
print(f"Total Return: {total_return:.2%}")

# Valor final del portafolio
initial_capital = 100
final_value = np.exp(total_log_return) * initial_capital
print(f"${initial_capital:.2f} -> ${final_value:.2f}")

# ### Curva de Equity

cum_trade_log_returns = btcusdt_test['trade_log_return'].cumsum()

plt.figure(figsize=(15, 6))
cum_trade_log_returns.plot()
plt.title('Cumulative Log Returns')
plt.ylabel('Cumulative Log Return')
plt.xlabel('Time')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()

# Curva de equity bruta
gross_equity_curve = np.exp(cum_trade_log_returns) * initial_capital

plt.figure(figsize=(15, 6))
gross_equity_curve.plot()
plt.title(f'Equity Curve (Starting Capital: ${initial_capital})')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Time')
plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
plt.show()

# ### Sharpe Ratio

# Sharpe Ratio bruto (por período)
sharpe_raw = btcusdt_test['trade_log_return'].mean() / btcusdt_test['trade_log_return'].std()
print(f"Raw Sharpe Ratio: {sharpe_raw:.4f}")

# Sharpe Ratio anualizado
trading_days_per_year = 365
hours_per_day = 24
periods_per_year = trading_days_per_year * hours_per_day

sharpe_annual = sharpe_raw * np.sqrt(periods_per_year)
print(f"Annualized Sharpe Ratio: {sharpe_annual:.2f}")

# **Interpretación**:
# - Sharpe < 0: Perdiendo dinero en promedio
# - Sharpe 0-1: Retornos ajustados por riesgo por debajo del promedio
# - Sharpe 1-2: Buenos retornos ajustados por riesgo
# - Sharpe > 2: Excelentes retornos ajustados por riesgo
#
# ---
#
# ## Consideración de Costos de Transacción
#
# ### Taker vs Maker
#
# - **Taker**: Toma liquidez (órdenes de mercado), paga comisiones más altas
# - **Maker**: Agrega liquidez (órdenes límite), paga comisiones más bajas (¡a veces negativas!)
#
# Los costos de transacción pueden convertir un edge positivo en negativo:
# - Las comisiones taker reducen las ganancias y amplifican las pérdidas
# - Un valor esperado positivo pequeño puede volverse negativo después de comisiones
#
# **Punto Clave**: La viabilidad de una estrategia depende fuertemente de los costos de ejecución.
#
# ---
#
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Implementación de Producto Punto
#
# Implementa la multiplicación matriz-vector manualmente.

X = [
    [-0.1, -0.01],
    [0.2, 0.5]
]
w = [-0.5, -0.25]
y_hat = []

# TODO: Escribe un bucle para calcular y_hat = X @ w

# Verificar
expected = [-0.1 * -0.5 + -0.01 * -0.25, 0.2 * -0.5 + 0.5 * -0.25]
print(f"Expected: {expected}")
# print(f"Your result: {y_hat}")

# ### Ejercicio 2: Transpuesta de Matriz
#
# La transpuesta intercambia filas y columnas: $A^T_{ij} = A_{ji}$

X = [
    [-0.1, -0.01, -0.2],
    [0.2, 0.5, 0.1]
]
X_transpose = []

# TODO: Transponer X de forma (2,3) a (3,2)

# Verificar
expected = [[-0.1, 0.2], [-0.01, 0.5], [-0.2, 0.1]]
# X_transpose == expected

# ### Ejercicio 3: Producto de Hadamard (Elemento a Elemento)
#
# El producto de Hadamard multiplica elementos correspondientes.

y_true = [[0.01, -0.02], [-0.01, -0.03]]
y_hat = [[0.02, -0.03], [0.01, -0.01]]
error = []

# TODO: Calcular error = y_true - y_hat (elemento a elemento)

# Verificar
expected = [[0.01 - 0.02, -0.02 - (-0.03)], [-0.01 - 0.01, -0.03 - (-0.01)]]
# error == expected

# ---
#
# ## Puntos Clave
#
# 1. **La multiplicación matricial** es la operación central en modelos ML:
#    $$\hat{y} = X \cdot w + b$$
#
# 2. **Statistical edge** = retorno esperado positivo por operación
#    - El modelo encuentra patrones, la estrategia los explota
#
# 3. **Precisión direccional** mide con qué frecuencia predecimos correctamente
#    - DA > 50% sugiere poder predictivo
#
# 4. **Sharpe Ratio** mide retornos ajustados por riesgo:
#    $$SR = \frac{\bar{r}}{\sigma_r} \times \sqrt{T}$$
#
# 5. **Costos de transacción** pueden eliminar un statistical edge
#    - Considerar ejecución taker vs maker
#
# 6. **Fórmulas clave**:
#    - Trade return: $r_{trade} = \text{signal} \times r_{actual}$
#    - Valor esperado: $E[r_{trade}]$ > 0 significa statistical edge
#    - Retorno total: $R_{total} = e^{\sum r_t}$
#
# ---
#
# **Siguiente Módulo**: Classification - Predicción binaria y métricas de evaluación de modelos
