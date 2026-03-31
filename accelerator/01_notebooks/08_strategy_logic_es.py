#!/usr/bin/env python
# coding: utf-8

# # Módulo 08: Lógica de Estrategia
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [Visión General del Framework de Estrategia](#strategy-framework-overview)
# 3. [Preparación de Datos](#data-preparation)
# 4. [Entrenamiento del Modelo](#model-training)
# 5. [Señales de Entrada y Salida](#entry-and-exit-signals)
# 6. [Dimensionamiento de Posiciones](#trade-sizing)
# 7. [Apalancamiento](#leverage)
# 8. [Análisis de Rendimiento de la Estrategia](#strategy-performance-analysis)
# 9. [Ejercicios Prácticos](#practical-exercises)
# 10. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Entender el pipeline completo de una estrategia de trading
# - Implementar señales de entrada/salida basadas en tiempo y predicados
# - Aplicar dimensionamiento de posiciones estático y dinámico (compounding)
# - Entender la mecánica del apalancamiento y su impacto en los retornos
# - Calcular PnL bruto y curvas de equity
# - Evaluar el rendimiento de la estrategia con parámetros realistas

# ---
#
# ## Visión General del Framework de Estrategia
#
# Una estrategia de trading completa consiste en tres componentes principales:
#
# ```
# signal = model(features)    # Generar predicciones
# orders = strategy(signal)   # Convertir a decisiones de trading
# results = execute(orders)   # Ejecutar y rastrear PnL
# ```
#
# ### Decisiones Estratégicas Clave
#
# 1. **Señales de Entrada/Salida**: Cuándo entrar y salir de posiciones
# 2. **Dimensionamiento de Posiciones**: Cuánto capital asignar por operación
# 3. **Apalancamiento**: Cuánto capital prestado usar
#
# ### Tipos de Estrategia
#
# | Tipo | Descripción | Estructura de Comisiones |
# |------|-------------|--------------------------|
# | **Maker** | Provee liquidez (órdenes límite) | Comisiones más bajas, posibles rebates |
# | **Taker** | Consume liquidez (órdenes de mercado) | Comisiones más altas, ejecución garantizada |

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ---
#
# ## Preparación de Datos
#
# ### Cargando Datos de Mercado
#
# Usaremos datos de futuros perpetuos BTCUSDT para construir y probar nuestra estrategia.

url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

print(f"Data shape: {btcusdt.shape}")
print(f"Date range: {btcusdt.index.min()} to {btcusdt.index.max()}")
btcusdt.head()

# ### Calculando Log Returns
#
# Los log returns son preferidos por sus propiedades matemáticas:
#
# $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
#
# **Ventaja clave**: Los log returns son aditivos en el tiempo:
#
# $$r_{t_1 \to t_n} = \sum_{i=1}^{n} r_{t_i}$$

btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())
btcusdt[['close', 'close_log_return']].head()

# ### Creando Features con Lag
#
# Para nuestro modelo autorregresivo, usamos retornos pasados para predecir la dirección futura:

btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)

# Eliminar filas con valores NaN
btcusdt = btcusdt.dropna()

btcusdt[['close_log_return', 'close_log_return_lag_1',
         'close_log_return_lag_2', 'close_log_return_lag_3']].head()

# ### Target de Clasificación Binaria
#
# Convertimos retornos continuos en un problema de clasificación binaria:
#
# $$y_t = \begin{cases} 1 & \text{si } r_t > 0 \text{ (Long)} \\ 0 & \text{si } r_t \leq 0 \text{ (Short)} \end{cases}$$

btcusdt['close_log_return_dir'] = (btcusdt['close_log_return'] > 0).astype(int)

# Verificar balance de clases
print("Direction distribution:")
print(btcusdt['close_log_return_dir'].value_counts())
print(f"\nUp ratio: {btcusdt['close_log_return_dir'].mean():.2%}")

# ### Split Train/Test
#
# Para datos de series temporales, usamos un split temporal para prevenir look-ahead bias:
#
# ```
# Tiempo: t0 -------- t_split -------- t_end
# Train:  [============]
# Test:                 [================]
# ```

def time_split(df, train_size=0.75):
    """
    Split time series data into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    train_size : float
        Proportion of data for training (0 to 1)

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    split_idx = int(len(df) * train_size)
    return df[:split_idx].copy(), df[split_idx:].copy()


btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print(f"Training set: {len(btcusdt_train):,} samples")
print(f"Test set: {len(btcusdt_test):,} samples")
print(f"\nTrain period: {btcusdt_train.index.min()} to {btcusdt_train.index.max()}")
print(f"Test period: {btcusdt_test.index.min()} to {btcusdt_test.index.max()}")

# Verificar balance de clases en ambos sets
print("Training set direction distribution:")
print(btcusdt_train['close_log_return_dir'].value_counts())
print(f"\nTest set direction distribution:")
print(btcusdt_test['close_log_return_dir'].value_counts())

# ---
#
# ## Entrenamiento del Modelo
#
# ### Configuración de Reproducibilidad
#
# Establecer semillas aleatorias asegura resultados consistentes entre ejecuciones.

import random
import os
from sklearn.preprocessing import StandardScaler

# Configuración de reproducibilidad
SEED = 99
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ### Estandarización de Features
#
# La estandarización mejora la convergencia del gradient descent:
#
# $$x_{scaled} = \frac{x - \mu}{\sigma}$$

features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
target = 'close_log_return_dir'

# Ajustar scaler solo en datos de entrenamiento (prevenir data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(btcusdt_train[features].values)
X_test_scaled = scaler.transform(btcusdt_test[features].values)

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ### Modelo de Logistic Regression
#
# El modelo de logistic regression produce la probabilidad de movimiento alcista:
#
# $$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

class LogisticRegression(nn.Module):
    """
    Binary logistic regression classifier.

    Outputs raw logits (pre-sigmoid values).
    Use BCEWithLogitsLoss for numerical stability.
    """
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Inicializar modelo
n_features = len(features)
model = LogisticRegression(n_features)

# Función de pérdida: Binary Cross-Entropy con Logits
criterion = nn.BCEWithLogitsLoss()

# Optimizador: Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.001)

# ### Bucle de Entrenamiento
#
# Full-batch gradient descent por simplicidad:

n_epochs = 15000
print_every = 3000

print("Training Logistic Regression Model")
print("=" * 40)

for epoch in range(n_epochs):
    # Forward pass
    optimizer.zero_grad()
    y_pred_logits = model(X_train)
    loss = criterion(y_pred_logits, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print("=" * 40)
print("Training complete!")
print(f"\nFinal weights: {model.linear.weight.data.numpy().flatten()}")
print(f"Final bias: {model.linear.bias.data.item():.6f}")

# ### Evaluación del Modelo

from sklearn.metrics import confusion_matrix, classification_report

# Generar predicciones en el set de test
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_proba = torch.sigmoid(y_pred_logits)
    y_pred_binary = (y_pred_proba >= 0.5).float()

# Convertir a numpy
y_test_np = y_test.squeeze().numpy()
y_pred_binary_np = y_pred_binary.squeeze().numpy()
y_pred_proba_np = y_pred_proba.squeeze().numpy()

# Calcular métricas
accuracy = np.mean(y_pred_binary_np == y_test_np)

print("Model Evaluation on Test Set")
print("=" * 40)
print(f"Total samples: {len(X_test):,}")
print(f"Directional accuracy: {accuracy:.4f}")
print(f"\nPrediction distribution:")
print(f"  Long (1):  {int(np.sum(y_pred_binary_np == 1)):,}")
print(f"  Short (0): {int(np.sum(y_pred_binary_np == 0)):,}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_np, y_pred_binary_np)
print(cm)

# ---
#
# ## Señales de Entrada y Salida
#
# ### Agregando Predicciones a los Datos de Test

btcusdt_test['y_pred_binary'] = y_pred_binary_np
btcusdt_test['y_pred_proba'] = y_pred_proba_np

btcusdt_test[['close_log_return', 'y_pred_binary', 'y_pred_proba']].head(10)

# ### Convirtiendo Predicciones a Señales de Trading
#
# Mapear predicciones binarias a señales direccionales:
#
# $$\text{signal}_t = \begin{cases} +1 & \text{si predicción} = 1 \text{ (Long)} \\ -1 & \text{si predicción} = 0 \text{ (Short)} \end{cases}$$

btcusdt_test['dir_signal'] = np.where(btcusdt_test['y_pred_binary'] == 1, 1, -1)

print("Signal distribution:")
print(btcusdt_test['dir_signal'].value_counts())

# ### Tipos de Señales de Entrada/Salida
#
# #### 1. Señales Basadas en Tiempo
#
# Operar en cada paso temporal basándose en la predicción del modelo.
# Este es el enfoque por defecto que hemos estado usando.

# Basado en tiempo: Operar cada período
print("Time-based signals: Trading every period")
print(f"Total trades: {len(btcusdt_test):,}")

# #### 2. Señales Basadas en Predicados
#
# Solo operar cuando la confianza del modelo supera un umbral:
#
# $$\text{operar si } P(y=1|x) \geq \theta_{high} \text{ O } P(y=1|x) \leq \theta_{low}$$

# Solo operar cuando probabilidad >= 55% o <= 45%
theta_high = 0.55
theta_low = 0.45

high_confidence_mask = (btcusdt_test['y_pred_proba'] >= theta_high) | \
                       (btcusdt_test['y_pred_proba'] <= theta_low)

print(f"Predicate-based signals (confidence threshold: {theta_low}-{theta_high})")
print(f"Total high-confidence trades: {high_confidence_mask.sum():,}")
print(f"Filtered out: {(~high_confidence_mask).sum():,} low-confidence signals")

# ### Calculando Retornos por Operación
#
# Retorno por operación = Señal × Retorno de Mercado
#
# $$r_{trade,t} = \text{signal}_t \times r_{market,t}$$

btcusdt_test['trade_log_return'] = btcusdt_test['dir_signal'] * btcusdt_test['close_log_return']

# Comparar basado en tiempo vs basado en predicados
print("Strategy comparison:")
print(f"Time-based cumulative return: {btcusdt_test['trade_log_return'].sum():.4f}")
print(f"Predicate-based cumulative return: {btcusdt_test.loc[high_confidence_mask, 'trade_log_return'].sum():.4f}")

# Visualizar retornos acumulados
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Estrategia basada en tiempo
btcusdt_test['trade_log_return'].cumsum().plot(ax=axes[0])
axes[0].set_title('Time-Based Strategy: Cumulative Log Returns')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Cumulative Log Return')
axes[0].grid(True, alpha=0.3)

# Estrategia basada en predicados
btcusdt_test.loc[high_confidence_mask, 'trade_log_return'].cumsum().plot(ax=axes[1])
axes[1].set_title(f'Predicate-Based Strategy (θ={theta_low}-{theta_high}): Cumulative Log Returns')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cumulative Log Return')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---
#
# ## Dimensionamiento de Posiciones
#
# ### 1. Dimensionamiento Estático
#
# Tamaño de posición fijo para cada operación:
#
# $$\text{Valor de Posición} = \text{Tamaño Fijo}$$

# Tamaño de posición estático
INITIAL_CAPITAL = 50.0

btcusdt_test['pre_trade_value'] = INITIAL_CAPITAL
btcusdt_test['post_trade_value'] = np.exp(btcusdt_test['trade_log_return']) * INITIAL_CAPITAL

btcusdt_test[['pre_trade_value', 'post_trade_value', 'trade_log_return']].head(10)

# ### Calculando PnL Bruto
#
# $$\text{Gross P\&L}_t = \text{Post Trade Value}_t - \text{Pre Trade Value}_t$$

btcusdt_test['trade_gross_pnl'] = btcusdt_test['post_trade_value'] - btcusdt_test['pre_trade_value']

print("Static Sizing Performance Metrics")
print("=" * 40)
print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
print(f"Mean P&L per trade: ${btcusdt_test['trade_gross_pnl'].mean():.4f}")
print(f"Total P&L: ${btcusdt_test['trade_gross_pnl'].sum():.2f}")
print(f"Final equity: ${INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum():.2f}")
print(f"Total return: {(INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum()) / INITIAL_CAPITAL:.2%}")

# Graficar curva de equity (dimensionamiento estático)
btcusdt_test['equity_curve_static'] = INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].cumsum()

btcusdt_test['equity_curve_static'].plot(figsize=(15, 6))
plt.title('Equity Curve - Static Position Sizing')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ### 2. Dimensionamiento Dinámico (Compounding)
#
# El tamaño de posición crece con las ganancias acumuladas:
#
# $$\text{Valor de Posición}_t = \text{Capital Inicial} \times e^{\sum_{i=1}^{t-1} r_{trade,i}}$$
#
# Esto aprovecha la propiedad aditiva de los log returns.

# Log return acumulado para compounding
btcusdt_test['cum_trade_log_return'] = btcusdt_test['trade_log_return'].cumsum()

# Valor post-operación con compounding
btcusdt_test['post_trade_value_compound'] = np.exp(btcusdt_test['cum_trade_log_return']) * INITIAL_CAPITAL

# Valor pre-operación es el valor post-operación del período anterior
btcusdt_test['pre_trade_value_compound'] = btcusdt_test['post_trade_value_compound'].shift().fillna(INITIAL_CAPITAL)

# PnL bruto con compounding
btcusdt_test['trade_gross_pnl_compound'] = btcusdt_test['post_trade_value_compound'] - btcusdt_test['pre_trade_value_compound']

btcusdt_test[['trade_log_return', 'cum_trade_log_return',
              'pre_trade_value_compound', 'post_trade_value_compound']].head(10)

print("Dynamic (Compounding) Sizing Performance Metrics")
print("=" * 40)
print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
print(f"Mean P&L per trade: ${btcusdt_test['trade_gross_pnl_compound'].mean():.4f}")
print(f"Final equity: ${btcusdt_test['post_trade_value_compound'].iloc[-1]:.2f}")
print(f"Total return: {btcusdt_test['post_trade_value_compound'].iloc[-1] / INITIAL_CAPITAL:.2%}")

# Multiplicador compuesto
compound_return = np.exp(btcusdt_test['trade_log_return'].sum())
print(f"\nCompound multiplier: {compound_return:.4f}x")

# Comparar dimensionamiento estático vs dinámico
fig, ax = plt.subplots(figsize=(15, 6))

# Curva de equity con dimensionamiento estático
btcusdt_test['equity_curve_static'].plot(ax=ax, label='Static Sizing')

# Curva de equity con dimensionamiento dinámico
btcusdt_test['post_trade_value_compound'].plot(ax=ax, label='Dynamic (Compounding)')

plt.title('Equity Curves: Static vs Dynamic Position Sizing')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ---
#
# ## Apalancamiento
#
# ### Entendiendo el Apalancamiento
#
# El apalancamiento amplifica tanto ganancias como pérdidas:
#
# $$\text{Retorno Apalancado} = \text{Apalancamiento} \times \text{Retorno Sin Apalancar}$$
#
# $$\text{PnL Apalancado} = \text{Apalancamiento} \times \text{PnL Sin Apalancar}$$

# Ejemplo: Efecto del apalancamiento
equity = 50.0
leverage = 2.0

# Operación positiva
trade_pnl_positive = 10.0
print(f"Positive trade P&L: ${trade_pnl_positive:.2f}")
print(f"With {leverage}x leverage: ${trade_pnl_positive * leverage:.2f}")

# Operación negativa
trade_pnl_negative = -20.0
print(f"\nNegative trade P&L: ${trade_pnl_negative:.2f}")
print(f"With {leverage}x leverage: ${trade_pnl_negative * leverage:.2f}")

# ### Aplicando Apalancamiento a la Estrategia

LEVERAGE = 2.0

# Valores de posición apalancados
btcusdt_test['post_trade_value_leveraged'] = np.exp(btcusdt_test['cum_trade_log_return']) * INITIAL_CAPITAL * LEVERAGE
btcusdt_test['pre_trade_value_leveraged'] = btcusdt_test['post_trade_value_leveraged'].shift().fillna(INITIAL_CAPITAL * LEVERAGE)
btcusdt_test['trade_gross_pnl_leveraged'] = btcusdt_test['post_trade_value_leveraged'] - btcusdt_test['pre_trade_value_leveraged']

print(f"Leveraged Strategy Performance ({LEVERAGE}x)")
print("=" * 40)
print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
print(f"Initial position (leveraged): ${INITIAL_CAPITAL * LEVERAGE:.2f}")
print(f"Final equity (leveraged): ${btcusdt_test['post_trade_value_leveraged'].iloc[-1]:.2f}")
print(f"Total return: {btcusdt_test['post_trade_value_leveraged'].iloc[-1] / INITIAL_CAPITAL:.2%}")

# Comparar sin apalancamiento vs con apalancamiento
fig, ax = plt.subplots(figsize=(15, 6))

# Sin apalancamiento (1x)
(np.exp(btcusdt_test['trade_log_return'].cumsum()) * INITIAL_CAPITAL).plot(ax=ax, label='1x (No Leverage)')

# Con apalancamiento (2x)
(np.exp(btcusdt_test['trade_log_return'].cumsum()) * INITIAL_CAPITAL * LEVERAGE).plot(ax=ax, label=f'{LEVERAGE}x Leverage')

plt.title('Equity Curves: Unleveraged vs Leveraged')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ### Advertencia sobre Riesgo de Apalancamiento
#
# **Importante**: Mayor apalancamiento aumenta tanto los retornos potenciales COMO las pérdidas potenciales.
#
# | Apalancamiento | Ganancia 10% | Pérdida 10% |
# |----------------|-------------|-------------|
# | 1x | +10% | -10% |
# | 2x | +20% | -20% |
# | 5x | +50% | -50% |
# | 10x | +100% | -100% (liquidación) |

# ---
#
# ## Análisis de Rendimiento de la Estrategia
#
# ### Resumen Final de Rendimiento

# Calcular Sharpe Ratio
returns = btcusdt_test['trade_log_return']
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 24)  # Anualizado para datos por hora

print("=" * 50)
print("RESUMEN DE RENDIMIENTO DE LA ESTRATEGIA")
print("=" * 50)
print(f"\nPeríodo: {btcusdt_test.index.min()} to {btcusdt_test.index.max()}")
print(f"Total Trades: {len(btcusdt_test):,}")

print("\n--- Retornos ---")
print(f"Cumulative Log Return: {returns.sum():.4f}")
print(f"Compound Multiplier: {np.exp(returns.sum()):.4f}x")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")

print(f"\n--- Dimensionamiento Estático (${INITIAL_CAPITAL} inicial) ---")
print(f"Final Equity: ${INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum():.2f}")
print(f"Total Return: {(INITIAL_CAPITAL + btcusdt_test['trade_gross_pnl'].sum()) / INITIAL_CAPITAL:.2%}")

print(f"\n--- Dimensionamiento Dinámico con Compounding (${INITIAL_CAPITAL} inicial) ---")
print(f"Final Equity: ${btcusdt_test['post_trade_value_compound'].iloc[-1]:.2f}")
print(f"Total Return: {btcusdt_test['post_trade_value_compound'].iloc[-1] / INITIAL_CAPITAL:.2%}")

print(f"\n--- Con {LEVERAGE}x Apalancamiento (${INITIAL_CAPITAL} inicial) ---")
print(f"Final Equity: ${btcusdt_test['post_trade_value_leveraged'].iloc[-1]:.2f}")
print(f"Total Return: {btcusdt_test['post_trade_value_leveraged'].iloc[-1] / INITIAL_CAPITAL:.2%}")

# ### La Fórmula del Alpha
#
# La fórmula completa para alpha en trading:
#
# $$\text{Alpha} = \text{Statistical Edge} \times \text{Ejecución}$$
#
# Desglosando la ejecución:
#
# $$\text{Alpha} = \text{Statistical Edge} \times \text{Compounding} \times \text{Apalancamiento}$$
#
# Donde:
# - **Statistical Edge**: Capacidad del modelo para predecir la dirección (de Módulos 05-07)
# - **Compounding**: Reinvertir ganancias para crecer el tamaño de posición
# - **Apalancamiento**: Pedir prestado para amplificar retornos (y riesgos)

# ---
#
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Agregar Costos de Transacción (Estrategia Taker)
#
# Agrega comisiones taker para calcular retorno neto y PnL.
# Recuerda considerar **comisiones de ida y vuelta** (entrada + salida).
#
# Comisión taker típica: 0.04% - 0.075% por lado

# TODO: Implementar cálculo de comisiones taker
# taker_fee = 0.0005  # 0.05% por lado
# round_trip_fee = taker_fee * 2
#
# btcusdt_test['trade_net_return'] = btcusdt_test['trade_log_return'] - round_trip_fee
# Comparar retornos netos vs brutos

# ### Ejercicio 2: Análisis de Estrategia Maker
#
# Implementa una estrategia maker con órdenes límite.
# Las comisiones maker son típicamente más bajas (o incluso negativas con rebates).
#
# Considera:
# - Probabilidad de llenado (no todas las órdenes límite se ejecutan)
# - Posición en la cola
# - Selección adversa

# TODO: Implementar análisis de estrategia maker
# maker_fee = -0.0002  # -0.02% (rebate)
# fill_rate = 0.7  # Asumir 70% de tasa de llenado

# ### Ejercicio 3: Optimización de Timeframe
#
# Resamplear los datos a diferentes timeframes (4H, 1D) y comparar el rendimiento de la estrategia.
# Encontrar el timeframe óptimo para una estrategia taker que supere las comisiones.

# TODO: Implementar resampleo de timeframe
# btcusdt_4h = btcusdt.resample('4H').agg({
#     'open': 'first',
#     'high': 'max',
#     'low': 'min',
#     'close': 'last',
#     'volume': 'sum'
# })

# ### Ejercicio 4: Gestión de Riesgo
#
# Implementa un stop por drawdown máximo:
# - Dejar de operar si el drawdown excede 20%
# - Reducir tamaño de posición durante períodos de drawdown

# TODO: Implementar cálculo de drawdown y gestión de riesgo
# rolling_max = equity_curve.cummax()
# drawdown = (equity_curve - rolling_max) / rolling_max

# ---
#
# ## Puntos Clave
#
# 1. **Pipeline de Estrategia**: Señal → Orden → Ejecución
#    - Cada paso afecta el rendimiento final
#
# 2. **Señales de Entrada/Salida**:
#    - Basadas en tiempo: Operar cada período
#    - Basadas en predicados: Operar solo con señales de alta confianza
#
# 3. **Dimensionamiento de Posiciones**:
#    - Estático: Tamaño de posición fijo, crecimiento lineal
#    - Dinámico (Compounding): La posición crece con las ganancias, crecimiento exponencial
#
# 4. **Apalancamiento**:
#    - Amplifica tanto ganancias como pérdidas
#    - Mayor apalancamiento = mayor riesgo de liquidación
#
# 5. **La Fórmula del Alpha**:
#    $$\text{Alpha} = \text{Statistical Edge} \times \text{Compounding} \times \text{Apalancamiento}$$
#
# 6. **Métricas Clave**:
#    - Cumulative log return: $\sum r_t$
#    - Multiplicador compuesto: $e^{\sum r_t}$
#    - Sharpe Ratio: $\frac{\bar{r}}{\sigma_r}$
#
# 7. **Costos de Transacción**:
#    - Las comisiones maker vs taker impactan significativamente la rentabilidad
#    - Deben incluirse en cualquier backtesting realista
#
# ---
#
# **¡Felicidades!** Has completado el Quant Trading Accelerator.
#
# Ahora tienes el conocimiento fundamental para:
# - Construir modelos de trading cuantitativo
# - Implementar backtesting apropiado con cross-validation
# - Diseñar y evaluar estrategias de trading
# - Entender la gestión de riesgo y posiciones
#
# **Próximos Pasos**: ¡Aplica estos conceptos al trading real con gestión de riesgo apropiada!
