#!/usr/bin/env python
# coding: utf-8

# # Módulo 06: Classification
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [Classification vs Regression](#classification-vs-regression)
# 3. [Target de Clasificación Binaria](#binary-classification-target)
# 4. [Logistic Regression](#logistic-regression)
# 5. [Confusion Matrix](#confusion-matrix)
# 6. [ROC AUC](#roc-auc)
# 7. [Análisis de Rentabilidad](#profitability-analysis)
# 8. [Excess Predictability](#excess-predictability)
# 9. [Ejercicios Prácticos](#practical-exercises)
# 10. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Entender cuándo usar classification vs regression para trading
# - Entrenar un modelo de logistic regression en PyTorch
# - Evaluar el rendimiento del modelo usando métricas de confusion matrix
# - Entender y calcular el ROC AUC score
# - Analizar la rentabilidad del modelo más allá de la precisión
# - Calcular Excess Predictability (Estadístico de Gerko)

# Librerías principales
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
# ### Salida de Regression
# - Valor continuo: `0.0012` (log return predicho)
# - Útil para: Dimensionamiento de posiciones, predicciones exactas
#
# ### Salida de Classification
# - Clase discreta: `UP` (75% confianza) o `DOWN` (25% confianza)
# - Útil para: Señales claras de trading, calibración de probabilidad
#
# ### ¿Cuándo Usar Cuál?
#
# | Caso de Uso | Enfoque |
# |-------------|---------|
# | Predecir retorno exacto | Regression |
# | Predecir solo dirección | Classification |
# | Dimensionar posición basado en confianza | Classification (con probabilidades) |
# | Señales simples up/down | Classification |

# ---
#
# ## Target de Clasificación Binaria
#
# Convertimos retornos continuos en clases binarias:
# - **1** = Long (el precio sube)
# - **0** = Short (el precio baja)

# Cargar datos
url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')
btcusdt.head()

# Calcular log returns
btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())

# Crear features con lag
btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)

# Eliminar filas NaN
btcusdt = btcusdt.dropna()

# Crear target de clasificación binaria
# 1 = retorno positivo (ir long)
# 0 = retorno negativo (ir short)
btcusdt['close_log_return_dir'] = btcusdt['close_log_return'].map(lambda x: 1 if x > 0 else 0)
btcusdt[['close_log_return', 'close_log_return_dir']].head(10)

# ### Verificar Balance del Target
#
# Importante: Verificar si las clases están balanceadas (aproximadamente 50/50).

btcusdt['close_log_return_dir'].value_counts()

# Calcular ratio de balance
balance_ratio = btcusdt['close_log_return_dir'].value_counts(normalize=True)
print(f"Up ratio: {balance_ratio[1]:.2%}")
print(f"Down ratio: {balance_ratio[0]:.2%}")

# ---
#
# ## Split Train/Test

def time_split(x, train_size=0.75):
    """Split data chronologically for time series."""
    i = int(len(x) * train_size)
    return x[:i].copy(), x[i:].copy()

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print(f"Train samples: {len(btcusdt_train)}")
print(f"Test samples: {len(btcusdt_test)}")

# Verificar balance de clases en ambas particiones
print("\nTrain class distribution:")
print(btcusdt_train['close_log_return_dir'].value_counts())

print("\nTest class distribution:")
print(btcusdt_test['close_log_return_dir'].value_counts())

# ---
#
# ## Logistic Regression
#
# ### La Función Sigmoid
#
# La logistic regression usa la **función sigmoid** para mapear salidas a probabilidades:
#
# $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
#
# Donde $z = w \cdot x + b$ (combinación lineal)
#
# La sigmoid mapea cualquier número real al rango (0, 1), que interpretamos como probabilidad.

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
# ESTANDARIZACIÓN DE FEATURES
# ---
features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
target = 'close_log_return_dir'

# Ajustar scaler solo en datos de entrenamiento (¡prevenir data leakage!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(btcusdt_train[features].values)
X_test_scaled = scaler.transform(btcusdt_test[features].values)

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

# ---
# MODELO DE LOGISTIC REGRESSION
# ---
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

# Binary Cross-Entropy con Logits (numéricamente estable)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# ---
# BUCLE DE ENTRENAMIENTO
# ---
for epoch in range(15000):
    optimizer.zero_grad()

    # Forward pass: obtener logits
    y_pred_logits = model(X_train)

    # Calcular loss
    loss = criterion(y_pred_logits, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    if epoch % 3000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

print(f"\nFinal weight: {model.linear.weight.data}")
print(f"Final bias: {model.linear.bias.data}")

# ---
# GENERAR PREDICCIONES
# ---
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_proba = torch.sigmoid(y_pred_logits)  # Convertir a probabilidades
    y_pred_binary = (y_pred_proba >= 0.5).float()  # Umbral en 0.5

y_test_np = y_test.squeeze().numpy()
y_pred_binary_np = y_pred_binary.squeeze().numpy()
y_pred_proba_np = y_pred_proba.squeeze().numpy()

# ---
#
# ## Confusion Matrix
#
# La confusion matrix muestra predicción vs resultados reales:
#
# |  | Predicho DOWN | Predicho UP |
# |--|---------------|-------------|
# | **Real DOWN** | TN (True Negative) | FP (False Positive) |
# | **Real UP** | FN (False Negative) | TP (True Positive) |

cm = confusion_matrix(y_test_np, y_pred_binary_np)
print("Confusion Matrix:")
print(cm)

# Extraer valores
TN = cm[0][0]  # True Negative (Correctamente predicho DOWN)
FN = cm[1][0]  # False Negative (Predicho DOWN, era UP)
FP = cm[0][1]  # False Positive (Predicho UP, era DOWN)
TP = cm[1][1]  # True Positive (Correctamente predicho UP)

print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"False Positives (FP): {FP}")
print(f"True Positives (TP): {TP}")

# ### Métricas Clave

# Precisión General (Win Rate)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy (Win Rate): {accuracy:.2%}")

# Precision: Cuando predecimos UP, ¿con qué frecuencia acertamos?
precision_up = TP / (TP + FP) if (TP + FP) > 0 else 0
print(f"Precision (UP): {precision_up:.2%}")

# Recall (Sensibilidad): De todos los UPs reales, ¿cuántos detectamos?
recall_up = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f"Recall (UP): {recall_up:.2%}")

# Precision para DOWN
precision_down = TN / (TN + FN) if (TN + FN) > 0 else 0
print(f"Precision (DOWN): {precision_down:.2%}")

# Distribución de señales
long_ratio = (FP + TP) / (FN + TN + TP + FP)
short_ratio = (FN + TN) / (FN + TN + TP + FP)
print(f"Long signal ratio: {long_ratio:.2%}")
print(f"Short signal ratio: {short_ratio:.2%}")
print(f"Long/Short imbalance: {long_ratio/short_ratio:.2f}")

# ---
#
# ## ROC AUC
#
# La curva **ROC** (Receiver Operating Characteristic) muestra el trade-off entre
# True Positive Rate y False Positive Rate a diferentes umbrales.
#
# **AUC** (Area Under Curve) resume la curva:
# - AUC = 0.5: Adivinación aleatoria
# - AUC = 1.0: Discriminación perfecta
# - AUC < 0.5: Peor que el azar (predicciones invertidas)

auc = roc_auc_score(y_test, y_pred_proba_np)
print(f"ROC AUC Score: {auc:.4f}")

# Graficar Curva ROC
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
# ## Análisis de Rentabilidad
#
# **Importante**: ¡Alta precisión no garantiza ganancias!
#
# Necesitamos analizar el rendimiento real de trading.

# Agregar predicciones a datos de test
btcusdt_test['y_pred_binary'] = y_pred_binary_np
btcusdt_test['y_pred_proba'] = y_pred_proba_np

# Crear señal direccional: 1 para long, -1 para short
btcusdt_test['dir_signal'] = np.where(btcusdt_test['y_pred_binary'] == 1, 1, -1)

# Verificar distribución de señales
btcusdt_test['dir_signal'].value_counts()

# Calcular retornos por operación
btcusdt_test['trade_log_return'] = btcusdt_test['dir_signal'] * btcusdt_test['close_log_return']

# Retornos acumulados
btcusdt_test['cum_trade_log_return'] = btcusdt_test['trade_log_return'].cumsum()

plt.figure(figsize=(15, 8))
btcusdt_test['cum_trade_log_return'].plot()
plt.title('Cumulative Trade Log Returns')
plt.ylabel('Cumulative Log Return')
plt.xlabel('Time')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()

# Comparar con el activo subyacente
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

axes[0].set_title('Model Cumulative Returns')
btcusdt_test['cum_trade_log_return'].plot(ax=axes[0])
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

axes[1].set_title('BTC/USDT Price')
btcusdt_test['close'].plot(ax=axes[1])

plt.tight_layout()
plt.show()

# ### Curva de Equity y Métricas de Rendimiento

initial_capital = 100

# PnL Bruto
btcusdt_test['trade_gross_pnl'] = np.exp(btcusdt_test['cum_trade_log_return']) * initial_capital

plt.figure(figsize=(15, 8))
btcusdt_test['trade_gross_pnl'].plot()
plt.title(f'Equity Curve (Starting Capital: ${initial_capital})')
plt.ylabel('Portfolio Value ($)')
plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
plt.show()

# Sharpe Ratio
sharpe_raw = btcusdt_test['trade_log_return'].mean() / btcusdt_test['trade_log_return'].std()
sharpe_annual = sharpe_raw * np.sqrt(365 * 24)

print(f"Raw Sharpe: {sharpe_raw:.4f}")
print(f"Annualized Sharpe: {sharpe_annual:.2f}")

# Retorno compuesto total
total_compound_return = np.exp(btcusdt_test['trade_log_return'].sum())
print(f"Total Compound Return: {total_compound_return:.2%}")
print(f"Final Portfolio Value: ${total_compound_return * initial_capital:.2f}")

# ---
#
# ## Excess Predictability (Estadístico de Gerko)
#
# ¿Cuánto mejor es nuestro modelo comparado con adivinar al azar?
#
# ### Crear Benchmark Aleatorio

# Generar predicciones binarias aleatorias
rng = np.random.default_rng(SEED)
random_binary = rng.integers(low=0, high=2, size=len(btcusdt_test))

# Convertir a señales (-1, +1)
btcusdt_test['random_dir_signal'] = random_binary * 2 - 1
btcusdt_test['random_dir_signal'].value_counts()

# Retornos de operaciones aleatorias
btcusdt_test['random_trade_log_return'] = btcusdt_test['random_dir_signal'] * btcusdt_test['close_log_return']
btcusdt_test['cum_random_trade_log_return'] = btcusdt_test['random_trade_log_return'].cumsum()

# Graficar comparación
plt.figure(figsize=(15, 8))
btcusdt_test['cum_trade_log_return'].plot(label='Model')
btcusdt_test['cum_random_trade_log_return'].plot(label='Random')
plt.title('Model vs Random Benchmark')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()

# Excess Predictability
excess_predictability = btcusdt_test['cum_trade_log_return'] - btcusdt_test['cum_random_trade_log_return']

plt.figure(figsize=(15, 8))
excess_predictability.plot()
plt.title('Excess Predictability (Model - Random)')
plt.ylabel('Excess Log Return')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.show()

# Excess Predictability total
total_excess = btcusdt_test['trade_log_return'].sum() - btcusdt_test['random_trade_log_return'].sum()
print(f"Total Excess Predictability: {total_excess:.4f}")

# ---
#
# ## Descomposición de Retornos (Anatolyev)
#
# Los retornos pueden descomponerse en dirección y magnitud:
#
# $$y_t = \text{sign}(y_t) \times |y_t|$$
#
# - $\text{sign}(y_t)$ = Dirección (+1 o -1)
# - $|y_t|$ = Magnitud (tamaño absoluto)
#
# ¡Esto sugiere que podríamos construir modelos separados para dirección y magnitud!

# Ejemplo
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
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Excess Profitability
#
# Crea un estadístico de prueba comparando la rentabilidad del modelo con buy & hold.

# TODO: Comparar rentabilidad del modelo sobre buy & hold (HODL)
# Pista: Calcular retornos de buy & hold y restar de los retornos del modelo

# ### Ejercicio 2: Modelo de Regression para Magnitud
#
# Entrena un modelo de regression para predecir el tamaño absoluto de los log returns futuros.

# TODO: Crear target = |close_log_return|
# Entrenar un modelo de regression

# ### Ejercicio 3: Modelo Combinado de Dirección + Magnitud
#
# Usa tanto classification (dirección) como regression (magnitud) para predecir
# el log return descompuesto.
#
# $$\hat{y}_t = \hat{\text{sign}}(y_t) \times \hat{|y_t|}$$

# TODO: Combinar predicción de dirección del modelo de classification
# con predicción de magnitud del modelo de regression

# ---
#
# ## Puntos Clave
#
# 1. **Classification vs Regression**:
#    - Classification predice clases discretas (UP/DOWN)
#    - Produce scores de probabilidad para trading basado en confianza
#
# 2. **Logistic Regression**:
#    - Usa sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
#    - La salida es probabilidad de clase UP
#
# 3. **Métricas de Confusion Matrix**:
#    - Accuracy: Predicciones correctas generales
#    - Precision: Precisión de predicciones positivas
#    - Recall: Cobertura de positivos reales
#
# 4. **ROC AUC**:
#    - AUC = 0.5: Adivinación aleatoria
#    - AUC > 0.5: Mejor que el azar
#    - AUC = 1.0: Discriminación perfecta
#
# 5. **Excess Predictability**:
#    - Mide la mejora sobre un baseline aleatorio
#    - Crítico para evaluar el verdadero valor del modelo
#
# 6. **Descomposición de Retornos**: $y_t = \text{sign}(y_t) \times |y_t|$
#    - Se puede modelar dirección y magnitud por separado
#
# ---
#
# **Siguiente Módulo**: Cross-Validation - Técnicas robustas de evaluación de modelos
