#!/usr/bin/env python
# coding: utf-8

# # Módulo 01: Fundamentos de Python para Trading Cuantitativo
#
# **Quant Trading Accelerator** | Programa de Formación Interno
#
# ---

# ## Tabla de Contenidos
#
# 1. [Introducción y Objetivos de Aprendizaje](#introduction)
# 2. [Variables: Los Bloques Fundamentales](#variables)
# 3. [Tipos Numéricos: La Precisión Importa](#numeric-types)
# 4. [Operaciones Aritméticas: Cálculos Financieros](#arithmetic)
# 5. [Strings: Trabajando con Símbolos e Identificadores](#strings)
# 6. [Ejercicios Prácticos](#exercises)
# 7. [Puntos Clave](#takeaways)

# ---
#
# ## 1. Introducción y Objetivos de Aprendizaje <a name="introduction"></a>

# ### ¿Qué es el Trading Cuantitativo?
#
# El trading cuantitativo utiliza modelos matemáticos y algoritmos computacionales para identificar y ejecutar oportunidades de trading. El flujo de trabajo típico sigue este pipeline:
#
# ```
# Datos de Mercado → Feature Engineering → Modelo → Generación de Señales → Gestión de Riesgo → Ejecución
# ```
#
# Como quant, trabajarás a lo largo de todo este pipeline. Python es la lingua franca de las finanzas cuantitativas debido a sus:
#
# - Capacidades de **prototipado rápido**
# - **Ecosistema rico** (NumPy, Pandas, PyTorch, Polars)
# - **Preparación para producción** en sistemas de trading en vivo

# ### Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# 1. **Declarar y manipular variables** para almacenar datos de mercado
# 2. **Entender la precisión numérica** y sus implicaciones en cálculos financieros
# 3. **Realizar operaciones aritméticas** comunes en trading (retornos, PnL, dimensionamiento de posiciones)
# 4. **Parsear y manipular strings** para símbolos de ticker e identificadores de datos
# 5. **Aplicar estos conceptos** a escenarios reales de trading

# ### Prerequisitos
#
# - Conocimientos básicos de informática
# - Curiosidad intelectual
# - No se requiere experiencia previa en programación
#
# ---
#
# ## 2. Variables: Los Bloques Fundamentales <a name="variables"></a>

# ### ¿Qué es una Variable?
#
# Una **variable** es un contenedor con nombre que almacena datos en memoria. En sistemas de trading, las variables contienen información crítica como:
#
# - Precios actuales de activos
# - Tamaños de posición
# - Parámetros de riesgo
# - Identificadores de órdenes
#
# Piensa en las variables como cajas etiquetadas donde almacenas valores para usar después.

# ### Asignación Básica de Variables
#
# El operador de asignación `=` almacena un valor en una variable. El nombre de la variable va a la izquierda, el valor a la derecha.

# Almacenar un valor de precio en una variable llamada 'price'
# Esto podría representar el precio bid actual de un activo
price = 10.0

# Acceder al valor almacenado referenciando el nombre de la variable
price

# ### Reasignación de Variables
#
# Las variables pueden actualizarse. Esto es esencial para rastrear condiciones cambiantes del mercado.

# Simular un tick de precio: el precio aumenta en 0.50
price = price + 0.5

# La variable ahora contiene el valor actualizado
price

# ### Crear Nuevas Variables a Partir de Existentes
#
# Puedes derivar nuevas variables a partir de cálculos que involucren variables existentes.

# Calcular cuál era el precio antes del tick
# Esta es una operación común al computar cambios de precio
old_price = price - 0.5

old_price

# ### Importante: Valor vs. Referencia
#
# Cuando asignas una variable a otra, Python copia el **valor**, no la referencia (para tipos primitivos como números).

# Demostración del comportamiento de copia de valores
price = 10.0
new_price = price    # new_price obtiene una COPIA del valor de price (10.0)
price = 11.0         # Cambiar price NO afecta a new_price

# new_price aún contiene el valor original copiado
new_price

# price tiene el nuevo valor
price

# > **Implicación en Trading**: Este comportamiento es crucial al almacenar precios históricos. Copiar una variable crea una instantánea independiente, no una referencia vinculada.
#
# ---
#
# ## 3. Tipos Numéricos: La Precisión Importa <a name="numeric-types"></a>

# ### Por Qué Importan los Tipos Numéricos en Trading
#
# Los cálculos financieros requieren atención cuidadosa a la precisión numérica. Un error de redondeo de $0.01 por operación, compuesto a lo largo de millones de operaciones, puede resultar en pérdidas significativas o problemas regulatorios.
#
# Python tiene dos tipos numéricos principales:
#
# | Tipo | Descripción | Caso de Uso |
# |------|-------------|-------------|
# | `int` | Números enteros | Conteo de operaciones, IDs de órdenes |
# | `float` | Números decimales | Precios, retornos, cantidades |

# ### Verificar Tipos de Variables
#
# Usa la función `type()` para inspeccionar el tipo de dato de una variable.

# price es un float (número decimal)
type(price)

# Almacenar el número de operaciones ejecutadas hoy
no_trades = 100

no_trades

# no_trades es un integer (número entero)
type(no_trades)

# ### Conversión de Tipos (Casting)
#
# A veces necesitas convertir entre tipos. Esto se llama **casting**.

# Convertir integer a float
no_trades = float(no_trades)

type(no_trades)

# Observa el punto decimal indicando que ahora es un float
no_trades

# Convertir float a integer (trunca la parte decimal - ¡NO redondea!)
int(100.12)

# Convertir integer a float
float(100)

# Convertir string a float (común al parsear feeds de datos de mercado)
float("262.82")

# > **Advertencia**: `int()` trunca hacia cero, no redondea. `int(2.9)` retorna `2`, no `3`.

# ### Precisión de Punto Flotante
#
# Los números de punto flotante tienen precisión limitada. Esto puede causar comportamiento inesperado:

# Esto debería ser igual a 0.3, pero debido a la representación de punto flotante...
0.1 + 0.1 + 0.1

# > **Buena Práctica**: Para cálculos financieros que requieren precisión exacta, usa el módulo `decimal` o aritmética de enteros (ej., almacenar precios en centavos/pips).
#
# ---
#
# ## 4. Operaciones Aritméticas: Cálculos Financieros <a name="arithmetic"></a>

# ### Precedencia de Operadores
#
# Python sigue el orden estándar de operaciones matemáticas (PEMDAS/BODMAS):
#
# 1. **P**aréntesis
# 2. **E**xponentes
# 3. **M**ultiplicación / **D**ivisión
# 4. **A**dición / **S**ustracción

# La multiplicación ocurre antes que la adición
3 + 3 * 3  # = 3 + 9 = 12

# Cálculo explícito equivalente
3 + 9

# Usa paréntesis para cambiar el orden de operaciones
(3 + 3) * 3  # = 6 * 3 = 18

# Cálculo explícito equivalente
6 * 3

# ### Cálculos Financieros Comunes
#
# #### Calcular Valor Futuro con Tasa de Crecimiento
#
# Si el precio crece un 5%, el nuevo precio es:
#
# $$P_{new} = P_{old} \times (1 + r)$$
#
# donde $r$ es la tasa de crecimiento.

# Calcular precio después de un aumento del 5%
price * 1.05

# #### Aplicar un Descuento
#
# $$P_{discounted} = P_{original} \times (1 - d)$$
#
# donde $d$ es la tasa de descuento.

# Calcular precio después de un descuento del 10%
price * (1 - 0.1)

# #### Cálculo de Retorno Simple
#
# El retorno simple entre dos precios es:
#
# $$R = \frac{P_{t} - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

# Calcular retorno simple
price_t = 105.0
price_t_minus_1 = 100.0

simple_return = (price_t - price_t_minus_1) / price_t_minus_1
simple_return

# #### Cálculo de Log Return
#
# Los log returns son preferidos en finanzas cuantitativas porque son:
# - **Aditivos en el tiempo**: $r_{t_1 \to t_3} = r_{t_1 \to t_2} + r_{t_2 \to t_3}$
# - **Simétricos**: Una ganancia de +10% y una pérdida de -10% no suman cero con retornos simples
#
# $$r_{log} = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

import math

log_return = math.log(price_t / price_t_minus_1)
log_return

# ---
#
# ## 5. Strings: Trabajando con Símbolos e Identificadores <a name="strings"></a>

# ### ¿Qué son los Strings?
#
# Los **strings** son secuencias de caracteres, usados para:
#
# - Símbolos de ticker (AAPL, BTCUSDT, ES=F)
# - IDs de órdenes
# - Identificadores de exchanges
# - Mensajes de log

# Definir un símbolo de ticker
symbol = 'GOOG'

symbol

type(symbol)

# ### Concatenación de Strings
#
# Usa `+` para combinar strings. Esto es útil para construir nombres de columnas o identificadores.

# Construir un nombre de columna para una serie de precios
col = symbol + "_price"

col

# Alternativa: notación de prefijo
col = "price_" + symbol

col

# ### Métodos de String
#
# Los strings tienen métodos integrados para manipulación.
#
# #### El Método `.replace()`
#
# Reemplaza ocurrencias de un substring con otro string.

# Reemplazar guiones bajos con guiones
col.replace("_", "-")

# > **Importante**: Los métodos de string retornan un NUEVO string. El original no cambia.

# El string original no se modifica
col

# Para persistir el cambio, reasigna la variable
col = col.replace("_", "-")

col

# #### El Método `.split()`
#
# Divide un string en una lista de substrings basándose en un delimitador.

# Dividir por guión
col.split('-')

# Almacenar el resultado en una variable
tokens = col.split('-')

tokens

# #### Indexación de Listas
#
# Accede a elementos individuales usando notación de corchetes. **Los índices comienzan en 0.**

# Primer elemento (índice 0)
tokens[0]

# Segundo elemento (índice 1)
tokens[1]

# #### El Método `.find()` y Slicing de Strings
#
# Encuentra la posición de un substring y extrae porciones del string.

# Encontrar la posición del guión
i = col.find('-')

i

# Extraer todo antes del guión (slicing de string)
# Sintaxis: string[inicio:fin] - fin es exclusivo
col[0:i]

# Extraer todo después del guión
col[i+1:]

# ### Formateo de Strings
#
# Combinar strings con números requiere conversión explícita o f-strings.

price

# Esto causaría un TypeError:
# symbol + " price is " + price  # No se puede concatenar str y float

# Opción 1: Conversión explícita con str()
symbol + " price is " + str(price)

# Opción 2: f-strings (recomendado - más limpio y flexible)
f"{symbol} price is {price}"

# #### Formateo Avanzado con f-strings
#
# Los f-strings soportan especificadores de formato para controlar la salida.

# Formatear con 2 decimales
f"{symbol} price is ${price:.2f}"

# Formatear números grandes con separador de miles
volume = 1500000
f"{symbol} volume: {volume:,}"

# Formatear porcentajes
pct_change = 0.0523
f"{symbol} change: {pct_change:.2%}"

# ---
#
# ## 6. Ejercicios Prácticos <a name="exercises"></a>

# ### Ejercicio 1: Calcular Delta de Precio
#
# **Escenario**: Estás rastreando movimientos diarios de precio. Calcula el cambio absoluto de precio entre ayer y hoy.
#
# $$\Delta P = P_{today} - P_{yesterday}$$

price_today = 100.0
price_yesterday = 90.0

# TODO: Calcular el delta de precio
price_delta = 0.0  # Reemplaza esto con tu cálculo

# TU CÓDIGO AQUÍ

# Validación (debería retornar True)
price_delta == 10

# ### Ejercicio 2: Calcular Profit & Loss (PnL) Total
#
# **Escenario**: Has ejecutado 4 operaciones hoy con valores individuales de PnL. Calcula tu PnL total.
#
# $$\text{Total PnL} = \sum_{i=1}^{n} \text{PnL}_i$$

trade1_pnl = 1.2    # Operación ganadora
trade2_pnl = -2.0   # Operación perdedora
trade3_pnl = 3.0    # Operación ganadora
trade4_pnl = 8.5    # Operación ganadora

# TODO: Calcular el PnL total
total_pnl = 0.0  # Reemplaza esto con tu cálculo

# TU CÓDIGO AQUÍ

# Validación (debería retornar True)
total_pnl == 10.7

# ### Ejercicio 3: Parsear Datos de Mercado
#
# **Escenario**: Recibes datos de mercado como un string en el formato `"SYMBOL:PRICE"`. Parsea este string para extraer el símbolo y precio como variables separadas.

s = 'AAPL:262.82'

# TODO: Extraer símbolo y precio del string
symbol = ''   # Debería ser 'AAPL'
price = 0.0   # Debería ser 262.82

# TU CÓDIGO AQUÍ

# Validación (debería retornar True)
symbol == 'AAPL'

# Validación (debería retornar True)
price == 262.82

# ### Ejercicio 4 (Desafío): Calcular Valor de Posición
#
# **Escenario**: Tienes una posición en TSLA. Calcula el valor total de la posición y el PnL no realizado.
#
# $$\text{Valor de Posición} = \text{Cantidad} \times \text{Precio Actual}$$
#
# $$\text{PnL No Realizado} = \text{Cantidad} \times (\text{Precio Actual} - \text{Precio de Entrada})$$

ticker = "TSLA"
quantity = 150           # Número de acciones en cartera
entry_price = 180.50     # Precio al que se abrió la posición
current_price = 195.75   # Precio actual de mercado

# TODO: Calcular valor de posición y PnL no realizado
position_value = 0.0     # Reemplaza con tu cálculo
unrealized_pnl = 0.0     # Reemplaza con tu cálculo

# TU CÓDIGO AQUÍ

# Mostrar resultados usando f-strings con formato apropiado
f"{ticker} | Position Value: ${position_value:,.2f} | Unrealized PnL: ${unrealized_pnl:,.2f}"

# ---
#
# ## 7. Puntos Clave <a name="takeaways"></a>

# ### Resumen
#
# | Concepto | Puntos Clave |
# |----------|--------------|
# | **Variables** | Contenedores con nombre para datos; usa nombres descriptivos |
# | **Integers** | Números enteros; úsalos para conteos e IDs |
# | **Floats** | Números decimales; úsalos para precios y retornos |
# | **Conversión de Tipos** | `int()`, `float()`, `str()` para casting explícito |
# | **Aritmética** | Sigue PEMDAS; usa paréntesis para claridad |
# | **Strings** | Inmutables; los métodos retornan nuevos strings |
# | **f-strings** | Método preferido para formateo de strings |
#
# ### Buenas Prácticas para Código Quant
#
# 1. **Usa nombres descriptivos de variables**: `entry_price` no `p1`
# 2. **Sé explícito sobre los tipos**: Convierte strings a floats al parsear datos
# 3. **Usa f-strings para formateo**: Más limpio y mantenible
# 4. **Comenta tus fórmulas financieras**: Tu yo del futuro te lo agradecerá
# 5. **Prueba casos límite**: ¿Qué pasa cuando el precio es 0? ¿Negativo?
#
# ### ¿Qué Sigue?
#
# En el **Módulo 02**, cubriremos:
#
# - **Listas y Colecciones**: Almacenar múltiples precios, datos de series temporales
# - **Bucles**: Iterar sobre datos de operaciones
# - **Condicionales**: Implementar lógica de trading (if signal > threshold, buy)
#
# ---
#
# **Fin del Módulo 01**
#
# *¿Preguntas? Contacta al equipo de Quant Research.*
