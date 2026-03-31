#!/usr/bin/env python3
"""
Translate the Jupyter notebook from English to Spanish.
- Markdown cells: translate all text to Spanish (keep technical terms in English)
- Code cells: translate comments only (keep all code identical)
"""

import json
import re
import sys

NOTEBOOK_PATH = r"c:\development\quant-trading\accelerator\02_strategy\01-ml_model_pytorch_es.ipynb"

# =============================================================================
# MARKDOWN TRANSLATION MAP
# =============================================================================
# We'll do phrase-level replacements first (longer phrases before shorter ones),
# then sentence-level for common patterns.

MARKDOWN_PHRASES = [
    # Section headers and structural phrases
    ("Table of Contents", "Tabla de Contenidos"),
    ("Learning Objectives", "Objetivos de Aprendizaje"),
    ("By the end of this module", "Al finalizar este módulo"),
    ("Key Takeaways", "Puntos Clave"),
    ("Practical Exercises", "Ejercicios Prácticos"),
    ("Data Preparation", "Preparación de Datos"),
    ("Model Training", "Entrenamiento del Modelo"),
    ("Model Evaluation", "Evaluación del Modelo"),
    ("Key Insight", "Punto Clave"),
    ("Best Practice", "Buena Práctica"),
    ("Best Practices", "Buenas Prácticas"),
    ("Interpretation", "Interpretación"),
    ("Warning", "Advertencia"),
    ("Important", "Importante"),
    ("Summary", "Resumen"),
    ("Next Steps", "Próximos Pasos"),
    ("Next", "Siguiente"),
    ("Exercise", "Ejercicio"),
    ("Example", "Ejemplo"),
    ("Note", "Nota"),

    # Common full sentences and phrases (order matters - longer first)
    ("In this notebook", "En este notebook"),
    ("In this section", "En esta sección"),
    ("In this module", "En este módulo"),
    ("Let's start by", "Comencemos por"),
    ("Let's start", "Comencemos"),
    ("Let's begin by", "Comencemos por"),
    ("Let's begin", "Comencemos"),
    ("Let's define", "Definamos"),
    ("Let's create", "Creemos"),
    ("Let's look at", "Veamos"),
    ("Let's see", "Veamos"),
    ("Let's plot", "Grafiquemos"),
    ("Let's check", "Verifiquemos"),
    ("Let's train", "Entrenemos"),
    ("Let's evaluate", "Evaluemos"),
    ("Let's load", "Carguemos"),
    ("Let's build", "Construyamos"),
    ("Let's implement", "Implementemos"),
    ("Let's test", "Probemos"),
    ("Let's examine", "Examinemos"),
    ("Let's explore", "Exploremos"),
    ("Let's visualize", "Visualicemos"),
    ("Let's compare", "Comparemos"),
    ("Let's analyze", "Analicemos"),
    ("Let's calculate", "Calculemos"),
    ("Let's use", "Usemos"),
    ("Let's try", "Intentemos"),
    ("we will learn", "aprenderemos"),
    ("we will explore", "exploraremos"),
    ("we will build", "construiremos"),
    ("we will create", "crearemos"),
    ("we will use", "usaremos"),
    ("we will implement", "implementaremos"),
    ("we will train", "entrenaremos"),
    ("we will evaluate", "evaluaremos"),
    ("we will define", "definiremos"),
    ("we will see", "veremos"),
    ("We will learn", "Aprenderemos"),
    ("We will explore", "Exploraremos"),
    ("We will build", "Construiremos"),
    ("We will create", "Crearemos"),
    ("We will use", "Usaremos"),
    ("We will implement", "Implementaremos"),
    ("We will train", "Entrenaremos"),
    ("We will evaluate", "Evaluaremos"),
    ("We will define", "Definiremos"),
    ("We will see", "Veremos"),
    ("you will be able to", "serás capaz de"),
    ("you will learn", "aprenderás"),
    ("you will understand", "comprenderás"),
    ("you will know", "conocerás"),
    ("The following", "Lo siguiente"),
    ("the following", "lo siguiente"),
    ("As we can see", "Como podemos ver"),
    ("as we can see", "como podemos ver"),
    ("This means that", "Esto significa que"),
    ("this means that", "esto significa que"),
    ("This is because", "Esto es porque"),
    ("this is because", "esto es porque"),
    ("For example", "Por ejemplo"),
    ("for example", "por ejemplo"),
    ("In other words", "En otras palabras"),
    ("in other words", "en otras palabras"),
    ("In addition", "Además"),
    ("in addition", "además"),
    ("However", "Sin embargo"),
    ("Therefore", "Por lo tanto"),
    ("Furthermore", "Además"),
    ("Moreover", "Además"),
    ("Finally", "Finalmente"),
    ("First", "Primero"),
    ("Second", "Segundo"),
    ("Third", "Tercero"),
    ("Now we", "Ahora"),
    ("now we", "ahora"),
    ("Now let's", "Ahora"),
    ("Below", "A continuación"),
    ("below", "a continuación"),
    ("Above", "Arriba"),
    ("above", "arriba"),
    ("Here we", "Aquí"),
    ("here we", "aquí"),
    ("Notice that", "Observe que"),
    ("notice that", "observe que"),
    ("Remember that", "Recuerde que"),
    ("remember that", "recuerde que"),
    ("Make sure", "Asegúrese"),
    ("make sure", "asegúrese"),
]

def translate_markdown_cell(source_lines):
    """Translate a markdown cell's source lines to Spanish."""
    result = []
    for line in source_lines:
        translated = translate_markdown_line(line)
        result.append(translated)
    return result


def translate_markdown_line(line):
    """Translate a single markdown line to Spanish using comprehensive rules."""
    # Skip empty lines
    if not line.strip():
        return line

    # Skip lines that are only code (indented or fenced)
    if line.strip().startswith('```') or line.startswith('    '):
        return line

    # Skip lines that are just math formulas
    if line.strip().startswith('$$') or line.strip().startswith('$'):
        return line

    # Skip image/link-only lines
    if line.strip().startswith('!['):
        return line

    # Apply phrase replacements
    for eng, esp in MARKDOWN_PHRASES:
        line = line.replace(eng, esp)

    # Translate remaining common words in context (careful not to break technical terms)
    # These are applied only to natural language portions
    line = apply_word_translations(line)

    return line


# Common English->Spanish word/phrase translations for remaining text
WORD_TRANSLATIONS = [
    # Verbs and verb phrases
    (r'\bWe can see\b', 'Podemos ver'),
    (r'\bwe can see\b', 'podemos ver'),
    (r'\bWe can\b', 'Podemos'),
    (r'\bwe can\b', 'podemos'),
    (r'\bWe need to\b', 'Necesitamos'),
    (r'\bwe need to\b', 'necesitamos'),
    (r'\bWe also need\b', 'También necesitamos'),
    (r'\bwe also need\b', 'también necesitamos'),
    (r'\bWe also\b', 'También'),
    (r'\bwe also\b', 'también'),
    (r'\bcan be used\b', 'puede ser usado'),
    (r'\bis used to\b', 'se usa para'),
    (r'\bare used to\b', 'se usan para'),
    (r'\bhas been\b', 'ha sido'),
    (r'\bhave been\b', 'han sido'),
    (r'\bwill be\b', 'será'),
    (r'\bshould be\b', 'debería ser'),
    (r'\bcould be\b', 'podría ser'),
    (r'\bcan also\b', 'también puede'),
    (r'\bwhich means\b', 'lo que significa'),
    (r'\bwhich is\b', 'que es'),
    (r'\bwhich are\b', 'que son'),
    (r'\bthat is\b', 'es decir'),
    (r'\bthat are\b', 'que son'),
    (r'\bThis is\b', 'Esto es'),
    (r'\bthis is\b', 'esto es'),
    (r'\bThere are\b', 'Hay'),
    (r'\bthere are\b', 'hay'),
    (r'\bThere is\b', 'Hay'),
    (r'\bthere is\b', 'hay'),

    # Nouns and noun phrases
    (r'\bthe model\b', 'el modelo'),
    (r'\bThe model\b', 'El modelo'),
    (r'\bthe data\b', 'los datos'),
    (r'\bThe data\b', 'Los datos'),
    (r'\bthe results\b', 'los resultados'),
    (r'\bThe results\b', 'Los resultados'),
    (r'\bthe output\b', 'la salida'),
    (r'\bThe output\b', 'La salida'),
    (r'\bthe input\b', 'la entrada'),
    (r'\bThe input\b', 'La entrada'),
    (r'\bthe training\b', 'el entrenamiento'),
    (r'\bThe training\b', 'El entrenamiento'),
    (r'\bthe function\b', 'la función'),
    (r'\bThe function\b', 'La función'),
    (r'\bthe value\b', 'el valor'),
    (r'\bThe value\b', 'El valor'),
    (r'\bthe values\b', 'los valores'),
    (r'\bThe values\b', 'Los valores'),
    (r'\bthe parameters\b', 'los parámetros'),
    (r'\bThe parameters\b', 'Los parámetros'),
    (r'\bthe prediction\b', 'la predicción'),
    (r'\bThe prediction\b', 'La predicción'),
    (r'\bthe predictions\b', 'las predicciones'),
    (r'\bThe predictions\b', 'Las predicciones'),
    (r'\bthe performance\b', 'el rendimiento'),
    (r'\bThe performance\b', 'El rendimiento'),
    (r'\bthe strategy\b', 'la estrategia'),
    (r'\bThe strategy\b', 'La estrategia'),
    (r'\bthe signal\b', 'la señal'),
    (r'\bThe signal\b', 'La señal'),
    (r'\bthe signals\b', 'las señales'),
    (r'\bThe signals\b', 'Las señales'),
    (r'\bthe price\b', 'el precio'),
    (r'\bThe price\b', 'El precio'),
    (r'\bthe prices\b', 'los precios'),
    (r'\bThe prices\b', 'Los precios'),
    (r'\bthe return\b', 'el retorno'),
    (r'\bThe return\b', 'El retorno'),
    (r'\bthe returns\b', 'los retornos'),
    (r'\bThe returns\b', 'Los retornos'),
    (r'\bthe weights\b', 'los pesos'),
    (r'\bThe weights\b', 'Los pesos'),
    (r'\bthe loss\b', 'la pérdida'),
    (r'\bThe loss\b', 'La pérdida'),
    (r'\bthe network\b', 'la red'),
    (r'\bThe network\b', 'La red'),
    (r'\bthe layer\b', 'la capa'),
    (r'\bThe layer\b', 'La capa'),
    (r'\bthe layers\b', 'las capas'),
    (r'\bThe layers\b', 'Las capas'),
    (r'\bthe epoch\b', 'la época'),
    (r'\bThe epoch\b', 'La época'),
    (r'\bthe epochs\b', 'las épocas'),
    (r'\bThe epochs\b', 'Las épocas'),
    (r'\bthe optimization\b', 'la optimización'),
    (r'\bThe optimization\b', 'La optimización'),
    (r'\bthe algorithm\b', 'el algoritmo'),
    (r'\bThe algorithm\b', 'El algoritmo'),
    (r'\bthe code\b', 'el código'),
    (r'\bThe code\b', 'El código'),
    (r'\bthe process\b', 'el proceso'),
    (r'\bThe process\b', 'El proceso'),
    (r'\bthe step\b', 'el paso'),
    (r'\bThe step\b', 'El paso'),
    (r'\bthe steps\b', 'los pasos'),
    (r'\bThe steps\b', 'Los pasos'),
    (r'\bthe next step\b', 'el siguiente paso'),
    (r'\bThe next step\b', 'El siguiente paso'),
    (r'\bthe features\b', 'las features'),
    (r'\bThe features\b', 'Las features'),
    (r'\bthe target\b', 'el objetivo'),
    (r'\bThe target\b', 'El objetivo'),
    (r'\bthe plot\b', 'el gráfico'),
    (r'\bThe plot\b', 'El gráfico'),
    (r'\bthe chart\b', 'el gráfico'),
    (r'\bThe chart\b', 'El gráfico'),
    (r'\bthe graph\b', 'el gráfico'),
    (r'\bThe graph\b', 'El gráfico'),
    (r'\bthe table\b', 'la tabla'),
    (r'\bThe table\b', 'La tabla'),
    (r'\bthe column\b', 'la columna'),
    (r'\bThe column\b', 'La columna'),
    (r'\bthe columns\b', 'las columnas'),
    (r'\bThe columns\b', 'Las columnas'),
    (r'\bthe row\b', 'la fila'),
    (r'\bThe row\b', 'La fila'),
    (r'\bthe rows\b', 'las filas'),
    (r'\bThe rows\b', 'Las filas'),
    (r'\bthe dataset\b', 'el dataset'),
    (r'\bThe dataset\b', 'El dataset'),
    (r'\bthe file\b', 'el archivo'),
    (r'\bThe file\b', 'El archivo'),

    # Common adjectives
    (r'\bimportant\b', 'importante'),
    (r'\bnecessary\b', 'necesario'),
    (r'\bdifferent\b', 'diferente'),
    (r'\bsimilar\b', 'similar'),
    (r'\bsimple\b', 'simple'),
    (r'\bfollowing\b', 'siguiente'),
    (r'\bprevious\b', 'anterior'),
    (r'\bcurrent\b', 'actual'),
    (r'\bavailable\b', 'disponible'),
    (r'\bpossible\b', 'posible'),

    # Common connectors
    (r'\band then\b', 'y luego'),
    (r'\bbecause\b', 'porque'),
    (r'\balthough\b', 'aunque'),
    (r'\bbefore\b', 'antes de'),
    (r'\bafter\b', 'después de'),
    (r'\bduring\b', 'durante'),
    (r'\bbetween\b', 'entre'),
    (r'\bwithout\b', 'sin'),
    (r'\binstead of\b', 'en lugar de'),
    (r'\bInstead of\b', 'En lugar de'),
    (r'\bas well as\b', 'así como'),
    (r'\bin order to\b', 'para'),
]


def apply_word_translations(line):
    """Apply regex-based word translations to a line."""
    for pattern, replacement in WORD_TRANSLATIONS:
        line = re.sub(pattern, replacement, line)
    return line


# =============================================================================
# CODE COMMENT TRANSLATIONS
# =============================================================================

COMMENT_TRANSLATIONS = [
    # Common comment patterns (longer patterns first)
    ("# Load and prepare", "# Cargar y preparar"),
    ("# Load and preprocess", "# Cargar y preprocesar"),
    ("# Load the data", "# Cargar los datos"),
    ("# Load data", "# Cargar datos"),
    ("# Load libraries", "# Cargar librerías"),
    ("# Load", "# Cargar"),
    ("# Calculate the", "# Calcular el"),
    ("# Calculate returns", "# Calcular retornos"),
    ("# Calculate log returns", "# Calcular Log Returns"),
    ("# Calculate", "# Calcular"),
    ("# Create the", "# Crear el"),
    ("# Create a", "# Crear un"),
    ("# Create features", "# Crear features"),
    ("# Create", "# Crear"),
    ("# Train the model", "# Entrenar el modelo"),
    ("# Train model", "# Entrenar modelo"),
    ("# Train", "# Entrenar"),
    ("# Training loop", "# Bucle de entrenamiento"),
    ("# Training", "# Entrenamiento"),
    ("# Plot the", "# Graficar el"),
    ("# Plot results", "# Graficar resultados"),
    ("# Plot", "# Graficar"),
    ("# Check the", "# Verificar el"),
    ("# Check if", "# Verificar si"),
    ("# Check for", "# Verificar"),
    ("# Check", "# Verificar"),
    ("# Convert to", "# Convertir a"),
    ("# Convert", "# Convertir"),
    ("# Print the", "# Imprimir el"),
    ("# Print results", "# Imprimir resultados"),
    ("# Print", "# Imprimir"),
    ("# Set the", "# Establecer el"),
    ("# Set up", "# Configurar"),
    ("# Set random seed", "# Establecer semilla aleatoria"),
    ("# Set seed", "# Establecer semilla"),
    ("# Set", "# Establecer"),
    ("# Initialize the", "# Inicializar el"),
    ("# Initialize", "# Inicializar"),
    ("# Define the", "# Definir el"),
    ("# Define a", "# Definir un"),
    ("# Define model", "# Definir modelo"),
    ("# Define loss", "# Definir Loss"),
    ("# Define optimizer", "# Definir optimizador"),
    ("# Define", "# Definir"),
    ("# Import libraries", "# Importar librerías"),
    ("# Import", "# Importar"),
    ("# Prepare the", "# Preparar el"),
    ("# Prepare data", "# Preparar datos"),
    ("# Prepare", "# Preparar"),
    ("# Split the data", "# Dividir los datos"),
    ("# Split data", "# Dividir datos"),
    ("# Split", "# Dividir"),
    ("# Normalize the", "# Normalizar el"),
    ("# Normalize", "# Normalizar"),
    ("# Standardize", "# Estandarizar"),
    ("# Scale the", "# Escalar el"),
    ("# Scale", "# Escalar"),
    ("# Forward pass", "# Paso hacia adelante"),
    ("# Backward pass", "# Retropropagación"),
    ("# Backpropagation", "# Retropropagación"),
    ("# Update weights", "# Actualizar pesos"),
    ("# Update parameters", "# Actualizar parámetros"),
    ("# Update", "# Actualizar"),
    ("# Compute the", "# Calcular el"),
    ("# Compute loss", "# Calcular Loss"),
    ("# Compute", "# Calcular"),
    ("# Evaluate the", "# Evaluar el"),
    ("# Evaluate model", "# Evaluar modelo"),
    ("# Evaluate", "# Evaluar"),
    ("# Predict", "# Predecir"),
    ("# Make predictions", "# Hacer predicciones"),
    ("# Get the", "# Obtener el"),
    ("# Get predictions", "# Obtener predicciones"),
    ("# Get", "# Obtener"),
    ("# Return the", "# Retornar el"),
    ("# Return", "# Retornar"),
    ("# Save the", "# Guardar el"),
    ("# Save model", "# Guardar modelo"),
    ("# Save", "# Guardar"),
    ("# Display the", "# Mostrar el"),
    ("# Display", "# Mostrar"),
    ("# Show the", "# Mostrar el"),
    ("# Show", "# Mostrar"),
    ("# Visualize", "# Visualizar"),
    ("# Test the", "# Probar el"),
    ("# Test", "# Probar"),
    ("# Run the", "# Ejecutar el"),
    ("# Run", "# Ejecutar"),
    ("# Apply the", "# Aplicar el"),
    ("# Apply", "# Aplicar"),
    ("# Remove the", "# Eliminar el"),
    ("# Remove", "# Eliminar"),
    ("# Drop the", "# Eliminar el"),
    ("# Drop", "# Eliminar"),
    ("# Filter the", "# Filtrar el"),
    ("# Filter", "# Filtrar"),
    ("# Sort the", "# Ordenar el"),
    ("# Sort", "# Ordenar"),
    ("# Count the", "# Contar el"),
    ("# Count", "# Contar"),
    ("# Select the", "# Seleccionar el"),
    ("# Select", "# Seleccionar"),
    ("# Merge the", "# Fusionar el"),
    ("# Merge", "# Fusionar"),
    ("# Join the", "# Unir el"),
    ("# Join", "# Unir"),
    ("# Group by", "# Agrupar por"),
    ("# Group", "# Agrupar"),
    ("# Reshape the", "# Redimensionar el"),
    ("# Reshape", "# Redimensionar"),
    ("# Flatten the", "# Aplanar el"),
    ("# Flatten", "# Aplanar"),
    ("# Extract the", "# Extraer el"),
    ("# Extract", "# Extraer"),
    ("# Generate the", "# Generar el"),
    ("# Generate", "# Generar"),
    ("# Build the", "# Construir el"),
    ("# Build", "# Construir"),
    ("# Compile the", "# Compilar el"),
    ("# Compile", "# Compilar"),
    ("# Fit the", "# Ajustar el"),
    ("# Fit", "# Ajustar"),
    ("# Transform the", "# Transformar el"),
    ("# Transform", "# Transformar"),
    ("# Encode the", "# Codificar el"),
    ("# Encode", "# Codificar"),
    ("# Decode the", "# Decodificar el"),
    ("# Decode", "# Decodificar"),
    ("# Iterate over", "# Iterar sobre"),
    ("# Iterate through", "# Iterar a través de"),
    ("# Iterate", "# Iterar"),
    ("# Loop through", "# Iterar sobre"),
    ("# Loop over", "# Iterar sobre"),
    ("# Add the", "# Agregar el"),
    ("# Add a", "# Agregar un"),
    ("# Add", "# Agregar"),
    ("# Append", "# Agregar"),
    ("# Insert", "# Insertar"),
    ("# Delete", "# Eliminar"),
    ("# Clear the", "# Limpiar el"),
    ("# Clear", "# Limpiar"),
    ("# Reset the", "# Reiniciar el"),
    ("# Reset", "# Reiniciar"),
    ("# Enable", "# Habilitar"),
    ("# Disable", "# Deshabilitar"),
    ("# Configure the", "# Configurar el"),
    ("# Configure", "# Configurar"),
    ("# Setup the", "# Configurar el"),
    ("# Setup", "# Configurar"),
    ("# Suppress warnings", "# Suprimir advertencias"),
    ("# Suppress", "# Suprimir"),
    ("# Ignore warnings", "# Ignorar advertencias"),
    ("# Ignore", "# Ignorar"),
    ("# Handle the", "# Manejar el"),
    ("# Handle", "# Manejar"),
    ("# Process the", "# Procesar el"),
    ("# Process", "# Procesar"),
    ("# Analyze the", "# Analizar el"),
    ("# Analyze", "# Analizar"),
    ("# Compare the", "# Comparar el"),
    ("# Compare", "# Comparar"),
    ("# Validate the", "# Validar el"),
    ("# Validate", "# Validar"),
    ("# Verify the", "# Verificar el"),
    ("# Verify", "# Verificar"),
    ("# Ensure the", "# Asegurar el"),
    ("# Ensure", "# Asegurar"),
    ("# Assign the", "# Asignar el"),
    ("# Assign", "# Asignar"),
    ("# Map the", "# Mapear el"),
    ("# Map", "# Mapear"),
    ("# Read the", "# Leer el"),
    ("# Read", "# Leer"),
    ("# Write the", "# Escribir el"),
    ("# Write", "# Escribir"),
    ("# Open the", "# Abrir el"),
    ("# Open", "# Abrir"),
    ("# Close the", "# Cerrar el"),
    ("# Close", "# Cerrar"),
    ("# Connect to", "# Conectar a"),
    ("# Connect", "# Conectar"),
    ("# Download the", "# Descargar el"),
    ("# Download", "# Descargar"),
    ("# Upload the", "# Subir el"),
    ("# Upload", "# Subir"),
    ("# Fetch the", "# Obtener el"),
    ("# Fetch", "# Obtener"),
    ("# Send the", "# Enviar el"),
    ("# Send", "# Enviar"),
    ("# Receive the", "# Recibir el"),
    ("# Receive", "# Recibir"),
    ("# Wait for", "# Esperar"),
    ("# Wait", "# Esperar"),
    ("# Retry", "# Reintentar"),
    ("# Log the", "# Registrar el"),
    ("# Log", "# Registrar"),
    ("# Debug", "# Depurar"),
    ("# Monitor", "# Monitorear"),
    ("# Track the", "# Rastrear el"),
    ("# Track", "# Rastrear"),
    ("# Record the", "# Registrar el"),
    ("# Record", "# Registrar"),
    ("# Store the", "# Almacenar el"),
    ("# Store", "# Almacenar"),
    ("# Cache the", "# Almacenar en caché el"),
    ("# Cache", "# Almacenar en caché"),
    ("# Optimize the", "# Optimizar el"),
    ("# Optimize", "# Optimizar"),
    ("# Minimize the", "# Minimizar el"),
    ("# Minimize", "# Minimizar"),
    ("# Maximize the", "# Maximizar el"),
    ("# Maximize", "# Maximizar"),
    ("# Sample the", "# Muestrear el"),
    ("# Sample", "# Muestrear"),
    ("# Shuffle the", "# Mezclar el"),
    ("# Shuffle", "# Mezclar"),
    ("# Random", "# Aleatorio"),
    ("# Clone the", "# Clonar el"),
    ("# Clone", "# Clonar"),
    ("# Copy the", "# Copiar el"),
    ("# Copy", "# Copiar"),
    ("# Move the", "# Mover el"),
    ("# Move", "# Mover"),
    ("# Rename the", "# Renombrar el"),
    ("# Rename", "# Renombrar"),
    ("# Replace the", "# Reemplazar el"),
    ("# Replace", "# Reemplazar"),
    ("# Swap the", "# Intercambiar el"),
    ("# Swap", "# Intercambiar"),
    ("# Invert the", "# Invertir el"),
    ("# Invert", "# Invertir"),
    ("# Reverse the", "# Invertir el"),
    ("# Reverse", "# Invertir"),
    ("# Transpose the", "# Transponer el"),
    ("# Transpose", "# Transponer"),
    ("# Pad the", "# Rellenar el"),
    ("# Pad", "# Rellenar"),
    ("# Clip the", "# Recortar el"),
    ("# Clip", "# Recortar"),
    ("# Crop the", "# Recortar el"),
    ("# Crop", "# Recortar"),
    ("# Resize the", "# Redimensionar el"),
    ("# Resize", "# Redimensionar"),
    ("# Rotate the", "# Rotar el"),
    ("# Rotate", "# Rotar"),
    ("# Flip the", "# Voltear el"),
    ("# Flip", "# Voltear"),

    # Descriptive comment translations
    ("# This function", "# Esta función"),
    ("# This method", "# Este método"),
    ("# This class", "# Esta clase"),
    ("# This is", "# Esto es"),
    ("# This will", "# Esto va a"),
    ("# Returns the", "# Retorna el"),
    ("# Returns a", "# Retorna un"),
    ("# Returns", "# Retorna"),
    ("# Note:", "# Nota:"),
    ("# TODO:", "# POR HACER:"),
    ("# FIXME:", "# CORREGIR:"),
    ("# WARNING:", "# ADVERTENCIA:"),
    ("# IMPORTANT:", "# IMPORTANTE:"),
    ("# Example:", "# Ejemplo:"),
    ("# See:", "# Ver:"),
    ("# Ref:", "# Ref:"),
    ("# Source:", "# Fuente:"),

    # Number/position related
    ("# Number of", "# Número de"),
    ("# number of", "# número de"),
    ("# Size of", "# Tamaño de"),
    ("# size of", "# tamaño de"),
    ("# Length of", "# Longitud de"),
    ("# length of", "# longitud de"),
    ("# Index of", "# Índice de"),
    ("# index of", "# índice de"),

    # Data-related
    ("# Missing values", "# Valores faltantes"),
    ("# missing values", "# valores faltantes"),
    ("# Null values", "# Valores nulos"),
    ("# null values", "# valores nulos"),
    ("# Empty values", "# Valores vacíos"),
    ("# empty values", "# valores vacíos"),
    ("# Default value", "# Valor por defecto"),
    ("# default value", "# valor por defecto"),

    # ML-related
    ("# Learning rate", "# Tasa de aprendizaje"),
    ("# learning rate", "# tasa de aprendizaje"),
    ("# Batch size", "# Tamaño de lote"),
    ("# batch size", "# tamaño de lote"),
    ("# Hidden layers", "# Capas ocultas"),
    ("# hidden layers", "# capas ocultas"),
    ("# Hidden layer", "# Capa oculta"),
    ("# hidden layer", "# capa oculta"),
    ("# Output layer", "# Capa de salida"),
    ("# output layer", "# capa de salida"),
    ("# Input layer", "# Capa de entrada"),
    ("# input layer", "# capa de entrada"),
    ("# Activation function", "# Función de activación"),
    ("# activation function", "# función de activación"),
    ("# Loss function", "# Función de Loss"),
    ("# loss function", "# función de Loss"),
    ("# Gradient", "# Gradiente"),
    ("# gradient", "# gradiente"),
    ("# Zero gradients", "# Reiniciar gradientes"),
    ("# zero gradients", "# reiniciar gradientes"),
    ("# Early stopping", "# Parada temprana"),
    ("# early stopping", "# parada temprana"),
    ("# Overfitting", "# Sobreajuste"),
    ("# overfitting", "# sobreajuste"),
    ("# Underfitting", "# Subajuste"),
    ("# underfitting", "# subajuste"),
    ("# Regularization", "# Regularización"),
    ("# regularization", "# regularización"),
    ("# Dropout", "# Dropout"),
    ("# dropout", "# dropout"),
    ("# Tensor", "# Tensor"),
    ("# tensor", "# tensor"),
]


def translate_code_comment(line):
    """Translate a code comment line from English to Spanish."""
    # Find if this line has a comment
    stripped = line.lstrip()

    # Full-line comment
    if stripped.startswith('#'):
        indent = line[:len(line) - len(stripped)]

        # Try phrase-based translations
        for eng, esp in COMMENT_TRANSLATIONS:
            if stripped.startswith(eng) or stripped == eng.lstrip():
                rest = stripped[len(eng):]
                return indent + esp + rest + ("\n" if line.endswith("\n") and not (esp + rest).endswith("\n") else "")

        # If no match, try to translate common English words in the comment
        comment_text = stripped[1:].strip()  # Remove # and leading space
        if comment_text:
            translated = translate_comment_text(comment_text)
            has_newline = line.endswith("\n")
            result = indent + "# " + translated
            if has_newline and not result.endswith("\n"):
                result += "\n"
            elif not has_newline and result.endswith("\n"):
                result = result.rstrip("\n")
            return result
        return line

    # Inline comment (code followed by # comment)
    # Find the last # that's not inside a string
    code_part, comment_part = split_inline_comment(line)
    if comment_part is not None:
        translated_comment = translate_code_comment("# " + comment_part)
        # Remove the "# " we added and any trailing newline from the translated part
        translated_comment = translated_comment.strip()
        if translated_comment.startswith("# "):
            translated_comment = translated_comment[2:]
        has_newline = line.endswith("\n")
        result = code_part + "# " + translated_comment
        if has_newline and not result.endswith("\n"):
            result += "\n"
        return result

    return line


def split_inline_comment(line):
    """Split a line into code and inline comment parts. Returns (code, comment) or (line, None)."""
    in_single_quote = False
    in_double_quote = False
    i = 0
    text = line.rstrip('\n')
    while i < len(text):
        ch = text[i]
        if ch == '\\' and (in_single_quote or in_double_quote):
            i += 2
            continue
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif ch == '#' and not in_single_quote and not in_double_quote:
            # Check if this is an inline comment (has code before it)
            code_before = text[:i].strip()
            if code_before and not code_before.startswith('#'):
                comment_text = text[i+1:].strip()
                return text[:i], comment_text
        i += 1
    return line, None


# Additional comment text translations (for free-form comment text)
COMMENT_TEXT_TRANSLATIONS = [
    (r'\bLoad\b', 'Cargar'),
    (r'\bload\b', 'cargar'),
    (r'\bCalculate\b', 'Calcular'),
    (r'\bcalculate\b', 'calcular'),
    (r'\bCreate\b', 'Crear'),
    (r'\bcreate\b', 'crear'),
    (r'\bTrain\b', 'Entrenar'),
    (r'\btrain\b', 'entrenar'),
    (r'\bPlot\b', 'Graficar'),
    (r'\bplot\b', 'graficar'),
    (r'\bCheck\b', 'Verificar'),
    (r'\bcheck\b', 'verificar'),
    (r'\bConvert\b', 'Convertir'),
    (r'\bconvert\b', 'convertir'),
    (r'\bPrint\b', 'Imprimir'),
    (r'\bprint\b', 'imprimir'),
    (r'\bSet\b', 'Establecer'),
    (r'\bset\b', 'establecer'),
    (r'\bInitialize\b', 'Inicializar'),
    (r'\binitialize\b', 'inicializar'),
    (r'\bDefine\b', 'Definir'),
    (r'\bdefine\b', 'definir'),
    (r'\bImport\b', 'Importar'),
    (r'\bimport\b', 'importar'),
    (r'\bPrepare\b', 'Preparar'),
    (r'\bprepare\b', 'preparar'),
    (r'\bSplit\b', 'Dividir'),
    (r'\bsplit\b', 'dividir'),
    (r'\bNormalize\b', 'Normalizar'),
    (r'\bnormalize\b', 'normalizar'),
    (r'\bEvaluate\b', 'Evaluar'),
    (r'\bevaluate\b', 'evaluar'),
    (r'\bPredict\b', 'Predecir'),
    (r'\bpredict\b', 'predecir'),
    (r'\bSave\b', 'Guardar'),
    (r'\bsave\b', 'guardar'),
    (r'\bDisplay\b', 'Mostrar'),
    (r'\bdisplay\b', 'mostrar'),
    (r'\bShow\b', 'Mostrar'),
    (r'\bshow\b', 'mostrar'),
    (r'\bVisualize\b', 'Visualizar'),
    (r'\bvisualize\b', 'visualizar'),
    (r'\bRun\b', 'Ejecutar'),
    (r'\brun\b', 'ejecutar'),
    (r'\bApply\b', 'Aplicar'),
    (r'\bapply\b', 'aplicar'),
    (r'\bRemove\b', 'Eliminar'),
    (r'\bremove\b', 'eliminar'),
    (r'\bFilter\b', 'Filtrar'),
    (r'\bfilter\b', 'filtrar'),
    (r'\bSort\b', 'Ordenar'),
    (r'\bsort\b', 'ordenar'),
    (r'\bSelect\b', 'Seleccionar'),
    (r'\bselect\b', 'seleccionar'),
    (r'\bExtract\b', 'Extraer'),
    (r'\bextract\b', 'extraer'),
    (r'\bGenerate\b', 'Generar'),
    (r'\bgenerate\b', 'generar'),
    (r'\bBuild\b', 'Construir'),
    (r'\bbuild\b', 'construir'),
    (r'\bCompile\b', 'Compilar'),
    (r'\bcompile\b', 'compilar'),
    (r'\bTransform\b', 'Transformar'),
    (r'\btransform\b', 'transformar'),
    (r'\bUpdate\b', 'Actualizar'),
    (r'\bupdate\b', 'actualizar'),
    (r'\bCompute\b', 'Calcular'),
    (r'\bcompute\b', 'calcular'),
    (r'\bOptimize\b', 'Optimizar'),
    (r'\boptimize\b', 'optimizar'),
    (r'\bStore\b', 'Almacenar'),
    (r'\bstore\b', 'almacenar'),
    (r'\bAppend\b', 'Agregar'),
    (r'\bappend\b', 'agregar'),
    (r'\bInsert\b', 'Insertar'),
    (r'\binsert\b', 'insertar'),
    (r'\bDelete\b', 'Eliminar'),
    (r'\bdelete\b', 'eliminar'),
    (r'\bClear\b', 'Limpiar'),
    (r'\bclear\b', 'limpiar'),
    (r'\bReset\b', 'Reiniciar'),
    (r'\breset\b', 'reiniciar'),
    (r'\bEnable\b', 'Habilitar'),
    (r'\benable\b', 'habilitar'),
    (r'\bDisable\b', 'Deshabilitar'),
    (r'\bdisable\b', 'deshabilitar'),
    (r'\bConfigure\b', 'Configurar'),
    (r'\bconfigure\b', 'configurar'),
    (r'\bProcess\b', 'Procesar'),
    (r'\bprocess\b', 'procesar'),
    (r'\bAnalyze\b', 'Analizar'),
    (r'\banalyze\b', 'analizar'),
    (r'\bCompare\b', 'Comparar'),
    (r'\bcompare\b', 'comparar'),
    (r'\bValidate\b', 'Validar'),
    (r'\bvalidate\b', 'validar'),
    (r'\bTest\b', 'Probar'),
    (r'\btest\b', 'probar'),
    (r'\bVerify\b', 'Verificar'),
    (r'\bverify\b', 'verificar'),
    (r'\bAssign\b', 'Asignar'),
    (r'\bassign\b', 'asignar'),
    (r'\bMap\b', 'Mapear'),
    (r'\bmap\b', 'mapear'),
    (r'\bRead\b', 'Leer'),
    (r'\bread\b', 'leer'),
    (r'\bWrite\b', 'Escribir'),
    (r'\bwrite\b', 'escribir'),
    (r'\bFetch\b', 'Obtener'),
    (r'\bfetch\b', 'obtener'),
    (r'\bGet\b', 'Obtener'),
    (r'\bget\b', 'obtener'),
    (r'\bthe data\b', 'los datos'),
    (r'\bthe model\b', 'el modelo'),
    (r'\bthe results\b', 'los resultados'),
    (r'\bthe output\b', 'la salida'),
    (r'\bthe input\b', 'la entrada'),
    (r'\bthe features\b', 'las features'),
    (r'\bthe target\b', 'el objetivo'),
    (r'\bthe predictions\b', 'las predicciones'),
    (r'\bthe weights\b', 'los pesos'),
    (r'\bthe parameters\b', 'los parámetros'),
    (r'\bthe loss\b', 'la pérdida'),
    (r'\bthe network\b', 'la red'),
    (r'\bthe layers\b', 'las capas'),
    (r'\bthe layer\b', 'la capa'),
    (r'\bthe function\b', 'la función'),
    (r'\bthe values\b', 'los valores'),
    (r'\bthe columns\b', 'las columnas'),
    (r'\bthe column\b', 'la columna'),
    (r'\bthe rows\b', 'las filas'),
    (r'\bthe row\b', 'la fila'),
    (r'\bthe index\b', 'el índice'),
    (r'\bthe file\b', 'el archivo'),
    (r'\bthe path\b', 'la ruta'),
    (r'\bthe dataset\b', 'el dataset'),
    (r'\bthe table\b', 'la tabla'),
    (r'\bthe signal\b', 'la señal'),
    (r'\bthe signals\b', 'las señales'),
    (r'\bthe strategy\b', 'la estrategia'),
    (r'\bthe price\b', 'el precio'),
    (r'\bthe prices\b', 'los precios'),
    (r'\bthe returns\b', 'los retornos'),
    (r'\bthe return\b', 'el retorno'),
    (r'\bdata\b', 'datos'),
    (r'\bmodel\b', 'modelo'),
    (r'\bresults\b', 'resultados'),
    (r'\bfeatures\b', 'features'),
    (r'\btraining\b', 'entrenamiento'),
    (r'\btesting\b', 'prueba'),
    (r'\bvalidation\b', 'validación'),
    (r'\bpredictions\b', 'predicciones'),
    (r'\bweights\b', 'pesos'),
    (r'\bparameters\b', 'parámetros'),
    (r'\bloss\b', 'pérdida'),
    (r'\bnetwork\b', 'red'),
    (r'\blayer\b', 'capa'),
    (r'\bfunction\b', 'función'),
    (r'\bvalues\b', 'valores'),
    (r'\bcolumns\b', 'columnas'),
    (r'\brows\b', 'filas'),
    (r'\bindex\b', 'índice'),
    (r'\bfile\b', 'archivo'),
    (r'\bdataset\b', 'dataset'),
    (r'\bstrategy\b', 'estrategia'),
    (r'\bsignal\b', 'señal'),
    (r'\bsignals\b', 'señales'),
    (r'\bprice\b', 'precio'),
    (r'\bprices\b', 'precios'),
    (r'\breturns\b', 'retornos'),
    (r'\band\b', 'y'),
    (r'\bfor\b', 'para'),
    (r'\bwith\b', 'con'),
    (r'\bfrom\b', 'de'),
    (r'\binto\b', 'en'),
    (r'\beach\b', 'cada'),
    (r'\ball\b', 'todos'),
    (r'\bnew\b', 'nuevo'),
    (r'\busing\b', 'usando'),
    (r'\bbased on\b', 'basado en'),
    (r'\bin order to\b', 'para'),
    (r'\bas a\b', 'como un'),
    (r'\bif necessary\b', 'si es necesario'),
    (r'\bif needed\b', 'si es necesario'),
]


def translate_comment_text(text):
    """Translate free-form comment text."""
    for pattern, replacement in COMMENT_TEXT_TRANSLATIONS:
        text = re.sub(pattern, replacement, text)
    return text


def translate_code_cell(source_lines):
    """Translate comments in a code cell's source lines."""
    result = []
    in_multiline_string = False
    string_delimiter = None

    for line in source_lines:
        # Track multi-line strings (triple quotes)
        stripped = line.strip().rstrip('\n')

        if not in_multiline_string:
            # Check if this line starts a multi-line string
            if '"""' in stripped or "'''" in stripped:
                delimiter = '"""' if '"""' in stripped else "'''"
                count = stripped.count(delimiter)
                if count == 1:
                    in_multiline_string = True
                    string_delimiter = delimiter
                # If count >= 2, it opens and closes on same line

            # Only translate if not in a multi-line string
            if not in_multiline_string or '"""' in stripped or "'''" in stripped:
                # Check if it's a comment line (not inside a string)
                line_stripped = line.lstrip()
                if line_stripped.startswith('#'):
                    result.append(translate_code_comment(line))
                else:
                    # Check for inline comments
                    _, comment = split_inline_comment(line)
                    if comment is not None:
                        result.append(translate_code_comment(line))
                    else:
                        result.append(line)
            else:
                result.append(line)
        else:
            # Inside multi-line string - don't translate
            if string_delimiter in stripped:
                in_multiline_string = False
                string_delimiter = None
            result.append(line)

    return result


def translate_notebook(path):
    """Main function: load, translate, save notebook."""
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            cell['source'] = translate_markdown_cell(cell['source'])
        elif cell['cell_type'] == 'code':
            cell['source'] = translate_code_cell(cell['source'])

    with open(path, 'w', encoding='utf-8', newline='') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Successfully translated {path}")
    print(f"Total cells processed: {len(nb['cells'])}")


if __name__ == '__main__':
    translate_notebook(NOTEBOOK_PATH)
