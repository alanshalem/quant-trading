#!/usr/bin/env python
"""
Script para convertir archivos .py a notebooks .ipynb usando nbconvert.

Uso:
    python convert_to_notebooks.py

Esto convertira todos los archivos strategy_*.py y video*.py a notebooks.
"""

import subprocess
import sys
from pathlib import Path


def convert_py_to_ipynb(py_file: Path) -> bool:
    """
    Convierte un archivo .py a .ipynb usando jupytext.

    Args:
        py_file: Path al archivo .py

    Returns:
        True si la conversion fue exitosa
    """
    try:
        # Metodo 1: Usando jupytext (recomendado)
        result = subprocess.run(
            ['jupytext', '--to', 'notebook', str(py_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"[OK] Convertido: {py_file.name}")
            return True
        else:
            print(f"[!] jupytext fallo para {py_file.name}, intentando metodo alternativo...")

    except FileNotFoundError:
        print("[!] jupytext no encontrado, intentando metodo alternativo...")

    # Metodo 2: Usando p2j o py2nb
    try:
        result = subprocess.run(
            ['p2j', str(py_file)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[OK] Convertido con p2j: {py_file.name}")
            return True
    except FileNotFoundError:
        pass

    # Metodo 3: Conversion manual usando nbformat
    try:
        import nbformat as nbf

        # Leer el archivo Python
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Crear notebook
        nb = nbf.v4.new_notebook()

        # Parsear el contenido
        cells = []
        current_cell = []
        current_type = 'code'
        in_markdown = False

        for line in content.split('\n'):
            # Detectar marcadores de celda
            if line.strip().startswith('# In['):
                # Guardar celda anterior
                if current_cell:
                    cell_content = '\n'.join(current_cell).strip()
                    if cell_content:
                        if current_type == 'markdown':
                            cells.append(nbf.v4.new_markdown_cell(cell_content))
                        else:
                            cells.append(nbf.v4.new_code_cell(cell_content))
                    current_cell = []
                    current_type = 'code'
                    in_markdown = False
            elif line.startswith('# #') or (line.startswith('# ') and len(line) > 2 and line[2:].strip().startswith('#')):
                # Markdown header
                if current_cell and current_type == 'code':
                    cell_content = '\n'.join(current_cell).strip()
                    if cell_content:
                        cells.append(nbf.v4.new_code_cell(cell_content))
                    current_cell = []
                current_type = 'markdown'
                # Remover '# ' del inicio para markdown
                current_cell.append(line[2:] if line.startswith('# ') else line[1:])
            elif line.startswith('# ') and current_type == 'markdown':
                current_cell.append(line[2:] if line.startswith('# ') else line[1:])
            else:
                if current_type == 'markdown' and current_cell:
                    # Terminar celda markdown
                    cell_content = '\n'.join(current_cell).strip()
                    if cell_content:
                        cells.append(nbf.v4.new_markdown_cell(cell_content))
                    current_cell = []
                    current_type = 'code'

                if not (line.strip().startswith('#!') and 'python' in line):
                    if not (line.strip() == '# coding: utf-8'):
                        current_cell.append(line)

        # Guardar ultima celda
        if current_cell:
            cell_content = '\n'.join(current_cell).strip()
            if cell_content:
                if current_type == 'markdown':
                    cells.append(nbf.v4.new_markdown_cell(cell_content))
                else:
                    cells.append(nbf.v4.new_code_cell(cell_content))

        nb['cells'] = cells

        # Escribir notebook
        output_path = py_file.with_suffix('.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)

        print(f"[OK] Convertido manualmente: {py_file.name}")
        return True

    except ImportError:
        print("[ERROR] nbformat no instalado. Instalar con: pip install nbformat")
        return False
    except Exception as e:
        print(f"[ERROR] Fallo conversion de {py_file.name}: {e}")
        return False


def main():
    """Funcion principal."""
    print("="*60)
    print("Conversion de archivos Python a Jupyter Notebooks")
    print("="*60 + "\n")

    # Directorio actual
    current_dir = Path(__file__).parent

    # Archivos a convertir
    patterns = ['strategy_*.py', 'video*.py']

    files_to_convert = []
    for pattern in patterns:
        files_to_convert.extend(current_dir.glob(pattern))

    if not files_to_convert:
        print("No se encontraron archivos para convertir.")
        return

    print(f"Archivos encontrados: {len(files_to_convert)}\n")

    successful = 0
    failed = 0

    for py_file in sorted(files_to_convert):
        if convert_py_to_ipynb(py_file):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Conversion completada: {successful} exitosos, {failed} fallidos")
    print("="*60)

    # Instrucciones adicionales
    print("""
Metodos de conversion disponibles (en orden de preferencia):

1. jupytext (recomendado):
   pip install jupytext
   jupytext --to notebook archivo.py

2. p2j:
   pip install p2j
   p2j archivo.py

3. Manual (usando nbformat):
   pip install nbformat

Para instalar todas las herramientas:
   pip install jupytext p2j nbformat
""")


if __name__ == '__main__':
    main()
