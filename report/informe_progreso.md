# Informe de Progreso - Práctica 2: Minería de Contenido Web

## Estado Actual del Proyecto

Este documento resume el progreso realizado hasta la fecha en la Práctica 2 de Minería Web.

### 1. Configuración del Proyecto (Completado)

Se ha completado la configuración inicial del entorno de trabajo. Esto incluye:

- Inicialización del repositorio Git.
- Creación de la estructura de directorios del proyecto (`data`, `src`, `notebooks`, `results`, `report`).
- Configuración del entorno virtual de Python (`.venv`).
- Instalación de las librerías necesarias (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`) y generación del archivo `requirements.txt`.
- Creación de los archivos iniciales de documentación (`PLAN.md`, `TODO.md`, `.gitignore`).

### 2. Carga y Exploración Inicial de Datos (Completado)

Se ha realizado un Análisis Exploratorio de Datos (EDA) sobre el conjunto de datos reducido proporcionado.

- **Conjunto de datos:** `news_reducido.csv` (ubicado en el directorio `data`).
- **Tamaño:** 10,000 artículos de noticias.
- **Script EDA:** `src/01_eda.py`.
- **Hallazgos Principales:**
    - **Columnas:** Se identificaron 9 columnas, incluyendo `category`, `headline`, `text`, `short_description`.
    - **Categorías:** El dataset contiene 4 categorías únicas: 'POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL'.
    - **Balance de Clases:** El conjunto de datos está perfectamente balanceado, con 2500 artículos por categoría (ver `results/category_distribution.png`).
    - **Valores Faltantes:** Se detectaron valores faltantes en las columnas:
        - `text`: 41 valores faltantes.
        - `short_description`: 469 valores faltantes.
        - `authors`: 787 valores faltantes.
    - **Análisis de Campos de Texto:** Se analizaron las longitudes de `headline`, `text` y `short_description` (ver `results/text_length_distributions.png`). El campo `text` es el más extenso y contiene la mayor cantidad de información.
- **Decisión Campo de Texto:** Basado en el análisis y las indicaciones del guion (`guion.md`), se ha decidido utilizar principalmente el campo **`text`** para las tareas de clustering y clasificación.

### 3. Próximos Pasos: Preprocesamiento

La siguiente fase del proyecto se centrará en el preprocesamiento del campo `text`. Las tareas incluyen:

- Implementar funciones de limpieza de texto (eliminar caracteres especiales, HTML si existiera, etc.).
- Manejar los valores faltantes encontrados en el campo `text`.
- Crear un pipeline de tokenización.
- Implementar la eliminación de *stopwords*.
- Implementar *stemming* o lematización.
- Crear funciones de vectorización (Binaria, Frecuencia, TF-IDF).
- Construir y probar el pipeline completo de preprocesamiento con `scikit-learn`.

Este informe se actualizará a medida que se completen nuevas fases del proyecto. 