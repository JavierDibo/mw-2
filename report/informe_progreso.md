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

### 3. Preprocesamiento (Completado)

Se ha completado la fase de preprocesamiento del campo `text`. Los pasos realizados y los resultados clave se encuentran implementados en el script `src/02_preprocessing.py`:

- **Manejo de Valores Faltantes:** Los 41 valores `NaN` en la columna `text` se reemplazaron por strings vacíos.
- **Limpieza de Texto:** Se aplicó una limpieza básica para convertir a minúsculas, eliminar signos de puntuación y números, y normalizar espacios en blanco.
- **Tokenización:** El texto limpio se dividió en tokens (palabras) individuales usando `nltk.word_tokenize`.
- **Eliminación de Stopwords:** Se eliminaron las palabras comunes en inglés (e.g., "the", "is", "in") utilizando la lista de `stopwords` de NLTK.
- **Stemming:** Se aplicó el algoritmo PorterStemmer (`nltk.stem.PorterStemmer`) para reducir las palabras a su raíz (e.g., "running" -> "run").
- **Pipelines de Scikit-learn:** Se encapsularon todos los pasos anteriores (limpieza, tokenización, stopwords, stemming) dentro de una función `stemming_tokenizer` personalizada.
- **Funciones de Vectorización:** Se crearon funciones (`create_binary_pipeline`, `create_frequency_pipeline`, `create_tfidf_pipeline`) que generan `Pipeline` de scikit-learn. Cada pipeline utiliza el `stemming_tokenizer` y aplica la vectorización correspondiente (Binaria, Frecuencia o TF-IDF) para transformar el texto crudo en una representación numérica.

El resultado de esta fase es un conjunto de herramientas reutilizables (los pipelines) que preparan los datos de texto para los algoritmos de machine learning.

### 4. Próximos Pasos: Clustering

La siguiente fase se centrará en aplicar técnicas de agrupamiento (clustering) sobre los datos preprocesados, siguiendo las indicaciones del `guion.md` y el `PLAN.md`:

- **Algoritmo k-means:**
    - Aplicar sobre las representaciones Binaria, Frecuencia y TF-IDF (K=4).
    - Analizar la sensibilidad a diferentes inicializaciones (semillas).
    - Seleccionar el mejor agrupamiento y guardar asignaciones.
    - Visualizar clusters usando t-SNE.
    - Realizar evaluación interna (cohesión, separación) y externa (comparando con categorías reales).
- **Algoritmo Gaussian Mixture:**
    - Aplicar sobre la representación TF-IDF (n_components=4).
    - Evaluar con las mismas métricas que k-means.
    - Comparar resultados con k-means.

Este informe se actualizará a medida que se completen nuevas fases del proyecto. 