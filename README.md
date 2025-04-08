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

### 4. Clustering K-Means (Completado)

Se ha completado la ejecución y evaluación del algoritmo K-Means (con K=4) sobre las tres representaciones vectoriales (Binaria, Frecuencia, TF-IDF) utilizando el script `src/03_kmeans_clustering.py`. Los pasos realizados incluyen:

- **Análisis de Sensibilidad:** Se ejecutó K-Means con 5 semillas diferentes (0, 42, 123, 2024, 999) para cada representación, observando la variación en las métricas (especialmente ARI) para constatar la sensibilidad del algoritmo a la inicialización. Se confirmó que los resultados dependen de la semilla inicial.
- **Ejecución Principal (SEED=42):** Se realizó una ejecución final con `SEED=42` para obtener resultados reproducibles.
- **Guardado de Asignaciones:** Las asignaciones de cluster resultantes para la ejecución con `SEED=42` se guardaron en `results/kmeans_assignments_*.csv` para cada representación.
- **Visualización t-SNE:** Se generaron visualizaciones t-SNE para cada representación, coloreadas tanto por los clusters predichos (`results/tsne_kmeans_*.png`) como por las categorías reales (`results/tsne_true_categories_*.png`), facilitando la inspección visual de los resultados.
- **Evaluación Interna y Externa:** Se calcularon métricas de evaluación:
    - **Internas:** Silhouette Score, Davies-Bouldin Score.
    - **Externas:** Adjusted Rand Index (ARI), Homogeneidad, Completitud, V-measure.
- **Análisis de Resultados:** Se analizó el rendimiento para cada representación.
    - **Binaria y Frecuencia:** Mostraron bajo rendimiento general, con métricas intrínsecas y extrínsecas pobres, indicando clusters poco definidos y baja correspondencia con las categorías reales.
    - **TF-IDF:** A pesar de métricas intrínsecas bajas (Silhouette cercano a 0, Davies-Bouldin alto), las métricas extrínsecas (ARI=0.3301, V-measure=0.4237) fueron significativamente superiores, indicando que esta representación, aunque genera clusters menos separables geométricamente, captura mucho mejor la estructura temática real de las noticias para K-Means. Se concluyó que TF-IDF es la representación más efectiva para K-Means en este contexto.

### 5. Clustering Gaussian Mixture (Completado)

Siguiendo las indicaciones del guion y abordando el `MemoryError` inicial, se ejecutó el algoritmo Gaussian Mixture Model (GMM) en el script `src/03_kmeans_clustering.py`:

- **Representación:** Se utilizó exclusivamente la representación TF-IDF.
- **Reducción de Dimensionalidad:** Se aplicó `TruncatedSVD` a la matriz TF-IDF antes de GMM para evitar problemas de memoria. Se realizó un *tuning* previo (ver `src/04_gmm_svd_param_tuning.py`) probando `n_components` en [50, 100, 150, 200, 250]. **El valor `n_components=100` obtuvo los mejores resultados** (ARI=0.736, V-measure=0.694) y fue el seleccionado. La dimensionalidad final fue (10000, 100).
- **Ejecución GMM:** Se ajustó `GaussianMixture` con `n_components=4` (según guion) y `random_state=SEED` sobre los datos reducidos por SVD.
- **Evaluación:**
    - Se calcularon las métricas internas (Silhouette, Davies-Bouldin) sobre los datos **reducidos por SVD**.
    - Se calcularon las métricas externas (ARI, Homogeneidad, Completitud, V-measure) comparando las etiquetas predichas por GMM con las categorías reales.
- **Resultados (TF-IDF + SVD + GMM):**
    - Silhouette (SVD): 0.0148
    - Davies-Bouldin (SVD): 3.6438
    - **ARI: 0.7362**
    - **Homogeneidad: 0.6926**
    - **Completitud: 0.6956**
    - **V-measure: 0.6941**
- **Comparación K-Means vs GMM (TF-IDF):**
    - Basándose en las **métricas externas** (que son comparables), GMM aplicado sobre los datos TF-IDF reducidos con SVD (ARI=0.736) **superó notablemente** a K-Means aplicado sobre los datos TF-IDF completos (ARI=0.330). Esto sugiere que la combinación de SVD (para manejar la dimensionalidad) y GMM (con su mayor flexibilidad para modelar clusters) fue más efectiva para capturar la estructura de las categorías reales en este caso.
- **Guardado de Asignaciones:** Las asignaciones de GMM se guardaron en `results/gmm_assignments_TF-IDF_SVD.csv`.

### 6. Clasificación k-NN (Completado)

Se ha completado la ejecución y evaluación del algoritmo k-NN (`KNeighborsClassifier`) utilizando el script `src/05_classification.py`. Se usó validación cruzada estratificada de 5 folds y `GridSearchCV` para encontrar los mejores hiperparámetros (`n_neighbors`, `weights`, `p`) para cada representación vectorial.

- **Representaciones Probadas:** Binaria, Frecuencia, TF-IDF.
- **Labels:** Se utilizó `LabelEncoder` para convertir las categorías de texto a enteros, resolviendo errores previos.
- **Resultados y Parámetros Óptimos (`accuracy` media en 5-fold CV):
    - **Binaria:** 0.3081 (n_neighbors=3, p=1, weights=distance)
    - **Frecuencia:** **0.6239** (n_neighbors=3, p=2, weights=distance)
    - **TF-IDF:** 0.3297 (n_neighbors=3, p=2, weights=distance)
- **Análisis Preliminar k-NN:** La representación de **Frecuencia** obtuvo el mejor rendimiento para k-NN con una precisión del 62.4%. Las representaciones Binaria y TF-IDF mostraron resultados considerablemente más bajos para este algoritmo.
- **Modelos Guardados:** Los mejores pipelines (vectorizador + clasificador k-NN) para cada representación, junto con el `LabelEncoder`, se han guardado en el directorio `results/models/` utilizando `joblib` para su posible uso futuro.

### 7. Clasificación Naive Bayes (Completado)

Se ejecutaron los experimentos de clasificación con Naive Bayes utilizando el script `src/06_naive_bayes_classification.py`. Se probó `MultinomialNB` y `GaussianNB` con validación cruzada estratificada de 5 folds.

- **Representaciones Probadas:** Binaria, Frecuencia, TF-IDF.
- **`GaussianNB`:** Requiere datos densos. Se incluyó un paso de densificación en el pipeline. Se ejecutó con `n_jobs=2` para mitigar problemas de memoria, lo cual funcionó.
- **Resultados (`accuracy` media en 5-fold CV):
    - **Binaria:**
        - `MultinomialNB`: **0.9168**
        - `GaussianNB`: 0.8486
    - **Frecuencia:**
        - `MultinomialNB`: **0.9359**
        - `GaussianNB`: 0.8212
    - **TF-IDF:**
        - `MultinomialNB`: **0.9225**
        - `GaussianNB`: 0.8054
- **Análisis Naive Bayes:** `MultinomialNB` superó consistentemente a `GaussianNB` en todas las representaciones. El mejor rendimiento general se obtuvo con `MultinomialNB` y la representación de **Frecuencia** (93.6%). Los tiempos de ejecución fueron notablemente más rápidos que los de la búsqueda de hiperparámetros de k-NN.

### 8. Comparación k-NN vs Naive Bayes y Próximos Pasos

- **Comparación General Detallada:**
    - **Mejores Resultados:**
        - Mejor k-NN: Frecuencia, Accuracy = 0.6239 (Params: n=3, p=2, weights=distance)
        - Mejor Naive Bayes: Frecuencia (MultinomialNB), Accuracy = **0.9359**
    - **Superioridad de Naive Bayes (MultinomialNB):** `MultinomialNB` demostró ser un clasificador significativamente superior a k-NN para esta tarea, logrando una precisión mucho mayor (~93.6% vs ~62.4%). Esto se atribuye principalmente a:
        - *Manejo de Alta Dimensionalidad y Dispersión:* Naive Bayes maneja eficientemente el gran número de características (palabras) y la dispersión inherente de los datos textuales mediante su modelo probabilístico, mientras que k-NN sufre de la "maldición de la dimensionalidad" donde las distancias se vuelven menos significativas.
        - *Idoneidad del Modelo:* El modelo generativo de `MultinomialNB` basado en la frecuencia de palabras parece capturar mejor la señal temática que el enfoque basado en distancia de k-NN.
    - **Importancia de la Representación:**
        - *Frecuencia como Ganadora:* La representación de frecuencia simple (conteo de palabras tras preprocesamiento) proporcionó la mejor base para ambos algoritmos, aunque la diferencia fue mucho más pronunciada para `MultinomialNB`.
        - *TF-IDF:* Funcionó bien con `MultinomialNB` (92.3%) pero mal con k-NN (33.0%), posiblemente porque la normalización TF-IDF no mejoró la separabilidad por distancia para k-NN en este espacio de alta dimensión.
        - *Binaria:* Perdió demasiada información al ignorar la frecuencia, resultando en el peor o casi peor rendimiento para ambos clasificadores.
    - **Variantes de Naive Bayes:** `GaussianNB` tuvo un rendimiento notablemente inferior (~80-85%) debido a que su asunción de normalidad gaussiana para las características no se cumple en datos textuales. Además, la necesidad de densificar los datos aumentó su coste computacional sin mejorar su adecuación.
    - **Parámetros de k-NN:** El mejor k-NN consistentemente prefirió `k=3` y `weights='distance'`, sugiriendo que en este espacio de alta dimensión, solo los vecinos más inmediatos y cercanos proporcionan información fiable.
    - **Eficiencia Computacional:** Naive Bayes fue órdenes de magnitud más rápido de entrenar y evaluar que k-NN con `GridSearchCV`, destacando `MultinomialNB` por su eficiencia.
    - **Conclusión:** `MultinomialNB` con representación de **Frecuencia** es la combinación óptima de precisión y eficiencia para clasificar estas noticias con el preprocesamiento actual.
- **Próximos Pasos:**
    - **Documentación:** Expandir el análisis en la memoria final (PDF), incluyendo tablas de comparación detalladas y discusión de trade-offs.
    - **Competición Kaggle:** Abordar la competición utilizando el conjunto de datos completo, posiblemente explorando algoritmos más avanzados o técnicas de preprocesamiento/ingeniería de características adicionales basadas en estos hallazgos.

Este informe se actualizará a medida que se completen nuevas fases del proyecto. 