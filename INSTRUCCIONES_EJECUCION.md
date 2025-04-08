# Instrucciones para Configurar y Ejecutar el Experimento K-Means

Este documento describe los pasos para configurar el entorno y ejecutar el script de clustering K-Means (`src/03_kmeans_clustering.py`) en una nueva máquina.

## Prerrequisitos

- **Git:** Necesario para clonar el repositorio.
- **Python:** Versión 3.8 o superior recomendada. Asegúrate de que `python` y `pip` estén en el PATH del sistema.

## Pasos

1.  **Clonar el Repositorio:**
    Abre una terminal o consola y clona el repositorio (reemplaza `<URL_DEL_REPOSITORIO>` con la URL real):
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_DIRECTORIO_DEL_PROYECTO> # Entra al directorio recién clonado
    ```

2.  **Crear y Activar Entorno Virtual:**
    Es altamente recomendable usar un entorno virtual para aislar las dependencias del proyecto.
    ```bash
    python -m venv .venv
    ```
    Ahora, activa el entorno virtual. El comando varía según tu sistema operativo y shell:
    - **Windows (cmd.exe):**
      ```cmd
      .\.venv\Scripts\activate
      ```
    - **Windows (PowerShell):**
      ```powershell
      .\.venv\Scripts\Activate.ps1
      ```
      *(Nota: Si la ejecución de scripts está deshabilitada en PowerShell, puede que necesites ejecutar `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` primero)*
    - **Linux / macOS (bash/zsh):**
      ```bash
      source .venv/bin/activate
      ```
    Deberías ver `(.venv)` al principio de la línea de comandos si la activación fue exitosa.

3.  **Instalar Dependencias:**
    Con el entorno virtual activado, instala las librerías requeridas:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar Recursos NLTK:**
    El script necesita algunos datos de NLTK (tokenizador y stopwords). Ejecuta los siguientes comandos (dentro del entorno activado):
    ```bash
    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords
    ```

5.  **Ejecutar el Script K-Means:**
    El script se ejecuta desde el directorio `src`. Asegúrate de estar en el directorio raíz del proyecto (el que contiene `src`, `data`, etc.) y luego ejecuta:
    ```bash
    python src/03_kmeans_clustering.py
    ```
    *(Nota: Si encuentras problemas con la importación de `preprocessing` o el comando `python` no funciona directamente, asegúrate de que el entorno virtual esté activado correctamente. Como alternativa, podrías intentar ejecutar desde la raíz usando el path completo al python del venv, como hicimos en los pasos de depuración, pero la activación debería ser suficiente en una configuración manual estándar)*.

6.  **Observar la Salida:**
    - El script imprimirá en la consola el progreso, los resultados de la prueba de sensibilidad a las semillas, las métricas de evaluación (Silhouette, Davies-Bouldin, ARI, Homogeneity, Completeness, V-measure) para la ejecución con la semilla fija (`SEED=42`) para cada representación (Binaria, Frecuencia, TF-IDF).
    - **Importante:** El script puede tardar varios minutos en completarse, especialmente durante la vectorización y el cálculo de t-SNE.
    - **Posibles Errores de Memoria:** Durante nuestras pruebas, observamos errores de memoria al calcular la métrica Davies-Bouldin y al ejecutar t-SNE con la representación Binaria. Es posible que encuentres errores similares si la máquina no tiene suficiente RAM para manejar la matriz de datos densificada que requieren estas operaciones. El script está configurado para manejar estos errores e informar `-999` o un mensaje de error para esas métricas, pero continuará con el resto del proceso.

7.  **Ver Resultados:**
    Los archivos de salida se guardarán en el directorio `results`:
    - `kmeans_assignments_*.csv`: Archivos CSV con las asignaciones de cluster para cada documento.
    - `tsne_kmeans_*.png`: Gráficos t-SNE coloreados por los clusters predichos.
    - `tsne_true_categories_*.png`: Gráficos t-SNE coloreados por las categorías reales (para comparación visual).

¡Listo! Con estos pasos deberías poder replicar el entorno y ejecutar el experimento de clustering K-Means. 