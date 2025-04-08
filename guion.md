## Grado en Ingeniería Informatica

## Minería Web

## Curso 2024/

# Practica 2: Minería de Contenido de la

# Web

```
Universidad de Jaen
```

## 1. Introduccion

La extraccion de informacion de documentos web (y de documentos en general)
representa hoy en día un sector de gran relevancia economica, con un impacto signi-
ficativo en multiplesambitos de nuestra sociedad. Esta practica, conocida como min-
ería web, ofrece soluciones innovadoras a una amplia gama de desafíos, mejorando
nuestra interaccion con la informacion digital de manera profunda y variada.

Entre sus aplicaciones mas destacadas se encuentra la capacidad de personalizar
recomendaciones de productos, películas, artículos y otros contenidos, basandose
en las preferencias y gustos individuales. Esto no solo enriquece la experiencia del
usuario si no que tambien optimiza la eficacia de las plataformas que implementan
tales recomendaciones. Ademas, la minería web facilita la segmentacion de audien-
cias para campanas publicitarias, permitiendo a las empresas dirigir sus esfuerzos de ̃
manera mas precisa y eficiente. Esta capacidad de segmentacion se traduce en cam-
panas m ̃ as exitosas, al alcanzar a aquellos individuos mas susceptibles a la propuesta
comercial presentada. La evaluacion de productos y servicios mediante el analisis de
comentarios y opiniones en la web es otra ventaja significativa. Y un largo etcetera
de aplicaciones.

## 2. Objetivo de la practica

La finalidad de la practica es por un lado la familiarizacion del estudiante con
tecnicas basicas de minería de datos en general, y en particular con aquellas cen-
tradas en la extraccion de conocimiento en documentos web. Se pide por tanto que el
estudiante sea capaz de aplicar las diferentes tecnicas de preprocesamiento de doc-
umentos vistas en clase, así como la aplicacion de diferentes tecnicas de clustering y
clasificacion tambien analizadas en clase.

## 3. Descripcion y Estructura de la Practica

**La practica puede realizarse de manera individual o en grupos de 2 personas
maximo.**

En esta practica se trabajara principalmente con**Python** mediante el empleo de la
libreríascikit-learn. No obstante, el estudiante es libre de utilizar otra herramienta de
minería de datos si así lo desea. Para el desarrollo de esta practica, nos apoyaremos
en el tutorialWorking With Text Data( **Nota:** el tutorial es para una version antigua


de scikit-learn, la 1.4, no obstante, la mayor parte de su funcionalidad debería de
funcionar para versiones futuras) de scikit-learn que nos proporciona un tutorial paso
a paso de procesamiento de texto. Se recomienda que el estudiante realice el tutorial
paso a paso, pero prestando especial atencion al uso de la herramientaPipeline.
Asimismo, se recomienda que se instale la libreríaPandasynumpypara la gestion
sencilla y eficiente de datos. Para instalar todas estas librerias, simplemente ejecuta
esta orden en un nuevo entorno virtual de Python:

pip install scikit-learn pandas numpy

La estructura de la practica se organiza en tres partes disenadas para profundizar ̃
en la comprension y aplicacion de tecnicas avanzadas de procesamiento de textos y
analisis de datos:

```
Agrupamiento de Documentos : Esta seccion requiere que los estudiantes apliquen
diversas tecnicas de agrupamiento discutidas durante el curso para llevar a
cabo un estudio experimental. Deberan implementar diferentes algoritmos de
clustering sobre un conjunto de documentos, analizar los resultados obtenidos
de cada metodo y, finalmente, elaborar conclusiones fundamentadas sobre la
efectividad y las peculiaridades de cada algoritmo utilizado.
```
```
Clasificacion de Documentos : En este modulo, utilizando el mismo conjun-
to de documentos, se espera que los estudiantes empleen distintos algoritmos
de clasificacion estudiados en el curso. Similar al ejercicio de agrupamiento, es
esencial que se realice un analisis detallado de los resultados de clasificacion,
extrayendo conclusiones significativas sobre la precision, eficacia, y las limita-
ciones de los metodos aplicados.
```
```
Competicion de Kaggle : Para estimular la cooperacion y el intercambio de
conocimientos entre los participantes, se propone un desafío en la plataforma
Kaggle. El objetivo es mejorar los modelos de clasificacion de documentos de-
sarrollados previamente. Los estudiantes tendran que experimentar con diver-
sas estrategias de minería de datos y ajuste de parametros para lograr el mejor
desempeno posible, promoviendo as ̃í un ambiente de aprendizaje competitivo
y colaborativo.
```
El **preprocesamiento** de los datos se destaca como un componente fundamen-
tal en cada fase del proyecto. Se enfatiza la importancia de transformar el texto en
tokensa traves de tecnicas como la**eliminacion de stopwords, stemming, y otras**


**metodologías revisadas en el curso**. Ademas, se anima a los estudiantes a experi-
mentar con distintas representaciones de los datos, tales como la representacion**bi-
naria** , por **frecuencia** , o **TF-IDF** , para evaluar como cada enfoque afecta la calidad
del analisis y el conocimiento extraído de los documentos.

En todos los estudios realizados, se llevara a cabo un estudio con**Validacion
Cruzada Estratificada de 5 folds (5-SCV)** , revise la documentacion de la claseStratifiedKFold
en scikit-learn para ver como realizarlo. **Se recomienda utilizar un valor de semilla
fijo para que el estudio sea reproducible**.

### 3.1. Datos utilizados

Se utilizara un conjunto de datos de noticias del periodico Huffington Post des-
de el ano 2012 a 2022. El conjunto de datos completo se compone de aproximada- ̃
mente 200.000 documentos relacionados con 42 categorías diferentes. Debido al gran
tamano del conjunto de datos, para el apartado de agrupamiento y clasificaci ̃ on obli-
gatorio, se trabajara con un conjunto reducido de unos 10.000 documentos (el 5 %
del total) de 4 categorías distintas. Sin embargo, para la competicion de Kaggle, se
trabajara obligatoriamente con el conjunto de datos completo con el objetivo de que
suponga un reto.

El conjunto de datos para esta competicion se compone de registros que incluyen
los siguientes atributos:

```
category: categoría en la que se publico el artículo ( es nuestra clase ).
```
```
headline: el titular del artículo de noticias.
```
```
text: es el texto principal de la noticia.
```
```
authors: lista de autores que contribuyeron al artículo.
```
```
link: enlace al artículo original.
```
```
shortdescription: resumen del artículo de noticias.
```
```
date: fecha de publicacion del artículo.
```
El alumno es libre de utilizar aquellas variables que considere relevantes de cara
a la realizacion de la practica, justificando las decisiones llevadas a cabo.


### 3.2. Agrupamiento

Se deben realizar diferentes experimentos de agrupamiento sobre la version re-
ducida del conjunto de datos. Estos experimentos consistiran en aplicar el algoritmo
k-meanssobre diferentes representaciones del texto: binaria, frecuencia y TF-IDF. Se
debe ignorar el atributo de clase. Utiliza el valor K=4. Para cada representacion uti-
lizada, contestar a las siguientes cuestiones:

1. Utiliza diferentes semillas de numeros aleatorios para que los centroides ini-
    ciales cambien de lugar. Observa como cambian los resultados. ¿Por que el al-
    goritmok-meanses tan sensible a los cambios de configuracion iniciales?
2. Busca el agrupamiento mas adecuado y almacena las asignaciones de los gru-
    pos en un fichero. Esta asignacion nos servira para realizar la validacion externa
    posterior.
3. Visualiza los clusters obtenidos. Apoyate en el algoritmo t-SNE para esta tarea.
4. Realiza un analilsis del clustering obtenido, haz tanto una evaluacion interna
    (con metricas de cohesion y separacion), como una evaluacion externa (usan-
    do los grupos calculados anteriormente y las clases). Analiza los resultados y
    extrae conclusiones sobre los mismos.
5. Ejecuta, solo para la representacion TF-IDF, el algoritmo de mezcla de gaus-
    sianas, el cual implementa el algoritmo EM (claseGaussianMixtureen scikit-
    learn) usando 4 componentes de mezcla (ncomponents=4). Compara el resul-
    tado obtenido con k-NN.

### 3.3. Clasificacion

Al igual que en el apartado anterior, se trabajara sobre la version reducida del
conjunto de datos. En este caso, el trabajo a realizar consistira en aplicar el algoritmo
k-NN(KNeighborsClassifier) yNa ̈ıve Bayes. Del mismo modo, se aplicaran estos
algoritmos con las diferentes representaciones (binaria, frecuencia y TF-IDF) siempre
que sea posible.

```
Contestar a las siguientes cuestiones:
```
```
Respecto a k-NN, para cada representacion utilizada:
```

- Prueba con diferentes valores dek, esquemas de pesos y valorpque indica
    la potencia de la distancia de Minkowski. Se recomienda analizar al menos
    5 combinaciones diferentes. Comenta los resultados obtenidos y analiza-
    los. Identifica los parametros que han obtenido la maxima precision.

```
Respecto a Na ̈ıve Bayes, para cada representacion utilizada:
```
- Ejecuta la version Gaussiana (GaussianNB) y la version Multinomial (MultinomialNB)
    y compara su rendimiento. Comenta los resultados obtenidos.

```
Finalmente, compara los resultados obtenidos por k-NN y Na ̈ıve Bayes y de-
termina cual es el mejor algoritmo y la mejor representacion atendiendo a la
precision obtenida. Considera tambien otros aspectos como el tiempo de ejecu-
cion o el consumo de memoria (entre otros) para enriquecer tu analisis.
```
**3.3.1. Competicion de Kaggle**

Kaggle es una plataforma de competiciones de ciencia de datos muy conocida que
se caracteriza por sus competicion con premios millonarios creados por diferentes
empresas. En esta practica se trabajara sobre una competicion privada en donde se
usa el conjunto de datos completo en un problema de clasficacion, lo que supone to-
do un reto. La informacion adicional y los detalles de la competicion se establecen
dentro de la competicion creada en Kaggle para tal efecto. El alumnado es completa-
mente libre de utilizar todas las estrategias y metodos que conozca para maximizar
el rendimiento de la clasificacion, teniendo que explicar cada paso llevado a cabo. Se
anima a que todo el mundo participe y se recuerda que es un problema muy difícil
que puede llevar a mucha frustracion. Se recuerda que el objetivo final de este aparta-
do es el de aprender nuevas tecnicas y favorecer la cooperacion y colaboracion, por
lo que no importa que el resultado no sea muy bueno.

Al final de la competicion el ganador recibira como premio**0.75 puntos** extra so-
bre la nota final, el segundo clasificado **0.5 puntos** y el tercero **0.25 puntos** , siempre y
cuando obtengan mejoras respecto a lo realizado en los apartados anteriores. **Los tres
ganadores deberan exponer en una presentacion de unos 10 minutos aproximada-
mente su desarrollo.**

```
El enlace de la competicion es este: Competicion de Kaggle.
```

## 4. Documentacion y entrega

```
Se enviara un fichero ZIP que contenga lo siguiente:
```
```
Memoria en formato PDF con los pasos llevados a cabo en cada tarea, las tablas
de resultados y el analisis realizado.
```
```
Si se usa Python, codigo fuente en Python de los estudios experimentales real-
izados.
```
## 5. Envío

La fecha tope de entrega sera el día **22 de abril a las 23:59** en la tarea de PLATEA
habilitada a tal efecto.


