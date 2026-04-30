# CORA: Plataforma Interactiva de Preprocesamiento y Análisis Estadístico

[](https://cora-app-c-dia.streamlit.app/)
[](https://www.python.org/)
[](https://cora-app-c-dia.streamlit.app/)

**[Prueba CORA en vivo aquí](https://cora-app-c-dia.streamlit.app/)**

## ¿Qué es CORA?

**CORA** es una plataforma web interactiva de ciencia de datos construida con Streamlit. Diseñada para agilizar el ciclo de vida de los datos, CORA automatiza el trabajo pesado del preprocesamiento (limpieza, imputación de nulos, manejo de *outliers* y estandarización) y permite a los usuarios ejecutar modelos estadísticos complejos con una interfaz visual, intuitiva y *AI-Ready*.

Transformamos *datasets* caóticos en *insights* accionables en cuestión de segundos, sin necesidad de escribir una sola línea de código.

-----

## Arquitectura y Módulos Principales

El sistema está construido bajo una arquitectura modular, separando la interfaz de usuario (*Frontend*) de la lógica matemática (*Backend*). Cada módulo de análisis fue desarrollado por especialistas del equipo:

### 1\. Motor de Carga y Limpieza Inteligente

  * **Autor(es):** [Angel] ([@enyeel](https://github.com/enyeel))
  * **Descripción:** Pipeline de ingesta de datos (.csv/.xlsx) que detecta automáticamente columnas inútiles, aplica imputación inteligente de valores nulos (media/mediana/moda), estandarización Z-Score, One-Hot Encoding para variables categóricas y filtra registros anómalos o "webones". Integra el estado de la aplicación mediante `st.session_state` para un flujo ininterrumpido.

### 2\. Estadística Descriptiva

  * **Autor(es):** [Carlos] ([@GithubCarlos](https://www.google.com/search?q=link)) y [Fani] ([@GithubFani](https://www.google.com/search?q=link))
  * **Descripción:** Análisis exploratorio de datos (EDA) automatizado. Genera medidas de tendencia central, dispersión, asimetría y tablas de frecuencia dinámicas. Incluye visualizaciones interactivas para entender la distribución inicial de los datos.

### 3\. Análisis Factorial Exploratorio (AFE)

  * **Autor(es):** [Karina] ([@GithubKarina](https://www.google.com/search?q=link)) y [Emiliano] ([@GithubEmiliano](https://www.google.com/search?q=link))
  * **Descripción:** Módulo de reducción de dimensionalidad. Ejecuta pruebas de viabilidad (KMO y Bartlett), elimina multicolinealidad dinámicamente y calcula cargas factoriales utilizando rotación Varimax. Incluye la visualización del *Scree Plot* y mapas de calor interactivos.

### 4\. Análisis Discriminante Lineal (LDA)

  * **Autor(es):** [Lora/Yara] ([@GithubYara](https://www.google.com/search?q=link)) y [Sebastian] ([@GithubSebas](https://www.google.com/search?q=link))
  * **Descripción:** Algoritmo supervisado para clasificación y evaluación de variables predictoras. Genera funciones discriminantes, calcula centroides por grupo, matrices de confusión y ejecuta el test de Box M para evaluar la homogeneidad de las covarianzas.

### 5\. Análisis de Conglomerados (Clustering)

  * **Autor(es):** Fernando ([@GithubFer](https://www.google.com/search?q=link)) y Emilio ([@GithubEmilio](https://www.google.com/search?q=link))
  * **Descripción:** Segmentación mediante IA no supervisada. Soporta modelos no jerárquicos (K-Means) apoyados por el Método del Codo, y algoritmos jerárquicos ilustrados mediante Dendrogramas interactivos. Evalúa la calidad de la agrupación a través del *Silhouette Score*.

-----

## Stack Tecnológico

  * **Frontend y Framework Web:** `Streamlit`
  * **Manipulación de Datos:** `Pandas`, `NumPy`
  * **Machine Learning y Estadística:** `Scikit-Learn`, `SciPy`, `factor_analyzer`
  * **Visualización de Datos:** `Plotly`, `Seaborn`, `Matplotlib`

-----

## Instalación y Despliegue Local

Si deseas clonar este repositorio y ejecutar CORA en tu entorno local, sigue estos pasos:

1.  Clona este repositorio:
    ```
    git clone https://github.com/enyeel/cora-app.git
    cd cora-app
    ```

3.  Instala las dependencias necesarias:
    ```
    pip install -r requirements.txt
    ```

4.  Inicia la aplicación:
    ```
    streamlit run app.py
    ```

-----

## Equipo de Desarrollo y Propiedad Intelectual

**CORA** fue diseñado y desarrollado con orgullo por el equipo **DIA**.

Este software es un proyecto integral desarrollado para demostrar la aplicación práctica de modelado estadístico, automatización de *pipelines* de datos y despliegue de aplicaciones web analíticas.

### Créditos de Autoría y Legado

Todo el código fuente, la arquitectura modular, el diseño de la interfaz y la integración matemática es **propiedad intelectual exclusiva de los autores y fundadores del equipo DIA.** Este trabajo consolida meses de desarrollo estructurado y se presenta en honor al legado del programa de Técnico Superior Universitario (TSU) en **Ciencia de Datos Área Inteligencia Artificial (CDIA)**.

Quedan estrictamente reservados los derechos de autoría intelectual y patrimonial. Ninguna entidad, ajena a los desarrolladores aquí listados, está autorizada para comercializar, adjudicarse la autoría o vender licencias de este software sin el consentimiento explícito del equipo desarrollador.

**💎 Puro pinche CDIA.**
