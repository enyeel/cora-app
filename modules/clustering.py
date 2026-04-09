import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ====================================================================
# Data Preparation Module
# ====================================================================
def preparar_datos(df, columnas=None):
    """
    Prepares data for clustering:
    - Detects numeric columns if not specified
    - Removes rows with missing values
    - Applies standardization using Z-score
    """
    # If no columns specified, automatically detect numeric columns
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()

    # Clean data and remove missing values for selected columns only
    df_clean = df.dropna(subset=columnas).copy()

    # Apply statistical scaling
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_clean[columnas])

    return datos_escalados, df_clean, scaler, columnas

# ====================================================================
# Elbow Method Visualization
# ====================================================================
def generar_grafica_codo(datos_escalados, k_max=10):
    """Computes WCSS for different K values and generates elbow curve."""
    wcss = []
    for i in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(datos_escalados)
        wcss.append(kmeans.inertia_)

    fig = px.line(
        x=list(range(1, k_max + 1)),
        y=wcss,
        markers=True,
        title='Método del Codo (Análisis de Error WCSS)',
        labels={'x': 'Número de Clusters (K)', 'y': 'WCSS (Inercia)'},
        template='plotly_dark'
    )
    return fig

# ====================================================================
# K-Means Clustering Implementation
# ====================================================================
def aplicar_kmeans(df_limpio, datos_escalados, n_clusters):
    """Executes K-Means algorithm and assigns cluster labels."""
    modelo = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(datos_escalados)

    df_res = df_limpio.copy()
    df_res['Cluster'] = (etiquetas + 1).astype(str)

    score = silhouette_score(datos_escalados, etiquetas, sample_size=10000, random_state=42)
    return df_res, modelo, score

# ====================================================================
# Hierarchical Clustering Implementation
# ====================================================================
def aplicar_jerarquico(df_limpio, datos_escalados, n_clusters, metodo_enlace='ward', max_filas=10000):
    """Executes Hierarchical Clustering with linkage method selection and memory optimization."""
    if len(datos_escalados) > max_filas:
        indices = np.random.choice(len(datos_escalados), max_filas, replace=False)
        datos_proc = datos_escalados[indices]
        df_res = df_limpio.iloc[indices].copy()
    else:
        datos_proc = datos_escalados
        df_res = df_limpio.copy()

    # Set linkage method and perform clustering
    modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=metodo_enlace)
    etiquetas = modelo.fit_predict(datos_proc)

    # Adjust cluster labels to start from 1 instead of 0
    df_res['Cluster'] = (etiquetas + 1).astype(str)

    # Maintain score calculation for consistency
    score = silhouette_score(datos_proc, etiquetas, sample_size=2500, random_state=42)

    return df_res, score

# ====================================================================
# Scatter Plot Visualization with Centroids
# ====================================================================
def generar_grafica_clusters(df_resultado, x_col, y_col, modelo_kmeans=None, scaler=None):
    """Generates interactive scatter plot with cluster centroids if applicable."""
    fig = px.scatter(
        df_resultado, x=x_col, y=y_col, color='Cluster',
        template='plotly_dark',
        title="Visualización de Segmentos Resultantes"
    )

    if modelo_kmeans is not None and scaler is not None:
        centroides = scaler.inverse_transform(modelo_kmeans.cluster_centers_)
        # Centroids are calculated for all features used in clustering.
        # Here we plot only the positions corresponding to selected X and Y columns.
        fig.add_trace(go.Scatter(
            x=centroides[:, 0], y=centroides[:, 1], mode='markers',
            marker=dict(color='white', size=15, symbol='x'), name='Centroids'
        ))
    return fig

# ====================================================================
# Dendrogram Visualization
# ====================================================================
def generar_dendrograma(datos_escalados, metodo_enlace='ward', muestra_max=100):
    """Generates hierarchical tree visualization with specified linkage method."""
    if len(datos_escalados) > muestra_max:
        indices = np.random.choice(len(datos_escalados), muestra_max, replace=False)
        datos_ready = datos_escalados[indices]
    else:
        datos_ready = datos_escalados

    # Use linkage method and generate dendrogram
    Z = linkage(datos_ready, method=metodo_enlace)

    # Pass linkage matrix to create_dendrogram for correct calculation
    fig = ff.create_dendrogram(
        datos_ready,
        colorscale=px.colors.qualitative.Prism,
        linkagefun=lambda x: linkage(x, method=metodo_enlace)
    )

    fig.update_layout(
        template='plotly_dark',
        title=f"Dendrograma de Agrupación (Método: {metodo_enlace.capitalize()})",
        height=600
    )
    return fig

# ====================================================================
# Cluster Profile Analysis via Parallel Coordinates
# ====================================================================
def generar_grafica_perfiles(df_resultado, columnas):
    """Generates parallel coordinates plot to compare cluster mean profiles."""
    # Convertimos Cluster a numérico temporalmente para la escala de color
    df_resultado['Cluster_Num'] = df_resultado['Cluster'].astype(int)

    fig = px.parallel_coordinates(
        df_resultado,
        dimensions=columnas,
        color="Cluster_Num",
        title="Análisis de Perfiles: Comparativa de Medias por Segmento",
        template='plotly_dark',
        color_continuous_scale=px.colors.qualitative.Prism
    )
    return fig


# ========================================================================
# CORA - Clustering and Analysis Software
# Development Team: DIA
# ========================================================================
# CRÉDITOS DE AUTORÍA (EXCLUSIVOS DE ESTE MÓDULO):
# -- Angel Emilio Gutierrez Lozano (st4341@utr.edu.mx)
# -- Abraham Fernando Garcia Buendia (st4316@utr.edu.mx)
# FECHA DE DESARROLLO: 2026-04-01
# ------------------------------------------------------------------------
# DECLARACIÓN DE ALCANCE:
# Los autores arriba mencionados son responsables únicamente del 
# diseño, lógica y desarrollo del módulo de CLUSTERING. 
# Este incluye: algoritmos K-Means/Jerárquico, visualizaciones de 
# dispersión, método del codo y generación de dendrogramas.
#
# Este trabajo es un proyecto escolar desarrollado en honor al 
# legado del TSU en Ciencia de Datos Área Inteligencia Artificial (CDIA).
# Quedan reservados los derechos de autoría intelectual.
# ========================================================================