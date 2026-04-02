import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- PREPARACIÓN ---
def preparar_datos(df, columnas=None):
    """
    Paso: 'Configurar parámetros'.
    Si no se pasan columnas, el sistema detecta automáticamente las numéricas.
    """
    # Si el usuario no eligió columnas, tomamos todas las que sean números
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()

    # Recibir datos del usuario  y limpiar solo las necesarias
    df_clean = df.dropna(subset=columnas).copy()

    # Aplicar escalado estadístico
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_clean[columnas])

    return datos_escalados, df_clean, scaler, columnas

# --- GRÁFICA DEL CODO ---
def generar_grafica_codo(datos_escalados, k_max=10):
    """Calcula WCSS y retorna la gráfica para seleccionar K óptimo."""
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

# --- ALGORITMO: K-MEANS ---
def aplicar_kmeans(df_limpio, datos_escalados, n_clusters):
    """Ejecuta K-Means y asigna etiquetas a los datos."""
    modelo = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(datos_escalados)

    df_res = df_limpio.copy()
    df_res['Cluster'] = etiquetas.astype(str)

    score = silhouette_score(datos_escalados, etiquetas)
    return df_res, modelo, score

# --- ALGORITMO: JERÁRQUICO ---
def aplicar_jerarquico(df_limpio, datos_escalados, n_clusters, max_filas=10000):
    """Ejecuta Clustering Jerárquico con control de RAM."""
    if len(datos_escalados) > max_filas:
        indices = np.random.choice(len(datos_escalados), max_filas, replace=False)
        datos_proc = datos_escalados[indices]
        df_res = df_limpio.iloc[indices].copy()
    else:
        datos_proc = datos_escalados
        df_res = df_limpio.copy()

    modelo = AgglomerativeClustering(n_clusters=n_clusters)
    etiquetas = modelo.fit_predict(datos_proc)

    df_res['Cluster'] = etiquetas.astype(str)
    score = silhouette_score(datos_proc, etiquetas)
    return df_res, score

# ---  GRÁFICA: DISPERSIÓN ---
def generar_grafica_clusters(df_resultado, modelo_kmeans=None, scaler=None):
    """Genera el Scatter Plot interactivo con centroides si aplica."""
    fig = px.scatter(
        df_resultado, x='followers', y='popularity', color='Cluster',
        hover_data=['name'], template='plotly_dark',
        title="Visualización de Segmentos Resultantes"
    )

    if modelo_kmeans is not None and scaler is not None:
        centroides = scaler.inverse_transform(modelo_kmeans.cluster_centers_)
        fig.add_trace(go.Scatter(
            x=centroides[:, 0], y=centroides[:, 1], mode='markers',
            marker=dict(color='white', size=15, symbol='x'), name='Centroides'
        ))
    return fig

# ---  GRÁFICA: DENDROGRAMA ---
def generar_dendrograma(datos_escalados, muestra_max=100):
    """Genera el árbol jerárquico para análisis visual de distancias."""
    if len(datos_escalados) > muestra_max:
        indices = np.random.choice(len(datos_escalados), muestra_max, replace=False)
        datos_ready = datos_escalados[indices]
    else:
        datos_ready = datos_escalados

    Z = linkage(datos_ready, method='ward')
    fig = ff.create_dendrogram(datos_ready, colorscale=px.colors.qualitative.Prism)
    fig.update_layout(template='plotly_dark', title="Dendrograma de Agrupación Jerárquica")
    return fig


# ========================================================================
# SOFTWARE: CORA 
# EQUIPO DE DESARROLLO: DIA
# ------------------------------------------------------------------------
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