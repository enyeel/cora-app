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
    df_res['Cluster'] = (etiquetas + 1).astype(str)

    score = silhouette_score(datos_escalados, etiquetas, sample_size=10000, random_state=42)
    return df_res, modelo, score

# --- ALGORITMO: JERÁRQUICO ---
def aplicar_jerarquico(df_limpio, datos_escalados, n_clusters, metodo_enlace='ward', max_filas=10000):
    """Ejecuta Clustering Jerárquico con control de RAM y selección de método."""
    if len(datos_escalados) > max_filas:
        indices = np.random.choice(len(datos_escalados), max_filas, replace=False)
        datos_proc = datos_escalados[indices]
        df_res = df_limpio.iloc[indices].copy()
    else:
        datos_proc = datos_escalados
        df_res = df_limpio.copy()

    # Agregamos el parámetro 'linkage' para cumplir con la maestra
    modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=metodo_enlace)
    etiquetas = modelo.fit_predict(datos_proc)

    # Ajuste: Sumamos +1 para que el conteo empiece en 1 y no en 0
    df_res['Cluster'] = (etiquetas + 1).astype(str)

    # Mantenemos tu configuración de score que no daba problemas
    score = silhouette_score(datos_proc, etiquetas, sample_size=2500, random_state=42)

    return df_res, score

# --- GRÁFICA: DISPERSIÓN ---
def generar_grafica_clusters(df_resultado, x_col, y_col, modelo_kmeans=None, scaler=None):
    """Genera el Scatter Plot interactivo con centroides si aplica."""
    fig = px.scatter(
        df_resultado, x=x_col, y=y_col, color='Cluster',
        template='plotly_dark',
        title="Visualización de Segmentos Resultantes"
    )

    if modelo_kmeans is not None and scaler is not None:
        centroides = scaler.inverse_transform(modelo_kmeans.cluster_centers_)
        # NOTA: Los centroides se calculan para TODAS las columnas usadas.
        # Aquí graficamos solo las posiciones 0 (X) y 1 (Y) correspondientes a las columnas elegidas.
        fig.add_trace(go.Scatter(
            x=centroides[:, 0], y=centroides[:, 1], mode='markers',
            marker=dict(color='white', size=15, symbol='x'), name='Centroides'
        ))
    return fig

# ---  GRÁFICA: DENDROGRAMA ---
def generar_dendrograma(datos_escalados, metodo_enlace='ward', muestra_max=100):
    """Genera el árbol jerárquico para análisis visual de distancias con método dinámico."""
    if len(datos_escalados) > muestra_max:
        indices = np.random.choice(len(datos_escalados), muestra_max, replace=False)
        datos_ready = datos_escalados[indices]
    else:
        datos_ready = datos_escalados

    # Cambiamos 'ward' por el parámetro metodo_enlace
    Z = linkage(datos_ready, method=metodo_enlace)

    # IMPORTANTE: Pasamos 'Z' (la matriz de enlace) al create_dendrogram
    # para que la gráfica use el método de cálculo correcto.
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

# --- GRÁFICA: PERFILES / MEDIAS  ---
def generar_grafica_perfiles(df_resultado, columnas):
    """
    Genera un gráfico de perfiles promediados con leyenda discreta.
    Transforma los datos para que cada cluster sea una serie independiente.
    """
    # 1. Agrupamos por cluster y calculamos el promedio de las variables
    df_medias = df_resultado.groupby('Cluster')[columnas].mean().reset_index()
    
    # 2. Transformamos el DataFrame para que Plotly pueda crear la leyenda discreta
    df_melt = df_medias.melt(id_vars='Cluster', var_name='Variable', value_name='Promedio')

    # 3. Ordenamos por Cluster (numéricamente) para que la leyenda sea 1, 2, 3...
    df_melt['Cluster_Sort'] = df_melt['Cluster'].astype(int)
    df_melt = df_melt.sort_values(['Cluster_Sort', 'Variable'])

    # 4. Creamos la gráfica de líneas (Perfil de Medias)
    fig = px.line(
        df_melt, 
        x='Variable', 
        y='Promedio', 
        color='Cluster', 
        markers=True,    
        title="Análisis de Perfiles: ADN de los Segmentos (Promedios por Cluster)",
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig.update_layout(
        legend_title_text='Segmento (Cluster)',
        xaxis_title="Variables Analizadas",
        yaxis_title="Valor Promedio (Escala Original)"
    )
    
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