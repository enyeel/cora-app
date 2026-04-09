# pages/Análisis_de_Clústers.py
import streamlit as st
import pandas as pd
import numpy as np
from modules.layout import renderizar_df_paginado, render_sidebar
from modules.clustering import (
    preparar_datos, generar_grafica_codo, aplicar_kmeans, 
    aplicar_jerarquico, generar_grafica_clusters, generar_dendrograma,
    generar_grafica_perfiles # 🔥 AQUÍ IMPORTAMOS LA GRÁFICA NUEVA
)

st.set_page_config(page_title="Clustering | DIA", page_icon=None, layout="wide")

# Sidebar compartido
render_sidebar()

st.title("Análisis de segmentación (clustering)")
st.markdown("Agrupa tus datos automáticamente descubriendo patrones ocultos mediante métodos no supervisados.")

# Data Verification
if 'df_encoded' not in st.session_state:
    st.warning("No hay datos en memoria. Vaya a la página principal, cargue su dataset y confirme para continuar.")
    st.stop()

df = st.session_state['df_encoded']

# Model Configuration
st.header("Model Configuration")
col1, col2 = st.columns(2)

with col1:
    # Strict metadata filtering (only real numeric types)
    metadata = st.session_state.get('metadata', {}) 
    columnas_disponibles = [
        col for col in df.columns
        if col in metadata 
        and metadata[col]["tipo"] in ["numerico_continuo", "numerico_discreto"]
    ]

    if len(columnas_disponibles) < 2:
        st.error("Insufficient numeric variables detected for clustering. Minimum 2 required.")
        st.stop()

    cols_seleccionadas = st.multiselect(
        "Selecciona las variables numéricas a agrupar (Mínimo 2):",
        options=columnas_disponibles,
        default=columnas_disponibles[:2]
    )
    
    if len(cols_seleccionadas) < 2:
        st.warning("⚠️ Selecciona al menos 2 variables para continuar.")
        st.stop()
    
    datos_escalados, df_clean, scaler, cols_procesadas = preparar_datos(df, cols_seleccionadas)

with col2:
    algoritmo = st.radio("Selecciona el Algoritmo:", ["K-Means (Rápido, Distancias)", "Jerárquico (Árbol, Agrupación)"])
    
    # Selector de método de enlace (solo para jerárquico)
    metodo_enlace_en = 'ward'
    if "Jerárquico" in algoritmo:
        opciones_jerarquico = {
            "Ward (Minimiza varianza)": "ward",
            "Completo (Max. distancia entre grupos)": "complete",
            "Promedio (Distancia media)": "average",
            "Simple (Min. distancia entre grupos)": "single"
        }
        enlace_es = st.selectbox(
            "Método de enlace:", 
            options=list(opciones_jerarquico.keys())
        )
        metodo_enlace_en = opciones_jerarquico[enlace_es]

# Model Configuration
st.header("Diagnóstico del número óptimo de clusters")
st.markdown("Observe la gráfica sugerida para decidir cuántos grupos (K) formar.")

if "K-Means" in algoritmo:
    fig_codo = generar_grafica_codo(datos_escalados)
    st.plotly_chart(fig_codo, use_container_width=True)
else:
    st.info("**Nota sobre el Dendrograma:** Para evitar que tu navegador colapse, esta gráfica muestra una muestra representativa de 100 registros. Las agrupaciones principales (colores) son precisas, aunque la forma exacta de las ramas varíe.")
    # Pass linkage method in English
    fig_dendro = generar_dendrograma(datos_escalados, metodo_enlace=metodo_enlace_en)
    st.plotly_chart(fig_dendro, use_container_width=True)

st.divider()

# ====================================================================
# Model Execution
# ====================================================================
st.header("Model Execution")

col_k, col_btn = st.columns([3, 1])
with col_k:
    # Límite lógico del slider según la cantidad de datos limpios
    max_k_posible = min(15, len(df_clean) - 1)
    
    if max_k_posible < 2:
        st.error("🛑 Tienes muy pocos datos sin valores nulos. Imposible agrupar.")
        st.stop()

    k_elegido = st.slider(
        "Basado en el diagnóstico arriba, ¿Cuántos grupos deseas crear?", 
        min_value=2, 
        max_value=max_k_posible, 
        value=min(3, max_k_posible)
    )

with col_btn:
    st.write("")
    if st.button("Ejecutar agrupación", type="primary", use_container_width=True):
        
        with st.spinner("Calculando distancias espaciales..."):
            if "K-Means" in algoritmo:
                df_res, modelo, score = aplicar_kmeans(df_clean, datos_escalados, k_elegido)
                fig_res = generar_grafica_clusters(df_res, cols_procesadas[0], cols_procesadas[1], modelo, scaler)
            else:
                # Pass selected linkage method
                df_res, score = aplicar_jerarquico(df_clean, datos_escalados, k_elegido, metodo_enlace=metodo_enlace_en)
                fig_res = generar_grafica_clusters(df_res, cols_procesadas[0], cols_procesadas[1])
            
            st.session_state['cluster_resultados'] = {
                'df_final': df_res,
                'figura': fig_res,
                'score': score
            }

# Change detection
sello_oficial = st.session_state.get('sello_datos_confirmados', 'sin_sello')
# Add linkage to signature for proper state reset on changes
huella_cluster = f"{sello_oficial}_{algoritmo}_{metodo_enlace_en}_{str(cols_seleccionadas)}_{k_elegido}"

if st.session_state.get("cluster_huella") != huella_cluster:
    if 'cluster_resultados' in st.session_state:
        del st.session_state['cluster_resultados']
    st.session_state.cluster_huella = huella_cluster

# ====================================================================
# Results Visualization
# ====================================================================
if 'cluster_resultados' in st.session_state:
    st.divider()
    st.header("Resultados de la agrupación")
    
    resultados = st.session_state['cluster_resultados']
    
    # Metricas
    st.metric(label="Clustering Quality (Silhouette Score)", value=f"{resultados['score']:.3f}", 
              help="Values closer to 1.0 indicate better defined and separated groups.")
    
    # Scatter plot
    st.plotly_chart(resultados['figura'], use_container_width=True)
    
    # Cluster profile visualization
    st.markdown("### Cluster Profiles")
    st.markdown("Review average variable behavior within each group to understand cluster characteristics.")
    fig_perfiles = generar_grafica_perfiles(resultados['df_final'], cols_procesadas)
    st.plotly_chart(fig_perfiles, use_container_width=True)
    
    # Tabla final y exportación
    with st.expander("Ver y exportar datos agrupados"):
        df_humano = st.session_state['df_chido'].copy()
        df_humano['Cluster'] = resultados['df_final']['Cluster']
        
        renderizar_df_paginado(
            df_humano,
            height=350,
            page_size=200,
            key="cluster_resultados_tabla"
        )

        csv_cluster = df_humano.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos con etiquetas de cluster",
            data=csv_cluster,
            file_name="dataset_clusterizado_limpio.csv",
            mime="text/csv"
        )