import streamlit as st
import pandas as pd
import numpy as np
from modules.renderers import renderizar_df_paginado
from modules.clustering import (
    preparar_datos, generar_grafica_codo, aplicar_kmeans, 
    aplicar_jerarquico, generar_grafica_clusters, generar_dendrograma
)

st.set_page_config(page_title="Clustering | DIA", page_icon="🧩", layout="wide")

st.title("🧩 Análisis de Segmentación (Clustering)")
st.markdown("Agrupa tus datos automáticamente descubriendo patrones ocultos mediante IA no supervisada.")

# 1. VERIFICAR SI HAY DATOS (Usamos df_encoded por lo que platicamos)
if 'df_encoded' not in st.session_state:
    st.warning("⚠️ ¡Papi, espérate! No hay datos en memoria. Ve a la página principal, sube tu dataset y confírmalo.")
    st.stop()

df = st.session_state['df_encoded']

# 2. SECCIÓN A: CONFIGURACIÓN 
st.header("1. Configuración del Modelo ⚙️")
col1, col2 = st.columns(2)

with col1:
    # --- NUEVO FILTRO INTELIGENTE (VERSIÓN DICCIONARIO) ---
    columnas_disponibles = []
    metadata = st.session_state.get('metadata', None) 

    for col in df.columns:
        # 1. Verificamos si la columna es numérica
        es_numerica = pd.api.types.is_numeric_dtype(df[col])
        
        # 2. Verificamos en tu metadata (diccionario) que el 'tipo' NO sea 'id'
        es_id = False
        if metadata is not None and col in metadata:
            # Extraemos el valor de la llave 'tipo' para esa columna
            if metadata[col].get('tipo') == 'id':
                es_id = True
                
        # Si es un número real y NO está clasificada como ID, pasa
        if es_numerica and not es_id:
            columnas_disponibles.append(col)
    # --------------------------------

    # El multiselect ahora solo mostrará datos limpios
    cols_seleccionadas = st.multiselect(
        "Selecciona las variables numéricas a agrupar (Mínimo 2):",
        options=columnas_disponibles,
        default=columnas_disponibles[:2] if len(columnas_disponibles) >= 2 else columnas_disponibles
    )
    
    datos_escalados, df_clean, scaler, cols_procesadas = preparar_datos(df, cols_seleccionadas)

with col2:
    algoritmo = st.radio("Selecciona el Algoritmo:", ["K-Means (Rápido, Distancias)", "Jerárquico (Árbol, Agrupación)"])

# 4. SECCIÓN B: DIAGNÓSTICO DEL K IDEAL
st.header("2. Diagnóstico del Número Óptimo de Clusters 📊")
st.markdown("Observa la gráfica sugerida para decidir cuántos grupos (`K`) formar.")


if "K-Means" in algoritmo:
    fig_codo = generar_grafica_codo(datos_escalados)
    st.plotly_chart(fig_codo, width='stretch')
else:
    st.info("**Nota sobre el Dendrograma:** Para evitar que tu navegador colapse, esta gráfica muestra una muestra representativa de 100 registros. Las agrupaciones principales (colores) son precisas, aunque la forma exacta de las ramas varíe.")
    fig_dendro = generar_dendrograma(datos_escalados)
    st.plotly_chart(fig_dendro, width='stretch')

st.divider()

# 5. SECCIÓN C: EJECUCIÓN (CON PERSISTENCIA EN MEMORIA)
st.header("3. Ejecución del Modelo 🚀")

col_k, col_btn = st.columns([3, 1])
with col_k:
    k_elegido = st.slider("Basado en el diagnóstico arriba, ¿Cuántos grupos deseas crear?", min_value=2, max_value=15, value=3)

with col_btn:
    st.write("") # Espaciador para alinear el botón
    if st.button("🤖 ¡Ejecutar Agrupación!", type="primary", width='stretch'):
        
        with st.spinner("Calculando distancias espaciales..."):
            # Ejecutar el algoritmo correspondiente
            if "K-Means" in algoritmo:
                df_res, modelo, score = aplicar_kmeans(df_clean, datos_escalados, k_elegido)
                # Generamos gráfica parcheada
                fig_res = generar_grafica_clusters(df_res, cols_procesadas[0], cols_procesadas[1], modelo, scaler)
            else:
                df_res, score = aplicar_jerarquico(df_clean, datos_escalados, k_elegido)
                # Para jerárquico no hay centroides en Scikit-Learn
                fig_res = generar_grafica_clusters(df_res, cols_procesadas[0], cols_procesadas[1])
            
            # GUARDAMOS EN SESSION_STATE PARA VENCER LA AMNESIA
            st.session_state['cluster_resultados'] = {
                'df_final': df_res,
                'figura': fig_res,
                'score': score
            }

# 6. SECCIÓN D: RESULTADOS (Solo se muestra si hay algo guardado en memoria)
if 'cluster_resultados' in st.session_state:
    st.divider()
    st.header("4. Resultados de la Agrupación ✨")
    
    resultados = st.session_state['cluster_resultados']
    
    # Metricas
    st.metric(label="Calidad de la Agrupación (Silhouette Score)", value=f"{resultados['score']:.3f}", 
              help="Entre más cerca a 1.0, más definidos y separados están los grupos.")
    
    # Gráfica
    st.plotly_chart(resultados['figura'], width='stretch')
    
    # Tabla final y exportación
    with st.expander("Ver y Exportar Datos Agrupados"):
        # 1. Traemos a "El Chido" (Textos originales)
        df_humano = st.session_state['df_chido'].copy()
        
        # 2. Le pegamos la columna 'Cluster' que calculó nuestro modelo numérico
        df_humano['Cluster'] = resultados['df_final']['Cluster']
        
        # 3. Mostramos la tabla bonita
        renderizar_df_paginado(
            df_humano,
            height=350,
            page_size=200,
            key="cluster_resultados"
        )

        # 4. Exportamos el humano, no el encodeado
        csv_cluster = df_humano.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar Datos con Etiquetas de Cluster",
            data=csv_cluster,
            file_name="dataset_clusterizado_limpio.csv",
            mime="text/csv"
        )