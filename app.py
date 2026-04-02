import io
import streamlit as st
import pandas as pd
import numpy as np
from modules.cleaning import *

# =======================================================
# 🖥️ LA INTERFAZ DE USUARIO (UI) - BRANDING DIA
# =======================================================
# 1. Configuración de la pestaña del navegador
st.set_page_config(
    page_title="CORA | by DIA", 
    page_icon="☀️", 
    layout="wide"
)

# 2. El Menú Lateral (Sidebar) Corporativo
with st.sidebar:
    st.markdown("## CORA by ☀️ DIA")
    st.caption("**Data Intelligence & Analytics**")
    st.caption("📍 *Software desarrollado en el Bajío Valley*")
    st.divider()
    st.caption("© 2026 DIA. Todos los derechos reservados a los 9 fundadores.")

# 3. El Título Principal de la App
st.title("CORA Analysis")
st.markdown("*Powered by **DIA** - Algoritmos de vanguardia para datos impecables.*")
st.divider()

# La mochila de memoria
if 'datos_limpios' not in st.session_state:
    st.session_state['datos_limpios'] = None

# ==========================================
# PANTALLA PRINCIPAL: LIMPIEZA
# ==========================================
st.title("Limpieza de Datos")
st.markdown("*Módulo de procesamiento impulsado por **CORA**.*")

# 1. CAJITA DE CARGAR ARCHIVO
archivo_subido = st.file_uploader("Sube tu dataset sucio (CSV o Excel)", type=["csv", "xlsx"])

if archivo_subido is not None:
    df = pd.read_csv(
    io.BytesIO(archivo_subido.read()),
    encoding='latin1',
    engine='python'
)
    
    # ==========================================
    # DETECTRES DE ANOMALÍAS (Outliers y Webones)
    # ==========================================
    # 1. Corremos los detectores
    mapa_outliers, cols_con_outliers = detectar_outliers(df)
    filas_webones = detectar_webones(df)
    
    # 2. Calculamos las métricas para las tarjetitas
    total_filas = len(df)
    total_nulos = df.isna().sum().sum()
    total_outliers = mapa_outliers.sum().sum()
    total_webones = filas_webones.sum()

    # ==========================================
    # 🩻 SECCIÓN 1: RAYOS X (Visualización)
    # ==========================================
    st.header("1. Diagnóstico de Rayos X")
    st.markdown("CÓRTEX ha escaneado tu base de datos. Pasa el cursor sobre la tabla.")
    
    # Leyenda de colores
    st.markdown("🟥 **Rojo:** Nulos | 🟦 **Azul:** Outliers | 🟪 **Morado:** Filas de Varianza Nula (Webones)")

    # La función de Pintado Maestro
    def pintar_rayos_x(data):
        # Creamos un dataframe de puros strings vacíos para guardar los colores
        estilos = pd.DataFrame('', index=data.index, columns=data.columns)
        
        # Capa 1: Morado para filas completas de webones
        for idx in data.index[filas_webones]:
            estilos.loc[idx, :] = 'background-color: rgba(150, 75, 255, 0.3);'
            
        # Capa 2: Azul para celdas atípicas (Outliers)
        estilos[mapa_outliers] = 'background-color: rgba(75, 150, 255, 0.5); font-weight: bold;'
        
        # Capa 3: Rojo para celdas vacías (Nulos)
        estilos[data.isna()] = 'background-color: rgba(255, 75, 75, 0.6);'
        
        return estilos

    # Renderizamos la tabla con los colores aplicados
    st.dataframe(df.style.apply(pintar_rayos_x, axis=None), height=300, use_container_width=True)
    
    # Tarjetitas Inteligentes
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Filas", total_filas)
    col2.metric("Nulos Encontrados", total_nulos)
    col3.metric("Outliers Detectados", total_outliers)
    col4.metric("Usuarios Inválidos", total_webones)
    
    st.divider()

    # ==========================================
    # 🎯 SECCIÓN 2: CONFIGURACIÓN INTELIGENTE
    # ==========================================
    st.header("2. Tratamiento de Outliers")
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col_out_1, col_out_2 = st.columns([2, 1])
    with col_out_1:
        # ¡LA MAGIA AQUÍ! El select ya viene PRE-LLENADO con cols_con_outliers
        cols_outliers_elegidas = st.multiselect(
            "Columnas detectadas con anomalías:", 
            columnas_numericas, 
            default=cols_con_outliers # <-- CÓRTEX elige por el usuario
        )
        if not cols_con_outliers:
            st.success("¡Buenas noticias! No se detectaron outliers en ninguna columna.")
            
    with col_out_2:
        accion_outliers = st.radio("Acción a tomar:", ["Neutralizar (Convertir a NaN)", "Eliminar fila"], key="radio_out")

    st.divider()

    # 4. DETALLE Y CONFIGURACIÓN DE WEBONES (Straight-lining)
    st.header("3. Filtro de Varianza Nula (Straight-lining)")
    st.warning("Detecta usuarios que respondieron lo mismo en todas las preguntas de escala (Ej. 3,3,3,3).")
    columnas_todas = df.columns.tolist()
    
    col_web_1, col_web_2 = st.columns([2, 1])
    with col_web_1:
        cols_likert = st.multiselect("Selecciona las columnas de la encuesta (Likert):", columnas_todas)
    with col_web_2:
        accion_webones = st.radio("¿Qué hacemos con ellos?", ["Convertir sus respuestas a NaN", "Eliminar usuario"], index=0)

    st.divider()

    # 5. DETALLE Y CONFIGURACIÓN DE NULOS (Imputación)
    st.header("4. Imputación de Nulos")
    st.success("Configura cómo se rellenarán los vacíos (incluyendo los que generaron los pasos anteriores).")
    
    col_nul_1, col_nul_2 = st.columns(2)
    with col_nul_1:
        metodo_imputacion = st.selectbox("Método para variables numéricas:", ["Media (Promedio)", "Mediana", "Moda"])
    with col_nul_2:
        st.write(" ") # Espaciador
        st.checkbox("Forzar eliminación de filas si aún quedan nulos", value=False)

    st.divider()

    # BOTÓN MAESTRO DE EJECUCIÓN
    st.button("Ejecutar Limpieza y Estandarización", type="primary", use_container_width=True)

    # 6. VISTA FINAL ESTANDARIZADA
    st.header("5. Resultado Final (Datos AI-Ready)")
    st.markdown("Aquí se mostrará tu dataset limpio, codificado (One-Hot) y estandarizado (Z-Score), listo para descargar o exportar al módulo de Análisis.")
    
    # Placeholder visual (por ahora solo muestra el df original para que no se vea vacío)
    st.dataframe(df.head(10), use_container_width=True)