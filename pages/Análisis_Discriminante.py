
import streamlit as st
import pandas as pd
import numpy as np
from modules.discriminant import ejecutar_analisis_discriminante
from modules.layout import renderizar_df_paginado, render_sidebar

st.set_page_config(page_title="Análisis Discriminante", page_icon=None, layout="wide")

# Sidebar compartido
render_sidebar()

st.title("Análisis Discriminante")

# Verify clean data is in memory
if 'df_chido' not in st.session_state or st.session_state.df_chido is None:
    st.warning("⚠️ Primero carga y limpia los datos en la página principal.")
    st.stop()

metadata = st.session_state.metadata
df = st.session_state.df_chido

# Wrapper for caching analysis results
# Stores results in memory for instant retrieval with same parameters
@st.cache_data(show_spinner=False)
def ejecutar_analisis_cache(dataframe, objetivo, predictoras):
    return ejecutar_analisis_discriminante(dataframe, objetivo, predictoras)

# ====================================================================
# Target Variable Selection
# ====================================================================
st.subheader("Selecciona la variable a predecir (objetivo)")

# Allow selection from all available columns
todas_las_columnas = list(df.columns)

if len(todas_las_columnas) == 0:
    st.warning("⚠️ No se encontraron variables en el dataset.")
    st.stop()

# User guidance for target variable selection
st.info("💡 **Recomendación:** Elige la variable que define los **grupos o categorías** a clasificar en tu dataset. El Análisis Discriminante intentará predecir esta variable basándose en las demás.")

columna_objetivo = st.selectbox(
    "Variable dependiente (categórica / agrupador):",
    todas_las_columnas,
    key="disc_objetivo"
)

if columna_objetivo:
    distribucion = df[columna_objetivo].value_counts()
    
    # Visual validation for group definition
    if len(distribucion) > 20:
        st.warning(f"⚠️ Ojo: Esta variable tiene {len(distribucion)} grupos diferentes. El Análisis Discriminante suele volverse confuso o inestable con tantas categorías.")
    elif len(distribucion) < 2:
        st.error("🛑 Esta variable tiene menos de 2 grupos. Es imposible discriminar.")
        st.stop()
        
    st.write(f"**Grupos encontrados:** {len(distribucion)}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(distribucion.reset_index().rename(
            columns={'index': 'Grupo', columna_objetivo: 'Frecuencia'}
        ), use_container_width=True)
    with col2:
        st.write(f"**Variable:** `{columna_objetivo}`")
        st.write(f"**Tipo:** {df[columna_objetivo].dtype}")

st.markdown("---")

# ====================================================================
# Predictive Variables Selection
# ====================================================================
st.subheader("Selecciona las variables predictoras")

# ====================================================================
# Strict Variable Filtering: Numeric Only
# ====================================================================
columnas_numericas_validas = [
    col for col in df.columns
    if col in metadata 
    # Only allow continuous and discrete numeric types
    and metadata[col]["tipo"] in ["numerico_continuo", "numerico_discreto"] 
    and col != columna_objetivo
]

# Strict safety fallback
if not columnas_numericas_validas:
    columnas_numericas_validas = [
        col for col in df.select_dtypes(include=['number']).columns 
        if col != columna_objetivo and not col.lower().startswith("id")
    ]

# Stop if no numeric variables
if len(columnas_numericas_validas) < 1:
    st.error("No numeric variables found for use as predictors. This analysis requires continuous or discrete variables.")
    st.stop()

# ====================================================================
# Selection Mode
# ====================================================================

# UI for selection mode
modo_seleccion = st.radio(
    "Selection mode:",
    options=["Use all valid numeric variables", "Select manually"],
    horizontal=True
)

if modo_seleccion == "Use all valid numeric variables":
    variables_predictoras = columnas_numericas_validas
    st.info(f"✅ {len(variables_predictoras)} predictive variables will be used automatically (IDs and categorical variables excluded).")
    st.write(f"**Included variables:** {', '.join(variables_predictoras)}")
else:
    variables_predictoras = st.multiselect(
        "Select predictive variables:",
        options=columnas_numericas_validas,
        default=columnas_numericas_validas,
        help="Choose continuous or discrete variables to help classify the groups."
    )
    
    if len(variables_predictoras) < 2:
        st.warning("Recommendation: Select at least 2 predictive variables for optimal discriminant analysis.")

# ====================================================================
# Analysis Execution
# ====================================================================

with st.expander("Resumen de configuración"):
    st.write(f"**Variable objetivo:** `{columna_objetivo}`")
    st.write(f"**Variables predictoras:** {len(variables_predictoras)}")
    st.write(f"**Grupos a clasificar:** {df[columna_objetivo].nunique()}")
    st.write(f"**Total de casos:** {len(df)}")
    st.write(f"**Casos sin nulos:** {len(df[variables_predictoras + [columna_objetivo]].dropna())}")

# ====================================================================
# Change Detection and State Management
# ====================================================================
# Listen to official seal from main page
sello_oficial = st.session_state.get('sello_datos_confirmados', 'sin_sello')

# Create signature from seal + target + predictors
huella_actual = f"{sello_oficial}_{columna_objetivo}_{str(variables_predictoras)}"

if "disc_huella_df" not in st.session_state:
    st.session_state.disc_huella_df = huella_actual

# Detect changes to reset results
if st.session_state.disc_huella_df != huella_actual:
    if "disc_resultados" in st.session_state:
        del st.session_state["disc_resultados"]
    st.session_state.disc_huella_df = huella_actual

# Initialize results variable
if "disc_resultados" not in st.session_state:
    st.session_state.disc_resultados = None

# ====================================================================
# Execution Button and Results Processing
# ====================================================================
if st.button("Ejecutar análisis discriminante", type="primary", key="disc_ejecutar", disabled=not columna_objetivo or len(variables_predictoras) < 2):
    with st.spinner("Procesando análisis discriminante (puede tardar un momento)..."):
        # Call analysis function
        resultados = ejecutar_analisis_cache(
            df, 
            columna_objetivo, 
            variables_predictoras
        )
        # Store in session state for persistence
        st.session_state.disc_resultados = resultados


# ====================================================================
# Results Rendering
# ====================================================================
# Render if results exist in memory
if st.session_state.disc_resultados is not None:
    resultados = st.session_state.disc_resultados
    
    if "error" in resultados:
        st.error(resultados["error"])
    else:
        st.success("Análisis completado con éxito")
        st.markdown("---")
        
        st.subheader("Resultados del análisis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precisión Global", resultados["resumen"]["Precisión Global"])
        with col2:
            st.metric("Grupos", resultados["resumen"]["Número de Grupos"])
        with col3:
            st.metric("Bien Clasificados", resultados["resumen"]["Casos Bien Clasificados"])
        with col4:
            st.metric("Mal Clasificados", resultados["resumen"]["Casos Mal Clasificados"])
        
        # Convertimos la lista de tuplas en un diccionario para acceder fácil
        figuras_dict = dict(resultados["figuras"])

        # --- NUEVO EXPANDER: HISTORIAL DE CAMBIOS ---
        with st.expander("Historial de Cambios y Reagrupación"):
            st.write("Esta tabla muestra cómo se movieron los datos desde sus grupos originales hacia la clasificación final del modelo.")
            
            historial = resultados["historial_grupos"]
            st.dataframe(historial, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Análisis de movimientos:")
            
            # Lógica para explicar los cambios en lenguaje natural
            for original in historial.index:
                total_original = historial.loc[original].sum()
                # Ver cuántos se quedaron en su lugar (si la columna existe)
                quedaron = historial.loc[original, original] if original in historial.columns else 0
                movidos = total_original - quedaron
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(f"Grupo {original}", f"{total_original} iniciales")
                with col2:
                    if movidos == 0:
                        st.success(f"¡Perfecto! Los {total_original} se mantuvieron en su grupo.")
                    else:
                        st.warning(f"Se mantuvieron {quedaron} y se movieron {movidos} a otros grupos.")
                        
                        # Detalle de a dónde se fueron
                        otros = historial.loc[original][historial.loc[original] > 0]
                        for destino, cantidad in otros.items():
                            if destino != original:
                                st.write(f"• {cantidad} se pasaron al grupo **{destino}**")
        
        with st.expander("Funciones discriminantes"):
            st.dataframe(resultados["funciones_discriminantes"], width='stretch')
            if "Funciones Discriminantes" in figuras_dict:
                st.plotly_chart(figuras_dict["Funciones Discriminantes"], width='stretch')
        
        if "Variables Discriminantes" in figuras_dict:
            with st.expander("Variables más discriminantes"):
                st.plotly_chart(figuras_dict["Variables Discriminantes"], width='stretch')

        with st.expander("Centroides de grupos"):
            st.dataframe(resultados["centroides"], width='stretch')
            if "Perfil de Centroides" in figuras_dict:
                st.plotly_chart(figuras_dict["Perfil de Centroides"], width='stretch')
        
        with st.expander("Matriz de confusión"):
            st.dataframe(resultados["matriz_confusion"], width='stretch')
            if "Matriz de Confusión" in figuras_dict:
                st.plotly_chart(figuras_dict["Matriz de Confusión"], width='stretch')
        
        with st.expander("Test de Box M (homogeneidad de covarianzas)"):
            box_m = resultados["test_box_m"]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Estadístico M:** {box_m['estadistico_M']:.4f}" if not pd.isna(box_m['estadistico_M']) else "**Estadístico M:** No disponible")
                st.write(f"**Chi-cuadrado:** {box_m['chi2']:.4f}" if not pd.isna(box_m['chi2']) else "**Chi-cuadrado:** No disponible")
            with col2:
                st.write(f"**Grados de Libertad:** {box_m['grados_libertad']}")
                st.write(f"**Valor p:** {box_m['p_valor']:.6f}" if not pd.isna(box_m['p_valor']) else "**Valor p:** No disponible")
            st.write(f"**Interpretación:** {box_m['interpretacion']}")
        
        with st.expander("Clasificación por grupo"):
            df_clasificacion = pd.DataFrame(resultados["clasificacion_por_grupo"]).T
            st.dataframe(df_clasificacion, width='stretch')
            if "Precisión por Grupo" in figuras_dict:
                st.plotly_chart(figuras_dict["Precisión por Grupo"], width='stretch')
        
        if len(resultados["casos_mal_clasificados"]) > 0:
            with st.expander("Casos mal clasificados"):
                renderizar_df_paginado(resultados["casos_mal_clasificados"], page_size=10)
                # st.dataframe(resultados["casos_mal_clasificados"], width='stretch')
        else:
            st.success("¡Todos los casos fueron clasificados correctamente!")
        
        st.markdown("---")
