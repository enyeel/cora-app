import streamlit as st
import pandas as pd
import numpy as np
from modules.discriminant import ejecutar_analisis_discriminante

st.set_page_config(page_title="Análisis Discriminante", page_icon="🎯", layout="wide")

st.title("🎯 Análisis Discriminante")

# Verificar si hay datos limpios
if 'df_chido' not in st.session_state or st.session_state.df_chido is None:
    st.warning("⚠️ Primero carga y limpia los datos en la página principal.")
    st.stop()

df = st.session_state.df_chido

#  WRAPPER PARA CACHEAR LA FUNCIÓN DIOS 
# Esto guarda el resultado en memoria. Si metes los mismos datos, carga al instante.
@st.cache_data(show_spinner=False)
def ejecutar_analisis_cache(dataframe, objetivo, predictoras):
    return ejecutar_analisis_discriminante(dataframe, objetivo, predictoras)

# --- PASO 1: SELECCIONAR VARIABLE OBJETIVO ---
st.subheader("📌 Paso 1: Selecciona la variable a predecir (objetivo)")

def obtener_variables_objetivo_validas(dataframe):
    validas = []
    for col in dataframe.columns:
        nunique = dataframe[col].nunique()
        if 1 < nunique <= 20:
            if dataframe[col].value_counts().min() >= 2:
                validas.append(col)
    return validas

columnas_categoricas = obtener_variables_objetivo_validas(df)

if len(columnas_categoricas) == 0:
    st.warning("⚠️ No se encontraron variables válidas para usar como objetivo.")
    st.info("Requisitos: Entre 2 y 20 categorías, y al menos 2 datos por categoría.")
    st.stop()

columna_objetivo = st.selectbox(
    "Variable dependiente (categórica):",
    columnas_categoricas,
    key="disc_objetivo"
)

if columna_objetivo:
    distribucion = df[columna_objetivo].value_counts()
    st.write(f"**Grupos encontrados:** {len(distribucion)}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(distribucion.reset_index().rename(
            columns={'index': 'Grupo', columna_objetivo: 'Frecuencia'}
        ), width='stretch')
    with col2:
        st.write(f"**Variable:** `{columna_objetivo}`")
        st.write(f"**Tipo:** {df[columna_objetivo].dtype}")

st.markdown("---")

# --- PASO 2: SELECCIONA LAS VARIABLES PREDICTORAS ---
st.subheader("📊 Paso 2: Selecciona las variables predictoras")

#  FILTRO INTELIGENTE USANDO LOS METADATOS 
metadata = st.session_state.get('metadata', {})

# Filtramos usando el perfilador: solo numericos y que NO sean la variable objetivo
columnas_numericas_validas = [
    col for col in df.columns
    if col in metadata 
    and metadata[col]["tipo"] in ["numerico_continuo", "numerico_discreto"]
    and col != columna_objetivo
]

# Fallback de seguridad (por si el metadata no está cargado por alguna razón)
if not columnas_numericas_validas:
    columnas_numericas_validas = [
        col for col in df.select_dtypes(include=['number']).columns 
        if col != columna_objetivo and not col.lower().startswith("id")
    ]

# UI para seleccionar el modo
modo_seleccion = st.radio(
    "Modo de selección:",
    options=["Usar todas las variables numéricas válidas", "Seleccionar manualmente"],
    horizontal=True
)

if modo_seleccion == "Usar todas las variables numéricas válidas":
    variables_predictoras = columnas_numericas_validas
    st.info(f"✅ Se usarán {len(variables_predictoras)} variables predictoras automáticamente (se excluyeron IDs y variables categóricas).")
    st.write(f"**Variables incluidas:** {', '.join(variables_predictoras)}")
else:
    variables_predictoras = st.multiselect(
        "Selecciona las variables predictoras:",
        options=columnas_numericas_validas,
        default=columnas_numericas_validas,
        help="Elige las variables continuas o discretas que ayudarán a clasificar los grupos."
    )
    
    if len(variables_predictoras) < 2:
        st.warning("⚠️ Se recomienda seleccionar al menos 2 variables predictoras para un análisis discriminante óptimo.")

# --- PASO 3: EJECUTAR ANÁLISIS Y GESTIONAR SESSION STATE ---
st.subheader("🚀 Paso 3: Ejecutar Análisis")

with st.expander("📋 Resumen de configuración"):
    st.write(f"**Variable objetivo:** `{columna_objetivo}`")
    st.write(f"**Variables predictoras:** {len(variables_predictoras)}")
    st.write(f"**Grupos a clasificar:** {df[columna_objetivo].nunique()}")
    st.write(f"**Total de casos:** {len(df)}")
    st.write(f"**Casos sin nulos:** {len(df[variables_predictoras + [columna_objetivo]].dropna())}")

# DETECTOR DE CAMBIOS (STATE SIGNATURE)
# Ahora escuchamos al sello oficial de app.py, no a los switches en tiempo real
sello_oficial = st.session_state.get('sello_datos_confirmados', 'sin_sello')

# Nuestra Súper Huella junta: El Sello de Confirmación + Variable Objetivo + Predictoras elegidas
huella_actual = f"{sello_oficial}_{columna_objetivo}_{str(variables_predictoras)}"

if "disc_huella_df" not in st.session_state:
    st.session_state.disc_huella_df = huella_actual

# Si la huella actual no coincide con la guardada, algo cambió en el universo
if st.session_state.disc_huella_df != huella_actual:
    # Limpiamos los resultados obsoletos
    if "disc_resultados" in st.session_state:
        del st.session_state["disc_resultados"]
    # Actualizamos la memoria con la nueva huella
    st.session_state.disc_huella_df = huella_actual

# Inicializamos la variable de resultados si no existe
if "disc_resultados" not in st.session_state:
    st.session_state.disc_resultados = None

# EL BOTÓN SOLO CALCULA Y GUARDA EN SESSION STATE
if st.button("🔍 Ejecutar Análisis Discriminante", type="primary", key="disc_ejecutar"):
    with st.spinner("Procesando análisis discriminante (puede tardar un momento)..."):
        # Llamamos a la función
        resultados = ejecutar_analisis_cache(
            df, 
            columna_objetivo, 
            variables_predictoras
        )
        # Guardamos en session_state para que sobreviva a cambios de página
        st.session_state.disc_resultados = resultados


#  LA RENDERIZACIÓN SE HACE SI HAY RESULTADOS GUARDADOS 
# Ya no está anidado dentro del 'if st.button:', así que se dibuja aunque vengas de otra pestaña
if st.session_state.disc_resultados is not None:
    resultados = st.session_state.disc_resultados
    
    if "error" in resultados:
        st.error(resultados["error"])
    else:
        st.success("✅ Análisis completado con éxito")
        st.markdown("---")
        
        st.subheader("📊 Resultados del Análisis")
        
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
        
        with st.expander("📈 Funciones Discriminantes"):
            st.dataframe(resultados["funciones_discriminantes"], width='stretch')
            if "Funciones Discriminantes" in figuras_dict:
                st.plotly_chart(figuras_dict["Funciones Discriminantes"], width='stretch')
        
        if "Variables Discriminantes" in figuras_dict:
            with st.expander("📦 Variables Más Discriminantes"):
                st.plotly_chart(figuras_dict["Variables Discriminantes"], width='stretch')

        with st.expander("🎯 Centroides de Grupos"):
            st.dataframe(resultados["centroides"], width='stretch')
            if "Perfil de Centroides" in figuras_dict:
                st.plotly_chart(figuras_dict["Perfil de Centroides"], width='stretch')
        
        with st.expander("📊 Matriz de Confusión"):
            st.dataframe(resultados["matriz_confusion"], width='stretch')
            if "Matriz de Confusión" in figuras_dict:
                st.plotly_chart(figuras_dict["Matriz de Confusión"], width='stretch')
        
        with st.expander("🔬 Test de Box M (Homogeneidad de Covarianzas)"):
            box_m = resultados["test_box_m"]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Estadístico M:** {box_m['estadistico_M']:.4f}" if not pd.isna(box_m['estadistico_M']) else "**Estadístico M:** No disponible")
                st.write(f"**Chi-cuadrado:** {box_m['chi2']:.4f}" if not pd.isna(box_m['chi2']) else "**Chi-cuadrado:** No disponible")
            with col2:
                st.write(f"**Grados de Libertad:** {box_m['grados_libertad']}")
                st.write(f"**Valor p:** {box_m['p_valor']:.6f}" if not pd.isna(box_m['p_valor']) else "**Valor p:** No disponible")
            st.write(f"**Interpretación:** {box_m['interpretacion']}")
        
        with st.expander("📊 Clasificación por Grupo"):
            df_clasificacion = pd.DataFrame(resultados["clasificacion_por_grupo"]).T
            st.dataframe(df_clasificacion, width='stretch')
            if "Precisión por Grupo" in figuras_dict:
                st.plotly_chart(figuras_dict["Precisión por Grupo"], width='stretch')
        
        if len(resultados["casos_mal_clasificados"]) > 0:
            with st.expander("⚠️ Casos Mal Clasificados"):
                st.dataframe(resultados["casos_mal_clasificados"], width='stretch')
        else:
            st.success("✅ ¡Todos los casos fueron clasificados correctamente!")
        
        st.markdown("---")