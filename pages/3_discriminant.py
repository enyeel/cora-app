import streamlit as st
from modules.discriminant import ejecutar_analisis_discriminante

st.set_page_config(page_title="Análisis Discriminante", page_icon="🎯", layout="wide")

st.title("🎯 Análisis Discriminante")

# Verificar si hay datos limpios
if 'df_limpio' not in st.session_state or st.session_state.df_limpio is None:
    st.warning("⚠️ Primero carga y limpia los datos en la página principal.")
    st.stop()

df = st.session_state.df_limpio

st.subheader("🎯 Análisis Discriminante")
st.markdown("---")

# --- PASO 1: SELECCIONAR VARIABLE OBJETIVO ---
st.subheader("📌 Paso 1: Selecciona la variable a predecir (objetivo)")

columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

if len(columnas_categoricas) == 0:
    st.warning("⚠️ No se encontraron variables categóricas para usar como variable dependiente.")
    st.info("El análisis discriminante requiere una variable objetivo categórica (ejemplo: 'género', 'compra', 'segmento')")
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
        ), use_container_width=True)
    with col2:
        st.write(f"**Variable:** `{columna_objetivo}`")
        st.write(f"**Tipo:** {df[columna_objetivo].dtype}")

st.markdown("---")

# --- PASO 2: SELECCIONAR VARIABLES PREDICTORAS ---
st.subheader("📊 Paso 2: Selecciona las variables predictoras")

columnas_numericas = df.select_dtypes(include=[float, int]).columns.tolist()

if len(columnas_numericas) == 0:
    st.warning("⚠️ No hay variables numéricas para usar como predictoras.")
    st.stop()

modo_seleccion = st.radio(
    "Modo de selección:",
    ["Usar todas las variables numéricas", "Seleccionar manualmente"],
    horizontal=True,
    key="disc_modo"
)

if modo_seleccion == "Usar todas las variables numéricas":
    variables_predictoras = columnas_numericas
    st.info(f"✅ Se usarán todas las {len(variables_predictoras)} variables numéricas.")
    st.write("**Variables incluidas:**", ", ".join(variables_predictoras[:10]) + 
            ("..." if len(variables_predictoras) > 10 else ""))
else:
    variables_predictoras = st.multiselect(
        "Variables independientes (numéricas):",
        columnas_numericas,
        default=columnas_numericas[:min(5, len(columnas_numericas))],
        key="disc_predictoras"
    )
    
    if not variables_predictoras:
        st.warning("⚠️ Selecciona al menos una variable predictora.")
        st.stop()
    else:
        st.success(f"✅ Seleccionadas {len(variables_predictoras)} variables predictoras.")

st.markdown("---")

# --- PASO 3: EJECUTAR ANÁLISIS ---
st.subheader("🚀 Paso 3: Ejecutar Análisis")

with st.expander("📋 Resumen de configuración"):
    st.write(f"**Variable objetivo:** `{columna_objetivo}`")
    st.write(f"**Variables predictoras:** {len(variables_predictoras)}")
    st.write(f"**Grupos a clasificar:** {df[columna_objetivo].nunique()}")
    st.write(f"**Total de casos:** {len(df)}")
    st.write(f"**Casos sin valores nulos:** {len(df[variables_predictoras + [columna_objetivo]].dropna())}")

if st.button("🔍 Ejecutar Análisis Discriminante", type="primary", key="disc_ejecutar"):
    with st.spinner("Procesando análisis discriminante..."):
        resultados = ejecutar_analisis_discriminante(
            df, 
            columna_objetivo, 
            variables_predictoras
        )
    
    if "error" in resultados:
        st.error(resultados["error"])
    else:
        st.success("✅ Análisis completado con éxito")
        st.markdown("---")
        
        # Mostrar resultados en formato organizado
        st.subheader("📊 Resultados del Análisis")
        
        # Resumen en columnas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precisión Global", resultados["resumen"]["Precisión Global"])
        with col2:
            st.metric("Grupos", resultados["resumen"]["Número de Grupos"])
        with col3:
            st.metric("Bien Clasificados", resultados["resumen"]["Casos Bien Clasificados"])
        with col4:
            st.metric("Mal Clasificados", resultados["resumen"]["Casos Mal Clasificados"])
        
        # Funciones Discriminantes
        with st.expander("📈 Funciones Discriminantes", expanded=True):
            st.dataframe(resultados["funciones_discriminantes"], use_container_width=True)
        
        # Centroides
        with st.expander("🎯 Centroides de Grupos"):
            st.dataframe(resultados["centroides"], use_container_width=True)
        
        # Matriz de Confusión
        with st.expander("📊 Matriz de Confusión"):
            st.dataframe(resultados["matriz_confusion"], use_container_width=True)
        
        # Test de Box M
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
        
        # Clasificación por grupo
        with st.expander("📊 Clasificación por Grupo"):
            df_clasificacion = pd.DataFrame(resultados["clasificacion_por_grupo"]).T
            st.dataframe(df_clasificacion, use_container_width=True)
        
        # Casos mal clasificados
        if len(resultados["casos_mal_clasificados"]) > 0:
            with st.expander("⚠️ Casos Mal Clasificados"):
                st.dataframe(resultados["casos_mal_clasificados"], use_container_width=True)
        else:
            st.success("✅ ¡Todos los casos fueron clasificados correctamente!")
        
        # Gráficos
        st.subheader("📈 Visualización de Resultados")
        for titulo, fig in resultados["figuras"]:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")