import streamlit as st
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Importamos las funciones limpias de tu compa
from modules.factorial import (
    limpiar_datos, 
    eliminar_multicolinealidad, 
    normalizar, 
    generar_scree_plot, 
    heatmap_cargas, 
    generar_diagrama_plotly
)

from modules.layout import render_sidebar

st.set_page_config(page_title="Análisis Factorial", page_icon=None, layout="wide")

# Sidebar compartido
render_sidebar()

st.title("Análisis factorial exploratorio (AFE)")
st.markdown("Descubra las variables latentes (factores) que explican el comportamiento de sus datos.")

# ==========================================
# 1. PREPARACIÓN DE DATOS CON MANEJO DE ESTADO
# ==========================================
if 'df_encoded' not in st.session_state or st.session_state['df_encoded'] is None:
    st.warning("No hay datos confirmados en memoria. Vaya a la página principal, cargue y confirme su dataset para continuar.")
    st.stop() # 👈 Esto detiene la ejecución aquí mismo, evitando el crash.

df_base = st.session_state.get('df_encoded', None) 

sello_oficial = st.session_state.get('sello_datos_confirmados', 'sin_sello')
huella_fact = f"{sello_oficial}"

# Solo recalculamos la preparación si la huella cambió
if st.session_state.get('fact_huella_prep') != huella_fact:
    with st.spinner("Preparando y limpiando datos numéricos..."):
        try:
            df_raw = df_base.copy()
            df_limpio = limpiar_datos(df_raw)
            df_sin_multi = eliminar_multicolinealidad(df_limpio)
            
            borradas = set(df_limpio.columns) - set(df_sin_multi.columns)
            st.session_state['fact_cols_borradas'] = borradas
            
            df_listo = normalizar(df_sin_multi)
            
            st.session_state['df_fact_ready'] = df_listo
            st.session_state['fact_huella_prep'] = huella_fact # Guardamos la huella actual
            
            if 'fact_error' in st.session_state:
                del st.session_state['fact_error']
                
        except ValueError as e:
            st.session_state['fact_error'] = str(e)

# Si hubo un error en la limpieza, mostramos alerta y detenemos la app
if 'fact_error' in st.session_state:
    st.error(f"🚨 No se pudo procesar el Análisis Factorial: {st.session_state['fact_error']}")
    st.info("💡 Consejo: Sube un archivo que contenga al menos 2 columnas con valores numéricos.")
    st.stop()

df = st.session_state['df_fact_ready']

# 🛑 PARCHE 2: Mostramos visualmente si alguna columna fue asesinada por multicolinealidad
if st.session_state.get('fact_cols_borradas'):
    st.warning(f"⚠️ Se ignoraron estas variables por tener correlación casi perfecta (>0.95) con otras: **{', '.join(st.session_state['fact_cols_borradas'])}**")

# ==========================================
# 2. PRUEBAS DE VIABILIDAD (KMO y Bartlett)
# ==========================================
st.header("Diagnóstico de viabilidad")

kmo_all, kmo_model = calculate_kmo(df)
chi2, p_value = calculate_bartlett_sphericity(df)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Índice KMO", value=f"{kmo_model:.3f}", 
              help="Mayor a 0.6 es aceptable. Mayor a 0.8 es excelente.")
with col2:
    st.metric(label="Prueba de Bartlett (p-value)", value=f"{p_value:.5f}", 
              help="Debe ser menor a 0.05 para que el análisis sea válido.")

if kmo_model < 0.6:
    st.warning("⚠️ El KMO es bajo. Tus datos podrían no ser los mejores para un Análisis Factorial, pero puedes continuar bajo tu propio riesgo.")
else:
    st.success("✅ Los datos son viables para el análisis.")

st.divider()

# ==========================================
# 3. SCREE PLOT Y CONFIGURACIÓN
# ==========================================
st.header("Selección de factores")

col_plot, col_conf = st.columns([2, 1])

with col_plot:
    fig_scree, eigenvalues = generar_scree_plot(df)
    st.plotly_chart(fig_scree, width='stretch')

with col_conf:
    st.markdown("### Configuración")
    modo = st.radio("Método de selección:", ["Automático (Kaiser)", "Manual"])
    
    if modo == "Automático (Kaiser)":
        n_factores = max(sum(eigenvalues > 1), 1)
        st.info(f"El modelo sugiere **{n_factores} factores** basándose en la regla de Kaiser.")
    else:
        max_val = max(1, len(df.columns) // 2)
        default_val = min(2, max_val)
        n_factores = st.number_input(
            "Número de Factores:", 
            min_value=1, 
            max_value=max_val, 
            value=default_val
        )

    ejecutar = st.button("Ejecutar análisis", type="primary", width='stretch')

st.divider()

# ==========================================
# 4. RESULTADOS DEL MODELO
# ==========================================
if ejecutar:
    with st.spinner("Calculando cargas factoriales..."):
        fa = FactorAnalyzer(n_factors=n_factores, rotation='varimax', method='minres')
        fa.fit(df)
        
        cargas = pd.DataFrame(fa.loadings_, index=df.columns)
        cargas.columns = [f"Factor {i+1}" for i in range(n_factores)]
        
        st.header("Interpretación de factores")
        
        tab1, tab2 = st.tabs(["Mapa de Calor (Cargas)", "Diagrama de Barras"])
        
        with tab1:
            fig_heat = heatmap_cargas(cargas)
            st.plotly_chart(fig_heat, width='stretch')
            
        with tab2:
            fig_bar = generar_diagrama_plotly(cargas)
            st.plotly_chart(fig_bar, width='stretch')
            
        # ==========================================
        # 5. EXPORTACIÓN BLINDADA
        # ==========================================
        st.subheader("Datos transformados")
        
        # 🛑 PARCHE 3: Le agregamos index=df.index para que no se pierda la alineación
        factores_df = pd.DataFrame(fa.transform(df), columns=cargas.columns, index=df.index)
        
        df_export = st.session_state['df_original'].copy()
        for col in factores_df.columns:
            # Quitamos el .values para que Pandas respete los índices originales al pegar
            df_export[col] = factores_df[col]
            
        st.dataframe(df_export.head(50), width='stretch')
        
        st.download_button(
            label="Descargar datos con factores integrados",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name="dataset_factorial.csv",
            mime="text/csv"
        )
