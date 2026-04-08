import streamlit as st
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from modules.factorial import ordenar_matriz_cargas

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

st.title("Análisis Factorial Exploratorio (AFE)")
st.markdown("Descubra las variables latentes (factores) que explican el comportamiento de sus datos.")

# ==========================================
# 1. PREPARACIÓN DE DATOS CON MANEJO DE ESTADO
# ==========================================
if 'df_chido' not in st.session_state or st.session_state['df_chido'] is None:
    st.warning("No hay datos confirmados en memoria. Vaya a la página principal, cargue y confirme su dataset para continuar.")
    st.stop()

df_chido = st.session_state['df_chido']
metadata = st.session_state.get('metadata', {})

# 🔥 FILTRO ESTRICTO: Solo columnas que la metadata marque como "numerico"
columnas_validas = []
for col in df_chido.columns:
    info_columna = metadata.get(col, {})
    tipo = info_columna.get('tipo', '')
    
    # Si la palabra 'numerico' está en el tipo (ej. numerico_continuo, numerico_discreto)
    if 'numerico' in tipo:
        columnas_validas.append(col)

# Si no hay numéricas, detenemos la app
if len(columnas_validas) < 2:
    st.error("🚨 No hay suficientes variables numéricas en este dataset para hacer un Análisis Factorial. (Se requieren al menos 2).")
    st.stop()

# Creamos df_base solo con las columnas permitidas
df_base = df_chido[columnas_validas].copy()

# 🛡️ ESCUDO ANTI-TEXTO: Forzamos la conversión a número. 
# Si hay algún texto raro colado, lo vuelve NaN para que la función limpiar_datos lo rellene después.
df_base = df_base.apply(pd.to_numeric, errors='coerce')

sello_oficial = st.session_state.get('sello_datos_confirmados', 'sin_sello')
huella_fact = f"{sello_oficial}"
# ==========================================
# 2. PRUEBAS DE VIABILIDAD (KMO y Bartlett)
# ==========================================
st.header("Diagnóstico de viabilidad")

kmo_all, kmo_model = calculate_kmo(df_base)
chi2, p_value = calculate_bartlett_sphericity(df_base)

# 🔥 SÚPER PARCHE ANTI-NaN: Detener si la matemática explota
if pd.isna(kmo_model) or pd.isna(p_value):
    st.error("🚨 ¡Análisis Imposible! Las matemáticas del modelo fallaron (Resultado: NaN).")
    st.info("""
    **¿Por qué pasa esto?**
    1. Tienes más variables (columnas) que sujetos (filas).
    2. Hay multicolinealidad perfecta (una variable es combinación lineal de otras).
    3. Las variables seleccionadas no tienen **ninguna correlación** entre ellas.
    
    *Solución: Revisa tus datos, elimina variables redundantes o consigue una muestra más grande.*
    """)
    st.stop() # Corta la ejecución para que no salgan gráficas rotas

# Mostrar métricas
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Índice KMO", value=f"{kmo_model:.3f}", 
              help="Mayor a 0.6 es aceptable. Mayor a 0.8 es excelente.")
with col2:
    st.metric(label="Prueba de Bartlett (p-value)", value=f"{p_value:.50f}", 
              help="Debe ser menor a 0.05 para que el análisis sea válido.")

# 🛑 ALERTA DE KMO BASURA
if kmo_model < 0.5:
    st.error("🛑 ¡KMO Inaceptable! El valor es menor a 0.50. Esto significa que las variables comparten muy poca varianza y los resultados del Análisis Factorial carecerán de sentido.")
    st.stop() # Si el KMO es menor a 0.5, ni siquiera deberíamos dejarlo continuar
elif kmo_model < 0.6:
    st.warning("⚠️ El KMO es bajo (mediocre). Tus datos no son los mejores para un Análisis Factorial, pero puedes continuar bajo tu propio riesgo.")
else:
    st.success("✅ Los datos son viables para el análisis.")

st.divider()

# ==========================================
# 3. SCREE PLOT Y CONFIGURACIÓN
# ==========================================
st.header("Selección de factores")

col_plot, col_conf = st.columns([2, 1])

with col_plot:
    fig_scree, eigenvalues = generar_scree_plot(df_base)
    st.plotly_chart(fig_scree, width='stretch')

with col_conf:
    st.markdown("### Configuración")
    modo = st.radio("Método de selección:", ["Automático (Kaiser)", "Manual"])
    
    if modo == "Automático (Kaiser)":
        n_factores = max(sum(eigenvalues > 1), 1)
        st.info(f"El modelo sugiere **{n_factores} factores** basándose en la regla de Kaiser.")
    else:
        max_val = max(1, len(df_base.columns) // 2)
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
        st.title("Análisis Factorial Confirmatorio (AFC)")
        #st.markdown("Descubra las variables latentes (factores) que explican el comportamiento de sus datos.")


        fa = FactorAnalyzer(n_factors=n_factores, rotation='varimax', method='minres')
        fa.fit(df_base)
        
        cargas = pd.DataFrame(fa.loadings_, index=df_base.columns)
        cargas.columns = [f"Factor {i+1}" for i in range(n_factores)]
        
        st.header("Interpretación de factores")
        
        tab1, tab2 = st.tabs(["Mapa de Calor (Cargas)", "Diagrama de Senderos"])
        
        with tab1:
            fig_heat = heatmap_cargas(cargas)
            st.plotly_chart(fig_heat, width='stretch')
            
        with tab2:
            fig_bar = generar_diagrama_plotly(cargas)
            st.plotly_chart(fig_bar, width='content')
            
        # ==========================================
        # 5. EXPORTACIÓN AL ESTILO SPSS (LO QUE PIDIÓ LA MAESTRA)
        # ==========================================
        st.subheader("Matriz de Cargas Factoriales (Para exportar)")
        st.markdown("Tabla con los factores en columnas y variables en filas. Usa el control para ocultar los valores bajos.")

        # Slider para que la maestra juegue con el umbral (default 0.4)
        umbral = 0.4
        # umbral = st.slider("Ocultar cargas absolutas menores a:", min_value=0.0, max_value=0.9, value=0.4, step=0.05)

        # Aplicar el filtro: Si pasa el umbral lo hacemos texto con 4 decimales, si no, lo dejamos en blanco
        cargas_filtradas = cargas.copy()
        for col in cargas_filtradas.columns:
            cargas_filtradas[col] = cargas_filtradas[col].apply(lambda x: f"{x:.4f}" if abs(x) >= umbral else "")

        # Mostramos la tabla limpia en la UI
        st.dataframe(cargas_filtradas, width='stretch')
        
        # Botón de descarga
        st.download_button(
            label="Descargar Matriz de Cargas (.csv)",
            # MUY IMPORTANTE: index=True para que las variables salgan en la primera columna
            data=cargas_filtradas.to_csv(index=True).encode('utf-8'), 
            file_name="matriz_cargas_factorial.csv",
            mime="text/csv"
        )
