import streamlit as st
import pandas as pd

from modules.descriptive import (
    frequency_table, central_tendency, dispersion_measures, shape_measures,
    position_measures, interpret_shape, normality_tests, categorical_frequency_table,
    plot_categorical_bar, histogram_from_table, frequency_polygon, boxplot, ogive,
    scatter_plot, scatter_matrix, correlation_matrix, correlation_heatmap
)

st.set_page_config(page_title="Análisis Descriptivo | DIA", page_icon=None, layout="wide")

from modules.layout import render_sidebar

# Sidebar compartido
render_sidebar()

st.title("Análisis descriptivo y exploratorio")
st.markdown("Comprenda la distribución, forma y correlación de sus variables.")

# ==========================================
# 1. EL CADENERO (VALIDACIÓN DE DATOS)
# ==========================================
if 'df_chido' not in st.session_state or st.session_state['df_chido'] is None:
    st.warning("⚠️ ¡Alto ahí! No hay datos en memoria. Ve a la página principal, sube tu dataset y dale al botón de 'Confirmar y Mandar a Análisis'.")
    st.stop()

# Usamos df_chido porque queremos ver los nombres originales de las categorías, no los números encodeados
df = st.session_state['df_chido']
metadata = st.session_state.get('metadata', {})

# ==========================================
# 2. FILTRO INTELIGENTE DE COLUMNAS
# ==========================================
# Filtramos para no meter IDs ni cosas raras
columnas_numericas = []
columnas_categoricas = []

for col in df.columns:
    es_invalido = metadata.get(col, {}).get("tipo") == "id" or metadata.get(col, {}).get("tipo") == "categorico_alto"
    if not es_invalido:
        if pd.api.types.is_numeric_dtype(df[col]):
            columnas_numericas.append(col)
        else:
            columnas_categoricas.append(col)

# ==========================================
# 3. CONFIGURACIÓN UI
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    col_num = st.selectbox("Selecciona una columna numérica:", options=[None] + columnas_numericas)
with col2:
    col_cat = st.selectbox("Selecciona una columna categórica (opcional):", options=[None] + columnas_categoricas)
with col3:
    bins = st.slider("Número de intervalos (bins) para histograma:", min_value=5, max_value=50, value=10)

ejecutar = st.button("Ejecutar análisis", type="primary", width='stretch')
st.divider()

# ==========================================
# 4. GESTIÓN DEL CACHÉ Y HUELLAS (SIGNATURE)
# ==========================================
sello_oficial = st.session_state.get('sello_datos_confirmados', 'sin_sello')
huella_actual = f"{sello_oficial}_{col_num}_{col_cat}_{bins}"

if st.session_state.get('desc_huella') != huella_actual:
    if 'desc_resultados' in st.session_state:
        del st.session_state['desc_resultados']
    st.session_state['desc_huella'] = huella_actual

# ==========================================
# 5. EJECUCIÓN MATEMÁTICA Y GUARDADO
# ==========================================
if ejecutar and col_num is not None:
    with st.spinner("Masticando datos y generando gráficas..."):
        resultados = {}
        
        # Numéricos
        freq_tbl = frequency_table(df, col_num, bins=bins)
        resultados['freq_tbl'] = freq_tbl
        resultados['ct'] = central_tendency(df, col_num)
        resultados['dm'] = dispersion_measures(df, col_num)
        resultados['sm'] = shape_measures(df, col_num)
        resultados['interp'] = interpret_shape(
            resultados['sm'].loc[resultados['sm']['Measure']=='Skewness (Asimetría)','Value'].values[0],
            resultados['sm'].loc[resultados['sm']['Measure']=='Kurtosis (Curtosis)','Value'].values[0]
        )
        resultados['pm'] = position_measures(df, col_num)
        resultados['nt'] = normality_tests(df, col_num)
        
        # Gráficas numéricas
        resultados['fig_hist'] = histogram_from_table(freq_tbl, col_num)
        resultados['fig_poly'] = frequency_polygon(freq_tbl, col_num)
        resultados['fig_box'] = boxplot(df, col_num)
        resultados['fig_ogive'] = ogive(freq_tbl, col_num)
        
        # Correlaciones generales
        resultados['corr_df'] = correlation_matrix(df, method='pearson')
        resultados['fig_corr'] = correlation_heatmap(df, method='pearson')
        resultados['fig_matrix'] = scatter_matrix(df)
        
        # Categórico (Si existe)
        if col_cat:
            cat_tbl = categorical_frequency_table(df, col_cat)
            resultados['cat_tbl'] = cat_tbl
            resultados['fig_cat_bar'] = plot_categorical_bar(cat_tbl, col_cat)
            
        st.session_state['desc_resultados'] = resultados

# ==========================================
# 6. RENDERIZADO VISUAL DESDE MEMORIA
# ==========================================
if 'desc_resultados' in st.session_state:
    res = st.session_state['desc_resultados']
    
    st.header(f"Análisis de: {col_num}")
    
    # 6.1 ESTADÍSTICAS BÁSICAS (TABS)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tendencia Central y Dispersión", "Forma y Posición", "Prueba de Normalidad", "Tabla de Frecuencias", "Gráficas"])
    
    with tab1:
        c1, c2 = st.columns(2)
        c1.dataframe(res['ct'], width='stretch', hide_index=True)
        c2.dataframe(res['dm'], width='stretch', hide_index=True)
        
    with tab2:
        c1, c2 = st.columns(2)
        c1.dataframe(res['sm'], width='stretch', hide_index=True)
        c1.dataframe(res['interp'], width='stretch', hide_index=True)
        c2.dataframe(res['pm'], width='stretch', hide_index=True)
        
    with tab3:
        st.dataframe(res['nt'], width='stretch', hide_index=True)
        
    with tab4:
        st.dataframe(res['freq_tbl'], width='stretch', hide_index=True)
        
   
    # 6.2 GRÁFICAS DE DISTRIBUCIÓN
    st.subheader("Distribución de los datos")
    col_g1, col_g2 = st.columns(2)
    col_g1.plotly_chart(res['fig_hist'], width='stretch')
    col_g2.plotly_chart(res['fig_poly'], width='stretch')
    
    col_g3, col_g4 = st.columns(2)
    col_g3.plotly_chart(res['fig_box'], width='stretch')
    col_g4.plotly_chart(res['fig_ogive'], width='stretch')
    
    # 6.3 ANÁLISIS MULTIVARIABLE Y CORRELACIÓN
    st.divider()
    st.header("Correlaciones globales (todas las variables numéricas)")
    
    if res['fig_corr'] is not None:
        with st.expander("Ver matriz de correlación (heatmap)", expanded=True):
            st.plotly_chart(res['fig_corr'], width='stretch')
            
        with st.expander("Ver matriz de dispersión (pairplot)"):
            st.warning("Renderizar esta gráfica puede tomar unos segundos si hay muchas variables.")
            if res.get('fig_matrix'):
                st.plotly_chart(res['fig_matrix'], width='stretch')
    else:
        st.info("No hay suficientes variables numéricas para hacer correlaciones.")

    # 6.4 ANÁLISIS CATEGÓRICO
    if col_cat and 'cat_tbl' in res:
        st.divider()
        st.header(f"Análisis categórico: {col_cat}")
        cc1, cc2 = st.columns([1, 2])
        cc1.dataframe(res['cat_tbl'], width='stretch', hide_index=True)
        cc2.plotly_chart(res['fig_cat_bar'], width='stretch')

elif col_num is None:
    st.info("Seleccione una columna numérica y haga clic en 'Ejecutar análisis' para comenzar.")

