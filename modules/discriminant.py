import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st  # Importante para la selección
import warnings
warnings.filterwarnings('ignore')

# --- TU FUNCIÓN AUXILIAR PARA EL TEST DE BOX M (SIN CAMBIOS) ---
def box_m_test(X, y):
    groups = np.unique(y)
    n_groups = len(groups)
    n_features = X.shape[1]
    df = 0.5 * n_features * (n_features + 1) * (n_groups - 1)
    cov_pooled = np.zeros((n_features, n_features))
    total_df = 0
    for g in groups:
        n_g = np.sum(y == g)
        X_g = X[y == g]
        if n_g > n_features:
            cov_g = np.cov(X_g.T, ddof=1)
            cov_pooled += (n_g - 1) * cov_g
            total_df += n_g - 1
            
    if total_df == 0:
        return {"estadistico_M": np.nan, "grados_libertad": df, "chi2": np.nan, "p_valor": np.nan, "interpretacion": "No se pudo calcular"}
    
    cov_pooled /= total_df
    M = 0
    for g in groups:
        n_g = np.sum(y == g)
        X_g = X[y == g]
        if n_g > n_features:
            cov_g = np.cov(X_g.T, ddof=1)
            if np.linalg.det(cov_g) > 0:
                M += (n_g - 1) * np.log(np.linalg.det(cov_pooled) / np.linalg.det(cov_g))
    
    chi2_stat = M
    p_value = 1 - stats.chi2.cdf(chi2_stat, df) if not np.isnan(M) else np.nan
    
    return {
        "estadistico_M": M,
        "grados_libertad": df,
        "chi2": chi2_stat,
        "p_valor": p_value,
        "interpretacion": "⚠️ Rechaza H0 (matrices diferentes)" if p_value < 0.05 else "✅ No rechaza H0 (matrices homogéneas)"
    }

# --- TU FUNCIÓN DE ANÁLISIS (MANTENIENDO TODA LA LÓGICA) ---
def ejecutar_analisis_discriminante(df, columna_objetivo, variables_predictoras):
    if columna_objetivo not in df.columns:
        return {"error": f"La variable objetivo '{columna_objetivo}' no existe."}
    
    for var in variables_predictoras:
        if var not in df.columns:
            return {"error": f"La variable predictora '{var}' no existe."}
    
    grupos = df[columna_objetivo].value_counts()
    if len(grupos) < 2:
        return {"error": "Se necesitan al menos 2 grupos."}
    
    try:
        df_clean = df[variables_predictoras + [columna_objetivo]].dropna()
        if len(df_clean) == 0:
            return {"error": "No quedan datos tras limpiar nulos."}
        
        X = df_clean[variables_predictoras].values
        y = df_clean[columna_objetivo].values
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        y_pred = lda.predict(X)
        y_prob = lda.predict_proba(X) if hasattr(lda, 'predict_proba') else None
        
        # Extracción de resultados y tablas
        n_funciones = len(lda.classes_) - 1 if len(lda.classes_) > 2 else 1
        if lda.coef_.shape[0] == 1:
            coeficientes = pd.DataFrame(lda.coef_, columns=variables_predictoras, index=["Función 1"])
            constante = pd.DataFrame([lda.intercept_], columns=['Constante'], index=["Función 1"])
        else:
            coeficientes = pd.DataFrame(lda.coef_, columns=variables_predictoras, index=[f"Función {i+1}" for i in range(lda.coef_.shape[0])])
            constante = pd.DataFrame(lda.intercept_, columns=['Constante'], index=[f"Función {i+1}" for i in range(lda.intercept_.shape[0])])
        
        tabla_funciones = pd.concat([constante, coeficientes], axis=1)
        matriz_m = confusion_matrix(y, y_pred)
        clases = lda.classes_
        df_matriz = pd.DataFrame(matriz_m, index=[f"Real: {c}" for c in clases], columns=[f"Predicho: {c}" for c in clases])
        precision = accuracy_score(y, y_pred)
        medias_por_grupo = df_clean.groupby(columna_objetivo)[variables_predictoras].mean()
        desviaciones_por_grupo = df_clean.groupby(columna_objetivo)[variables_predictoras].std()
        centroides = pd.DataFrame(lda.means_, columns=variables_predictoras, index=clases)
        
        # Autovalores y Test Box M
        autovalores = []
        varianza_explicada = []
        if hasattr(lda, 'scalings_'):
            scalings = lda.scalings_
            if scalings.shape[1] > 0:
                autovalores = np.linalg.eigvals(scalings.T @ scalings)
                autovalores = np.sort(autovalores)[::-1][:n_funciones]
                varianza_explicada = autovalores / np.sum(autovalores) if np.sum(autovalores) > 0 else []

        box_m_resultados = box_m_test(X, y)
        analisis_casos = df_clean.copy()
        analisis_casos['PREDICCIÓN'] = y_pred
        mal_clasificados = analisis_casos[analisis_casos[columna_objetivo] != y_pred]
        
        # --- GENERACIÓN DE GRÁFICOS (TU LÓGICA ORIGINAL) ---
        figuras = []
        colores = px.colors.qualitative.Set1
        
        # Gráfico de precisión
        precision_grupos = pd.DataFrame({'Grupo': [str(c) for c in clases], 
                                        'Precisión (%)': [(np.sum((y == c) & (y_pred == c)) / np.sum(y == c)) * 100 for c in clases]})
        fig_precision = go.Figure(data=[go.Bar(x=precision_grupos['Grupo'], y=precision_grupos['Precisión (%)'], marker_color=colores)])
        fig_precision.update_layout(title="Precisión por Grupo")
        figuras.append(("Precisión por Grupo", fig_precision))
        
        # Matriz de Confusión
        fig_cm = go.Figure(data=go.Heatmap(z=matriz_m, x=[str(c) for c in clases], y=[str(c) for c in clases], colorscale='Blues'))
        figuras.append(("Matriz de Confusión", fig_cm))

        # Mapa de funciones discriminantes (con tu truco de muestreo)
        if n_funciones >= 2 and len(variables_predictoras) >= 2:
            scores = lda.transform(X)
            df_scores = pd.DataFrame(scores, columns=[f'DF{i+1}' for i in range(scores.shape[1])])
            df_scores['Grupo'] = y
            LIMITE_PUNTOS = 1500
            df_plot = df_scores.groupby('Grupo', group_keys=False).apply(lambda x: x.sample(min(len(x), int(LIMITE_PUNTOS/len(clases))), random_state=42)) if len(df_scores) > LIMITE_PUNTOS else df_scores
            
            fig_scores = go.Figure()
            for i, grupo in enumerate(clases):
                df_g = df_plot[df_plot['Grupo'] == grupo]
                fig_scores.add_trace(go.Scattergl(x=df_g['DF1'], y=df_g['DF2'], mode='markers', name=str(grupo)))
            figuras.append(("Funciones Discriminantes", fig_scores))

        # Perfil de centroides (Radar)
        if len(clases) >= 2 and len(variables_predictoras) >= 2:
            centroides_norm = (centroides - centroides.min()) / (centroides.max() - centroides.min() + 1e-9)
            fig_radar = go.Figure()
            for i, grupo in enumerate(clases):
                fig_radar.add_trace(go.Scatterpolar(r=centroides_norm.loc[grupo].values, theta=centroides_norm.columns, fill='toself', name=str(grupo)))
            figuras.append(("Perfil de Centroides", fig_radar))

        return {
            "resumen": {
                "Metodología": "LDA", "Variable Objetivo": columna_objetivo, "Variables Predictoras": variables_predictoras,
                "Número de Grupos": len(clases), "Precisión Global": f"{precision:.2%}", "Número de Casos Analizados": len(df_clean),
                "Casos Bien Clasificados": len(df_clean) - len(mal_clasificados), "Casos Mal Clasificados": len(mal_clasificados)
            },
            "funciones_discriminantes": tabla_funciones, "centroides": centroides, "matriz_confusion": df_matriz,
            "test_box_m": box_m_resultados, "figuras": figuras, "precision": precision
        }
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# --- BLOQUE NUEVO: INTERFAZ DE USUARIO PARA ELEGIR COLUMNAS ---
def mostrar_interfaz_discriminante(df):
    st.subheader("Configuración del Análisis Discriminante")
    
    # 1. El usuario elige la columna objetivo
    columna_y = st.selectbox(
        "Selecciona la Variable Objetivo (Categoría/Grupo):",
        options=df.columns,
        index=0,
        help="Esta es la variable que define los grupos (ej. Especies, Riesgo, etc.)"
    )
    
    # 2. El usuario elige las variables predictoras (solo mostramos numéricas por defecto)
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    variables_x = st.multiselect(
        "Selecciona las Variables Predictoras (Numéricas):",
        options=columnas_numericas,
        default=columnas_numericas[:min(5, len(columnas_numericas))],
        help="Selecciona las variables que ayudarán a distinguir entre los grupos."
    )
    
    if st.button("🚀 Ejecutar Análisis con estas columnas"):
        if not variables_x:
            st.error("Debes seleccionar al menos una variable predictora.")
            return

        resultados = ejecutar_analisis_discriminante(df, columna_y, variables_x)
        
        if "error" in resultados:
            st.error(resultados["error"])
        else:
            # Mostrar Resumen
            st.success(f"Análisis completado. Precisión: {resultados['resumen']['Precisión Global']}")
            
            # Mostrar Gráficos
            for titulo, fig in resultados['figuras']:
                st.write(f"### {titulo}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar Tablas
            with st.expander("Ver tablas de resultados"):
                st.write("**Funciones Discriminantes**")
                st.dataframe(resultados['funciones_discriminantes'])
                st.write("**Matrices de Confusión**")
                st.dataframe(resultados['matriz_confusion'])
                st.write("**Test de Box M**")
                st.write(resultados['test_box_m'])

# Para usarlo en tu archivo pages/ o Inicio.py:
if 'df' in st.session_state:
    mostrar_interfaz_discriminante(st.session_state.df)
