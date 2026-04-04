import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# --- FUNCIÓN AUXILIAR PARA EL TEST DE BOX M ---
def box_m_test(X, y):
    """
    Calcula el test de Box M para la homogeneidad de las matrices de covarianza.
    """
    groups = np.unique(y)
    n_groups = len(groups)
    n_features = X.shape[1]
    
    # Calcular los grados de libertad
    df = 0.5 * n_features * (n_features + 1) * (n_groups - 1)
    
    # Calcular el estadístico M
    M = 0
    valid_groups = 0
    for g in groups:
        n_g = np.sum(y == g)
        X_g = X[y == g]
        if n_g > 1 and len(X_g) > n_features:
            cov_g = np.cov(X_g.T)
            if np.linalg.det(cov_g) > 0:
                valid_groups += 1
    
    if valid_groups < 2:
        return {
            "estadistico_M": np.nan,
            "grados_libertad": df,
            "chi2": np.nan,
            "p_valor": np.nan,
            "interpretacion": "No se pudo calcular (posiblemente por falta de datos en algunos grupos)"
        }
    
    # Calcular la matriz de covarianza combinada
    cov_pooled = np.cov(X.T)
    
    M = 0
    for g in groups:
        n_g = np.sum(y == g)
        X_g = X[y == g]
        if n_g > 1:
            cov_g = np.cov(X_g.T)
            if np.linalg.det(cov_g) > 0 and np.linalg.det(cov_pooled) > 0:
                M += (n_g - 1) * np.log(np.linalg.det(cov_pooled) / np.linalg.det(cov_g))
    
    chi2_stat = M
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return {
        "estadistico_M": M,
        "grados_libertad": df,
        "chi2": chi2_stat,
        "p_valor": p_value,
        "interpretacion": "⚠️ Se rechaza H0 (matrices diferentes) - considerar QDA" 
                         if p_value < 0.05 
                         else "✅ No se rechaza H0 (matrices homogéneas) - válido para LDA"
    }

def mostrar_interfaz_analisis_discriminante(df):
    """
    Muestra la interfaz de usuario para el análisis discriminante.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos limpios.
    """
    
    st.subheader("🎯 Análisis Discriminante")
    st.markdown("---")
    
    # --- PASO 1: SELECCIONAR VARIABLE OBJETIVO ---
    st.subheader("📌 Paso 1: Selecciona la variable a predecir (objetivo)")
    
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(columnas_categoricas) == 0:
        st.warning("⚠️ No se encontraron variables categóricas para usar como variable dependiente.")
        st.info("El análisis discriminante requiere una variable objetivo categórica (ejemplo: 'género', 'compra', 'segmento')")
        return
    
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
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columnas_numericas) == 0:
        st.warning("⚠️ No hay variables numéricas para usar como predictoras.")
        return
    
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
            return
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

def ejecutar_analisis_discriminante(df, columna_objetivo, variables_predictoras):
    """
    Ejecuta el análisis discriminante lineal (LDA) con las variables seleccionadas.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos limpios.
    columna_objetivo : str
        Nombre de la variable dependiente (categórica).
    variables_predictoras : list
        Lista de nombres de variables independientes (numéricas).
    
    Retorna:
    --------
    dict
        Diccionario con todos los resultados del análisis.
    """
    
    # --- VALIDACIONES ---
    if columna_objetivo not in df.columns:
        return {"error": f"La variable objetivo '{columna_objetivo}' no existe en el dataset."}
    
    for var in variables_predictoras:
        if var not in df.columns:
            return {"error": f"La variable predictora '{var}' no existe en el dataset."}
    
    # Verificar grupos
    grupos = df[columna_objetivo].value_counts()
    if len(grupos) < 2:
        return {"error": f"La variable '{columna_objetivo}' tiene solo {len(grupos)} grupo(s). Se necesitan al menos 2 grupos."}
    
    if any(grupos < 2):
        grupos_pequenos = grupos[grupos < 2].index.tolist()
        return {"error": f"Los grupos {grupos_pequenos} tienen menos de 2 observaciones."}
    
    try:
        # --- PREPARACIÓN DE DATOS ---
        df_clean = df[variables_predictoras + [columna_objetivo]].dropna()
        
        if len(df_clean) == 0:
            return {"error": "Después de eliminar valores nulos, no quedan datos para analizar."}
        
        X = df_clean[variables_predictoras].values
        y = df_clean[columna_objetivo].values
        
        grupos_limpios = pd.Series(y).value_counts()
        if any(grupos_limpios < 2):
            return {"error": "Después de eliminar valores nulos, algunos grupos quedaron con menos de 2 observaciones."}
        
        if len(df_clean) <= len(variables_predictoras):
            return {"error": f"El número de observaciones ({len(df_clean)}) debe ser mayor que el número de variables predictoras ({len(variables_predictoras)})."}
        
        # --- ENTRENAMIENTO ---
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        
        # --- PREDICCIONES ---
        y_pred = lda.predict(X)
        y_prob = lda.predict_proba(X) if hasattr(lda, 'predict_proba') else None
        
        # --- EXTRACCIÓN DE RESULTADOS ---
        n_funciones = len(lda.classes_) - 1 if len(lda.classes_) > 2 else 1
        
        # Funciones Discriminantes
        if lda.coef_.shape[0] == 1:
            coeficientes = pd.DataFrame(lda.coef_, columns=variables_predictoras, index=[f"Función 1"])
            constante = pd.DataFrame([lda.intercept_], columns=['Constante'], index=[f"Función 1"])
        else:
            coeficientes = pd.DataFrame(lda.coef_, columns=variables_predictoras, 
                                        index=[f"Función {i+1}" for i in range(lda.coef_.shape[0])])
            constante = pd.DataFrame(lda.intercept_, columns=['Constante'], 
                                     index=[f"Función {i+1}" for i in range(lda.intercept_.shape[0])])
        
        tabla_funciones = pd.concat([constante, coeficientes], axis=1)
        
        # Matriz de Confusión
        matriz_m = confusion_matrix(y, y_pred)
        clases = lda.classes_
        df_matriz = pd.DataFrame(matriz_m, index=[f"Real: {c}" for c in clases], 
                                 columns=[f"Predicho: {c}" for c in clases])
        
        precision = accuracy_score(y, y_pred)
        
        # Estadísticas por grupo
        medias_por_grupo = df_clean.groupby(columna_objetivo)[variables_predictoras].mean()
        desviaciones_por_grupo = df_clean.groupby(columna_objetivo)[variables_predictoras].std()
        
        # Centroides
        centroides = pd.DataFrame(lda.means_, columns=variables_predictoras, index=clases)
        
        # Autovalores
        if hasattr(lda, 'scalings_'):
            scalings = lda.scalings_
            if scalings.shape[1] > 0:
                autovalores = np.linalg.eigvals(scalings.T @ scalings)
                autovalores = np.sort(autovalores)[::-1][:n_funciones]
                varianza_explicada = autovalores / np.sum(autovalores) if np.sum(autovalores) > 0 else []
            else:
                autovalores = []
                varianza_explicada = []
        else:
            autovalores = []
            varianza_explicada = []
        
        # Test de Box M
        box_m_resultados = box_m_test(X, y)
        
        # Casos mal clasificados
        analisis_casos = df_clean.copy()
        analisis_casos['PREDICCIÓN'] = y_pred
        if y_prob is not None:
            prob_asignada = []
            for i in range(len(y_pred)):
                idx_clase = list(clases).index(y_pred[i])
                prob_asignada.append(y_prob[i][idx_clase])
            analisis_casos['PROBABILIDAD'] = prob_asignada
        mal_clasificados = analisis_casos[analisis_casos[columna_objetivo] != y_pred]
        
        # Estadísticas por grupo
        clasificacion_por_grupo = {}
        for grupo in clases:
            mask_grupo = (y == grupo)
            total_grupo = np.sum(mask_grupo)
            correctos_grupo = np.sum((y[mask_grupo] == y_pred[mask_grupo]))
            clasificacion_por_grupo[str(grupo)] = {
                "total": int(total_grupo),
                "correctos": int(correctos_grupo),
                "porcentaje_acierto": f"{correctos_grupo/total_grupo*100:.1f}%" if total_grupo > 0 else "0%"
            }
        
        # --- GRÁFICOS ---
        figuras = []
        colores = px.colors.qualitative.Set1
        
        # Gráfico de precisión por grupo
        precision_grupos = pd.DataFrame({
            'Grupo': list(clasificacion_por_grupo.keys()),
            'Precisión (%)': [float(c['porcentaje_acierto'].replace('%', '')) for c in clasificacion_por_grupo.values()]
        })
        
        fig_precision = go.Figure(data=[
            go.Bar(x=precision_grupos['Grupo'], y=precision_grupos['Precisión (%)'],
                   marker_color=colores[:len(precision_grupos)],
                   text=precision_grupos['Precisión (%)'], textposition='auto')
        ])
        fig_precision.update_layout(title="Precisión de Clasificación por Grupo", height=400, width=600)
        figuras.append(("Precisión por Grupo", fig_precision))
        
        # Matriz de confusión
        fig_cm = go.Figure(data=go.Heatmap(z=matriz_m, x=[str(c) for c in clases], y=[str(c) for c in clases],
                                           text=matriz_m, texttemplate="%{text}", colorscale='Blues'))
        fig_cm.update_layout(title="Matriz de Confusión", height=500, width=600)
        figuras.append(("Matriz de Confusión", fig_cm))
        
        # Boxplots de variables más discriminantes
        importancias = np.abs(lda.coef_.mean(axis=0)) if lda.coef_.ndim > 1 else np.abs(lda.coef_)
        indices_top = np.argsort(importancias)[::-1][:min(3, len(variables_predictoras))]
        vars_top = [variables_predictoras[i] for i in indices_top]
        
        if len(vars_top) > 0:
            fig_box = make_subplots(rows=1, cols=len(vars_top), subplot_titles=vars_top, shared_yaxes=True)
            for idx, var in enumerate(vars_top):
                for i, grupo in enumerate(clases):
                    datos_grupo = df_clean[df_clean[columna_objetivo] == grupo][var].dropna()
                    fig_box.add_trace(go.Box(y=datos_grupo, name=str(grupo), 
                                             marker_color=colores[i % len(colores)],
                                             showlegend=(idx == 0)), row=1, col=idx+1)
            fig_box.update_layout(title="Variables Más Discriminantes", height=500, width=800)
            figuras.append(("Variables Discriminantes", fig_box))
        
        # Mapa de funciones discriminantes
        if n_funciones >= 2 and len(variables_predictoras) >= 2:
            scores = lda.transform(X)
            df_scores = pd.DataFrame(scores, columns=[f'DF{i+1}' for i in range(scores.shape[1])])
            df_scores['Grupo'] = y
            
            fig_scores = go.Figure()
            for i, grupo in enumerate(clases):
                df_grupo = df_scores[df_scores['Grupo'] == grupo]
                fig_scores.add_trace(go.Scatter(x=df_grupo['DF1'], y=df_grupo['DF2'], mode='markers',
                                                name=str(grupo), marker=dict(color=colores[i % len(colores)], size=8)))
                fig_scores.add_trace(go.Scatter(x=[df_grupo['DF1'].mean()], y=[df_grupo['DF2'].mean()],
                                                mode='markers', name=f'Centroide {grupo}',
                                                marker=dict(color=colores[i % len(colores)], size=15, symbol='x')))
            fig_scores.update_layout(title="Mapa de Funciones Discriminantes", height=600, width=700)
            figuras.append(("Funciones Discriminantes", fig_scores))
        
        # Gráfico radial de centroides
        if len(clases) >= 2 and len(variables_predictoras) >= 2:
            centroides_norm = (centroides - centroides.min()) / (centroides.max() - centroides.min())
            fig_radar = go.Figure()
            for i, grupo in enumerate(clases):
                fig_radar.add_trace(go.Scatterpolar(r=centroides_norm.loc[grupo].values,
                                                    theta=centroides_norm.columns, fill='toself',
                                                    name=str(grupo), line=dict(color=colores[i % len(colores)])))
            fig_radar.update_layout(title="Perfil de Centroides", height=500, width=700)
            figuras.append(("Perfil de Centroides", fig_radar))
        
        # --- RETORNAR RESULTADOS ---
        return {
            "resumen": {
                "Metodología": "Análisis Discriminante Lineal (LDA)",
                "Variable Objetivo": columna_objetivo,
                "Variables Predictoras": variables_predictoras,
                "Número de Grupos": len(clases),
                "Grupos": list(map(str, clases)),
                "Precisión Global": f"{precision:.2%}",
                "Número de Casos Analizados": len(df_clean),
                "Casos Bien Clasificados": len(df_clean) - len(mal_clasificados),
                "Casos Mal Clasificados": len(mal_clasificados)
            },
            "funciones_discriminantes": tabla_funciones,
            "centroides": centroides,
            "medias_por_grupo": medias_por_grupo,
            "desviaciones_por_grupo": desviaciones_por_grupo,
            "matriz_confusion": df_matriz,
            "varianza_explicada": varianza_explicada,
            "autovalores": autovalores,
            "clasificacion_por_grupo": clasificacion_por_grupo,
            "test_box_m": box_m_resultados,
            "casos_mal_clasificados": mal_clasificados,
            "figuras": figuras,
            "precision": precision,
            "n_funciones": n_funciones
        }
        
    except Exception as e:
        import traceback
        return {"error": f"Error en el análisis: {str(e)}"}