import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

# IMPORTAMOS NUESTRAS ARMAS
from modules.cleaning import detectar_outliers, detectar_webones
from modules.cleaning_motor import aplicar_estructural, aplicar_outliers, aplicar_webones, aplicar_nulos

# =======================================================
# 🖥️ LA INTERFAZ DE USUARIO (UI) - BRANDING DIA
# =======================================================
st.set_page_config(page_title="CEXO | by DIA", page_icon="☀️", layout="wide")

with st.sidebar:
    st.markdown("## CEXO by ☀️ DIA")
    st.caption("**Data Intelligence & Analytics**")
    st.caption("📍 *Software desarrollado en el Bajío Valley*")
    st.divider()
    st.caption("© 2026 DIA. Todos los derechos reservados a los 11 fundadores (CDIA).")

st.title("CEXO Analysis")
st.markdown("*Powered by **DIA** - Algoritmos de vanguardia para datos impecables.*")
st.divider()

st.title("Limpieza de Datos")

archivo_subido = st.file_uploader("Sube tu dataset sucio (CSV o Excel)", type=["csv", "xlsx"])

if archivo_subido is not None:
    try:
        if archivo_subido.name.endswith('.csv'):
            df_raw = pd.read_csv(archivo_subido)
        else:
            df_raw = pd.read_excel(archivo_subido)
    except Exception:
        st.error("🚨 Archivo dañado o ilegible.")
        st.stop()

    # 🧠 TRUCO 1: Guardamos la memoria config en el session_state para curar la amnesia
    if 'nombre_archivo' not in st.session_state or st.session_state['nombre_archivo'] != archivo_subido.name:
        st.session_state['df_original'] = df_raw.copy()
        st.session_state['nombre_archivo'] = archivo_subido.name
        
        # Reiniciamos la memoria de decisiones al subir nuevo archivo
        st.session_state['pipeline_config'] = {
            "estructural": {"enabled": True, "drop_cols": [], "coerce_cols": []}, # <-- NUEVO
            "outliers": {"enabled": True, "acciones_por_fila": {}},
            "webones": {"enabled": True, "acciones_por_fila": {}},
            "imputacion": {"enabled": True, "estrategia_global": "Media", "acciones_por_columna": {}}
        }

        # Recalculamos los detectores
        mapa_out, cols_out = detectar_outliers(df_raw)
        st.session_state['mapa_outliers'] = mapa_out
        st.session_state['cols_con_outliers'] = cols_out
        st.session_state['filas_webones'] = detectar_webones(df_raw)

    # Recuperamos las variables
    df_original = st.session_state['df_original']
    mapa_outliers = st.session_state['mapa_outliers']
    filas_webones = st.session_state['filas_webones']
    pipeline_config = st.session_state['pipeline_config'] # 🧠 Referencia directa a la memoria viva

    # Métricas Base
    total_filas = len(df_original)
    total_nulos = df_original.isna().sum().sum()
    total_outliers = mapa_outliers.sum().sum()
    total_webones = filas_webones.sum()

    # ==========================================
    # 🩻 SECCIÓN 1: RAYOS X
    # ==========================================
    st.header("1. Diagnóstico de Rayos X")
    st.markdown("🟥 **Rojo:** Nulos | 🟦 **Azul:** Outliers | 🟪 **Morado:** Filas de Varianza Nula")

    def pintar_rayos_x(data):
        estilos = pd.DataFrame('', index=data.index, columns=data.columns)
        webones_alineados = filas_webones.loc[data.index]
        outliers_alineados = mapa_outliers.loc[data.index, data.columns]
        
        for idx in data.index[webones_alineados]:
            estilos.loc[idx, :] = 'background-color: rgba(150, 75, 255, 0.3);'
        estilos[outliers_alineados] = 'background-color: rgba(75, 150, 255, 0.5); font-weight: bold;'
        estilos[data.isna()] = 'background-color: rgba(255, 75, 75, 0.6);'
        return estilos

    formatos_columnas = {
        col: "{:.0f}" if not df_original[col].dropna().empty and all(x.is_integer() for x in df_original[col].dropna()) else "{}"
        for col in df_original.select_dtypes(include=[np.number]).columns
    }

    st.dataframe(
        df_original.style.apply(pintar_rayos_x, axis=None).format(formatter=formatos_columnas, na_rep="NaN"), 
        height=350, use_container_width=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Filas", total_filas)
    col2.metric("Nulos Encontrados", total_nulos)
    col3.metric("Outliers Detectados", total_outliers)
    col4.metric("Usuarios Inválidos", total_webones)

    st.divider()

    # ==========================================
    # 👻 SECCIÓN 1.5: LIMPIEZA ESTRUCTURAL 
    # ==========================================
    cols_100_nulos = []
    cols_peligro_nulos = []
    coercion_info = {} # Guardará el nombre de la columna y un DataFrame con los intrusos

    # 🕵️‍♂️ LÓGICA DE DETECCIÓN AUTOMÁTICA
    for col in df_original.columns:
        pct_nulos = df_original[col].isnull().mean()
        
        # Detectar vacías
        if pct_nulos == 1.0:
            cols_100_nulos.append(col)
        elif pct_nulos >= 0.60:
            cols_peligro_nulos.append(col)

        # Detectar coerción y atrapar a los rebeldes
        if not is_numeric_dtype(df_original[col]):
            s_orig = df_original[col]
            s_num = pd.to_numeric(s_orig, errors='coerce')
            
            # Máscara: Era un dato válido originalmente, pero to_numeric lo destrozó (lo hizo NaN)
            mask_rebeldes = s_orig.notna() & s_num.isna()
            
            # Validamos que al menos haya un número real y calculamos la tasa de éxito
            if s_num.notna().any():
                tasa_exito = s_num.notna().sum() / s_orig.notna().sum()
                # Si la mayoría son números y hay al menos un rebelde
                if tasa_exito > 0.50 and mask_rebeldes.any():
                    # Guardamos las filas exactas donde están los weyes raros
                    df_rebeldes = df_original.loc[mask_rebeldes, [col]]
                    coercion_info[col] = df_rebeldes

    # 🖥️ RENDERIZADO CONDICIONAL
    if cols_100_nulos or cols_peligro_nulos or coercion_info:
        st.header("1.5. Limpieza Estructural (Auto-Detect)")
        st.info("💡 **CEXO AI** encontró anomalías estructurales antes de empezar el análisis.")

        pipeline_config["estructural"]["drop_cols"] = []
        pipeline_config["estructural"]["coerce_cols"] = []

        if cols_100_nulos:
            st.error(f"💀 **Columnas 100% Vacías:** `{', '.join(cols_100_nulos)}`\n\n*Eliminadas automáticamente.*")
            pipeline_config["estructural"]["drop_cols"].extend(cols_100_nulos)

        if cols_peligro_nulos:
            st.warning("⚠️ **Columnas Zombie (>60% vacías):** Te sugerimos eliminarlas.")
            for col in cols_peligro_nulos:
                pct = df_original[col].isnull().mean() * 100
                if st.checkbox(f"Eliminar `{col}` ({pct:.0f}% nulos)", value=True, key=f"drop_{col}"):
                    pipeline_config["estructural"]["drop_cols"].append(col)
            st.divider()

        if coercion_info:
            st.success("🔄 **Datos Intrusos Detectados:**")
            st.write("Se encontraron textos en columnas numéricas. Revisa a los sospechosos:")
            
            # Hacemos una tablita y un checkbox independiente por cada columna afectada
            for col, df_rebeldes in coercion_info.items():
                st.markdown(f"### 🕵️ Columna: `{col}`")
                
                # Mostramos los datos raros en un mini dataframe
                st.dataframe(df_rebeldes, use_container_width=True)
                
                # Checkbox con la cantidad exacta de valores a neutralizar
                if st.checkbox(f"Neutralizar estos {len(df_rebeldes)} valores a NaN", value=True, key=f"coerce_{col}"):
                    pipeline_config["estructural"]["coerce_cols"].append(col)
                
                st.divider()

    # 🚀 APLICAMOS LA MAGIA ESTRUCTURAL
    df_estructural = aplicar_estructural(df_original, pipeline_config)
    
    # 🧠 RECALCULAMOS DETECTORES (Super importante para las sig. secciones)
    mapa_outliers, cols_con_outliers = detectar_outliers(df_estructural)
    filas_webones = detectar_webones(df_estructural)

    # ==========================================
    # 🎯 SECCIÓN 2: TRATAMIENTO QUIRÚRGICO (OUTLIERS)
    # ==========================================
    tratar_activado = st.toggle("Habilitar Tratamiento de Outliers", value=pipeline_config["outliers"]["enabled"])
    pipeline_config["outliers"]["enabled"] = tratar_activado

    if tratar_activado:
        st.markdown("### 2. Tratamiento Quirúrgico de Outliers")
        filas_con_outliers = mapa_outliers.any(axis=1)
        df_contexto_outliers = df_estructural.loc[filas_con_outliers].copy()

        if not df_contexto_outliers.empty:
            df_contexto_outliers = df_contexto_outliers.reset_index()
            df_contexto_outliers.rename(columns={'index': '_orig_index'}, inplace=True)
            df_contexto_outliers.insert(0, '¿Qué hacemos?', 'Neutralizar valor (NaN)')

            def pintar_editor(data):
                estilos = pd.DataFrame('', index=data.index, columns=data.columns)
                indices_originales = data['_orig_index']
                for idx_actual, row in data.iterrows():
                    idx_real = indices_originales[idx_actual]
                    for col in data.columns:
                        if col in mapa_outliers.columns and mapa_outliers.loc[idx_real, col]:
                            estilos.loc[idx_actual, col] = 'background-color: rgba(75, 150, 255, 0.5); font-weight: bold;'
                        elif pd.isna(row[col]):
                            estilos.loc[idx_actual, col] = 'background-color: rgba(255, 75, 75, 0.6);'
                return estilos

            cols_intocables = [col for col in df_contexto_outliers.columns if col != '¿Qué hacemos?']

            df_editado = st.data_editor(
                df_contexto_outliers.style.apply(pintar_editor, axis=None).format(formatter=formatos_columnas, na_rep="NaN"),
                column_config={
                    "¿Qué hacemos?": st.column_config.SelectboxColumn(
                        "Seleccionar Acción",
                        options=["Neutralizar valor (NaN)", "Eliminar fila completa", "Ignorar (Dejar como está)"],
                        required=True,
                    ),
                    "_orig_index": None, 
                },
                disabled=cols_intocables,
                hide_index=True,
                use_container_width=True,
                key="editor_outliers"
            )

            # Llenamos memoria
            pipeline_config["outliers"]["acciones_por_fila"].clear()
            for _, row in df_editado.iterrows():
                orig_idx = row.get('_orig_index')
                accion = row.get('¿Qué hacemos?')
                if accion != "Ignorar (Dejar como está)":
                    pipeline_config["outliers"]["acciones_por_fila"][orig_idx] = accion
        else:
            st.success("¡Base de datos impecable! No se detectaron outliers.")
    else:
        st.markdown("### ~~2. Tratamiento Quirúrgico de Outliers~~")
        st.info("⚠️ El tratamiento de outliers está desactivado.")

    # 🧠 TRUCO 2: Aplicar HASTA EL FINAL de la sección, ya que recolectamos las decisiones
    df_sin_outliers = aplicar_outliers(df_estructural, mapa_outliers, pipeline_config)

    st.divider()

    # ==========================================
    # 🎯 SECCIÓN 3: FILTRO DE VARIANZA NULA (WEBONES)
    # ==========================================
    tratar_webones = st.toggle("Habilitar Filtro de Usuarios Inválidos (Straight-lining)", value=pipeline_config["webones"]["enabled"])
    pipeline_config["webones"]["enabled"] = tratar_webones

    if tratar_webones:
        st.markdown("### 3. Filtro de Varianza Nula (Straight-lining)")
        
        # Filtramos sobre df_sin_outliers
        df_contexto_webones = df_sin_outliers.loc[filas_webones[filas_webones].index.intersection(df_sin_outliers.index)].copy()
        
        if not df_contexto_webones.empty:
            st.markdown("👀 **Vista de Inspección:** Usuarios que respondieron lo mismo en toda la encuesta.")
            df_contexto_webones = df_contexto_webones.reset_index()
            df_contexto_webones.rename(columns={'index': '_orig_index'}, inplace=True)
            df_contexto_webones.insert(0, '¿Qué hacemos?', 'Eliminar fila completa') 
            
            def pintar_editor_webones(data):
                estilos = pd.DataFrame('', index=data.index, columns=data.columns)
                estilos.loc[:, :] = 'background-color: rgba(150, 75, 255, 0.3);'
                return estilos
            
            cols_intocables_w = [col for col in df_contexto_webones.columns if col != '¿Qué hacemos?']
            
            df_editado_w = st.data_editor(
                df_contexto_webones.style.apply(pintar_editor_webones, axis=None).format(formatter=formatos_columnas, na_rep="NaN"),
                column_config={
                    "¿Qué hacemos?": st.column_config.SelectboxColumn(
                        "Seleccionar Acción",
                        options=["Eliminar fila completa", "Ignorar (Dejar como está)"],
                        required=True,
                    ),
                    "_orig_index": None,
                },
                disabled=cols_intocables_w,
                hide_index=True,
                use_container_width=True,
                key="editor_webones" 
            )
            
            pipeline_config["webones"]["acciones_por_fila"].clear()
            for _, row in df_editado_w.iterrows():
                orig_idx = row.get('_orig_index')
                accion = row.get('¿Qué hacemos?')
                if accion != "Ignorar (Dejar como está)":
                    pipeline_config["webones"]["acciones_por_fila"][orig_idx] = accion
        else:
            st.success("¡Excelente! No se detectaron usuarios con varianza nula (o ya fueron eliminados).")
    else:
        st.markdown("### ~~3. Filtro de Varianza Nula~~")
        st.info("⚠️ Filtro desactivado.")

    # 🧠 Aplicamos HASTA EL FINAL de la sección 3
    df_sin_webones = aplicar_webones(df_sin_outliers, pipeline_config)

    st.divider()

    # ==========================================
    # 🎯 SECCIÓN 4: TRATAMIENTO DE NULOS
    # ==========================================
    tratar_nulos = st.toggle("Habilitar Tratamiento de Valores Nulos", value=pipeline_config["imputacion"]["enabled"])
    pipeline_config["imputacion"]["enabled"] = tratar_nulos

    if tratar_nulos:
        st.markdown("### 4. Tratamiento de Valores Nulos")
        
        col_glob1, col_glob2 = st.columns([1, 2])
        estrategia_global = col_glob1.radio(
            "Estrategia Global por defecto para variables numéricas:", 
            options=["Media", "Mediana"], 
            horizontal=True,
            index=0 if pipeline_config["imputacion"]["estrategia_global"] == "Media" else 1
        )
        pipeline_config["imputacion"]["estrategia_global"] = estrategia_global
        
        # Leemos los nulos del DF que ya viene curado de la Sección 3
        filas_con_nulos = df_sin_webones[df_sin_webones.isnull().any(axis=1)]
        
        if not filas_con_nulos.empty:
            st.markdown("👀 **Vista de Contexto:** Estas son las filas que contienen valores faltantes (originales o inyectados).")
            
            def pintar_celdas_nulas(val):
                return 'background-color: rgba(255, 75, 75, 0.3)' if pd.isna(val) else ''
            
            metodo_map = getattr(filas_con_nulos.style, 'map', getattr(filas_con_nulos.style, 'applymap', None))
            st.dataframe(
                metodo_map(pintar_celdas_nulas).format(formatter=formatos_columnas, na_rep="NaN"),
                height=250, use_container_width=True
            )
            
            st.markdown("🛠️ **Acciones por Columna:**")
            nulos_por_col = df_sin_webones.isnull().sum()
            cols_con_nulos = nulos_por_col[nulos_por_col > 0].index.tolist()
            
            df_nulos_contexto = pd.DataFrame({
                "Columna": cols_con_nulos,
                "Tipo de Dato": [str(df_sin_webones[col].dtype) for col in cols_con_nulos],
                "Nulos Iniciales": nulos_por_col[cols_con_nulos].values,
                "Acción a tomar": ["Usar Estrategia Global"] * len(cols_con_nulos)
            })
            
            def pintar_editor_nulos(data):
                estilos = pd.DataFrame('', index=data.index, columns=data.columns)
                estilos.loc[:, :] = 'background-color: rgba(255, 165, 0, 0.2);' 
                return estilos
            
            df_editado_nulos = st.data_editor(
                df_nulos_contexto.style.apply(pintar_editor_nulos, axis=None),
                column_config={
                    "Acción a tomar": st.column_config.SelectboxColumn(
                        "Estrategia Específica",
                        options=["Usar Estrategia Global", "Imputar por Media", "Imputar por Mediana", "Eliminar filas con nulos", "Ignorar (Dejar nulo)"],
                        required=True,
                    ),
                    "Columna": st.column_config.Column(disabled=True),
                    "Tipo de Dato": st.column_config.Column(disabled=True),
                    "Nulos Iniciales": st.column_config.Column(disabled=True),
                },
                hide_index=True, use_container_width=True, key="editor_nulos"
            )
            
            pipeline_config["imputacion"]["acciones_por_columna"].clear()
            for _, row in df_editado_nulos.iterrows():
                pipeline_config["imputacion"]["acciones_por_columna"][row['Columna']] = row['Acción a tomar']
        else:
            st.success("¡Excelente! En este punto del pipeline ya no existen valores nulos.")
    else:
        st.markdown("### ~~4. Tratamiento de Valores Nulos~~")
        st.info("⚠️ Módulo desactivado.")

    # 🧠 Aplicamos HASTA EL FINAL de la sección 4
    df_imputado = aplicar_nulos(df_sin_webones, pipeline_config)

    st.divider()

    # ==========================================
    # 🎯 SECCIÓN 5: RESULTADO FINAL (MAGIA EN VIVO)
    # ==========================================
    st.header("5. Resultado Final en Tiempo Real (Datos AI-Ready)")
    df_final = df_imputado.copy() 
    
    st.markdown("CEXO ha procesado tus decisiones en vivo. Así se ve tu dataset en este momento:")
    
    filas_finales = len(df_final)
    nulos_finales = df_final.isna().sum().sum()
    
    filas_sobrevivientes = df_final.index.intersection(mapa_outliers.index)
    outliers_restantes = 0
    for col in mapa_outliers.columns:
        if col in df_final.columns:
            # Si era outlier, sobrevivió a la purga, y su valor YA NO ES NULO y CAMBIÓ del original (fue imputado), ya no es outlier.
            # Solo contamos como outlier si sigue siendo exactamente igual al original.
            es_outlier_original = mapa_outliers.loc[filas_sobrevivientes, col]
            sigue_igual = df_final.loc[filas_sobrevivientes, col] == df_original.loc[filas_sobrevivientes, col]
            outliers_restantes += (es_outlier_original & sigue_igual).sum()
            
    webones_restantes = filas_webones.loc[filas_sobrevivientes].sum()

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Filas Finales", filas_finales, delta=int(filas_finales - total_filas))
    col_r2.metric("Nulos Actuales", nulos_finales, delta=int(nulos_finales - total_nulos), delta_color="inverse")
    col_r3.metric("Outliers Restantes", outliers_restantes, delta=int(outliers_restantes - total_outliers), delta_color="inverse")
    col_r4.metric("Inválidos Restantes", int(webones_restantes), delta=int(webones_restantes - total_webones), delta_color="inverse")

    # 🎨 PINTAMOS LA TABLA FINAL (Sólo lo que sigue sucio)
    def pintar_final(data):
        estilos = pd.DataFrame('', index=data.index, columns=data.columns)
        webones_alineados = filas_webones.loc[data.index]
        
        for idx in data.index[webones_alineados]:
            estilos.loc[idx, :] = 'background-color: rgba(150, 75, 255, 0.3);'
            
        for col in data.columns:
            if col in mapa_outliers.columns:
                es_outlier_original = mapa_outliers.loc[data.index, col]
                sigue_igual = data[col] == df_original.loc[data.index, col]
                estilos.loc[es_outlier_original & sigue_igual, col] = 'background-color: rgba(75, 150, 255, 0.5); font-weight: bold;'
                
        estilos[data.isna()] = 'background-color: rgba(255, 75, 75, 0.6);'
        return estilos

    st.dataframe(
        df_final.style.apply(pintar_final, axis=None).format(formatter=formatos_columnas, na_rep="NaN"), 
        height=300, 
        use_container_width=True
    )

    if nulos_finales > 0:
        st.warning(f"⚠️ **Aviso:** Tu dataset final aún tiene **{nulos_finales}** valores nulos. Asegúrate de que los módulos posteriores de análisis soporten datos faltantes.")