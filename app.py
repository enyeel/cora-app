import streamlit as st
import pandas as pd
import numpy as np

# IMPORTAMOS NUESTRAS ARMAS
from modules.cleaning import detectar_outliers, detectar_webones
from modules.cleaning_motor import ejecutar_pipeline_maestro

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

# ==========================================
# ⚙️ INICIALIZACIÓN DE LA MEMORIA CENTRAL
# ==========================================
pipeline_config = {
    "outliers": {"enabled": True, "acciones_por_fila": {}},
    "webones": {"enabled": True, "acciones_por_fila": {}}, # <-- ESTO CAMBIÓ
    "imputacion": {"enabled": False, "metodo_num": "Media", "forzar_drop": False}
}

st.title("Limpieza de Datos")

archivo_subido = st.file_uploader("Sube tu dataset sucio (CSV o Excel)", type=["csv", "xlsx"])

if archivo_subido is not None:
    # Leemos el archivo una sola vez (evita consumir el buffer varias veces)
    try:
        if archivo_subido.name.endswith('.csv'):
            df_raw = pd.read_csv(archivo_subido)
        else:
            df_raw = pd.read_excel(archivo_subido)
    except Exception:
        st.error("🚨 Archivo dañado o ilegible.")
        st.stop()

    # Si es un archivo nuevo (nombre distinto) actualizamos la memoria y
    # reiniciamos el historial quirúrgico y los detectores pesados.
    if 'nombre_archivo' not in st.session_state or st.session_state['nombre_archivo'] != archivo_subido.name:
        st.session_state['df_original'] = df_raw.copy()
        st.session_state['nombre_archivo'] = archivo_subido.name
        if 'config_quirurgica' in st.session_state:
            del st.session_state['config_quirurgica']

        # Recalculamos los detectores para el nuevo archivo
        mapa_out, cols_out = detectar_outliers(df_raw)
        st.session_state['mapa_outliers'] = mapa_out
        st.session_state['cols_con_outliers'] = cols_out
        st.session_state['filas_webones'] = detectar_webones(df_raw)
    else:
        # Si por alguna razón faltan variables en sesión, las calculamos/aseguramos
        if 'df_original' not in st.session_state:
            st.session_state['df_original'] = df_raw.copy()
        if 'mapa_outliers' not in st.session_state or 'filas_webones' not in st.session_state:
            mapa_out, cols_out = detectar_outliers(st.session_state['df_original'])
            st.session_state['mapa_outliers'] = mapa_out
            st.session_state['cols_con_outliers'] = cols_out
            st.session_state['filas_webones'] = detectar_webones(st.session_state['df_original'])

    # Recuperamos las variables de la memoria
    df = st.session_state['df_original']
    mapa_outliers = st.session_state['mapa_outliers']
    filas_webones = st.session_state['filas_webones']

    # Métricas
    total_filas = len(df)
    total_nulos = df.isna().sum().sum()
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
        col: "{:.0f}" if not df[col].dropna().empty and all(x.is_integer() for x in df[col].dropna()) else "{}"
        for col in df.select_dtypes(include=[np.number]).columns
    }

    st.dataframe(
        df.style.apply(pintar_rayos_x, axis=None).format(formatter=formatos_columnas, na_rep="NaN"), 
        height=350, use_container_width=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Filas", total_filas)
    col2.metric("Nulos Encontrados", total_nulos)
    col3.metric("Outliers Detectados", total_outliers)
    col4.metric("Usuarios Inválidos", total_webones)
    
    st.divider()

    # ==========================================
    # 🎯 SECCIÓN 2: TRATAMIENTO QUIRÚRGICO
    # ==========================================
    tratar_activado = st.toggle("Habilitar Tratamiento de Outliers", value=True)
    pipeline_config["outliers"]["enabled"] = tratar_activado

    if tratar_activado:
        st.markdown("### 2. Tratamiento Quirúrgico de Outliers")
        filas_con_outliers = mapa_outliers.any(axis=1)
        df_contexto_outliers = df.loc[filas_con_outliers].copy()

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

            # 🔥 AL CAMBIAR ALGO AQUÍ, STREAMLIT RECARGA Y ACTUALIZA EL RESULTADO FINAL ABAJO
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
                use_container_width=True
            )

            # 🛠️ LLENAMOS EL DICCIONARIO REACTIVO EN TIEMPO REAL
            for _, row in df_editado.iterrows():
                orig_idx = row.get('_orig_index')
                accion = row.get('¿Qué hacemos?')
                if accion != "Ignorar (Dejar como está)":
                    pipeline_config["outliers"]["acciones_por_fila"][orig_idx] = accion

        else:
            st.success("¡Base de datos impecable! No se detectaron outliers.")
    else:
        st.markdown("### ~~2. Tratamiento Quirúrgico de Outliers~~")
        st.info("⚠️ El tratamiento de outliers está desactivado. Los valores atípicos pasarán intactos.")

    st.divider()

    # ==========================================
    # 🎯 SECCIONES 3 Y 4 (Las conectaremos después)
    # ==========================================
    # st.header("3. Filtro de Varianza Nula (Straight-lining)") ...
    # st.header("4. Imputación de Nulos") ...

    # ==========================================
    # 🎯 SECCIÓN 5: RESULTADO FINAL (MAGIA EN VIVO)
    # ==========================================
    st.header("5. Resultado Final en Tiempo Real (Datos AI-Ready)")
    
    # 🚀 EL ORQUESTADOR ENTRA EN ACCIÓN
    df_limpio_actual = ejecutar_pipeline_maestro(
        st.session_state['df_original'], 
        mapa_outliers, 
        pipeline_config
    )
    
    st.markdown("CEXO está procesando tus decisiones en vivo. Así se ve tu dataset en este momento:")
    
    # --- 🧮 CÁLCULO DE MÉTRICAS ACTUALIZADAS ---
    filas_finales = len(df_limpio_actual)
    nulos_finales = df_limpio_actual.isna().sum().sum()
    
    # Outliers restantes: contamos los que eran True en el mapa original 
    # y que la celda actual SIGUE existiendo y NO es nula.
    filas_sobrevivientes = df_limpio_actual.index.intersection(mapa_outliers.index)
    outliers_restantes = 0
    for col in mapa_outliers.columns:
        if col in df_limpio_actual.columns:
            # Sumamos las celdas que eran outlier Y que en el df limpio no son nulas
            outliers_restantes += (mapa_outliers.loc[filas_sobrevivientes, col] & df_limpio_actual.loc[filas_sobrevivientes, col].notna()).sum()
            
    # Webones restantes: los que sobrevivieron al drop
    webones_restantes = filas_webones.loc[filas_sobrevivientes].sum()

    # --- 🃏 PINTAMOS LAS TARJETITAS CON DELTAS ---
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    # Delta inverso para que si bajan los nulos/outliers se ponga verde
    col_r1.metric("Filas Finales", filas_finales, delta=int(filas_finales - total_filas))
    col_r2.metric("Nulos Actuales", nulos_finales, delta=int(nulos_finales - total_nulos), delta_color="inverse")
    col_r3.metric("Outliers Restantes", outliers_restantes, delta=int(outliers_restantes - total_outliers), delta_color="inverse")
    col_r4.metric("Inválidos Restantes", int(webones_restantes), delta=int(webones_restantes - total_webones), delta_color="inverse")

    # --- 🎨 PINTAMOS LA TABLA FINAL ---
    st.dataframe(
        df_limpio_actual.style.apply(pintar_rayos_x, axis=None).format(formatter=formatos_columnas, na_rep="NaN"), 
        height=300, 
        use_container_width=True
    )