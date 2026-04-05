import streamlit as st
import pandas as pd
import numpy as np

from modules.cleaning import detectar_outliers, detectar_webones, codificar_categoricos_inteligente, estandarizar_zscore, detectar_anomalias_estructurales
from modules.cleaning_motor import aplicar_estructural, aplicar_outliers, aplicar_webones, aplicar_nulos
from modules.dataset_profiler import analizar_dataframe
from modules.renderers import renderizar_df_paginado, pintor_universal

# =======================================================
# ⚙️ GESTIÓN DE ESTADO (SESSION STATE)
# =======================================================
def inicializar_sesion(df_raw, nombre_archivo):
    """Configura la memoria por primera vez cuando se sube un archivo nuevo."""
    if 'nombre_archivo' not in st.session_state or st.session_state['nombre_archivo'] != nombre_archivo:
        st.session_state.clear() # Limpieza nuclear de sesiones viejas
        
        st.session_state['df_original'] = df_raw.copy()
        st.session_state['nombre_archivo'] = nombre_archivo
        
        st.session_state['pipeline_config'] = {
            "estructural": {"enabled": True, "drop_cols": [], "coerce_cols": []},
            "outliers": {"enabled": True, "acciones_por_fila": {}},
            "webones": {"enabled": True, "acciones_por_fila": {}},
            "imputacion": {"enabled": True, "estrategia_global": "Media", "acciones_por_columna": {}}
        }

        # Ejecutamos detectores base
        st.session_state['mapa_outliers'], st.session_state['cols_con_outliers'] = detectar_outliers(df_raw)
        st.session_state['filas_webones'] = detectar_webones(df_raw)
        st.session_state['metadata'] = analizar_dataframe(df_raw)
        st.session_state['anomalias'] = detectar_anomalias_estructurales(df_raw)

# =======================================================
# 🖥️ INTERFAZ DE USUARIO (UI)
# =======================================================
st.set_page_config(page_title="CEXO | by DIA", page_icon="☀️", layout="wide")

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Software de Estadística", page_icon="📊", layout="wide")

# --- FUNCIÓN DE LIMPIEZA (Tu otra chamba) ---
def limpiar_datos(df):
    # Aquí meterás tu lógica: quitar nulos, cambiar tipos de datos, etc.
    # Por ahora, solo regresamos el mismo dataframe.
    df_limpio = df.dropna() 
    return df_limpio

# --- EL MENÚSITO OBVIS (Barra Lateral) ---
st.sidebar.title("⚙️ Menú de Opciones")
st.sidebar.markdown("---")

archivo_subido = st.file_uploader(
    "Sube tu dataset sucio (CSV o Excel)", 
    type=['csv', 'xlsx']
)

if archivo_subido is not None:
    try:
        if archivo_subido.name.endswith('.csv'):
            df_raw = pd.read_csv(archivo_subido)
        else:
            df_raw = pd.read_excel(archivo_subido)
    except Exception:
        st.error("🚨 Archivo dañado o ilegible.")
        st.stop()

    inicializar_sesion(df_raw, archivo_subido.name)

if 'df_original' in st.session_state:
    df_original = st.session_state['df_original']
    mapa_outliers = st.session_state['mapa_outliers']
    filas_webones = st.session_state['filas_webones']
    pipeline_config = st.session_state['pipeline_config']
    anomalias = st.session_state['anomalias']

    total_filas = len(df_original)
    total_nulos = df_original.isna().sum().sum()
    total_outliers = mapa_outliers.sum().sum()
    total_webones = filas_webones.sum()

    st.success(f"🗃️ Trabajando con el dataset en memoria: **{st.session_state['nombre_archivo']}**")

    if st.button("Limpiar memoria y empezar de cero", type="secondary"):
        st.session_state.clear()
        st.rerun()
        
    st.divider()

    # ==========================================
    # SECCIÓN: DIAGNÓSTICO
    # ==========================================
    st.header("Diagnóstico de Rayos X")
    
    st.info("💡 Código de colores en la tabla:\n- **Rojo:** Valores nulos.\n- **Azul:** Valores atípicos (Outliers).\n- **Morado:** Usuarios con varianza nula (Straight-lining).")

    formatos_columnas = {
        col: "{:.0f}" if not df_original[col].dropna().empty and all(float(x).is_integer() for x in df_original[col].dropna()) else "{}"
        for col in df_original.select_dtypes(include=[np.number]).columns
    }

    renderizar_df_paginado(
        df_original,
        pintar_func=lambda d: pintor_universal(d, df_original, mapa_outliers, filas_webones),
        style_mode="apply",
        formatter=formatos_columnas,
        height=350,
        key="rayosx"
    )
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Filas", total_filas)
    col2.metric("Nulos Encontrados", total_nulos)
    col3.metric("Outliers Detectados", total_outliers)
    col4.metric("Usuarios Inválidos", total_webones)

    st.divider()

    st.subheader("Perfil Automático del Dataset")
    st.dataframe(pd.DataFrame(st.session_state.metadata).T)

    st.divider()

    # ==========================================
    # SECCIÓN: LIMPIEZA ESTRUCTURAL 
    # ==========================================
    if anomalias["vacias"] or anomalias["zombies"] or anomalias["coercion"]:
        st.header("Limpieza Estructural Automática")
        st.info("💡 **CEXO AI** encontró anomalías estructurales antes de empezar el análisis.")

        pipeline_config["estructural"]["drop_cols"] = []
        pipeline_config["estructural"]["coerce_cols"] = []

        if anomalias["vacias"]:
            st.error(f"💀 **Columnas 100% Vacías:** `{', '.join(anomalias['vacias'])}`\n\n*Eliminadas automáticamente.*")
            pipeline_config["estructural"]["drop_cols"].extend(anomalias["vacias"])

        if anomalias["zombies"]:
            st.warning("⚠️ **Columnas Zombie (>60% vacías):** Te sugerimos eliminarlas.")
            for col in anomalias["zombies"]:
                pct = df_original[col].isnull().mean() * 100
                if st.checkbox(f"Eliminar `{col}` ({pct:.0f}% nulos)", value=True, key=f"drop_{col}"):
                    pipeline_config["estructural"]["drop_cols"].append(col)
            st.divider()

        if anomalias["coercion"]:
            st.warning("⚠️ **Datos Intrusos Detectados:** Se encontraron textos en columnas numéricas. Revisa a los sospechosos:")
            for col, df_rebeldes in anomalias["coercion"].items():
                st.markdown(f"**Columna:** `{col}`")
                st.dataframe(df_rebeldes, width='stretch')
                if st.checkbox(f"Neutralizar estos {len(df_rebeldes)} valores a NaN", value=True, key=f"coerce_{col}"):
                    pipeline_config["estructural"]["coerce_cols"].append(col)
                st.divider()

    df_estructural = aplicar_estructural(df_original, pipeline_config)
    
    # Recalculamos detectores intermedios
    mapa_outliers, _ = detectar_outliers(df_estructural)
    filas_webones = detectar_webones(df_estructural)

    # ==========================================
    # SECCIÓN: OUTLIERS
    # ==========================================
    tratar_activado = st.toggle("Habilitar Tratamiento de Outliers", value=pipeline_config["outliers"]["enabled"])
    pipeline_config["outliers"]["enabled"] = tratar_activado

    st.header("Tratamiento Quirúrgico de Outliers")

    if tratar_activado:
        filas_con_outliers = mapa_outliers.any(axis=1)
        df_contexto_outliers = df_estructural.loc[filas_con_outliers].copy()

        if not df_contexto_outliers.empty:
            df_contexto_outliers = df_contexto_outliers.reset_index().rename(columns={'index': '_orig_index'})
            df_contexto_outliers.insert(0, '¿Qué hacemos?', 'Neutralizar valor (NaN)')

            cols_intocables = [col for col in df_contexto_outliers.columns if col != '¿Qué hacemos?']

            df_editado = st.data_editor(
                df_contexto_outliers.style.apply(
                    lambda d: pintor_universal(d, df_estructural, mapa_outliers, None, pintar_webones=False), 
                    axis=None
                ).format(formatter=formatos_columnas, na_rep="NaN"),
                column_config={
                    "¿Qué hacemos?": st.column_config.SelectboxColumn("Seleccionar Acción", options=["Neutralizar valor (NaN)", "Eliminar fila completa", "Ignorar (Dejar como está)"], required=True),
                    "_orig_index": None, 
                },
                disabled=cols_intocables, hide_index=True, width='stretch', key="editor_outliers"
            )

            pipeline_config["outliers"]["acciones_por_fila"].clear()
            for _, row in df_editado.iterrows():
                if row.get('¿Qué hacemos?') != "Ignorar (Dejar como está)":
                    pipeline_config["outliers"]["acciones_por_fila"][row.get('_orig_index')] = row.get('¿Qué hacemos?')
        else:
            st.success("✅ Base de datos impecable. No se detectaron outliers.")
    else:
        st.markdown("~~Tratamiento Quirúrgico de Outliers~~")
        st.info("ℹ️ El tratamiento de outliers está desactivado.")

    df_sin_outliers = aplicar_outliers(df_estructural, mapa_outliers, pipeline_config)
    st.divider()

    # ==========================================
    # SECCIÓN: FILTRO DE VARIANZA NULA
    # ==========================================
    tratar_webones = st.toggle("Habilitar Filtro de Usuarios Inválidos (Straight-lining)", value=pipeline_config["webones"]["enabled"])
    pipeline_config["webones"]["enabled"] = tratar_webones

    st.header("Filtro de Varianza Nula (Straight-lining)")

    if tratar_webones:
        df_contexto_webones = df_sin_outliers.loc[filas_webones[filas_webones].index.intersection(df_sin_outliers.index)].copy()
        
        if not df_contexto_webones.empty:
            st.markdown("**Vista de Inspección:** Usuarios que respondieron lo mismo en toda la encuesta.")
            df_contexto_webones = df_contexto_webones.reset_index().rename(columns={'index': '_orig_index'})
            df_contexto_webones.insert(0, '¿Qué hacemos?', 'Eliminar fila completa') 
            
            cols_intocables_w = [col for col in df_contexto_webones.columns if col != '¿Qué hacemos?']
            
            df_editado_w = st.data_editor(
                df_contexto_webones.style.apply(
                    lambda d: pintor_universal(d, None, None, filas_webones, pintar_nulos=False, pintar_outliers=False), 
                    axis=None
                ).format(formatter=formatos_columnas, na_rep="NaN"),
                column_config={
                    "¿Qué hacemos?": st.column_config.SelectboxColumn("Seleccionar Acción", options=["Eliminar fila completa", "Ignorar (Dejar como está)"], required=True),
                    "_orig_index": None,
                },
                disabled=cols_intocables_w, hide_index=True, width='stretch', key="editor_webones" 
            )
            
            pipeline_config["webones"]["acciones_por_fila"].clear()
            for _, row in df_editado_w.iterrows():
                if row.get('¿Qué hacemos?') != "Ignorar (Dejar como está)":
                    pipeline_config["webones"]["acciones_por_fila"][row.get('_orig_index')] = row.get('¿Qué hacemos?')
        else:
            st.success("✅ No se detectaron usuarios con varianza nula (o ya fueron eliminados).")
    else:
        st.markdown("~~Filtro de Varianza Nula~~")
        st.info("ℹ️ Filtro desactivado.")

    df_sin_webones = aplicar_webones(df_sin_outliers, pipeline_config)
    st.divider()

    # ==========================================
    # SECCIÓN: TRATAMIENTO DE NULOS
    # ==========================================
    tratar_nulos = st.toggle("Habilitar Tratamiento de Valores Nulos", value=pipeline_config["imputacion"]["enabled"])
    pipeline_config["imputacion"]["enabled"] = tratar_nulos

    st.header("Tratamiento de Valores Nulos")

    if tratar_nulos:
        col_glob1, col_glob2 = st.columns([1, 2])
        estrategia_global = col_glob1.radio(
            "Estrategia global por defecto para numéricas:", 
            options=["Media", "Mediana"], horizontal=True,
            index=0 if pipeline_config["imputacion"]["estrategia_global"] == "Media" else 1
        )
        pipeline_config["imputacion"]["estrategia_global"] = estrategia_global
        
        filas_con_nulos = df_sin_webones[df_sin_webones.isnull().any(axis=1)]
        
        if not filas_con_nulos.empty:
            st.markdown("**Vista de Contexto:** Filas que contienen valores faltantes.")
            
            renderizar_df_paginado(
                filas_con_nulos,
                pintar_func=lambda v: 'background-color: rgba(255, 75, 75, 0.3)' if pd.isna(v) else '',
                style_mode="map", formatter=formatos_columnas, height=250, key="nulos"
            )
            
            st.markdown("**Acciones por Columna:**")
            nulos_por_col = df_sin_webones.isnull().sum()
            cols_con_nulos = nulos_por_col[nulos_por_col > 0].index.tolist()
            
            df_nulos_contexto = pd.DataFrame({
                "Columna": cols_con_nulos,
                "Tipo de Dato": [str(df_sin_webones[col].dtype) for col in cols_con_nulos],
                "Nulos Iniciales": nulos_por_col[cols_con_nulos].values,
                "Acción a tomar": ["Usar Estrategia Global"] * len(cols_con_nulos)
            })
            
            df_editado_nulos = st.data_editor(
                df_nulos_contexto.style.apply(lambda d: pd.DataFrame('background-color: rgba(255, 165, 0, 0.2);', index=d.index, columns=d.columns), axis=None),
                column_config={
                    "Acción a tomar": st.column_config.SelectboxColumn("Estrategia Específica", options=["Usar Estrategia Global", "Imputar por Media", "Imputar por Mediana", "Eliminar filas con nulos", "Ignorar (Dejar nulo)"], required=True),
                    "Columna": st.column_config.Column(disabled=True),
                    "Tipo de Dato": st.column_config.Column(disabled=True),
                    "Nulos Iniciales": st.column_config.Column(disabled=True),
                },
                hide_index=True, width='stretch', key="editor_nulos"
            )
            
            pipeline_config["imputacion"]["acciones_por_columna"].clear()
            for _, row in df_editado_nulos.iterrows():
                pipeline_config["imputacion"]["acciones_por_columna"][row['Columna']] = row['Acción a tomar']
        else:
            st.success("✅ En este punto del pipeline ya no existen valores nulos.")
    else:
        st.markdown("~~Tratamiento de Valores Nulos~~")
        st.info("ℹ️ Módulo desactivado.")

    df_imputado = aplicar_nulos(df_sin_webones, pipeline_config)
    st.divider()

    # ==========================================
    # SECCIÓN: RESULTADO FINAL
    # ==========================================
    st.header("Resultado Final en Tiempo Real")
    df_final = df_imputado.copy() 
    
    filas_finales = len(df_final)
    nulos_finales = df_final.isna().sum().sum()
    
    filas_sobrevivientes = df_final.index.intersection(mapa_outliers.index)
    outliers_restantes = sum((mapa_outliers.loc[filas_sobrevivientes, col] & (df_final.loc[filas_sobrevivientes, col] == df_original.loc[filas_sobrevivientes, col])).sum() for col in mapa_outliers.columns if col in df_final.columns)
    webones_restantes = filas_webones.loc[filas_sobrevivientes].sum()

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Filas Finales", filas_finales, delta=int(filas_finales - total_filas))
    col_r2.metric("Nulos Actuales", nulos_finales, delta=int(nulos_finales - total_nulos), delta_color="inverse")
    col_r3.metric("Outliers Restantes", outliers_restantes, delta=int(outliers_restantes - total_outliers), delta_color="inverse")
    col_r4.metric("Inválidos Restantes", int(webones_restantes), delta=int(webones_restantes - total_webones), delta_color="inverse")

    renderizar_df_paginado(
        df_final,
        pintar_func=lambda d: pintor_universal(d, df_original, mapa_outliers, filas_webones),
        style_mode="apply", formatter=formatos_columnas, height=300, key="final"
    )

    if nulos_finales > 0:
        st.warning(f"⚠️ **Aviso:** Tu dataset final aún tiene **{nulos_finales}** valores nulos. Asegúrate de que los módulos posteriores de análisis soporten datos faltantes.")
    
    st.divider()

    # ==========================================
    # SECCIÓN: EXPORTAR Y ANALIZAR
    # ==========================================
    st.header("Exportar y Analizar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Descargar Dataset Limpio (CSV)",
            data=df_imputado.to_csv(index=False).encode('utf-8'),
            file_name="dataset_al_ready_cora.csv",
            mime="text/csv", width='stretch'
        )

    with col2:
        if st.button("Confirmar y Mandar a Análisis", type="primary", width='stretch'):
            st.session_state['df_chido'] = df_imputado.copy()
            st.session_state['df_encoded'] = codificar_categoricos_inteligente(df_imputado, st.session_state['metadata'])
            st.session_state['df_scaled'] = estandarizar_zscore(st.session_state['df_encoded'], metadata=st.session_state.get('metadata'), columnas_excluir=['ID_Usuario'])

            st.success("✅ ¡Dataset bloqueado! Versiones matemáticas generadas en memoria.")
            st.balloons()
            
    if 'df_chido' in st.session_state:
        st.divider()
        st.subheader("Versiones en Memoria")

        tabs = st.tabs(["Limpio", "One-Hot Encoded", "Estandarizado (Scaled)"])

        with tabs[0]:
            try: renderizar_df_paginado(st.session_state.get('df_chido'), formatter=formatos_columnas, height=300, key="result_chido")
            except Exception: st.dataframe(st.session_state.get('df_chido'))

        with tabs[1]:
            try: renderizar_df_paginado(st.session_state.get('df_encoded'), height=300, key="result_encoded")
            except Exception: st.dataframe(st.session_state.get('df_encoded'))

        with tabs[2]:
            try: renderizar_df_paginado(st.session_state.get('df_scaled'), height=300, key="result_scaled")
            except Exception: st.dataframe(st.session_state.get('df_scaled'))

# --- MÓDULOS DE TUS COMPAS ---
else:
    st.info("👆 Sube un dataset para comenzar el proceso de limpieza.")
    st.stop()
