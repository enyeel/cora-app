import streamlit as st
import pandas as pd

# =======================================================
# FUNCIONES DE RENDERIZADO Y LAYOUT COMPARTIDO
# =======================================================
def pintor_universal(data, df_orig=None, mapa_out=None, filas_web=None, pintar_nulos=True, pintar_outliers=True, pintar_webones=True):
    """Genera los estilos CSS para cualquier tabla de la app basado en reglas."""
    estilos = pd.DataFrame('', index=data.index, columns=data.columns)
    
    if '_orig_index' in data.columns:
        idx_real = data['_orig_index'].values
    else:
        idx_real = data.index

    if pintar_webones and filas_web is not None:
        es_webon = filas_web.loc[idx_real].values
        estilos.loc[es_webon, :] = 'background-color: rgba(150, 75, 255, 0.3);'

    if pintar_outliers and mapa_out is not None and df_orig is not None:
        for col in data.columns:
            if col in mapa_out.columns:
                es_outlier = mapa_out.loc[idx_real, col].values
                val_actual = data[col].values
                val_orig = df_orig.loc[idx_real, col].values
                sigue_igual = (val_actual == val_orig)
                estilos.loc[es_outlier & sigue_igual, col] = 'background-color: rgba(75, 150, 255, 0.5); font-weight: bold;'

    if pintar_nulos:
        estilos[data.isna()] = 'background-color: rgba(255, 75, 75, 0.6);'
        
    return estilos


def renderizar_df_paginado(
    df,
    pintar_func=None,
    style_mode="apply",
    formatter=None,
    height=300,
    page_size=200,
    key="tabla"
):
    total_filas = len(df)
    total_paginas = max(1, (total_filas - 1) // page_size + 1)

    col1, col2 = st.columns([1,4])

    with col1:
        pagina = st.number_input(
            "Página",
            min_value=1,
            max_value=total_paginas,
            value=1,
            key=f"{key}_pagina"
        )
    with col2:
        st.caption(f"Mostrando filas {(pagina-1)*page_size+1} - {min(pagina*page_size,total_filas)} de {total_filas}")

    start = (pagina-1) * page_size
    end = start + page_size

    df_page = df.iloc[start:end]

    if pintar_func:
        styler = df_page.style

        if style_mode == "apply":
            styler = styler.apply(pintar_func, axis=None)
        elif style_mode == "map":
            metodo_map = getattr(styler, "map", getattr(styler, "applymap", None))
            styler = metodo_map(pintar_func)
        if formatter:
            styler = styler.format(formatter=formatter, na_rep="NaN")

        st.dataframe(
            styler,
            height=height,
            width="stretch"
        )

    else:
        if formatter:
            df_page = df_page.style.format(formatter=formatter, na_rep="NaN")

        st.dataframe(
            df_page,
            height=height,
            width="stretch"
        )


def render_sidebar():
    """Renderiza el sidebar estandar de la aplicación.

    Mantiene branding y enlaces; sin emojis para conservar tono profesional.
    """
    with st.sidebar:
        st.markdown("## CORA by DIA")
        st.caption("Data Intelligence & Analytics")
        st.caption("Software desarrollado en el Bajío Valley")
        st.divider()
        st.caption("© 2026 DIA. Todos los derechos reservados a los 11 fundadores (CDIA).")
