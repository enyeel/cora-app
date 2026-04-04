import streamlit as st

def renderizar_df_paginado(
    df,
    pintar_func=None,
    style_mode="apply",   # "apply" o "map"
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

    # aplicar estilos
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
