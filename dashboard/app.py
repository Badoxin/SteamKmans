# Dashboard: Segmentacion de juegos de Steam con K-Means
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Segmentacion de juegos de Steam", layout="wide")

COLORES = {"Premium consolidados": "#2a6f97", "Indies de nicho": "#6a994e",
           "Baratos populares": "#f4a259", "Mal calificados": "#bc4749"}
ORDEN = list(COLORES.keys())


@st.cache_data
def cargar():
    datos = pd.read_csv(RAIZ / "datos" / "steam_procesado.csv", encoding="utf-8")
    metricas = pd.read_csv(RAIZ / "resultados" / "metricas_k.csv")
    return datos, metricas


df_total, metricas = cargar()

# Barra lateral: pagina y 3 filtros
pagina = st.sidebar.radio("Pagina", ["1. Analisis general", "2. Segmentacion K-Means"])
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

# Filtro 1: segmento
segmentos = st.sidebar.multiselect("Segmento", ORDEN, default=ORDEN)
# Filtro 2: rango de precio
p_max = float(df_total["price"].max())
rango_precio = st.sidebar.slider("Precio (USD)", 0.0, p_max, (0.0, p_max), step=0.5)
# Filtro 3: anio de lanzamiento
a_min, a_max = int(df_total["release_year"].min()), int(df_total["release_year"].max())
rango_anio = st.sidebar.slider("Anio de lanzamiento", a_min, a_max, (a_min, a_max))

df = df_total[
    df_total["segmento"].isin(segmentos)
    & df_total["price"].between(*rango_precio)
    & df_total["release_year"].between(*rango_anio)
]

# Titulo
st.title("Segmentacion de juegos de Steam mediante K-Means")
if df.empty:
    st.warning("No hay juegos con esos filtros.")
    st.stop()

# PAGINA 1: ANALISIS GENERAL
if pagina.startswith("1"):
    st.header("Analisis general")

    # 4 KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Juegos analizados", f"{len(df):,}")
    c2.metric("Precio mediano", "$" + f"{df['price'].median():.2f}")
    c3.metric("% positivo mediano", f"{df['rating_ratio'].median():.1f}%")
    c4.metric("Horas jugadas (mediana)", f"{df['average_playtime'].median() / 60:.1f} h")

    # Visualizaciones generales
    g1, g2 = st.columns(2)
    fig = px.histogram(df[df["price"] < 40], x="price", nbins=40,
                       title="Distribucion del precio (menores a 40 USD)",
                       labels={"price": "Precio (USD)"},
                       color_discrete_sequence=["#2a6f97"])
    g1.plotly_chart(fig, width="stretch")
    por_anio = df.groupby("release_year").size().reset_index(name="juegos")
    fig = px.bar(por_anio, x="release_year", y="juegos", title="Juegos por anio de lanzamiento",
                 labels={"release_year": "Anio", "juegos": "Juegos"},
                 color_discrete_sequence=["#2a6f97"])
    g2.plotly_chart(fig, width="stretch")

    # Estadisticas generales
    st.subheader("Estadisticas descriptivas")
    cols = ["price", "rating_ratio", "total_ratings", "owners_mid",
            "average_playtime", "achievements"]
    st.dataframe(df[cols].describe().T.round(1), width="stretch")

# PAGINA 2: SEGMENTACION K-MEANS
else:
    st.header("Segmentacion K-Means")
    conteo = df["segmento"].value_counts()
    k = int(df_total["cluster"].nunique())
    silueta = float(metricas.loc[metricas["K"] == k, "silhouette"].iloc[0])

    # 4 KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segmentos (K)", k)
    c2.metric("Coeficiente de silueta", f"{silueta:.3f}")
    c3.metric("Segmento mas grande", conteo.index[0])
    c4.metric("Juegos filtrados", f"{len(df):,}")

    # Distribucion de registros por cluster y grafica de dispersion
    a1, a2 = st.columns([1, 2])
    dist = conteo.reindex([s for s in ORDEN if s in conteo.index]).reset_index()
    dist.columns = ["segmento", "juegos"]
    fig = px.bar(dist, x="segmento", y="juegos", color="segmento", color_discrete_map=COLORES,
                 text="juegos", title="Juegos por segmento", labels={"segmento": ""})
    fig.update_layout(showlegend=False)
    a1.plotly_chart(fig, width="stretch")
    fig = px.scatter(df, x="total_ratings", y="rating_ratio", color="segmento",
                     color_discrete_map=COLORES, hover_name="name", log_x=True, opacity=0.6,
                     category_orders={"segmento": ORDEN},
                     title="Dispersion de los segmentos",
                     labels={"total_ratings": "Resenas totales (escala log)",
                             "rating_ratio": "% de resenas positivas"})
    fig.update_traces(marker=dict(size=5))
    a2.plotly_chart(fig, width="stretch")

    # Dos graficas para comparar las caracteristicas de los grupos
    medianas = df.groupby("segmento")[["price", "rating_ratio", "average_playtime",
                                       "owners_mid"]].median()
    medianas = medianas.reindex([s for s in ORDEN if s in medianas.index]).reset_index()
    medianas["horas"] = medianas["average_playtime"] / 60
    b1, b2 = st.columns(2)
    fig = px.bar(medianas, x="segmento", y="price", color="segmento",
                 color_discrete_map=COLORES, text_auto=".2f",
                 title="Precio mediano por segmento (USD)", labels={"segmento": "", "price": "USD"})
    fig.update_layout(showlegend=False)
    b1.plotly_chart(fig, width="stretch")
    fig = px.bar(medianas, x="segmento", y="rating_ratio", color="segmento",
                 color_discrete_map=COLORES, text_auto=".1f",
                 title="% de resenas positivas mediano por segmento",
                 labels={"segmento": "", "rating_ratio": "%"})
    fig.update_layout(showlegend=False)
    b2.plotly_chart(fig, width="stretch")

    # Tabla de estadisticas descriptivas de los clusteres (medianas)
    st.subheader("Estadisticas descriptivas por segmento (medianas)")
    tabla = df.groupby("segmento").agg(
        juegos=("appid", "count"), precio=("price", "median"),
        porcentaje_positivo=("rating_ratio", "median"), resenas=("total_ratings", "median"),
        propietarios=("owners_mid", "median"),
        horas_jugadas=("average_playtime", lambda s: round(s.median() / 60, 1)),
        logros=("achievements", "median"),
    ).round(2).reindex([s for s in ORDEN if s in df["segmento"].unique()])
    st.dataframe(tabla, width="stretch")

    # Descripcion de los grupos (calculada con todo el dataset)
    m = df_total.groupby("segmento").agg(
        juegos=("appid", "count"), precio=("price", "median"), positivo=("rating_ratio", "median"),
        resenas=("total_ratings", "median"), propietarios=("owners_mid", "median"),
        horas=("average_playtime", lambda s: round(s.median() / 60, 1)),
        logros=("achievements", "median"))
    gratis = (df_total["price"] == 0).groupby(df_total["segmento"]).mean() * 100

    def datos(seg):
        r = m.loc[seg]
        return (f"{int(r['juegos']):,} juegos. Mediana de precio ${r['precio']:.2f}, "
                f"{r['positivo']:.1f}% de resenas positivas, {r['resenas']:,.0f} resenas, "
                f"{r['propietarios']:,.0f} propietarios y {r['horas']:.1f} horas jugadas.")

    st.subheader("Descripcion de los grupos")
    st.markdown(
        "- **Premium consolidados:** juegos de precio alto, con la mejor calificacion, mucha "
        "audiencia y las mayores horas jugadas. " + datos("Premium consolidados") + "\n"
        "- **Indies de nicho:** el grupo mas grande; precio medio y buena calificacion, pero poca "
        "audiencia. " + datos("Indies de nicho") + "\n"
        "- **Baratos populares:** muy baratos (" + f"{gratis['Baratos populares']:.0f}% gratuitos) "
        "con buena calificacion y mucha audiencia. " + datos("Baratos populares") + "\n"
        "- **Mal calificados:** calificacion muy por debajo del resto y pocas resenas. "
        + datos("Mal calificados")
    )

    # Hallazgos y recomendaciones
    pr, ind, bar, mal = (m.loc[s] for s in ORDEN)
    st.subheader("Principales hallazgos")
    st.markdown(
        f"1. **El precio alto no atrae a mas jugadores:** los Baratos populares tienen "
        f"{bar['propietarios']:,.0f} propietarios medianos, igual que los Premium "
        f"({pr['propietarios']:,.0f}), pero cuestan ${bar['precio']:.2f} contra ${pr['precio']:.2f}.\n"
        f"2. **Los Indies de nicho son buenos pero poco visibles:** su calificacion "
        f"({ind['positivo']:.1f}%) es parecida a la de los Premium ({pr['positivo']:.1f}%), pero "
        f"tienen solo {ind['resenas']:,.0f} resenas y {ind['propietarios']:,.0f} propietarios.\n"
        f"3. **Los Mal calificados se distinguen por su calificacion:** {mal['positivo']:.1f}% de "
        f"resenas positivas contra mas de 80% en los demas grupos."
    )
    st.subheader("Recomendaciones")
    st.markdown(
        "1. **Estudios pequenos:** invertir en visibilidad (ofertas, demos, campanas), porque su "
        "calidad ya es comparable a la de los Premium.\n"
        "2. **Precio:** un precio cercano a 15 USD solo se sostiene con mucho contenido y alta "
        "calidad; con menos contenido, un modelo barato o gratuito alcanza una audiencia similar.\n"
        "3. **Plataforma:** destacar juegos con buena calificacion pero pocas resenas ayudaria a "
        "que los Indies de nicho lleguen a mas jugadores."
    )
