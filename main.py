import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats# Configuración de la página
st.set_page_config(
    page_title="Análisis Predictivo de Precios y Reseñas en Airbnb",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5A5F;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.8rem;
        color: #484848;
        padding: 0.5rem 0;
        border-bottom: 2px solid #FF5A5F;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #484848;
        padding: 0.3rem 0;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F7F7F7;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF5A5F;
    }
    .metric-label {
        font-size: 1rem;
        color: #767676;
    }
    .footer {
        text-align: center;
        padding: 1rem 0;
        color: #767676;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)# Título e introducción
st.markdown('<div class="main-header">Análisis de Datos de Airbnb en España 2024</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("""
    <div class="info-box">
    Bienvenido al dashboard interactivo para el análisis de datos de Airbnb en diferentes ciudades de España (2024).  
    Este proyecto, parte de mi TFG, explora la <b>predicción de precios</b> mediante modelos de aprendizaje automático y el 
    <b>análisis de reseñas</b> usando procesamiento de lenguaje natural.
    </div>
    """, unsafe_allow_html=True)st.markdown("<p style='text-align: right;'><i>Autor: Ángel Soto García - Grado en Ciencia de Datos - UOC</i></p>", unsafe_allow_html=True)# Diccionario de ciudades y URLs
ciudades_urls = {
    "Barcelona": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_barcelona.parquet",
    "Euskadi": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_euskadi.parquet",
    "Girona": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_girona.parquet",
    "Madrid": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_madrid.parquet",
    "Mallorca": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_mallorca.parquet",
    "Menorca": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_menorca.parquet",
    "Málaga": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_málaga.parquet",
    "Sevilla": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_sevilla.parquet",
    "Valencia": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_valencia.parquet"
}# Sidebar para selección de ciudad y filtros
st.sidebar.markdown("<h2 style='text-align: center; color: #FF5A5F;'>Controles</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Selección de Ciudad</h3>", unsafe_allow_html=True)
ciudad_seleccionada = st.sidebar.selectbox("Selecciona una ciudad:", list(ciudades_urls.keys()))# Carga de datos con barra de progreso
with st.spinner(f"Cargando datos de {ciudad_seleccionada}..."):
    try:
        data = pd.read_parquet(ciudades_urls[ciudad_seleccionada])
        st.sidebar.success(f"Datos de {ciudad_seleccionada} cargados correctamente.")
    except Exception as e:
        st.error(f"Error al cargar los datos de {ciudad_seleccionada}: {e}")
        st.stop()# Validar columnas esenciales
required_columns = ["neighbourhood_cleansed", "room_type", "price", "number_of_reviews", "minimum_nights", "latitude", "longitude"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Faltan las siguientes columnas en los datos: {', '.join(missing_columns)}")
    st.stop()# Limpiar datos
data["neighbourhood_cleansed"] = data["neighbourhood_cleansed"].astype(str).replace("nan", None)
neighborhoods_options = [n for n in data["neighbourhood_cleansed"].unique() if n is not None]
room_type_options = [str(room) for room in data["room_type"].unique() if pd.notna(room) and room is not None]# Convertir columnas numéricas y manejar valores no válidos
for col in ["price", "latitude", "longitude", "number_of_reviews", "minimum_nights"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")# Filtros en sidebar
st.sidebar.markdown("<h3>Filtros</h3>", unsafe_allow_html=True)
neighborhoods = st.sidebar.multiselect(
    "Seleccionar vecindarios",
    options=neighborhoods_options,
    default=neighborhoods_options[:5] if len(neighborhoods_options) > 5 else neighborhoods_options
)
room_types = st.sidebar.multiselect(
    "Seleccionar tipos de habitación",
    options=room_type_options,
    default=room_type_options
)
price_min = float(data["price"].min()) if not data["price"].isna().all() else 0.0
price_max = float(data["price"].max()) if not data["price"].isna().all() else 1000.0
price_range = st.sidebar.slider(
    "Rango de precios (€)",
    min_value=int(price_min),
    max_value=min(int(price_max), 1000),
    value=(int(price_min), min(int(price_max), 500)),
    step=10
)
min_reviews = st.sidebar.slider(
    "Número mínimo de reseñas",
    min_value=0,
    max_value=int(data["number_of_reviews"].max()) if not data["number_of_reviews"].isna().all() else 100,
    value=0
)
min_nights_range = st.sidebar.slider(
    "Rango de noches mínimas",
    min_value=int(data["minimum_nights"].min()) if not data["minimum_nights"].isna().all() else 1,
    max_value=min(int(data["minimum_nights"].max()), 30) if not data["minimum_nights"].isna().all() else 30,
    value=(1, 7)
)# Filtrar datos
filtered_data = data[
    (data["neighbourhood_cleansed"].isin(neighborhoods)) &
    (data["price"].ge(price_range[0])) &
    (data["price"].le(price_range[1])) &
    (data["room_type"].isin(room_types)) &
    (data["number_of_reviews"].ge(min_reviews)) &
    (data["minimum_nights"].ge(min_nights_range[0])) &
    (data["minimum_nights"].le(min_nights_range[1]))
].copy()# Verificar si hay datos filtrados
if len(filtered_data) == 0:
    st.warning("No hay datos que cumplan con los filtros seleccionados. Ajusta los filtros e intenta de nuevo.")
    st.stop()# Procesar variables
filtered_data["host_since"] = pd.to_datetime(filtered_data["host_since"], errors="coerce")
filtered_data["host_age_years"] = (datetime.now() - filtered_data["host_since"]).dt.days / 365
filtered_data["occupancy_rate"] = (365 - filtered_data["availability_365"]) / 365
filtered_data["price_per_person"] = filtered_data["price"] / filtered_data["accommodates"].replace(0, 1)
filtered_data["log_price"] = np.log1p(filtered_data["price"])# Manejar host_response_rate
if "host_response_rate" in filtered_data.columns and filtered_data["host_response_rate"].dtype == object:
    filtered_data["host_response_rate"] = filtered_data["host_response_rate"].str.rstrip("%").astype(float) / 100# Procesar amenidades
if "amenities" in filtered_data.columns:
    def parse_amenities(amenities):
        if isinstance(amenities, str):
            try:
                return eval(amenities) if amenities else []
            except:
                return []
        return amenities if isinstance(amenities, list) else []

filtered_data["amenities"] = filtered_data["amenities"].apply(parse_amenities)
all_amenities = []
for amenity_list in filtered_data["amenities"]:
    if isinstance(amenity_list, list):
        all_amenities.extend(amenity_list)
common_amenities = [item[0] for item in Counter(all_amenities).most_common(10)]
for amenity in common_amenities:
    filtered_data[f"has_{amenity}"] = filtered_data["amenities"].apply(lambda x: amenity in x if isinstance(x, list) else False)

# Verificar si hay suficientes datos filtrados
if len(filtered_data) < 5:
    st.warning(f"Solo hay {len(filtered_data)} alojamientos con los filtros actuales. Por favor, ajusta los filtros para ver más datos.")# Panel de métricas resumidas
st.markdown('<div class="subheader">Métricas Clave</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">€{filtered_data['price'].median():.2f}</div>
        <div class="metric-label">Precio Mediano</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(filtered_data)}</div>
        <div class="metric-label">Alojamientos</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_data['review_scores_rating'].mean():.1f}</div>
        <div class="metricAlles: Puntuación Media</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_data['occupancy_rate'].mean():.1%}</div>
        <div class="metric-label">Ocupación Media</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_data['host_age_years'].mean():.1f}</div>
        <div class="metric-label">Años de Anfitrión</div>
    </div>
    """, unsafe_allow_html=True)# Sección de visualizaciones
st.markdown(f'<div class="subheader">Visualizaciones para {ciudad_seleccionada}</div>', unsafe_allow_html=True)
tabs = st.tabs([
    "Distribución Geográfica",
    "Análisis de Precios",
    "Características del Alojamiento",
    "Análisis de Reseñas",
    "Modelo Predictivo"
])with tabs[0]:
    st.markdown('<div class="section-header">Distribución Geográfica de Alojamientos</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        # Verificar datos para el mapa
        if (len(filtered_data) > 0 and
            not filtered_data[["latitude", "longitude", "price"]].isna().any().any()):
            map_data = filtered_data.sample(min(len(filtered_data), 1000)).copy()
            map_data = map_data.dropna(subset=["latitude", "longitude", "price"])
            map_data["latitude"] = map_data["latitude"].astype(float)
            map_data["longitude"] = map_data["longitude"].astype(float)
            map_data["price"] = map_data["price"].astype(float)

        fig = px.scatter_mapbox(
            map_data,
            lat="latitude",
            lon="longitude",
            color="price",
            size="number_of_reviews",
            hover_name="name",
            hover_data=["price", "room_type", "review_scores_rating"],
            zoom=11,
            color_continuous_scale=px.colors.sequential.Plasma,
            opacity=0.7,
            title=""
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay suficientes datos válidos con coordenadas o precios para mostrar el mapa.")

with col2:
    st.markdown('<div class="section-header">Distribución por Vecindario</div>', unsafe_allow_html=True)
    neighbourhood_counts = filtered_data["neighbourhood_cleansed"].value_counts().head(10)
    fig = px.bar(
        x=neighbourhood_counts.values,
        y=neighbourhood_counts.index,
        orientation="h",
        labels={"x": "Número de Alojamientos", "y": "Vecindario"},
        color=neighbourhood_counts.values,
        color_continuous_scale=px.colors.sequential.Viridis,
        title=""
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Distribución de Precios</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Distribución Original (Asimétrica)", "Distribución Log-transformada"))
        fig.add_trace(
            go.Histogram(
                x=filtered_data["price"].clip(upper=filtered_data["price"].quantile(0.95)),
                nbinsx=30,
                marker_color="#FF5A5F",
                name="Precio Original"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=filtered_data["log_price"],
                nbinsx=30,
                marker_color="#00A699",
                name="Log-Precio"
            ),
            row=2, col=1
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.9rem;">
         <b>Nota sobre la distribución de precios:</b> La variable precio muestra una marcada asimetría positiva (cola derecha), 
        común en datos económicos. La transformación logarítmica ayuda a normalizar esta distribución, 
        facilitando el análisis estadístico y mejorando los modelos predictivos.
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Precios por Vecindario</div>', unsafe_allow_html=True)
    price_by_neighbourhood = filtered_data.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False).head(10)
    fig = px.bar(
        x=price_by_neighbourhood.values,
        y=price_by_neighbourhood.index,
        orientation="h",
        labels={"x": "Precio Mediano (€)", "y": "Vecindario"},
        color=price_by_neighbourhood.values,
        olor_continuous_scale=px.colors.sequential.Plasma,
        title=""
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Comparación por Tipo de Habitación</div>', unsafe_allow_html=True)
    fig = px.box(
        filtered_data,
        x="room_type",
        y="price",
        color="room_type",
        labels={"price": "Precio (€)", "room_type": "Tipo de Habitación"},
        title="",
        category_orders={"room_type": sorted(filtered_data["room_type"].unique())},
        points="outliers"
    )
    fig.update_layout(xaxis={"categoryorder": "total descending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Capacidad vs. Precio</div>', unsafe_allow_html=True)
        fig = px.scatter(
            filtered_data,
            x="accommodates",
            y="price",
            color="room_type",
            size="review_scores_rating",
            hover_name="name",
            hover_data=["bedrooms", "bathrooms", "price"],
            opacity=0.7,
            title=""
        )
        fig.add_trace(
            go.Scatter(
                x=sorted(filtered_data["accommodates"].unique()),
                y=[filtered_data[filtered_data["accommodates"]==accom]["price"].median() for accom in sorted(filtered_data["accommodates"].unique())],
                mode="lines+markers",
                name="Precio Mediano",
                line=dict(color="red", width=2, dash="dot")
            )
        )
        fig.update_layout(
            xaxis_title="Capacidad (personas)",
            yaxis_title="Precio (€)",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">Relación Dormitorios/Baños vs. Precio</div>', unsafe_allow_html=True)
    fig = px.scatter(
        filtered_data,
        x="bedrooms",
        y="bathrooms",
        size="price",
        color="price",
        hover_name="name",
        color_continuous_scale=px.colors.sequential.Viridis,
        title=""
    )
    fig.update_layout(
        xaxis_title="Número de Dormitorios",
        yaxis_title="Número de Baños",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="section-header">Amenidades más Comunes</div>', unsafe_allow_html=True)
    if "amenities" in filtered_data.columns and common_amenities:
        amenities_df = pd.DataFrame(Counter(all_amenities).most_common(15), columns=["amenity", "count"])
        fig = px.bar(
            amenities_df,
            x="count",
            y="amenity",
            orientation="h",
            color="count",
            color_continuous_scale=px.colors.sequential.Viridis,
            title=""
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos de amenidades disponibles para esta ciudad.")

with col2:
    st.markdown('<div class="section-header">Disponibilidad vs. Precio</div>', unsafe_allow_html=True)
    fig = px.scatter(
        filtered_data,
        x="availability_365",
        y="price",
        color="room_type",
        opacity=0.7,
        labels={"availability_365": "Disponibilidad (días/año)", "price": "Precio (€)"},
        title=""
    )
    x_range = np.array([0, 365])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_data["availability_365"].fillna(0),
        filtered_data["price"].fillna(filtered_data["price"].mean())
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=slope * x_range + intercept,
            mode="lines",
            name=f"Tendencia (r={r_value:.2f})",
            line=dict(color="red", width=2, dash="dot")
        )
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    # Datos para análisis de reseñas
    data_dia_semana = pd.DataFrame({
        "Día": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
        "vader_compound": [0.776859, 0.783723, 0.776271, 0.773506, 0.775032, 0.776225, 0.764861],
        "num_reseñas": [7200, 7500, 7300, 7100, 7400, 7600, 6900]
    })

data_clusters = pd.DataFrame({
    "cluster": [0, 1, 2],
    "count": [7130.0, 3837.0, 39033.0],
    "mean": [0.853238, 0.720461, 0.765594],
    "std": [0.130881, 0.207740, 0.300997],
    "min": [-0.8899, -0.7579, -0.9835],
    "25%": [0.7906, 0.5267, 0.7096],
    "50%": [0.8977, 0.7783, 0.8885],
    "75%": [0.9460, 0.8885, 0.9524],
    "max": [0.9970, 0.9948, 0.9986]
})

data_mensual = pd.DataFrame({
    "year_month": [
        "2011-01", "2011-04", "2011-05", "2011-06", "2011-07", "2011-08", "2011-09", "2011-11", "2011-12",
        "2012-01", "2012-02", "2012-03", "2012-05", "2012-06", "2012-07", "2012-08", "2012-09", "2012-10",
        "2012-11", "2012-12", "2013-01", "2013-02", "2013-03", "2013-04", "2013-05", "2013-06", "2013-07",
        "2013-08", "2013-09", "2013-10", "2013-11", "2013-12", "2014-01", "2014-02", "2014-03", "2014-04",
        "2014-05", "2014-06", "2014-07", "2014-08", "2014-09", "2014-10", "2014-11", "2014-12", "2015-01",
        "2015-02", "2015-03", "2015-04", "2015-05", "2015-06", "2015-07", "2015-08", "2015-09", "2015-10",
        "2015-11", "2015-12", "2016-01", "2016-02", "2016-03", "2016-04", "2016-05", "2016-06", "2016-07",
        "2016-08", "2016-09", "2016-10", "2016-11", "2016-12", "2017-01", "2017-02", "2017-03", "2017-04",
        "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01",
        "2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08", "2018-09", "2018-10",
        "2018-11", "2018-12", "2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06", "2019-07",
        "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04",
        "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01",
        "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08", "2021-09", "2021-10",
        "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07",
        "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04",
        "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01",
        "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10",
        "2024-11", "2024-12"
    ],
    "vader_compound": [
        0.930000, 0.982000, 0.895700, 0.841100, 0.897150, 0.926600, 0.925667, 0.950100, 0.950100,
        0.929650, 0.977467, 0.968800, 0.879814, 0.954233, 0.922167, 0.930478, 0.938389, 0.958967,
        0.955725, 0.867740, 0.868100, 0.903800, 0.884836, 0.867258, 0.894208, 0.941038, 0.928197,
        0.933975, 0.928989, 0.920045, 0.892260, 0.857892, 0.873842, 0.890317, 0.915682, 0.931698,
        0.904630, 0.917828, 0.895731, 0.877517, 0.908437, 0.884002, 0.920922, 0.846467, 0.880280,
        0.886132, 0.877107, 0.884918, 0.894229, 0.877178, 0.841583, 0.874817, 0.890381, 0.879286,
        0.889026, 0.838844, 0.845016, 0.857054, 0.870728, 0.887189, 0.874846, 0.850709, 0.842540,
        0.840764, 0.847850, 0.839682, 0.808977, 0.811831, 0.786819, 0.816883, 0.803289, 0.819001,
        0.793763, 0.816881, 0.786060, 0.809665, 0.808683, 0.833374, 0.808800, 0.772921, 0.795341,
        0.788197, 0.780953, 0.786639, 0.795594, 0.824602, 0.794505, 0.793590, 0.800579, 0.813679,
        0.791136, 0.759824, 0.749889, 0.763619, 0.770189, 0.767264, 0.786655, 0.785676, 0.796091,
        0.788698, 0.797587, 0.782660, 0.765938, 0.778965, 0.774635, 0.768214, 0.730183, 0.688162,
        0.670969, 0.652791, 0.720687, 0.650906, 0.678824, 0.688786, 0.732879, 0.660308, 0.700060,
        0.620922, 0.747775, 0.721271, 0.745978, 0.742302, 0.745000, 0.716813, 0.751618, 0.729861,
        0.734373, 0.737536, 0.716178, 0.721339, 0.726500, 0.762065, 0.786046, 0.784476, 0.775399,
        0.760688, 0.761779, 0.750654, 0.733250, 0.735556, 0.726192, 0.726420, 0.747397, 0.753837,
        0.767235, 0.764159, 0.770953, 0.736370, 0.773050, 0.765165, 0.749125, 0.720443, 0.730310,
        0.742275, 0.738527, 0.761762, 0.769454, 0.762163, 0.760203, 0.744064, 0.728979, 0.681660,
        0.440333, 0.698543
    ]
})

resumen_general = {
    "total_reseñas": 50000,
    "total_usuarios": 49812,
    "periodo": {"inicio": "2011-01-04", "fin": "2024-12-25"},
    "promedio_sentimiento": 0.7746285680000001,
    "sentimiento_min": -0.9835,
    "sentimiento_max": 0.9986
}

clusters = {
    "cluster_0": {
        "num_reseñas": 7130,
        "porcentaje": 14.26,
        "sentimiento_promedio": 0.8532384852734924,
        "palabras_clave": ["great", "location", "great location", "stay", "place", "apartment", "host", "great place", "clean", "recommend"]
    },
    "cluster_1": {
        "num_reseñas": 3837,
        "porcentaje": 7.674,
        "sentimiento_promedio": 0.7204611154547824,
        "palabras_clave": ["good", "location", "good location", "apartment", "place", "stay", "clean", "nice", "host", "good place"]
    },
    "cluster_2": {
        "num_reseñas": 39033,
        "porcentaje": 78.066,
        "sentimiento_promedio": 0.7655939512720007,
        "palabras_clave": ["apartment", "stay", "place", "nice", "location", "clean", "recommend", "great", "perfect", "host"]
    }
}

temas_principales = {
    "Tema_1": "0.028*\"check\" + 0.021*\"help\" + 0.020*\"give\" + 0.017*\"time\" + 0.015*\"arrive\" + 0.012*\"leave\" + 0.011*\"arrival\" + 0.011*\"not\" + 0.011*\"question\" + 0.010*\"early\"",
    "Tema_2": "0.030*\"room\" + 0.018*\"not\" + 0.016*\"bed\" + 0.016*\"bathroom\" + 0.016*\"small\" + 0.013*\"kitchen\" + 0.013*\"night\" + 0.011*\"work\" + 0.010*\"shower\" + 0.010*\"bedroom\"",
    "Tema_3": "0.074*\"accommodation\" + 0.051*\"pleasant\" + 0.028*\"welcome\" + 0.025*\"locate\" + 0.016*\"description\" + 0.014*\"foot\" + 0.露天直播间: 0.012*\"functional\" + 0.011*\"photo\" + 0.011*\"available\" + 0.011*\"practical\"",
    "Tema_4": "0.040*\"apartment\" + 0.038*\"great\" + 0.037*\"stay\" + 0.036*\"location\" + 0.027*\"place\" + 0.027*\"good\" + 0.025*\"clean\" + 0.020*\"recommend\" + 0.019*\"nice\" + 0.017*\"host\"",
    "Tema_5": "0.064*\"house\" + 0.034*\"attentive\" + 0.033*\"hostel\" + 0.020*\"attention\" + 0.018*\"department\" + 0.014*\"floor\" + 0.012*\"position\" + 0.011*\"doubt\" + 0.011*\"meter\" + 0.010*\"wide\""
}

st.subheader("Análisis de Clusters Identificados")
st.markdown("""
Esta sección presenta un análisis detallado de los clusters de reseñas identificados, incluyendo la actividad, el sentimiento promedio por día de la semana, la distribución de sentimientos por cluster y la evolución temporal del sentimiento.
""")

# Resumen General
st.markdown("### Resumen General")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Reseñas", resumen_general["total_reseñas"])
with col2:
    st.metric("Total de Usuarios", resumen_general["total_usuarios"])
with col3:
    st.metric("Promedio de Sentimiento", f"{resumen_general['promedio_sentimiento']:.3f}")
st.markdown(f"*Período de Análisis*: {resumen_general['periodo']['inicio']} - {resumen_general['periodo']['fin']}")
st.markdown(f"*Rango de Sentimiento*: {resumen_general['sentimiento_min']:.4f} a {resumen_general['sentimiento_max']:.4f}")

# Información de Clusters
st.markdown("### Detalles de los Clusters")
for cluster, info in clusters.items():
    st.markdown(f"{cluster.replace('_', ' ').title()}")
    st.markdown(f"- *Número de Reseñas*: {info['num_reseñas']} ({info['porcentaje']:.2f}%)")
    st.markdown(f"- *Sentimiento Promedio*: {info['sentimiento_promedio']:.3f}")
    st.markdown(f"- *Palabras Clave*: {', '.join(info['palabras_clave'])}")

# Temas Principales
st.markdown("### Temas Principales Identificados")
for tema, descripcion in temas_principales.items():
    st.markdown(f"{tema}: {descripcion}")

# Gráfico 1: Actividad y Sentimiento por Día de la Semana
st.markdown("### Actividad y Sentimiento por Día de la Semana")
st.markdown("""
*Descripción*: Gráfico combinado que muestra la actividad y el sentimiento promedio de las reseñas por día de la semana. Las barras indican el número de reseñas, mientras que la línea roja muestra el sentimiento promedio (calculado con VADER).  
*Información Adicional*: Período de análisis: Semanal (por día)  
Método de análisis de sentimiento: VADER
""")
fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=data_dia_semana["Día"],
        y=data_dia_semana["num_reseñas"],
        name="Número de Reseñas",
        marker_color="skyblue"
    )
)
fig1.add_trace(
    go.Scatter(
        x=data_dia_semana["Día"],
        y=data_dia_semana["vader_compound"],
        name="Sentimiento Promedio",
        line=dict(color="red", width=2),
        yaxis="y2"
    )
)
fig1.update_layout(
    title="Actividad y Sentimiento por Día de la Semana",
    xaxis=dict(title="Día de la Semana"),
    yaxis=dict(title="Número de Reseñas", titlefont=dict(color="skyblue"), tickfont=dict(color="skyblue")),
    yaxis2=dict(title="Sentimiento Promedio (VADER)", titlefont=dict(color="red"), tickfont=dict(color="red"), overlaying="y", side="right"),
    legend=dict(x=0.01, y=0.99),
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Distribución de Sentimiento por Cluster
st.markdown("### Distribución de Sentimiento por Cluster")
st.markdown("""
*Descripción*: Diagrama de caja que muestra la distribución de las puntuaciones de sentimiento (calculadas con VADER) para cada cluster de reseñas. Permite comparar el rango y la mediana del sentimiento entre clusters.  
*Información Adicional*: Número de clusters: 3  
Método de análisis de sentimiento: VADER
""")
fig2 = go.Figure()
for cluster in data_clusters["cluster"]:
    cluster_data = data_clusters[data_clusters["cluster"] == cluster]
    fig2.add_trace(
        go.Box(
            y=[
                cluster_data["min"].iloc[0],
                cluster_data["25%"].iloc[0],
                cluster_data["50%"].iloc[0],
                cluster_data["75%"].iloc[0],
                cluster_data["max"].iloc[0]
            ],
            name=f"Cluster {cluster}",
            boxpoints=False
        )
    )
fig2.update_layout(
    title="Distribución de Sentimiento por Cluster",
    yaxis=dict(title="Sentimiento (VADER)"),
    xaxis=dict(title="Cluster"),
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3: Evolución del Sentimiento Mensual
st.markdown("### Evolución del Sentimiento Mensual")
st.markdown("""
*Descripción*: Gráfico de líneas que muestra la evolución del sentimiento promedio (calculado con VADER) de las reseñas a lo largo del tiempo, agrupado por mes.  
*Información Adicional*: Período de análisis: Mensual  
Método de análisis de sentimiento: VADER
""")
fig3 = px.line(
    data_mensual,
    x="year_month",
    y="vader_compound",
    title="Evolución del Sentimiento Promedio Mensual",
    labels={"year_month": "Mes", "vader_compound": "Sentimiento Promedio (VADER)"}
)
fig3.update_xaxes(tickangle=45)
fig3.update_layout(height=500)
st.plotly_chart(fig3, use_container_width=True)

with tabs[4]:
    st.markdown('<div class="section-header">Modelo Predictivo</div>', unsafe_allow_html=True)
    st.info("La implementación del modelo predictivo está en desarrollo. Se incluirán modelos de regresión para predecir precios basados en las características del alojamiento.")# Pie de página
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")

