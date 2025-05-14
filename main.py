import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats

# Configuración de la página
st.set_page_config(
    page_title="Análisis Predictivo de Precios y Reseñas en Airbnb",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
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
    """, unsafe_allow_html=True)

# Título e introducción
st.markdown('<div class="main-header">Análisis de Datos de Airbnb en España 2024</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("""
    <div class="info-box">
    Bienvenido al dashboard interactivo para el análisis de datos de Airbnb en diferentes ciudades de España (2024).  
    Este proyecto, parte de mi TFG, explora la <b>predicción de precios</b> mediante modelos de aprendizaje automático y el 
    <b>análisis de reseñas</b> usando procesamiento de lenguaje natural.
    </div>
    """, unsafe_allow_html=True)
st.markdown("<p style='text-align: right;'><i>Autor: Ángel Soto García - Grado en Ciencia de Datos - UOC</i></p>", unsafe_allow_html=True)

# Diccionario de ciudades y URLs
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
}

# Sidebar para selección de ciudad y filtros
st.sidebar.markdown("<h2 style='text-align: center; color: #FF5A5F;'>Controles</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Selección de Ciudad</h3>", unsafe_allow_html=True)
ciudad_seleccionada = st.sidebar.selectbox("Selecciona una ciudad:", list(ciudades_urls.keys()))

# Carga de datos con barra de progreso
with st.spinner(f"Cargando datos de {ciudad_seleccionada}..."):
    try:
        data = pd.read_parquet(ciudades_urls[ciudad_seleccionada])
        st.sidebar.success(f"Datos de {ciudad_seleccionada} cargados correctamente.")
    except Exception as e:
        st.error(f"Error al cargar los datos de {ciudad_seleccionada}: {e}")
        st.stop()

# Validar columnas esenciales
required_columns = ["neighbourhood_cleansed", "room_type", "price", "number_of_reviews", "minimum_nights", "latitude", "longitude"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Faltan las siguientes columnas en los datos: {', '.join(missing_columns)}")
    st.stop()

# Limpiar datos
data["neighbourhood_cleansed"] = data["neighbourhood_cleansed"].astype(str).replace("nan", None)
neighborhoods_options = [n for n in data["neighbourhood_cleansed"].unique() if n is not None]
room_type_options = [str(room) for room in data["room_type"].unique() if pd.notna(room) and room is not None]

# Convertir columnas numéricas y manejar valores no válidos
numeric_columns = [
    "price", "latitude", "longitude", "number_of_reviews", "minimum_nights", "maximum_nights",
    "accommodates", "bathrooms", "bedrooms", "beds", "host_listings_count", "host_total_listings_count",
    "availability_365", "review_scores_rating", "review_scores_location", "review_scores_communication",
    "review_scores_cleanliness", "review_scores_checkin"
]
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Convertir tasas porcentuales
if "host_response_rate" in data.columns and data["host_response_rate"].dtype == object:
    data["host_response_rate"] = data["host_response_rate"].str.rstrip("%").astype(float) / 100
if "host_acceptance_rate" in data.columns and data["host_acceptance_rate"].dtype == object:
    data["host_acceptance_rate"] = data["host_acceptance_rate"].str.rstrip("%").astype(float) / 100

# Filtros en sidebar
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
)

# Filtrar datos
filtered_data = data[
    (data["neighbourhood_cleansed"].isin(neighborhoods)) &
    (data["price"].ge(price_range[0])) &
    (data["price"].le(price_range[1])) &
    (data["room_type"].isin(room_types)) &
    (data["number_of_reviews"].ge(min_reviews)) &
    (data["minimum_nights"].ge(min_nights_range[0])) &
    (data["minimum_nights"].le(min_nights_range[1]))
].copy()

# Verificar si hay datos filtrados
if len(filtered_data) == 0:
    st.warning("No hay datos que cumplan con los filtros seleccionados. Ajusta los filtros e intenta de nuevo.")
    st.stop()

# Procesar variables adicionales
filtered_data["host_since"] = pd.to_datetime(filtered_data["host_since"], errors="coerce")
filtered_data["host_age_years"] = (datetime.now() - filtered_data["host_since"]).dt.days / 365
filtered_data["occupancy_rate"] = (365 - filtered_data["availability_365"]) / 365 if "availability_365" in filtered_data.columns else np.nan
filtered_data["price_per_person"] = filtered_data["price"] / filtered_data["accommodates"].replace(0, 1) if "accommodates" in filtered_data.columns else np.nan
filtered_data["log_price"] = np.log1p(filtered_data["price"])
if "last_scraped" in filtered_data.columns:
    filtered_data["last_scraped"] = pd.to_datetime(filtered_data["last_scraped"], errors="coerce")

# Procesar amenidades
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
    st.warning(f"Solo hay {len(filtered_data)} alojamientos con los filtros actuales. Por favor, ajusta los filtros para ver más datos.")

# Panel de métricas resumidas
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
    avg_rating = filtered_data["review_scores_rating"].mean() if "review_scores_rating" in filtered_data.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_rating:.1f}</div>
        <div class="metric-label">Puntuación Media</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    avg_occupancy = filtered_data["occupancy_rate"].mean() if "occupancy_rate" in filtered_data.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_occupancy:.1%}</div>
        <div class="metric-label">Ocupación Media</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_data['host_age_years'].mean():.1f}</div>
        <div class="metric-label">Años de Anfitrión</div>
    </div>
    """, unsafe_allow_html=True)

# Sección de visualizaciones
st.markdown(f'<div class="subheader">Visualizaciones para {ciudad_seleccionada}</div>', unsafe_allow_html=True)
tabs = st.tabs([
    "Distribución Geográfica",
    "Análisis de Precios",
    "Características del Alojamiento",
    "Características del Anfitrión",
    "Análisis de Reseñas",
    "Modelo Predictivo"
])

# Pestaña 1: Distribución Geográfica
with tabs[0]:
    st.markdown('<div class="section-header">Distribución Geográfica de Alojamientos</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        if (len(filtered_data) > 0 and
            "latitude" in filtered_data.columns and
            "longitude" in filtered_data.columns and
            "price" in filtered_data.columns and
            not filtered_data[["latitude", "longitude", "price"]].isna().all().any()):
            try:
                map_data = filtered_data.sample(min(len(filtered_data), 1000)).copy()
                map_data = map_data.dropna(subset=["latitude", "longitude", "price"])
                map_data["latitude"] = map_data["latitude"].astype(float)
                map_data["longitude"] = map_data["longitude"].astype(float)
                map_data["price"] = map_data["price"].astype(float)
                if len(map_data) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scattermapbox(
                        lat=map_data["latitude"],
                        lon=map_data["longitude"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=map_data["price"],
                            colorscale="Viridis",
                            opacity=0.7,
                            colorbar=dict(title="Precio (€)")
                        ),
                        text=map_data.apply(
                            lambda row: f"Nombre: {row.get('name', 'N/A')}<br>"
                                        f"Precio: €{row.get('price', 0):.2f}<br>"
                                        f"Tipo: {row.get('room_type', 'N/A')}<br>"
                                        f"Puntuación: {row.get('review_scores_rating', 0):.1f}",
                            axis=1
                        ),
                        hoverinfo="text"
                    ))
                    fig.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=dict(
                            center=dict(
                                lat=map_data["latitude"].mean(),
                                lon=map_data["longitude"].mean()
                            ),
                            zoom=11
                        ),
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos válidos para mostrar el mapa.")
            except Exception as e:
                st.error(f"Error al generar el mapa: {e}")
                if len(filtered_data) > 0:
                    fig = px.scatter(
                        filtered_data.sample(min(len(filtered_data), 1000)),
                        x="longitude",
                        y="latitude",
                        color="price",
                        size="number_of_reviews",
                        hover_name="name",
                        title="Distribución de Alojamientos",
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Faltan datos de latitud, longitud o precio para mostrar el mapa.")

    with col2:
        st.markdown('<div class="section-header">Distribución por Vecindario</div>', unsafe_allow_html=True)
        if "neighbourhood_cleansed" in filtered_data.columns:
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
        else:
            st.info("La columna 'neighbourhood_cleansed' no está disponible.")

# Pestaña 2: Análisis de Precios
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Distribución de Precios</div>', unsafe_allow_html=True)
        if "price" in filtered_data.columns:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Distribución Original", "Distribución Log-transformada"))
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
        else:
            st.info("La columna 'price' no está disponible.")

        st.markdown('<div class="section-header">Precio vs. Disponibilidad Anual</div>', unsafe_allow_html=True)
        if "availability_365" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["availability_365", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="availability_365",
                    y="price",
                    labels={"availability_365": "Disponibilidad (días/año)", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'availability_365' o 'price'.")

    with col2:
        st.markdown('<div class="section-header">Precios por Vecindario</div>', unsafe_allow_html=True)
        if "neighbourhood_cleansed" in filtered_data.columns and "price" in filtered_data.columns:
            price_by_neighbourhood = filtered_data.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=price_by_neighbourhood.values,
                y=price_by_neighbourhood.index,
                orientation="h",
                labels={"x": "Precio Mediano (€)", "y": "Vecindario"},
                color=price_by_neighbourhood.values,
                color_continuous_scale=px.colors.sequential.Plasma,
                title=""
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Faltan las columnas 'neighbourhood_cleansed' o 'price'.")

        st.markdown('<div class="section-header">Precios por Tipo de Habitación</div>', unsafe_allow_html=True)
        if "room_type" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["room_type", "price"])
            if len(plot_data) > 0:
                fig = px.box(
                    plot_data,
                    x="room_type",
                    y="price",
                    color="room_type",
                    labels={"price": "Precio (€)", "room_type": "Tipo de Habitación"},
                    title="",
                    category_orders={"room_type": sorted(plot_data["room_type"].unique())},
                    points="outliers"
                )
                fig.update_layout(xaxis={"categoryorder": "total descending"}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'room_type' o 'price'.")

# Pestaña 3: Características del Alojamiento
with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Distribución de Tipos de Propiedad</div>', unsafe_allow_html=True)
        if "property_type" in filtered_data.columns:
            property_counts = filtered_data["property_type"].value_counts().head(10)
            fig = px.bar(
                x=property_counts.index,
                y=property_counts.values,
                labels={"x": "Tipo de Propiedad", "y": "Número de Alojamientos"},
                color=property_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'property_type' no está disponible.")

        st.markdown('<div class="section-header">Amenidades más Comunes</div>', unsafe_allow_html=True)
        if "amenities" in filtered_data.columns and len(common_amenities) > 0:
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
            st.info("No hay datos de amenidades disponibles.")

        st.markdown('<div class="section-header">Precio por Número de Habitaciones</div>', unsafe_allow_html=True)
        if "bedrooms" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["bedrooms", "price"])
            if len(plot_data) > 0:
                fig = px.box(
                    plot_data,
                    x="bedrooms",
                    y="price",
                    labels={"bedrooms": "Número de Habitaciones", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'bedrooms' o 'price'.")

    with col2:
        st.markdown('<div class="section-header">Precio vs. Capacidad</div>', unsafe_allow_html=True)
        if "accommodates" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["accommodates", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="accommodates",
                    y="price",
                    labels={"accommodates": "Capacidad", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'accommodates' o 'price'.")

        st.markdown('<div class="section-header">Distribución de Camas</div>', unsafe_allow_html=True)
        if "beds" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="beds",
                nbins=20,
                labels={"beds": "Número de Camas"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'beds' no está disponible.")

        st.markdown('<div class="section-header">Distribución de Baños</div>', unsafe_allow_html=True)
        if "bathrooms" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="bathrooms",
                nbins=20,
                labels={"bathrooms": "Número de Baños"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'bathrooms' no está disponible.")

# Pestaña 4: Características del Anfitrión
with tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Tasa de Respuesta del Anfitrión</div>', unsafe_allow_html=True)
        if "host_response_rate" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="host_response_rate",
                nbins=20,
                labels={"host_response_rate": "Tasa de Respuesta (%)"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_response_rate' no está disponible.")

        st.markdown('<div class="section-header">Tiempo de Respuesta del Anfitrión</div>', unsafe_allow_html=True)
        if "host_response_time" in filtered_data.columns:
            response_time_counts = filtered_data["host_response_time"].value_counts()
            fig = px.bar(
                x=response_time_counts.index,
                y=response_time_counts.values,
                labels={"x": "Tiempo de Respuesta", "y": "Número de Anfitriones"},
                color=response_time_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_response_time' no está disponible.")

        st.markdown('<div class="section-header">Antigüedad del Anfitrión</div>', unsafe_allow_html=True)
        if "host_age_years" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="host_age_years",
                nbins=20,
                labels={"host_age_years": "Años como Anfitrión"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_age_years' no está disponible.")

    with col2:
        st.markdown('<div class="section-header">Tasa de Aceptación vs. Precio</div>', unsafe_allow_html=True)
        if "host_acceptance_rate" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_acceptance_rate", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="host_acceptance_rate",
                    y="price",
                    labels={"host_acceptance_rate": "Tasa de Aceptación (%)", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'host_acceptance_rate' o 'price'.")

        st.markdown('<div class="section-header">Número de Listados vs. Precio</div>', unsafe_allow_html=True)
        if "host_listings_count" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_listings_count", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="host_listings_count",
                    y="price",
                    labels={"host_listings_count": "Número de Listados", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'host_listings_count' o 'price'.")

        st.markdown('<div class="section-header">Total de Listados del Anfitrión</div>', unsafe_allow_html=True)
        if "host_total_listings_count" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="host_total_listings_count",
                nbins=20,
                labels={"host_total_listings_count": "Total de Listados"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_total_listings_count' no está disponible.")

# Pestaña 5: Análisis de Reseñas
with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Distribución de Número de Reseñas</div>', unsafe_allow_html=True)
        if "number_of_reviews" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="number_of_reviews",
                nbins=30,
                labels={"number_of_reviews": "Número de Reseñas"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'number_of_reviews' no está disponible.")

        st.markdown('<div class="section-header">Puntuación General vs. Precio</div>', unsafe_allow_html=True)
        if "review_scores_rating" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_rating", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="review_scores_rating",
                    y="price",
                    labels={"review_scores_rating": "Puntuación General", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_rating' o 'price'.")

        st.markdown('<div class="section-header">Puntuación de Limpieza por Tipo de Habitación</div>', unsafe_allow_html=True)
        if "review_scores_cleanliness" in filtered_data.columns and "room_type" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_cleanliness", "room_type"])
            if len(plot_data) > 0:
                fig = px.box(
                    plot_data,
                    x="room_type",
                    y="review_scores_cleanliness",
                    labels={"room_type": "Tipo de Habitación", "review_scores_cleanliness": "Puntuación de Limpieza"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_cleanliness' o 'room_type'.")

    with col2:
        st.markdown('<div class="section-header">Puntuación de Ubicación por Vecindario</div>', unsafe_allow_html=True)
        if "review_scores_location" in filtered_data.columns and "neighbourhood_cleansed" in filtered_data.columns:
            top_neighbourhoods = filtered_data["neighbourhood_cleansed"].value_counts().head(10).index
            plot_data = filtered_data[filtered_data["neighbourhood_cleansed"].isin(top_neighbourhoods)].dropna(subset=["review_scores_location"])
            if len(plot_data) > 0:
                fig = px.box(
                    plot_data,
                    x="neighbourhood_cleansed",
                    y="review_scores_location",
                    labels={"neighbourhood_cleansed": "Vecindario", "review_scores_location": "Puntuación de Ubicación"},
                    title=""
                )
                fig.update_layout(xaxis={"categoryorder": "total descending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_location' o 'neighbourhood_cleansed'.")

        st.markdown('<div class="section-header">Puntuación de Comunicación vs. Precio</div>', unsafe_allow_html=True)
        if "review_scores_communication" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_communication", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="review_scores_communication",
                    y="price",
                    labels={"review_scores_communication": "Puntuación de Comunicación", "price": "Precio (€)"},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_communication' o 'price'.")

        st.markdown('<div class="section-header">Puntuación de Check-in</div>', unsafe_allow_html=True)
        if "review_scores_checkin" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="review_scores_checkin",
                nbins=20,
                labels={"review_scores_checkin": "Puntuación de Check-in"},
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'review_scores_checkin' no está disponible.")

# Pestaña 6: Modelo Predictivo
with tabs[5]:
    st.markdown('<div class="section-header">Modelo Predictivo</div>', unsafe_allow_html=True)
    st.info("La implementación del modelo predictivo está en desarrollo. Se incluirán modelos de regresión para predecir precios basados en las características del alojamiento.")

    st.markdown('<div class="section-header">Distribución de Noches Mínimas</div>', unsafe_allow_html=True)
    if "minimum_nights" in filtered_data.columns:
        fig = px.histogram(
            filtered_data,
            x="minimum_nights",
            nbins=30,
            labels={"minimum_nights": "Noches Mínimas"},
            title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'minimum_nights' no está disponible.")

    st.markdown('<div class="section-header">Distribución de Noches Máximas</div>', unsafe_allow_html=True)
    if "maximum_nights" in filtered_data.columns:
        fig = px.histogram(
            filtered_data,
            x="maximum_nights",
            nbins=30,
            labels={"maximum_nights": "Noches Máximas"},
            title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'maximum_nights' no está disponible.")

    st.markdown('<div class="section-header">Última Fecha de Recopilación</div>', unsafe_allow_html=True)
    if "last_scraped" in filtered_data.columns:
        last_scraped_counts = filtered_data["last_scraped"].value_counts().sort_index()
        fig = px.bar(
            x=last_scraped_counts.index,
            y=last_scraped_counts.values,
            labels={"x": "Fecha de Recopilación", "y": "Número de Alojamientos"},
            title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'last_scraped' no está disponible.")

# Pie de página
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")
