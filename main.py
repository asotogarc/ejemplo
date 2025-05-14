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
    page_title="Análisis de Precios y Reseñas en Airbnb",
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
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        color: #484848;
        padding: 0.3rem 0;
        margin-top: 1rem;
        text-align: center;
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
        background-color: #6057FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .importance-box {
        background-color: #F7F7F7;
        border-left: 5px solid #FF5A5F;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .divider-horizontal {
        border-top: 1px solid #ddd;
        margin: 1rem 0;
    }
    .divider-vertical {
        border-left: 1px solid #ddd;
        height: 100%;
        margin: 0 1rem;
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
    Panel interactivo para el análisis de un conjunto datos de Airbnb en diferentes ciudades de España (2024).  
    </div>
    """, unsafe_allow_html=True)



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
    "Puntuación, Limpieza y Ubicación",
    "Características temporales",
    "Características de Usuarios"
])

# Pestaña 1: Distribución Geográfica
with tabs[0]:
#    st.markdown('<div class="section-header">Distribución Geográfica de Alojamientos</div>', unsafe_allow_html=True)
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
                            lambda row: 
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
                        height=500,
                        title=dict(text="Distribución Geográfica de Alojamientos", font=dict(color="white"), x=0.5)
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
                    fig.update_layout(title=dict(text="Distribución de Alojamientos", font=dict(color="white"), x=0.5))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Faltan datos de latitud, longitud o precio para mostrar el mapa.")
    
    with col2:
        if "neighbourhood_cleansed" in filtered_data.columns:
            neighbourhood_counts = filtered_data["neighbourhood_cleansed"].value_counts().head(10)
            fig = px.bar(
                x=neighbourhood_counts.values,
                y=neighbourhood_counts.index,
                orientation="h",
                labels={"x": "Número de Alojamientos", "y": "Vecindario"},
                color=neighbourhood_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Distribución por Vecindario"
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                title=dict(text="Distribución por Vecindario", font=dict(color="white"), x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'neighbourhood_cleansed' no está disponible.")

# Pestaña 2: Análisis de Precios
with tabs[1]:
    col1, col2 = st.columns([1, 1])
    with col1:
        if "price" in filtered_data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Distribución Original", "Distribución Log-transformada")
            )
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
            fig.update_layout(
                height=500,
                showlegend=False,
                title=dict(text="Distribución de Precios de Alojamientos", font=dict(color="white"), x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'price' no está disponible.")
        
        

        if "availability_365" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["availability_365", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="availability_365",
                    y="price",
                    labels={"availability_365": "Disponibilidad (días/año)", "price": "Precio (€)"},
                    title="Relación entre Precio y Disponibilidad Anual"
                )
                fig.update_layout(title=dict(text="Relación entre Precio y Disponibilidad Anual", font=dict(color="white"), x=0.5))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'availability_365' o 'price'.")

    with col2:
        if "neighbourhood_cleansed" in filtered_data.columns and "price" in filtered_data.columns:
            price_by_neighbourhood = filtered_data.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=price_by_neighbourhood.values,
                y=price_by_neighbourhood.index,
                orientation="h",
                labels={"x": "Precio Mediano (€)", "y": "Vecindario"},
                color=price_by_neighbourhood.values,
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Precios Medios por Vecindario"
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                title=dict(text="Precios Medios por Vecindario", font=dict(color="white"), x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Faltan las columnas 'neighbourhood_cleansed' o 'price'.")
        
        
        

        if "room_type" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["room_type", "price"])
            if len(plot_data) > 0:
                min_points = 1
                category_counts = plot_data["room_type"].value_counts()
                valid_categories = category_counts[category_counts >= min_points].index.tolist()
                if valid_categories:
                    plot_data_filtered = plot_data[plot_data["room_type"].isin(valid_categories)]
                    try:
                        fig = px.box(
                            plot_data_filtered,
                            x="room_type",
                            y="price",
                            color="room_type",
                            labels={"price": "Precio (€)", "room_type": "Tipo de Habitación"},
                            title="Distribución de Precios por Tipo de Habitación",
                            category_orders={"room_type": sorted(valid_categories)},
                            points="outliers"
                        )
                        fig.update_layout(
                            xaxis={"categoryorder": "total descending"},
                            showlegend=False,
                            title=dict(text="Distribución de Precios por Tipo de Habitación", font=dict(color="white"), x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de caja: {e}")
                        st.write("Valores únicos en 'room_type':", plot_data_filtered["room_type"].unique())
                else:
                    st.warning("No hay tipos de habitación con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'room_type' o 'price'.")
        
        
        
        st.markdown('<div class="section-header">Importancia de Características</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="importance-box">
        <h3>Ranking de características por importancia (Gain):</h3>
        <ul>
            <li><b>property_type</b>: 78926.31 (Normalizado: 28.96%)</li>
            <li><b>room_type</b>: 34114.39 (Normalizado: 12.52%)</li>
            <li><b>accommodates</b>: 18599.50 (Normalizado: 6.83%)</li>
            <li><b>distance_to_center</b>: 18420.12 (Normalizado: 6.76%)</li>
            <li><b>total_amenities</b>: 17949.60 (Normalizado: 6.59%)</li>
            <li><b>scaled_log_distance</b>: 15124.26 (Normalizado: 5.55%)</li>
            <li><b>minimum_nights</b>: 13345.00 (Normalizado: 4.90%)</li>
            <li><b>maximum_nights</b>: 9672.04 (Normalizado: 3.55%)</li>
            <li><b>neighborhood_cluster</b>: 8574.30 (Normalizado: 3.15%)</li>
            <li><b>ciudad</b>: 6455.99 (Normalizado: 2.37%)</li>
            <li><b>log_accommodates</b>: 5992.53 (Normalizado: 2.20%)</li>
            <li><b>scaled_distance_to_center</b>: 5727.14 (Normalizado: 2.10%)</li>
            <li><b>scaled_total_amenities</b>: 5582.59 (Normalizado: 2.05%)</li>
            <li><b>log_distance</b>: 5135.69 (Normalizado: 1.88%)</li>
            <li><b>log_minimum_nights</b>: 4305.02 (Normalizado: 1.58%)</li>
            <li><b>essential_score</b>: 4167.03 (Normalizado: 1.53%)</li>
            <li><b>bathroom_per_person</b>: 2884.72 (Normalizado: 1.06%)</li>
            <li><b>log_maximum_nights</b>: 2685.00 (Normalizado: 0.99%)</li>
            <li><b>bathrooms</b>: 1846.04 (Normalizado: 0.68%)</li>
            <li><b>accommodates_squared</b>: 1767.03 (Normalizado: 0.65%)</li>
            <li><b>bedrooms</b>: 1722.41 (Normalizado: 0.63%)</li>
            <li><b>bed_bath_product</b>: 1719.05 (Normalizado: 0.63%)</li>
            <li><b>bed_to_bedroom_ratio</b>: 1437.47 (Normalizado: 0.53%)</li>
            <li><b>luxury_score</b>: 1251.96 (Normalizado: 0.46%)</li>
            <li><b>person_per_bed</b>: 1096.63 (Normalizado: 0.40%)</li>
            <li><b>person_per_bedroom</b>: 1091.58 (Normalizado: 0.40%)</li>
            <li><b>bed_accom_ratio</b>: 958.30 (Normalizado: 0.35%)</li>
            <li><b>beds</b>: 695.56 (Normalizado: 0.26%)</li>
            <li><b>distance_to_center_squared</b>: 440.89 (Normalizado: 0.16%)</li>
            <li><b>bedrooms_squared</b>: 412.67 (Normalizado: 0.15%)</li>
            <li><b>bathrooms_squared</b>: 266.47 (Normalizado: 0.10%)</li>
            <li><b>beds_squared</b>: 130.00 (Normalizado: 0.05%)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Pestaña 3: Características del Alojamiento
with tabs[2]:
    col1, col2 = st.columns([1, 1])
    with col1:

        if "property_type" in filtered_data.columns:
            property_counts = filtered_data["property_type"].value_counts().head(10)
            fig = px.bar(
                x=property_counts.index,
                y=property_counts.values,
                labels={"x": "Tipo de Propiedad", "y": "Número de Alojamientos"},
                color=property_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Tipos de Propiedad Más Comunes"
            )
            fig.update_layout(title=dict(text="Tipos de Propiedad Más Comunes", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'property_type' no está disponible.")
        
        

        if "amenities" in filtered_data.columns and len(common_amenities) > 0:
            amenities_df = pd.DataFrame(Counter(all_amenities).most_common(15), columns=["amenity", "count"])
            fig = px.bar(
                amenities_df,
                x="count",
                y="amenity",
                orientation="h",
                color="count",
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Amenidades Más Frecuentes en Alojamientos"
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                title=dict(text="Amenidades Más Frecuentes en Alojamientos", font=dict(color="white"), x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de amenidades disponibles.")
        
        
        
        st.markdown('<div class="section-header">Distribución de Precios según Número de Habitaciones</div>', unsafe_allow_html=True)
        if "bedrooms" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["bedrooms", "price"])
            if len(plot_data) > 0:
                min_points = 1
                category_counts = plot_data["bedrooms"].value_counts()
                valid_categories = category_counts[category_counts >= min_points].index.tolist()
                if valid_categories:
                    plot_data_filtered = plot_data[plot_data["bedrooms"].isin(valid_categories)]
                    try:
                        fig = px.box(
                            plot_data_filtered,
                            x="bedrooms",
                            y="price",
                            labels={"bedrooms": "Número de Habitaciones", "price": "Precio (€)"},
                            title="Distribución de Precios según Número de Habitaciones"
                        )
                        fig.update_layout(title=dict(text="Distribución de Precios según Número de Habitaciones", font=dict(color="white"), x=0.5))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de caja: {e}")
                        st.write("Valores únicos en 'bedrooms':", plot_data_filtered["bedrooms"].unique())
                else:
                    st.warning("No hay números de habitaciones con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'bedrooms' o 'price'.")

    with col2:
        if "accommodates" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["accommodates", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="accommodates",
                    y="price",
                    labels={"accommodates": "Capacidad", "price": "Precio (€)"},
                    title="Relación entre Precio y Capacidad de Alojamiento"
                )
                fig.update_layout(title=dict(text="Relación entre Precio y Capacidad de Alojamiento", font=dict(color="white"), x=0.5))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'accommodates' o 'price'.")
        
        
        
        if "beds" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="beds",
                nbins=20,
                labels={"beds": "Número de Camas"},
                title="Distribución del Número de Camas"
            )
            fig.update_layout(title=dict(text="Distribución del Número de Camas", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'beds' no está disponible.")
        
        
        
        
        if "bathrooms" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="bathrooms",
                nbins=20,
                labels={"bathrooms": "Número de Baños"},
                title="Distribución del Número de Baños"
            )
            fig.update_layout(title=dict(text="Distribución del Número de Baños", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'bathrooms' no está disponible.")

# Pestaña 4: Características del Anfitrión
with tabs[3]:
    col1, col2 = st.columns([1, 1])
    with col1:
        if "host_response_rate" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="host_response_rate",
                nbins=20,
                labels={"host_response_rate": "Tasa de Respuesta (%)"},
                title="Distribución de la Tasa de Respuesta del Anfitrión"
            )
            fig.update_layout(title=dict(text="Distribución de la Tasa de Respuesta del Anfitrión", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_response_rate' no está disponible.")
        
        
        if "host_response_time" in filtered_data.columns:
            response_time_counts = filtered_data["host_response_time"].value_counts()
            fig = px.bar(
                x=response_time_counts.index,
                y=response_time_counts.values,
                labels={"x": "Tiempo de Respuesta", "y": "Número de Anfitriones"},
                color=response_time_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Tiempo de Respuesta del Anfitrión"
            )
            fig.update_layout(title=dict(text="Tiempo de Respuesta del Anfitrión", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_response_time' no está disponible.")
        

        
        if "host_age_years" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="host_age_years",
                nbins=20,
                labels={"host_age_years": "Años como Anfitrión"},
                title="Antigüedad de los Anfitriones"
            )
            fig.update_layout(title=dict(text="Antigüedad de los Anfitriones", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_age_years' no está disponible.")

    with col2:

        if "host_acceptance_rate" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_acceptance_rate", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="host_acceptance_rate",
                    y="price",
                    labels={"host_acceptance_rate": "Tasa de Aceptación (%)", "price": "Precio (€)"},
                    title="Relación entre Tasa de Aceptación y Precio"
                )
                fig.update_layout(title=dict(text="Relación entre Tasa de Aceptación y Precio", font=dict(color="white"), x=0.5))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'host_acceptance_rate' o 'price'.")
        

        

        if "host_listings_count" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_listings_count", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="host_listings_count",
                    y="price",
                    labels={"x": "Número de Listados", "y": "Precio (€)"},
                    title="Relación entre Número de Listados y Precio"
                )
                fig.update_layout(title=dict(text="Relación entre Número de Listados y Precio", font=dict(color="white"), x=0.5))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'host_listings_count' o 'price'.")
        
        
        if "host_total_listings_count" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="host_total_listings_count",
                nbins=20,
                labels={"host_total_listings_count": "Total de Listados"},
                title="Distribución del Total de Listados por Anfitrión"
            )
            fig.update_layout(title=dict(text="Distribución del Total de Listados por Anfitrión", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'host_total_listings_count' no está disponible.")

# Pestaña 5: Análisis de Reseñas
with tabs[4]:
    col1, col2 = st.columns([1, 1])
    with col1:
        if "number_of_reviews" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="number_of_reviews",
                nbins=30,
                labels={"number_of_reviews": "Número de Reseñas"},
                title="Distribución del Número de Reseñas"
            )
            fig.update_layout(title=dict(text="Distribución del Número de Reseñas", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'number_of_reviews' no está disponible.")
        
        
        

        if "review_scores_rating" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_rating", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="review_scores_rating",
                    y="price",
                    labels={"review_scores_rating": "Puntuación General", "price": "Precio (€)"},
                    title="Relación entre Puntuación General y Precio"
                )
                fig.update_layout(title=dict(text="Relación entre Puntuación General y Precio", font=dict(color="white"), x=0.5))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_rating' o 'price'.")
        
        
        

        if "review_scores_cleanliness" in filtered_data.columns and "room_type" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_cleanliness", "room_type"])
            if len(plot_data) > 0:
                min_points = 1
                category_counts = plot_data["room_type"].value_counts()
                valid_categories = category_counts[category_counts >= min_points].index.tolist()
                if valid_categories:
                    plot_data_filtered = plot_data[plot_data["room_type"].isin(valid_categories)]
                    try:
                        fig = px.box(
                            plot_data_filtered,
                            x="room_type",
                            y="review_scores_cleanliness",
                            labels={"room_type": "Tipo de Habitación", "review_scores_cleanliness": "Puntuación de Limpieza"},
                            title="Puntuación de Limpieza según Tipo de Habitación"
                        )
                        fig.update_layout(title=dict(text="Puntuación de Limpieza según Tipo de Habitación", font=dict(color="white"), x=0.5))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de caja: {e}")
                        st.write("Valores únicos en 'room_type':", plot_data_filtered["room_type"].unique())
                else:
                    st.warning("No hay tipos de habitación con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_cleanliness' o 'room_type'.")

    with col2:

        if "review_scores_location" in filtered_data.columns and "neighbourhood_cleansed" in filtered_data.columns:
            top_neighbourhoods = filtered_data["neighbourhood_cleansed"].value_counts().head(10).index
            plot_data = filtered_data[filtered_data["neighbourhood_cleansed"].isin(top_neighbourhoods)].dropna(subset=["review_scores_location"])
            if len(plot_data) > 0:
                min_points = 1
                category_counts = plot_data["neighbourhood_cleansed"].value_counts()
                valid_categories = category_counts[category_counts >= min_points].index.tolist()
                if valid_categories:
                    plot_data_filtered = plot_data[plot_data["neighbourhood_cleansed"].isin(valid_categories)]
                    try:
                        fig = px.box(
                            plot_data_filtered,
                            x="neighbourhood_cleansed",
                            y="review_scores_location",
                            labels={"neighbourhood_cleansed": "Vecindario", "review_scores_location": "Puntuación de Ubicación"},
                            title="Puntuación de Ubicación por Vecindario"
                        )
                        fig.update_layout(
                            xaxis={"categoryorder": "total descending"},
                            title=dict(text="Puntuación de Ubicación por Vecindario", font=dict(color="white"), x=0.5)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de caja: {e}")
                        st.write("Valores únicos en 'neighbourhood_cleansed':", plot_data_filtered["neighbourhood_cleansed"].unique())
                else:
                    st.warning("No hay vecindarios con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_location' o 'neighbourhood_cleansed'.")
        
        
        

        if "review_scores_communication" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_communication", "price"]).sample(min(1000, len(filtered_data)))
            if len(plot_data) > 0:
                fig = px.scatter(
                    plot_data,
                    x="review_scores_communication",
                    y="price",
                    labels={"review_scores_communication": "Puntuación de Comunicación", "price": "Precio (€)"},
                    title="Relación entre Puntuación de Comunicación y Precio"
                )
                fig.update_layout(title=dict(text="Relación entre Puntuación de Comunicación y Precio", font=dict(color="white"), x=0.5))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_communication' o 'price'.")
        
        
        
        if "review_scores_checkin" in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x="review_scores_checkin",
                nbins=20,
                labels={"review_scores_checkin": "Puntuación de Check-in"},
                title="Distribución de la Puntuación de Check-in"
            )
            fig.update_layout(title=dict(text="Distribución de la Puntuación de Check-in", font=dict(color="white"), x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'review_scores_checkin' no está disponible.")

# Pestaña 6: Características temporales
with tabs[5]:
    if "minimum_nights" in filtered_data.columns:
        fig = px.histogram(
            filtered_data,
            x="minimum_nights",
            nbins=30,
            labels={"minimum_nights": "Noches Mínimas"},
            title="Distribución de Noches Mínimas Requeridas"
        )
        fig.update_layout(title=dict(text="Distribución de Noches Mínimas Requeridas", font=dict(color="white"), x=0.5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'minimum_nights' no está disponible.")
    
    st.markdown('<div class="divider-horizontal"></div>', unsafe_allow_html=True)
    
    if "maximum_nights" in filtered_data.columns:
        fig = px.histogram(
            filtered_data,
            x="maximum_nights",
            nbins=30,
            labels={"maximum_nights": "Noches Máximas"},
            title="Distribución de Noches Máximas Permitidas"
        )
        fig.update_layout(title=dict(text="Distribución de Noches Máximas Permitidas", font=dict(color="white"), x=0.5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'maximum_nights' no está disponible.")
    
    st.markdown('<div class="divider-horizontal"></div>', unsafe_allow_html=True)
    

    if "last_scraped" in filtered_data.columns:
        last_scraped_counts = filtered_data["last_scraped"].value_counts().sort_index()
        fig = px.bar(
            x=last_scraped_counts.index,
            y=last_scraped_counts.values,
            labels={"x": "Fecha de Recopilación", "y": "Número de Alojamientos"},
            title="Frecuencia de Última Fecha de Recopilación de Datos"
        )
        fig.update_layout(title=dict(text="Frecuencia de Última Fecha de Recopilación de Datos", font=dict(color="white"), x=0.5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("La columna 'last_scraped' no está disponible.")

# Pestaña 7: Características de Usuarios
# Pestaña 7: Características de Usuarios
with tabs[6]:
    st.markdown('<div class="subheader">Características de Usuarios</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown("""
        <div class="info-box">
        <h3>Resumen General de Reseñas</h3>
        <ul>
            <li><b>Total de Reseñas:</b> 50,000</li>
            <li><b>Total de Usuarios:</b> 49,812</li>
            <li><b>Período de Análisis:</b> 04/01/2011 - 25/12/2024</li>
            <li><b>Promedio de Sentimiento:</b> 0.775</li>
            <li><b>Sentimiento Mínimo:</b> -0.9835</li>
            <li><b>Sentimiento Máximo:</b> 0.9986</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        

        st.markdown("""
        <div class="info-box">
        <h3>Cluster 0</h3>
        <ul>
            <li><b>Número de Reseñas:</b> 7,130 (14.26%)</li>
            <li><b>Sentimiento Promedio:</b> 0.853</li>
            <li><b>Palabras Clave:</b> great, location, great location, stay, place, apartment, host, great place, clean, recommend</li>
        </ul>
        <h3>Cluster 1</h3>
        <ul>
            <li><b>Número de Reseñas:</b> 3,837 (7.674%)</li>
            <li><b>Sentimiento Promedio:</b> 0.720</li>
            <li><b>Palabras Clave:</b> good, location, good location, apartment, place, stay, clean, nice, host, good place</li>
        </ul>
        <h3>Cluster 2</h3>
        <ul>
            <li><b>Número de Reseñas:</b> 39,033 (78.066%)</li>
            <li><b>Sentimiento Promedio:</b> 0.766</li>
            <li><b>Palabras Clave:</b> apartment, stay, place, nice, location, clean, recommend, great, perfect, host</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-box">
        <h3>Temas Identificados</h3>
        <ul>
            <li><b>Tema 1:</b> Check-in y asistencia (check, help, give, time, arrive, leave, arrival, question, early)</li>
            <li><b>Tema 2:</b> Características del alojamiento (room, bed, bathroom, small, kitchen, night, work, shower, bedroom)</li>
            <li><b>Tema 3:</b> Experiencia general (accommodation, pleasant, welcome, locate, description, foot, functional, photo, available, practical)</li>
            <li><b>Tema 4:</b> Valoración positiva (apartment, great, stay, location, place, good, clean, recommend, nice, host)</li>
            <li><b>Tema 5:</b> Atención y espacio (house, attentive, hostel, attention, department, floor, position, doubt, meter, wide)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        


        # Datos del gráfico
        review_data = pd.DataFrame({
            "Día": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
            "review_id": [8257, 6485, 6278, 6228, 6607, 6539, 9606],
            "vader_compound": [0.776859, 0.783723, 0.776271, 0.773506, 0.775032, 0.776225, 0.764861]
        })
        # Crear gráfico combinado
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Barras para el número de reseñas
        fig.add_trace(
            go.Bar(
                x=review_data["Día"],
                y=review_data["review_id"],
                name="Número de Reseñas",
                marker_color="#FF5A5F"
            ),
            secondary_y=False
        )
        # Línea para el sentimiento promedio
        fig.add_trace(
            go.Scatter(
                x=review_data["Día"],
                y=review_data["vader_compound"],
                name="Sentimiento Promedio",
                line=dict(color="#00A699", width=3)
            ),
            secondary_y=True
        )
        # Actualizar diseño
        fig.update_layout(
            title=dict(text="Actividad y Sentimiento por Día de la Semana", font=dict(color="white"), x=0.5),
            xaxis_title="Día de la Semana",
            yaxis_title="Número de Reseñas",
            yaxis2_title="Sentimiento Promedic",
            height=500
        )
        fig.update_yaxes(title_text="Número de Reseñas", secondary_y=False)
        fig.update_yaxes(title_text="Sentimiento Promedio", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>Descripción:</b> Gráfico combinado que muestra la actividad y el sentimiento promedio de las reseñas por día de la semana. Las barras indican el número de reseñas, mientras que la línea muestra el sentimiento promedio (calculado con VADER).<br>
        <b>Información Adicional:</b><br>
        - Período de análisis: Semanal (por día)<br>
        - Método de análisis de sentimiento: VADER
        </div>
        """, unsafe_allow_html=True)

        

        # Datos del gráfico de evolución del sentimiento
        sentiment_data = pd.DataFrame({
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
        # Convertir year_month a formato datetime para mejor manejo
        sentiment_data["year_month"] = pd.to_datetime(sentiment_data["year_month"], format="%Y-%m")
        # Crear gráfico de líneas
        fig = px.line(
            sentiment_data,
            x="year_month",
            y="vader_compound",
            labels={"year_month": "Fecha", "vader_compound": "Sentimiento Promedio (VADER)"},
            title="Evolución del Sentimiento Promedio por Mes"
        )
        fig.update_traces(line=dict(color="#00A699", width=3))
        fig.update_layout(
            title=dict(text="Evolución del Sentimiento Promedio por Mes", font=dict(color="white"), x=0.5),
            xaxis_title="Fecha",
            yaxis_title="Sentimiento Promedio (VADER)",
            height=500,
            xaxis=dict(
                tickformat="%Y-%m",
                dtick="M12",  # Mostrar etiquetas cada 12 meses
                tickangle=45
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>Descripción:</b> Gráfico de líneas que muestra la evolución del sentimiento promedio (calculado con VADER) de las reseñas a lo largo del tiempo, agrupado por mes.<br>
        <b>Información Adicional:</b><br>
        - Período de análisis: Mensual (04/01/2011 - 25/12/2024)<br>
        - Método de análisis de sentimiento: VADER
        </div>
        """, unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.markdown("TFG - Análisis de Precios y Reseñas en Airbnb | Ángel Soto García")
