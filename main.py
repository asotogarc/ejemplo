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
            plot_data = filtered_data.dropna(subset=["availability_365", "price"]).copy()
            if len(plot_data) > 0:
                # Crear rangos de disponibilidad (intervalos de 50 días)
                bins = [0, 50, 100, 150, 200, 250, 300, 365]
                labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-365']
                plot_data['availability_range'] = pd.cut(plot_data['availability_365'], bins=bins, labels=labels, include_lowest=True)
                
                # Filtrar rangos con suficientes datos (mínimo 5 puntos por rango)
                min_points = 5
                category_counts = plot_data['availability_range'].value_counts()
                valid_categories = category_counts[category_counts >= min_points].index.tolist()
                plot_data_filtered = plot_data[plot_data['availability_range'].isin(valid_categories)]
                
                if len(plot_data_filtered) > 0 and len(valid_categories) > 0:
                    try:
                        # Crear gráfico de caja
                        fig = px.box(
                            plot_data_filtered,
                            x='availability_range',
                            y='price',
                            labels={'availability_range': 'Disponibilidad Anual (días)', 'price': 'Precio (€)'},
                            title='Distribución de Precios por Rango de Disponibilidad Anual',
                            category_orders={'availability_range': labels}  # Mantener el orden de los rangos
                        )
                        fig.update_layout(
                            xaxis_title='Disponibilidad Anual (días)',
                            yaxis_title='Precio (€)',
                            title=dict(text='Distribución de Precios por Rango de Disponibilidad Anual', font=dict(color='white'), x=0.5),
                            showlegend=False
                        )
                        # Limitar el eje Y para evitar valores extremos (opcional)
                        fig.update_yaxes(range=[0, plot_data_filtered['price'].quantile(0.95)])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de caja: {e}")
                        st.write("Valores únicos en 'availability_range':", plot_data_filtered['availability_range'].unique())
                else:
                    st.warning("No hay rangos de disponibilidad con suficientes datos para mostrar el gráfico.")
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
            plot_data = filtered_data.dropna(subset=["accommodates", "price"]).copy()
            if len(plot_data) > 0:
                # Filtrar capacidades con suficientes datos (mínimo 5 puntos)
                min_points = 5
                category_counts = plot_data["accommodates"].value_counts()
                valid_categories = category_counts[category_counts >= min_points].index.tolist()
                plot_data_filtered = plot_data[plot_data["accommodates"].isin(valid_categories)]
                
                if len(plot_data_filtered) > 0 and len(valid_categories) > 0:
                    try:
                        # Crear gráfico de caja
                        fig = px.box(
                            plot_data_filtered,
                            x="accommodates",
                            y="price",
                            labels={"accommodates": "Capacidad (Personas)", "price": "Precio (€)"},
                            title="Distribución de Precios por Capacidad de Alojamiento",
                            category_orders={"accommodates": sorted(valid_categories)}  # Ordenar capacidades
                        )
                        fig.update_layout(
                            xaxis_title="Capacidad (Personas)",
                            yaxis_title="Precio (€)",
                            title=dict(text="Distribución de Precios por Capacidad de Alojamiento", font=dict(color="white"), x=0.5),
                            showlegend=False
                        )
                        # Limitar el eje Y para evitar valores extremos
                        fig.update_yaxes(range=[0, plot_data_filtered["price"].quantile(0.95)])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de caja: {e}")
                        st.write("Valores únicos en 'accommodates':", plot_data_filtered["accommodates"].unique())
                else:
                    st.warning("No hay valores de capacidad con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'accommodates' o 'price'.")        
        
        
        if "beds" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["beds", "price"]).copy()
            if len(plot_data) > 0:
                # Limitar el número de camas a un máximo razonable (por ejemplo, 10) y agrupar valores mayores
                plot_data["beds"] = plot_data["beds"].clip(upper=10)
                # Filtrar valores con suficientes datos (mínimo 5 puntos)
                min_points = 5
                category_counts = plot_data["beds"].value_counts()
                valid_beds = category_counts[category_counts >= min_points].index.tolist()
                plot_data_filtered = plot_data[plot_data["beds"].isin(valid_beds)]
                
                if len(plot_data_filtered) > 0 and len(valid_beds) > 0:
                    try:
                        # Crear gráfico de violín
                        fig = px.violin(
                            plot_data_filtered,
                            x="beds",
                            y="price",
                            box=True,  # Incluir caja dentro del violín para mostrar mediana y cuartiles
                            points=False,  # No mostrar puntos individuales para mantener claridad
                            labels={"beds": "Número de Camas", "price": "Precio (€)"},
                            title="Distribución de Precios por Número de Camas"
                        )
                        fig.update_traces(
                            fillcolor='rgba(255, 90, 95, 0.2)',  # Color suave para los violines
                            line_color='#FF5A5F',
                            line_width=2
                        )
                        fig.update_layout(
                            xaxis_title="Número de Camas",
                            yaxis_title="Precio (€)",
                            title=dict(text="Distribución de Precios por Número de Camas", font=dict(color="white"), x=0.5),
                            xaxis=dict(tickmode="linear", dtick=1),  # Asegurar etiquetas enteras
                            yaxis=dict(range=[0, plot_data_filtered["price"].quantile(0.95)]),  # Limitar eje Y
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de violín: {e}")
                        st.write("Valores únicos en 'beds':", plot_data_filtered["beds"].unique())
                else:
                    st.warning("No hay valores de número de camas con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'beds' o 'price'.")
        
        
        
        if "bathrooms" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["bathrooms"]).copy()
            if len(plot_data) > 0:
                # Limitar el número de baños a un máximo razonable (por ejemplo, 5) y agrupar valores mayores
                plot_data["bathrooms"] = plot_data["bathrooms"].clip(upper=5)
                # Filtrar valores con suficientes datos (mínimo 5 puntos)
                min_points = 5
                category_counts = plot_data["bathrooms"].value_counts()
                valid_bathrooms = category_counts[category_counts >= min_points].index.tolist()
                plot_data_filtered = plot_data[plot_data["bathrooms"].isin(valid_bathrooms)]
                
                if len(plot_data_filtered) > 0 and len(valid_bathrooms) > 0:
                    try:
                        # Recalcular conteos después de filtrar
                        bathrooms_counts = plot_data_filtered["bathrooms"].value_counts().sort_index()
                        # Preparar datos para el gráfico de donut
                        donut_data = pd.DataFrame({
                            "Número de Baños": bathrooms_counts.index,
                            "Conteo": bathrooms_counts.values
                        })
                        # Crear gráfico de donut
                        fig = px.pie(
                            donut_data,
                            names="Número de Baños",
                            values="Conteo",
                            title="Proporción de Alojamientos por Número de Baños",
                            hole=0.4,  # Crear el hueco para el efecto donut
                            color_discrete_sequence=px.colors.sequential.Viridis
                        )
                        fig.update_traces(
                            textinfo="percent+label",
                            textposition="inside",
                            showlegend=True
                        )
                        fig.update_layout(
                            title=dict(text="Proporción de Alojamientos por Número de Baños", font=dict(color="white"), x=0.5),
                            showlegend=True,
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al generar el gráfico de donut: {e}")
                        st.write("Valores únicos en 'bathrooms':", plot_data_filtered["bathrooms"].unique())
                else:
                    st.warning("No hay valores de número de baños con suficientes datos para mostrar el gráfico.")
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("La columna 'bathrooms' no está disponible.")        


# Pestaña 4: Características del Anfitrión
with tabs[3]:
    col1, col2 = st.columns([1, 1])
    with col1:
        if "host_response_rate" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_response_rate"]).copy()
            if len(plot_data) > 0:
                try:
                    # Convertir tasa de respuesta a porcentaje (si no está en 0-100)
                    if plot_data["host_response_rate"].max() <= 1:
                        plot_data["host_response_rate"] = plot_data["host_response_rate"] * 100
                    # Crear rangos de tasa de respuesta
                    bins = [0, 50, 80, 95, 100]
                    labels = ["0-50%", "50-80%", "80-95%", "95-100%"]
                    plot_data["response_range"] = pd.cut(
                        plot_data["host_response_rate"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Contar alojamientos por rango
                    response_counts = plot_data["response_range"].value_counts().sort_index()
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    valid_ranges = response_counts[response_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["response_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Recalcular conteos
                        response_counts_filtered = plot_data_filtered["response_range"].value_counts().sort_index()
                        # Preparar datos para gráfico de dona
                        donut_data = pd.DataFrame({
                            "Rango": response_counts_filtered.index,
                            "Conteo": response_counts_filtered.values
                        })
                        # Crear gráfico de dona
                        fig = px.pie(
                            donut_data,
                            names="Rango",
                            values="Conteo",
                            title="Proporción de Anfitriones por Tasa de Respuesta",
                            hole=0.4,  # Efecto dona
                            color_discrete_sequence=["#FF5A5F", "#00A699", "#484848", "#767676"]
                        )
                        fig.update_traces(
                            textinfo="percent+label",
                            textposition="inside",
                            textfont=dict(color="white"),
                            hovertemplate="%{label}: %{value} anfitriones (%{percent})"
                        )
                        fig.update_layout(
                            title=dict(text="Proporción de Anfitriones por Tasa de Respuesta", font=dict(color="white"), x=0.5),
                            showlegend=True,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de tasa de respuesta con suficientes datos (mínimo 5 puntos por rango).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de dona: {str(e)}")
                    st.write("Estadísticas de 'host_response_rate':", plot_data["host_response_rate"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
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
            plot_data = filtered_data.dropna(subset=["host_age_years"]).copy()
            if len(plot_data) > 0:
                try:
                    # Crear rangos de antigüedad
                    bins = [0, 2, 5, 10, float("inf")]
                    labels = ["0-2 años", "2-5 años", "5-10 años", ">10 años"]
                    plot_data["age_range"] = pd.cut(plot_data["host_age_years"], bins=bins, labels=labels, include_lowest=True)
                    # Contar alojamientos por rango
                    age_counts = plot_data["age_range"].value_counts().sort_index()
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    valid_ranges = age_counts[age_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["age_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Recalcular conteos
                        age_counts_filtered = plot_data_filtered["age_range"].value_counts().sort_index()
                        # Preparar datos para gráfico de anillos
                        ring_data = pd.DataFrame({
                            "Rango de Antigüedad": age_counts_filtered.index,
                            "Conteo": age_counts_filtered.values
                        })
                        # Crear gráfico de anillos
                        fig = px.pie(
                            ring_data,
                            names="Rango de Antigüedad",
                            values="Conteo",
                            title="Proporción de Alojamientos por Antigüedad del Anfitrión",
                            hole=0.5,  # Crear efecto de anillo
                            color_discrete_sequence=["#FF5A5F", "#00A699", "#484848", "#767676"]
                        )
                        fig.update_traces(
                            textinfo="percent+label",
                            textposition="inside",
                            textfont=dict(color="white"),
                            hovertemplate="%{label}: %{value} alojamientos (%{percent})"
                        )
                        fig.update_layout(
                            title=dict(text="Proporción de Alojamientos por Antigüedad del Anfitrión", font=dict(color="white"), x=0.5),
                            showlegend=True,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de antigüedad con suficientes datos para mostrar el gráfico.")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de anillos: {e}")
                    st.write("Valores únicos en 'host_age_years':", plot_data["host_age_years"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("La columna 'host_age_years' no está disponible.")
    
    with col2:

        if "host_acceptance_rate" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_acceptance_rate", "price"]).copy()
            if len(plot_data) > 0:
                try:
                    # Convertir tasa de aceptación a porcentaje (si no está en 0-100)
                    if plot_data["host_acceptance_rate"].max() <= 1:
                        plot_data["host_acceptance_rate"] = plot_data["host_acceptance_rate"] * 100
                    # Crear rangos de tasa de aceptación
                    bins = [0, 50, 80, 100]
                    labels = ["0-50%", "50-80%", "80-100%"]
                    plot_data["acceptance_range"] = pd.cut(
                        plot_data["host_acceptance_rate"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    category_counts = plot_data["acceptance_range"].value_counts()
                    valid_ranges = category_counts[category_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["acceptance_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Calcular mediana de precios y conteo por rango
                        bubble_data = plot_data_filtered.groupby("acceptance_range").agg(
                            median_price=("price", "median"),
                            count=("price", "count")
                        ).reset_index()
                        # Normalizar tamaños de burbujas
                        max_size = 50
                        min_size = 10
                        bubble_data["size"] = min_size + (bubble_data["count"] / bubble_data["count"].max()) * (max_size - min_size)
                        # Crear gráfico de burbujas
                        fig = go.Figure()
                        for i, row in bubble_data.iterrows():
                            fig.add_trace(
                                go.Scatter(
                                    x=[row["acceptance_range"]],
                                    y=[row["median_price"]],
                                    mode="markers",
                                    marker=dict(
                                        size=row["size"],
                                        color=["#FF5A5F", "#00A699", "#484848"][i % 3],
                                        opacity=0.8,
                                        line=dict(width=1, color="#FFFFFF")
                                    ),
                                    text=f"{row['acceptance_range']}: {int(row['count'])} alojamientos, Precio mediano: €{row['median_price']:.0f}",
                                    hoverinfo="text",
                                    name=row["acceptance_range"]
                                )
                            )
                        # Actualizar diseño
                        fig.update_layout(
                            xaxis_title="Tasa de Aceptación",
                            yaxis_title="Precio Mediano (€)",
                            title=dict(text="Relación entre Tasa de Aceptación y Precio", font=dict(color="white"), x=0.5),
                            yaxis=dict(range=[0, bubble_data["median_price"].quantile(0.95) * 1.2]),
                            showlegend=True,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50),
                            xaxis=dict(tickangle=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de tasa de aceptación con suficientes datos (mínimo 5 puntos por rango).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de burbujas: {str(e)}")
                    st.write("Estadísticas de 'host_acceptance_rate':", plot_data["host_acceptance_rate"].describe())
                    st.write("Estadísticas de 'price':", plot_data["price"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'host_acceptance_rate' o 'price'.")
        

        if "host_listings_count" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["host_listings_count", "price"]).copy()
            if len(plot_data) > 0:
                try:
                    # Limitar número de listados a un máximo razonable (por ejemplo, 10)
                    plot_data["host_listings_count"] = plot_data["host_listings_count"].clip(upper=10)
                    # Filtrar valores con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    category_counts = plot_data["host_listings_count"].value_counts()
                    valid_listings = sorted(category_counts[category_counts >= min_points].index.tolist())
                    plot_data_filtered = plot_data[plot_data["host_listings_count"].isin(valid_listings)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_listings) > 0:
                        # Calcular mediana de precios y conteo por número de listados
                        tile_data = plot_data_filtered.groupby("host_listings_count").agg(
                            median_price=("price", "median"),
                            count=("price", "count")
                        ).reset_index()
                        # Normalizar valores para tamaños y colores
                        tile_data["size"] = np.sqrt(tile_data["count"] / tile_data["count"].max()) * 80  # Escala ajustada
                        # Asignar colores manualmente según precio mediano
                        max_price = tile_data["median_price"].max()
                        min_price = tile_data["median_price"].min()
                        tile_data["color"] = tile_data["median_price"].apply(
                            lambda x: (
                                "#00A699" if x <= min_price + (max_price - min_price) / 3 else
                                "#484848" if x <= min_price + 2 * (max_price - min_price) / 3 else
                                "#FF5A5F"
                            )
                        )
                        # Ajustar posiciones Y para evitar superposición
                        tile_data["y_pos"] = 1 + (tile_data.index % 2) * 0.2 - 0.1  # Alternar posiciones Y
                        # Crear gráfico de mosaico
                        fig = go.Figure()
                        for i, row in tile_data.iterrows():
                            # Formato compacto del texto
                            text = f"€{row['median_price']:.0f} ({int(row['count'])})"
                            # Ajustar tamaño de fuente según tamaño de burbuja
                            font_size = min(12, max(8, row['size'] / 5))
                            fig.add_trace(
                                go.Scatter(
                                    x=[row["host_listings_count"]],
                                    y=[row["y_pos"]],
                                    mode="markers+text",
                                    marker=dict(
                                        size=row["size"],
                                        color=row["color"],
                                        opacity=0.9,
                                        line=dict(width=1, color="#FFFFFF"),
                                        showscale=False  # Eliminar barra de color
                                    ),
                                    text=[text],
                                    textposition="middle center",
                                    textfont=dict(color="white", size=font_size),
                                    hoverinfo="text",
                                    hovertext=f"{int(row['host_listings_count'])} listados: €{row['median_price']:.0f}, {int(row['count'])} alojamientos",
                                    showlegend=False  # Eliminar leyenda
                                )
                            )
                        # Actualizar diseño
                        fig.update_layout(
                            xaxis_title="Número de Listados",
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.5, 1.5]),
                            title=dict(text="Precios (mediana) por Número de Listados", font=dict(color="white"), x=0.5),
                            xaxis=dict(
                                tickmode="linear",
                                dtick=1,
                                range=[min(valid_listings)-0.7, max(valid_listings)+0.7]  # Más espacio
                            ),
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay valores de número de listados con suficientes datos (mínimo 5 puntos por categoría).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de mosaico: {str(e)}")
                    st.write("Estadísticas de 'host_listings_count':", plot_data["host_listings_count"].describe())
                    st.write("Estadísticas de 'price':", plot_data["price"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'host_listings_count' o 'price'.")
# Pestaña 5: Análisis de Reseñas
with tabs[4]:
    col1, col2 = st.columns([1, 1])
    with col1:
        if "number_of_reviews" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["number_of_reviews"]).copy()
            if len(plot_data) > 0:
                try:
                    # Crear rangos de número de reseñas
                    bins = [0, 10, 50, 100, float("inf")]
                    labels = ["0-10", "10-50", "50-100", ">100"]
                    plot_data["reviews_range"] = pd.cut(
                        plot_data["number_of_reviews"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Contar alojamientos por rango
                    reviews_counts = plot_data["reviews_range"].value_counts().sort_index()
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    valid_ranges = reviews_counts[reviews_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["reviews_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Recalcular conteos
                        reviews_counts_filtered = plot_data_filtered["reviews_range"].value_counts().sort_index()
                        # Preparar datos para gráfico de dona
                        donut_data = pd.DataFrame({
                            "Rango": reviews_counts_filtered.index,
                            "Conteo": reviews_counts_filtered.values
                        })
                        # Crear gráfico de dona
                        fig = px.pie(
                            donut_data,
                            names="Rango",
                            values="Conteo",
                            title="Proporción de Alojamientos por Número de Reseñas",
                            hole=0.4,
                            color_discrete_sequence=["#FF5A5F", "#00A699", "#484848", "#767676"]
                        )
                        fig.update_traces(
                            textinfo="percent+label",
                            textposition="inside",
                            textfont=dict(color="white"),
                            hovertemplate="%{label}: %{value} alojamientos (%{percent})"
                        )
                        fig.update_layout(
                            title=dict(text="Proporción de Alojamientos por Número de Reseñas", font=dict(color="white"), x=0.5),
                            showlegend=True,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de número de reseñas con suficientes datos (mínimo 5 puntos por rango).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de dona: {str(e)}")
                    st.write("Estadísticas de 'number_of_reviews':", plot_data["number_of_reviews"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("La columna 'number_of_reviews' no está disponible.")        
        
        

        if "review_scores_rating" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_rating", "price"]).copy()
            if len(plot_data) > 0:
                try:
                    # Normalizar puntuaciones si están en escala 0-5 o 0-10
                    max_score = plot_data["review_scores_rating"].max()
                    if max_score <= 5:
                        plot_data["review_scores_rating"] = plot_data["review_scores_rating"] * 20
                    elif max_score <= 10:
                        plot_data["review_scores_rating"] = plot_data["review_scores_rating"] * 10
                    # Crear rangos de puntuación general
                    bins = [0, 80, 90, 100]
                    labels = ["0-80", "80-90", "90-100"]
                    plot_data["rating_range"] = pd.cut(
                        plot_data["review_scores_rating"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    category_counts = plot_data["rating_range"].value_counts()
                    valid_ranges = category_counts[category_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["rating_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Calcular mediana de precios y conteo por rango
                        bubble_data = plot_data_filtered.groupby("rating_range").agg(
                            median_price=("price", "median"),
                            count=("price", "count")
                        ).reset_index()
                        # Normalizar tamaños de burbujas
                        max_size = 50
                        min_size = 10
                        bubble_data["size"] = min_size + (bubble_data["count"] / bubble_data["count"].max()) * (max_size - min_size)
                        # Crear gráfico de burbujas
                        fig = go.Figure()
                        for i, row in bubble_data.iterrows():
                            fig.add_trace(
                                go.Scatter(
                                    x=[row["rating_range"]],
                                    y=[row["median_price"]],
                                    mode="markers",
                                    marker=dict(
                                        size=row["size"],
                                        color=["#FF5A5F", "#00A699", "#484848"][i % 3],
                                        opacity=0.8,
                                        line=dict(width=1, color="#FFFFFF")
                                    ),
                                    text=f"{row['rating_range']}: {int(row['count'])} alojamientos, €{row['median_price']:.0f}",
                                    hoverinfo="text",
                                    showlegend=False
                                )
                            )
                        # Actualizar diseño
                        fig.update_layout(
                            xaxis_title="Puntuación General",
                            yaxis_title="Precio Mediano (€)",
                            title=dict(text="Relación entre Puntuación General y Precio", font=dict(color="white"), x=0.5),
                            yaxis=dict(range=[0, bubble_data["median_price"].quantile(0.95) * 1.2]),
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de puntuación general con suficientes datos (mínimo 5 puntos por rango).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de burbujas: {str(e)}")
                    st.write("Estadísticas de 'review_scores_rating':", plot_data["review_scores_rating"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_rating' o 'price'.")
        
        
        



    with col2:

        # Posibles nombres alternativos para las columnas
        location_columns = [
            "review_scores_location", "location_score", "review_location", 
            "location_rating", "review_scores_location_score"
        ]
        neighbourhood_columns = ["neighbourhood", "neighborhood", "neighbourhood_cleansed", "neighborhood_cleansed"]
        
        # Buscar columnas válidas
        location_col = next((col for col in location_columns if col in filtered_data.columns), None)
        neighbourhood_col = next((col for col in neighbourhood_columns if col in filtered_data.columns), None)
        
        if location_col and neighbourhood_col:
            plot_data = filtered_data.dropna(subset=[location_col, neighbourhood_col]).copy()
            if len(plot_data) > 0:
                try:
                    # Normalizar puntuaciones si están en escala 0-5 o 0-10
                    max_score = plot_data[location_col].max()
                    if max_score <= 5:
                        plot_data[location_col] = plot_data[location_col] * 20
                    elif max_score <= 10:
                        plot_data[location_col] = plot_data[location_col] * 10
                    # Calcular puntuación promedio por vecindario
                    location_scores = plot_data.groupby(neighbourhood_col)[location_col].agg(
                        mean="mean",
                        count="count"
                    ).reset_index()
                    # Filtrar vecindarios con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    location_scores = location_scores[location_scores["count"] >= min_points]
                    # Ordenar por puntuación descendente y limitar a top 10
                    location_scores = location_scores.sort_values("mean", ascending=False).head(10)
                    
                    if len(location_scores) > 0:
                        # Crear gráfico de barras horizontales
                        fig = px.bar(
                            location_scores,
                            y=neighbourhood_col,
                            x="mean",
                            labels={neighbourhood_col: "Vecindario", "mean": "Puntuación Promedio de Ubicación (0-100)"},
                            title="Puntuación de Ubicación por Vecindario",
                            color="mean",
                            color_continuous_scale=["#00A699", "#FF5A5F"]
                        )
                        fig.update_layout(
                            xaxis_title="Puntuación Promedio de Ubicación (0-100)",
                            yaxis_title="Vecindario",
                            title=dict(text="Puntuación de Ubicación por Vecindario", font=dict(color="white"), x=0.5),
                            xaxis=dict(range=[0, 100]),
                            yaxis=dict(autorange="reversed"),  # Ordenar de mayor a menor
                            showlegend=False,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        fig.update_traces(
                            text=location_scores["mean"].round(1),
                            textposition="auto",
                            textfont=dict(color="white")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay vecindarios con suficientes datos (mínimo 5 puntos por vecindario).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de barras: {str(e)}")
                    st.write(f"Estadísticas de '{location_col}':", plot_data[location_col].describe())
                    st.write("Vecindarios únicos:", plot_data[neighbourhood_col].unique())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            missing_cols = []
            if not location_col:
                missing_cols.append("columna de puntuación de ubicación (se intentó: " + ", ".join(location_columns) + ")")
            if not neighbourhood_col:
                missing_cols.append("columna de vecindario (se intentó: " + ", ".join(neighbourhood_columns) + ")")
            st.info(f"Faltan las siguientes columnas: {', '.join(missing_cols)}.")
        
        
        

        if "review_scores_communication" in filtered_data.columns and "price" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_communication", "price"]).copy()
            if len(plot_data) > 0:
                try:
                    # Normalizar puntuaciones si están en escala 0-5 o 0-10
                    max_score = plot_data["review_scores_communication"].max()
                    if max_score <= 5:
                        plot_data["review_scores_communication"] = plot_data["review_scores_communication"] * 20
                    elif max_score <= 10:
                        plot_data["review_scores_communication"] = plot_data["review_scores_communication"] * 10
                    # Crear rangos de puntuación de comunicación
                    bins = [0, 80, 90, 100]
                    labels = ["0-80", "80-90", "90-100"]
                    plot_data["comm_range"] = pd.cut(
                        plot_data["review_scores_communication"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    category_counts = plot_data["comm_range"].value_counts()
                    valid_ranges = category_counts[category_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["comm_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Calcular mediana de precios por rango
                        line_data = plot_data_filtered.groupby("comm_range")["price"].median().reset_index()
                        # Crear gráfico de líneas suavizadas
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=line_data["comm_range"],
                                y=line_data["price"],
                                mode="lines+markers",
                                line=dict(color="#FF5A5F", width=3, shape="spline"),
                                marker=dict(size=10, color="#00A699", line=dict(width=1, color="#FFFFFF")),
                                name="Precio Mediano"
                            )
                        )
                        # Actualizar diseño
                        fig.update_layout(
                            xaxis_title="Puntuación de Comunicación",
                            yaxis_title="Precio Mediano (€)",
                            title=dict(text="Relación entre Puntuación de Comunicación y Precio", font=dict(color="white"), x=0.5),
                            yaxis=dict(range=[0, line_data["price"].quantile(0.95) * 1.2]),
                            showlegend=False,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de puntuación de comunicación con suficientes datos (mínimo 5 puntos por rango).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de líneas: {str(e)}")
                    st.write("Estadísticas de 'review_scores_communication':", plot_data["review_scores_communication"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("Faltan las columnas 'review_scores_communication' o 'price'.")        
        
        
        if "review_scores_checkin" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["review_scores_checkin"]).copy()
            if len(plot_data) > 0:
                try:
                    # Normalizar puntuaciones si están en escala 0-5 o 0-10
                    max_score = plot_data["review_scores_checkin"].max()
                    if max_score <= 5:
                        plot_data["review_scores_checkin"] = plot_data["review_scores_checkin"] * 20
                    elif max_score <= 10:
                        plot_data["review_scores_checkin"] = plot_data["review_scores_checkin"] * 10
                    # Crear rangos de puntuación de check-in
                    bins = [0, 80, 90, 100]
                    labels = ["0-80", "80-90", "90-100"]
                    plot_data["checkin_range"] = pd.cut(
                        plot_data["review_scores_checkin"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Contar alojamientos por rango
                    checkin_counts = plot_data["checkin_range"].value_counts().sort_index()
                    # Filtrar rangos con suficientes datos (mínimo 5 puntos)
                    min_points = 5
                    valid_ranges = checkin_counts[checkin_counts >= min_points].index.tolist()
                    plot_data_filtered = plot_data[plot_data["checkin_range"].isin(valid_ranges)]
                    
                    if len(plot_data_filtered) > 0 and len(valid_ranges) > 0:
                        # Recalcular conteos
                        checkin_counts_filtered = plot_data_filtered["checkin_range"].value_counts().sort_index()
                        # Preparar datos para gráfico de dona
                        donut_data = pd.DataFrame({
                            "Rango": checkin_counts_filtered.index,
                            "Conteo": checkin_counts_filtered.values
                        })
                        # Crear gráfico de dona
                        fig = px.pie(
                            donut_data,
                            names="Rango",
                            values="Conteo",
                            title="Proporción de Alojamientos por Puntuación de Check-in",
                            hole=0.4,
                            color_discrete_sequence=["#FF5A5F", "#00A699", "#484848"]
                        )
                        fig.update_traces(
                            textinfo="percent+label",
                            textposition="inside",
                            textfont=dict(color="white", size=14),
                            hovertemplate="%{label}: %{value} alojamientos (%{percent})"
                        )
                        fig.update_layout(
                            title=dict(text="Proporción de Alojamientos por Puntuación de Check-in", font=dict(color="white"), x=0.5),
                            showlegend=True,
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=100, b=50, l=50, r=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay rangos de puntuación de check-in con suficientes datos (mínimo 5 puntos por rango).")
                except Exception as e:
                    st.error(f"Error al generar el gráfico de dona: {str(e)}")
                    st.write("Estadísticas de 'review_scores_checkin':", plot_data["review_scores_checkin"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el gráfico.")
        else:
            st.info("La columna 'review_scores_checkin' no está disponible.")
# Pestaña 6: Características temporales
with tabs[5]:
        if "minimum_nights" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["minimum_nights"]).copy()
            if len(plot_data) > 0:
                try:
                    # Filtrar valores extremos (>365 noches)
                    plot_data = plot_data[plot_data["minimum_nights"] <= 365]
                    if len(plot_data) > 0:
                        # Calcular mediana y moda
                        median_nights = plot_data["minimum_nights"].median()
                        mode_nights = plot_data["minimum_nights"].mode()[0]
                        # Crear rangos de noches mínimas
                        bins = [0, 2, 7, 365]
                        labels = ["1-2 noches", "3-7 noches", ">7 noches"]
                        plot_data["min_nights_range"] = pd.cut(
                            plot_data["minimum_nights"], bins=bins, labels=labels, include_lowest=True
                        )
                        # Calcular porcentajes
                        percentages = plot_data["min_nights_range"].value_counts(normalize=True) * 100
                        # Generar output textual
                        text_output = (
                            f"**Distribución de Noches Mínimas Requeridas**\n\n"
                            f"- **Mediana**: {median_nights:.0f} noches\n"
                            f"- **Moda**: {mode_nights:.0f} noches\n"
                            f"- **Distribución**:\n"
                            f"  - 1-2 noches: {percentages.get('1-2 noches', 0):.1f}%\n"
                            f"  - 3-7 noches: {percentages.get('3-7 noches', 0):.1f}%\n"
                            f"  - >7 noches: {percentages.get('>7 noches', 0):.1f}%"
                        )
                        st.markdown(text_output)
                    else:
                        st.warning("No hay datos válidos de noches mínimas después de filtrar valores extremos.")
                except Exception as e:
                    st.error(f"Error al generar el resumen: {str(e)}")
                    st.write("Estadísticas de 'minimum_nights':", plot_data["minimum_nights"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el resumen.")
        else:
            st.info("La columna 'minimum_nights' no está disponible.")

        if "maximum_nights" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["maximum_nights"]).copy()
            if len(plot_data) > 0:
                try:
                    # Tratar valores extremos (>1125 como "sin límite")
                    plot_data["maximum_nights"] = plot_data["maximum_nights"].clip(upper=1125)
                    # Calcular mediana y moda
                    median_nights = plot_data["maximum_nights"].median()
                    mode_nights = plot_data["maximum_nights"].mode()[0]
                    # Crear rangos de noches máximas
                    bins = [0, 30, 365, 1125]
                    labels = ["≤30 noches", "31-365 noches", ">365 noches"]
                    plot_data["max_nights_range"] = pd.cut(
                        plot_data["maximum_nights"], bins=bins, labels=labels, include_lowest=True
                    )
                    # Calcular porcentajes
                    percentages = plot_data["max_nights_range"].value_counts(normalize=True) * 100
                    # Generar output textual
                    text_output = (
                        f"**Distribución de Noches Máximas Permitidas**\n\n"
                        f"- **Mediana**: {median_nights:.0f} noches\n"
                        f"- **Moda**: {mode_nights:.0f} noches\n"
                        f"- **Distribución**:\n"
                        f"  - ≤30 noches: {percentages.get('≤30 noches', 0):.1f}%\n"
                        f"  - 31-365 noches: {percentages.get('31-365 noches', 0):.1f}%\n"
                        f"  - >365 noches: {percentages.get('>365 noches', 0):.1f}%"
                    )
                    st.markdown(text_output)
                except Exception as e:
                    st.error(f"Error al generar el resumen: {str(e)}")
                    st.write("Estadísticas de 'maximum_nights':", plot_data["maximum_nights"].describe())
            else:
                st.warning("No hay datos suficientes para mostrar el resumen.")
        else:
            st.info("La columna 'maximum_nights' no está disponible.")


        if "last_scraped" in filtered_data.columns:
            plot_data = filtered_data.dropna(subset=["last_scraped"]).copy()
            if len(plot_data) > 0:
                try:
                    plot_data["last_scraped"] = pd.to_datetime(plot_data["last_scraped"], errors="coerce")
                    date_counts = plot_data["last_scraped"].value_counts()
                    most_common_date = date_counts.index[0].strftime("%Y-%m-%d")
                    most_common_percentage = (date_counts.iloc[0] / len(plot_data) * 100)
                    unique_dates = len(date_counts)
                    text_output = (
                        f"**Frecuencia de Última Fecha de Recopilación**\n\n"
                        f"- **Fecha más común**: {most_common_date} ({most_common_percentage:.1f}% de los alojamientos)\n"
                        f"- **Fechas únicas**: {unique_dates}"
                    )
                    st.markdown(text_output)
                except Exception as e:
                    st.error(f"Error al generar el resumen: {str(e)}")
                    st.write("Valores únicos de 'last_scraped':", plot_data["last_scraped"].unique())
            else:
                st.warning("No hay datos suficientes para mostrar el resumen.")
        else:
            st.info("La columna 'last_scraped' no está disponible.")

# Pestaña 7: Características de Usuarios
# Pestaña 7: Características de Usuarios
with tabs[6]:
    st.markdown('<div class="subheader">Características de Usuarios</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    # Columna 1: Gráficos para Resumen General y Clusters
    with col1:
        # Gráfico 1: Resumen General de Reseñas
        resumen_data = pd.DataFrame({
            "Métrica": ["Total de Reseñas", "Total de Usuarios"],
            "Valor": [50000, 49812]
        })
        fig_resumen = px.bar(
            resumen_data,
            x="Métrica",
            y="Valor",
            title="Resumen General de Reseñas",
            color="Métrica",
            color_discrete_sequence=px.colors.sequential.Viridis,
            text=resumen_data["Valor"].round(4)
        )
        fig_resumen.update_traces(textposition="auto")
        fig_resumen.update_layout(
            xaxis_title="Métrica",
            yaxis_title="Valor",
            showlegend=False,
            title_x=0.5
        )
        st.plotly_chart(fig_resumen, use_container_width=True)

        # Gráfico 2: Clusters de Reseñas
        clusters_data = pd.DataFrame({
            "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
            "Número de Reseñas": [7130, 3837, 39033],
            "Sentimiento Promedio": [0.853, 0.720, 0.766]
        })
        fig_clusters = go.Figure()
        fig_clusters.add_trace(
            go.Bar(
                x=clusters_data["Cluster"],
                y=clusters_data["Número de Reseñas"],
                name="Número de Reseñas",
                marker_color="#FF5A5F",
                text=clusters_data["Número de Reseñas"],
                textposition="auto"
            )
        )
        fig_clusters.add_trace(
            go.Scatter(
                x=clusters_data["Cluster"],
                y=clusters_data["Sentimiento Promedio"],
                name="Sentimiento Promedio",
                yaxis="y2",
                line=dict(color="#00A699", width=3),
                mode="lines+markers",
                marker=dict(size=8)
            )
        )
        fig_clusters.update_layout(
            title="Clusters de Reseñas",
            xaxis_title="Cluster",
            yaxis_title="Número de Reseñas",
            yaxis2=dict(
                title="Sentimiento Promedio",
                overlaying="y",
                side="right"
            ),
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

    # Columna 2: Gráfico para Temas y Gráficos Originales
    with col2:
        # Gráfico 3: Temas Identificados
        temas_data = pd.DataFrame({
            "Tema": ["Check-in y asistencia", "Características del alojamiento", "Experiencia general", "Valoración positiva", "Atención y espacio"],
            "Importancia": [5, 8, 6, 10, 4]  # Valores ficticios
        })
        fig_temas = px.bar(
            temas_data,
            y="Tema",
            x="Importancia",
            title="Temas Identificados",
            orientation="h",
            color="Tema",
            color_discrete_sequence=px.colors.sequential.Inferno,
            text=temas_data["Importancia"]
        )
        fig_temas.update_traces(textposition="auto")
        fig_temas.update_layout(
            xaxis_title="Recurrencia",
            yaxis_title="Tema",
            showlegend=False,
            title_x=0.5
        )
        st.plotly_chart(fig_temas, use_container_width=True)

        # Gráfico 4: Actividad y Sentimiento por Día de la Semana
        review_data = pd.DataFrame({
            "Día": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
            "review_id": [8257, 6485, 6278, 6228, 6607, 6539, 9606],
            "vader_compound": [0.776859, 0.783723, 0.776271, 0.773506, 0.775032, 0.776225, 0.764861]
        })
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=review_data["Día"],
                y=review_data["review_id"],
                name="Número de Reseñas",
                marker_color="#FF5A5F",
                text=review_data["review_id"],
                textposition="auto"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=review_data["Día"],
                y=review_data["vader_compound"],
                name="Sentimiento Promedio",
                line=dict(color="#00A699", width=3),
                mode="lines+markers",
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        fig.update_layout(
            title=dict(text="Actividad y Sentimiento por Día de la Semana", font=dict(color="white"), x=0.5),
            xaxis_title="Día de la Semana",
            yaxis_title="Número de Reseñas",
            yaxis2_title="Sentimiento Promedio",
            height=500,
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Número de Reseñas", secondary_y=False)
        fig.update_yaxes(title_text="Sentimiento Promedio", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Gráfico 5: Evolución del Sentimiento Promedio por Mes
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
        sentiment_data["year_month"] = pd.to_datetime(sentiment_data["year_month"], format="%Y-%m")
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

# Pie de página
st.markdown("---")
st.markdown("TFG - Análisis de Precios y Reseñas en Airbnb | Ángel Soto García")
