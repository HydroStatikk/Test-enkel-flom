import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import hmac
from utils import haversine, calculate_metrics

# Set page configuration
st.set_page_config(
    page_title="Enkelflom-local",
    page_icon="🌊",
    layout="wide"
)

# Application title and description
st.title("Enkelflom-local")
st.markdown("""
Adjust the parameters to see detailed flood analysis results.
""")

# User Authentication as a sidebar dialog
with st.sidebar:
    st.markdown("<h2 style='color: #1E88E5; font-size: 28px;'>HydroStatikk</h2>", unsafe_allow_html=True)
    
    # Check authentication state
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.subheader("Authentication Required")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log In"):
            if hmac.compare_digest(username, "Internal@AFRYuser1220") and \
               hmac.compare_digest(password, "HSNorway@2025ver"):
                st.session_state["password_correct"] = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    else:
        # Logout button in the sidebar
        if st.button("Log Out"):
            st.session_state["password_correct"] = False
            st.rerun()
        
        # Geographical parameters
        st.subheader("Geographical Parameters")
        user_lat = st.number_input("Latitude", value=59.729121, format="%.6f", 
                                help="The latitude coordinate of your location")
        user_lon = st.number_input("Longitude", value=11.030474, format="%.6f", 
                               help="The longitude coordinate of your location")
        radius_km = st.slider("Radius (km)", min_value=1, max_value=500, value=95, 
                           help="The radius to search for measurement stations")
        
        # Catchment area
        catchment_area_km2 = st.number_input("Catchment Area (km²)", min_value=0.1, value=106.0, format="%.1f", 
                                      help="The catchment area for your analysis in square kilometers")
        
        # Factors
        st.subheader("Adjustment Factors")
        climate_factor = st.number_input("Climate Factor", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.1f",
                              help="Factor to adjust for climate considerations")
        safety_factor = st.number_input("Safety Factor", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.1f",
                              help="Factor to adjust for safety considerations")
        
        # Weighting parameters
        st.subheader("Weighting Parameters")
        distance_scaling_factor = st.slider("Distance Scaling Factor", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                      help="Controls how quickly weight decays with distance (smaller values prioritize closer stations)")

# Main content area
if st.session_state["password_correct"]:
    try:
        # Load data directly from the file path
        file_path = "attached_assets/Data.xlsx"
        st.info(f"Trying to load data from {file_path}")
        try:
            df = pd.read_excel(file_path)
            st.success(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        except Exception as data_load_error:
            st.error(f"Error loading Excel file: {data_load_error}")
            st.stop()
        
        # Apply the function to filter stations within the radius
        st.info("Filtering stations based on distance")
        df["distance_to_user"] = df.apply(lambda row: haversine(user_lat, user_lon, row["latitude"], row["longitude"]), axis=1)
        df_filtered = df[df["distance_to_user"] <= radius_km].copy()
        
        if df_filtered.empty:
            st.warning(f"No stations found within {radius_km} km radius. Please increase your search radius or check your coordinates.")
        else:
            # Calculate weights and metrics
            # Calculate exponential weight based on distance
            df_filtered.loc[:, "locality_weight"] = np.exp(-df_filtered["distance_to_user"] / distance_scaling_factor)
            
            # Set the weight scaled directly to the locality weight (no scaling factor)
            df_filtered.loc[:, "locality_weight_scaled"] = df_filtered["locality_weight"] * 1.0  # Just using 1.0 as default value
            
            # Normalize the weights to make sure they sum up to 1
            df_filtered.loc[:, "locality_weight_normalized"] = df_filtered["locality_weight_scaled"] / df_filtered["locality_weight_scaled"].sum()
            
            # Make sure we have proper station names (use stationName column if it exists)
            if 'stationName' in df_filtered.columns:
                df_filtered.loc[:, 'station_name'] = df_filtered['stationName']
            elif 'station_name' not in df_filtered.columns:
                # If no station name column, create one with a generic name and index
                df_filtered.loc[:, 'station_name'] = df_filtered.index.map(lambda x: f"Station {x+1}")
            
            # Calculate the weighted average of specific discharge using locality weight
            weighted_avg_specific_discharge = np.average(df_filtered["specificDischarge"], weights=df_filtered["locality_weight_normalized"])
            
            # Calculate average and standard deviation of specificDischarge
            average_discharge = df_filtered["specificDischarge"].mean()
            std_discharge = df_filtered["specificDischarge"].std()
            
            # Calculate average discharge over the catchment area using weighted average
            average_discharge_catchment = weighted_avg_specific_discharge * catchment_area_km2 / 1000  # Convert to cubic meters per second
            
            # Display map with stations
            st.subheader("Station Map")
            
            # Create a folium map centered on user location
            m = folium.Map(location=[user_lat, user_lon], zoom_start=8)
            
            # Add a marker for user location
            folium.Marker(
                [user_lat, user_lon],
                popup="Your Location",
                icon=folium.Icon(color="red", icon="home")
            ).add_to(m)
            
            # Add a circle for the radius
            folium.Circle(
                location=[user_lat, user_lon],
                radius=radius_km * 1000,  # Convert to meters
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.2
            ).add_to(m)
            
            # Add markers for each station
            for _, row in df_filtered.iterrows():
                # Create a wider popup with more information
                popup_content = f"""
                <div style="width: 300px;">
                    <h4>Station Information</h4>
                    <b>Name:</b> {row['station_name']}<br>
                    <b>Distance:</b> {row['distance_to_user']:.2f} km<br>
                    <b>Latitude:</b> {row["latitude"]:.6f}<br>
                    <b>Longitude:</b> {row["longitude"]:.6f}<br>
                    <b>Specific Discharge:</b> {row['specificDischarge']:.2f} l/s/km²<br>
                    <b>Weight:</b> {row['locality_weight_normalized']:.4f}
                </div>
                """
                
                folium.Marker(
                    [row["latitude"], row["longitude"]],
                    popup=folium.Popup(popup_content, max_width=350),
                    icon=folium.Icon(color="blue", icon="tint")
                ).add_to(m)
            
            # Create two columns for map and local basin information
            map_col, info_col = st.columns([7, 3])
            
            with map_col:
                # Display the map with dimensions that match the entire column width
                st_data = st_folium(m, width="100%", height=600)
                # Number of stations
                st.info(f"Found {len(df_filtered)} stations within {radius_km} km radius.")
                
            with info_col:
                # Display basic metrics
                st.subheader("Local Basin Information")
                
                metrics_container = st.container()
                metrics_container.metric("Weighted Average Specific Discharge", f"{weighted_avg_specific_discharge:.2f} l/s/km²")
                metrics_container.metric("Average Specific Discharge", f"{average_discharge:.2f} l/s/km²")
                metrics_container.metric("Standard Deviation", f"{std_discharge:.2f} l/s/km²")
                metrics_container.metric("Avg Discharge Over Catchment", f"{average_discharge_catchment:.2f} m³/s")
            
            # List of the columns to calculate the average and standard deviation
            columns_to_calculate = [
                "Qm/Qn", "Q5/Qn", "Q10/Qn", "Q20/Qn", "Q50/Qn", "Q100/Qn", "Q200/Qn"
            ]
            
            # List of the columns for hourly discharge
            columns_to_calculate_hourly = [
                "Qm/Qn (kul)", "Q5/Qn (kul)", "Q10/Qn (kul)", "Q20/Qn (kul)", "Q50/Qn (kul)", "Q100/Qn (kul)", "Q200/Qn (kul)"
            ]
            
            # Define display names for each column
            column_display_names = {
                "Qm/Qn": "Middle Flood",
                "Q5/Qn": "5-Year Flood",
                "Q10/Qn": "10-Year Flood",
                "Q20/Qn": "20-Year Flood",
                "Q50/Qn": "50-Year Flood",
                "Q100/Qn": "100-Year Flood",
                "Q200/Qn": "200-Year Flood",
                "Qm/Qn (kul)": "Middle Flood (Hourly)",
                "Q5/Qn (kul)": "5-Year Flood (Hourly)",
                "Q10/Qn (kul)": "10-Year Flood (Hourly)",
                "Q20/Qn (kul)": "20-Year Flood (Hourly)",
                "Q50/Qn (kul)": "50-Year Flood (Hourly)",
                "Q100/Qn (kul)": "100-Year Flood (Hourly)",
                "Q200/Qn (kul)": "200-Year Flood (Hourly)"
            }
            
            # Initialize lists to store data for charts
            labels = []
            flood_discharge_values = []
            deviated_values = []
            dimensioned_values = []
            dimensioned_deviated_values = []
            
            for column in columns_to_calculate:
                # Calculate metrics for this column
                metrics = calculate_metrics(
                    df_filtered, 
                    column, 
                    average_discharge_catchment, 
                    climate_factor, 
                    safety_factor
                )
                
                # Extract metrics
                average_value = metrics["average_value"]
                std_dev = metrics["std_dev"]
                result = metrics["result"]
                flood_discharge = metrics["flood_discharge"]
                deviated_flood_discharge = metrics["deviated_flood_discharge"]
                dimensioned_flood_discharge = metrics["dimensioned_flood_discharge"]
                dimensioned_deviated_flood_discharge = metrics["dimensioned_deviated_flood_discharge"]
                
                # Store values for charts
                labels.append(column_display_names[column])
                flood_discharge_values.append(flood_discharge)
                deviated_values.append(deviated_flood_discharge)
                dimensioned_values.append(dimensioned_flood_discharge)
                dimensioned_deviated_values.append(dimensioned_deviated_flood_discharge)
            
            # Create a line chart for the discharge values - showing only dimensioned values
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=labels, 
                y=dimensioned_values,
                mode='lines+markers',
                name='Dimensioned Flood Discharge',
                line=dict(width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=labels, 
                y=dimensioned_deviated_values,
                mode='lines+markers',
                name='Dimensioned Deviated Flood Discharge',
                line=dict(width=3, dash='dash')
            ))
            
            fig.update_layout(
                title="Flood (døgnmiddel) Values",
                xaxis_title="Flood Type",
                yaxis_title="Discharge (m³/s)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                ),
                hovermode="x"
            )
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Display flood values graph
                st.subheader("Flood (døgnmiddel)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed flood values in a table
                st.subheader("Flood (døgnmiddel) Values")
                
                # Create a DataFrame for the table display
                flood_data = {
                    "Flood Type": labels,
                    "Flood Discharge (m³/s)": [f"{val:.2f}" for val in flood_discharge_values],
                    "Deviated Flood Discharge (m³/s)": [f"{val:.2f}" for val in deviated_values],
                    "Dimensioned Flood Discharge (m³/s)": [f"{val:.2f}" for val in dimensioned_values],
                    "Dimensioned Deviated Flood Discharge (m³/s)": [f"{val:.2f}" for val in dimensioned_deviated_values]
                }
                
                # Display the table
                st.dataframe(flood_data, use_container_width=True)
                
                # Hourly Discharge Calculations
                st.subheader("Flood (kulminasjon)")
                
                # Initialize lists for hourly values
                hourly_labels = []
                hourly_flood_values = []
                hourly_deviated_values = []
                hourly_dimensioned_values = []
                hourly_dimensioned_deviated_values = []
                
                for column in columns_to_calculate_hourly:
                    # Calculate metrics for hourly columns
                    average_value = df_filtered[column].mean()
                    std_dev = df_filtered[column].std()
                    
                    # Calculate the result for each column (average_value + std_dev) / average_value
                    result = (average_value + std_dev) / average_value
                    
                    # Calculate flood discharge for hourly values
                    hourly_flood_discharge = average_discharge_catchment * average_value
                    
                    # Calculate deviated flood discharge
                    deviated_hourly_flood_discharge = hourly_flood_discharge * result
                    
                    # Calculate dimensioned hourly flood discharge
                    dimensioned_hourly_flood_discharge = hourly_flood_discharge * climate_factor * safety_factor
                    
                    # Calculate dimensioned deviated hourly flood discharge
                    dimensioned_deviated_hourly_flood_discharge = deviated_hourly_flood_discharge * climate_factor * safety_factor
                    
                    # Store values for charts
                    hourly_labels.append(column_display_names[column])
                    hourly_flood_values.append(hourly_flood_discharge)
                    hourly_deviated_values.append(deviated_hourly_flood_discharge)
                    hourly_dimensioned_values.append(dimensioned_hourly_flood_discharge)
                    hourly_dimensioned_deviated_values.append(dimensioned_deviated_hourly_flood_discharge)
                
                # Create a line chart for the hourly discharge values - showing only dimensioned values
                hourly_fig = go.Figure()
                
                hourly_fig.add_trace(go.Scatter(
                    x=hourly_labels, 
                    y=hourly_dimensioned_values,
                    mode='lines+markers',
                    name='Dimensioned Hourly Flood Discharge',
                    line=dict(width=3, color='#FF6B6B')
                ))
                
                hourly_fig.add_trace(go.Scatter(
                    x=hourly_labels, 
                    y=hourly_dimensioned_deviated_values,
                    mode='lines+markers',
                    name='Dimensioned Deviated Hourly Flood Discharge',
                    line=dict(width=3, dash='dash', color='#4ECDC4')
                ))
                
                hourly_fig.update_layout(
                    title="Flood (kulminasjon) Values",
                    xaxis_title="Flood Type",
                    yaxis_title="Discharge (m³/s)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0
                    ),
                    hovermode="x"
                )
                
                # Display the hourly flood values graph
                st.plotly_chart(hourly_fig, use_container_width=True)
                
                # Display hourly discharge values in a table
                hourly_data = {
                    "Flood Type": hourly_labels,
                    "Hourly Flood Discharge (m³/s)": [f"{val:.2f}" for val in hourly_flood_values],
                    "Hourly Deviated Flood Discharge (m³/s)": [f"{val:.2f}" for val in hourly_deviated_values],
                    "Hourly Dimensioned Flood Discharge (m³/s)": [f"{val:.2f}" for val in hourly_dimensioned_values],
                    "Hourly Dimensioned Deviated Flood Discharge (m³/s)": [f"{val:.2f}" for val in hourly_dimensioned_deviated_values]
                }
                
                # Display the hourly table
                st.dataframe(hourly_data, use_container_width=True)
            
            with col2:
                
                # Land use parameters table
                st.subheader("Land Use Parameters Statistics")
                
                # Define the land use columns and their display names
                land_use_columns = {
                    "percentAgricul": "Agricultural Land", 
                    "percentBog": "Bog/Wetland", 
                    "percentEffLake": "Effective Lake", 
                    "percentForest": "Forest", 
                    "percentLake": "Lake", 
                    "percentMountain": "Mountain/Highland", 
                    "percentUrban": "Urban Area"
                }
                
                # Create a dictionary to store the stats
                land_use_stats = {
                    "Land Use Type": [],
                    "Average (%)": [],
                    "Standard Deviation (%)": []
                }
                
                # Calculate statistics for each land use parameter
                for col, display_name in land_use_columns.items():
                    if col in df_filtered.columns:
                        avg_value = df_filtered[col].mean()
                        std_value = df_filtered[col].std()
                        
                        # Add to the stats dictionary
                        land_use_stats["Land Use Type"].append(display_name)
                        land_use_stats["Average (%)"].append(f"{avg_value:.2f}")
                        land_use_stats["Standard Deviation (%)"].append(f"{std_value:.2f}")
                
                # Create a simple bar chart for land use parameters with both average and standard deviation
                if land_use_stats["Land Use Type"]:
                    land_use_df = pd.DataFrame(land_use_stats)
                    
                    # Convert values to numeric
                    land_use_df["Average (%)"] = land_use_df["Average (%)"].apply(lambda x: float(x))
                    land_use_df["Standard Deviation (%)"] = land_use_df["Standard Deviation (%)"].apply(lambda x: float(x))
                    
                    # Create a simple bar chart using Plotly with both average and std dev
                    fig_land_use = go.Figure()
                    
                    # Add bar for averages
                    fig_land_use.add_trace(go.Bar(
                        x=land_use_df["Land Use Type"],
                        y=land_use_df["Average (%)"],
                        name="Average",
                        marker_color='rgb(55, 83, 109)',
                        text=land_use_df["Average (%)"].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto',
                    ))
                    
                    # Add bar for standard deviations
                    fig_land_use.add_trace(go.Bar(
                        x=land_use_df["Land Use Type"],
                        y=land_use_df["Standard Deviation (%)"],
                        name="Standard Deviation",
                        marker_color='rgb(26, 118, 255)',
                        text=land_use_df["Standard Deviation (%)"].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto',
                    ))
                    
                    # Customize layout
                    fig_land_use.update_layout(
                        title="Land Use Statistics",
                        xaxis_tickangle=-45,
                        yaxis_title="Percentage (%)",
                        xaxis_title="",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        barmode='group'
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig_land_use, use_container_width=True)
            
            # Display explanation
            st.subheader("Understanding the Results")
            st.markdown("""
            ### Key Concepts
            
            - **Specific Discharge**: The amount of water flowing from a unit area of the catchment (liters per second per square kilometer).
            - **Catchment Area**: The area of land where water from rain or snowmelt drains into a specific body of water.
            - **Flood Discharge**: Volume of water flowing through a river/stream during flood events (cubic meters per second).
            - **Return Period**: Average time between flood events of a certain magnitude (e.g., Q100 = once in 100 years flood).
            
            ### Factors
            
            - **Climate Factor**: Adjustment for future climate change impacts on rainfall and runoff patterns.
            - **Safety Factor**: Additional margin of safety for critical infrastructure or sensitive areas.
            
            ### Land Use Impact
            
            Land use affects how water moves through a catchment:
            - Forests slow runoff and increase infiltration
            - Urban areas increase runoff speed and volume
            - Lakes provide natural flood attenuation
            """)
            
    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    # Show message when not authenticated
    st.info("Please log in using the sidebar to access the application.")
    st.markdown("""
    ### About Enkelflom-local
    
    This application provides hydrological flood analysis for specific locations in Norway. 
    Using data from measurement stations, it calculates flood discharge metrics and risk assessment.
    
    To access the full functionality, please log in with your provided credentials.
    """)