import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Radius of Earth in km
    R = 6371
    
    # Convert decimal degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in km
    return R * c

def calculate_metrics(df, column, average_discharge_catchment, climate_factor, safety_factor):
    """
    Calculate flood metrics for a given column
    
    Parameters:
    df (DataFrame): Filtered DataFrame containing station data
    column (str): The column name to calculate metrics for
    average_discharge_catchment (float): Average discharge over the catchment area
    climate_factor (float): Factor to adjust for climate considerations
    safety_factor (float): Factor to adjust for safety considerations
    
    Returns:
    dict: Dictionary containing calculated metrics
    """
    # Calculate average and standard deviation
    average_value = df[column].mean()
    std_dev = df[column].std()
    
    # Calculate the required expression for the column
    result = (average_value + std_dev) / average_value
    
    # Calculate flood discharge by multiplying average discharge over the catchment area by the Qm/Qn value
    flood_discharge = average_discharge_catchment * average_value
    
    # Calculate deviated flood discharge
    deviated_flood_discharge = flood_discharge * result
    
    # Calculate dimensioned flood discharge
    dimensioned_flood_discharge = flood_discharge * climate_factor * safety_factor
    
    # Calculate dimensioned deviated flood discharge
    dimensioned_deviated_flood_discharge = deviated_flood_discharge * climate_factor * safety_factor
    
    # Return all metrics in a dictionary
    return {
        "average_value": average_value,
        "std_dev": std_dev,
        "result": result,
        "flood_discharge": flood_discharge,
        "deviated_flood_discharge": deviated_flood_discharge,
        "dimensioned_flood_discharge": dimensioned_flood_discharge,
        "dimensioned_deviated_flood_discharge": dimensioned_deviated_flood_discharge
    }
