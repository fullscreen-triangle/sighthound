import matplotlib.pyplot as plt
import folium
import json

import numpy as np
import pandas as pd


def plot_gps_trace(data, optimal_path=None, dubins_path=None, save_path="gps_trace_plot.png"):
    """
    Plot GPS trace, optimal path, and Dubin's path on a 2D plot.
    Args:
        data: DataFrame containing GPS points with latitude and longitude.
        optimal_path: List of (latitude, longitude) tuples for the optimal path.
        dubins_path: List of (x, y) tuples for the Dubin's path.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(data["longitude"], data["latitude"], label="GPS Trace", color="blue")

    if optimal_path:
        opt_lon, opt_lat = zip(*optimal_path)
        plt.plot(opt_lon, opt_lat, label="Optimal Path", color="green")

    if dubins_path:
        dub_lon, dub_lat = zip(*dubins_path)
        plt.plot(dub_lon, dub_lat, label="Dubin's Path", color="red")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title("GPS Trace and Paths")
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def save_map(data, triangulated_position=None, optimal_path=None, dubins_path=None, save_path="activity_map.html"):
    """
    Save an interactive map with GPS trace, triangulated position, and paths.
    Args:
        data: DataFrame containing GPS points.
        triangulated_position: Dictionary with triangulated position (optional).
        optimal_path: List of (latitude, longitude) tuples for the optimal path (optional).
        dubins_path: List of (x, y) tuples for the Dubin's path (optional).
        save_path: Path to save the map.
    Returns:
        A Folium map object.
    """
    # Ensure latitude and longitude columns exist and are valid
    if "latitude" not in data.columns or "longitude" not in data.columns:
        raise ValueError("Data must contain 'latitude' and 'longitude' columns.")

    # Filter out invalid rows with NaN values for latitude or longitude
    valid_data = data.dropna(subset=["latitude", "longitude"])

    # Generate GPS trace as a list of (latitude, longitude) tuples
    points = list(zip(valid_data["latitude"].tolist(), valid_data["longitude"].tolist()))

    # Create the Folium map centered on the mean latitude and longitude
    m = folium.Map(location=[valid_data["latitude"].mean(), valid_data["longitude"].mean()], zoom_start=15)

    # Add the GPS trace
    if points:
        folium.PolyLine(points, color="blue", weight=2.5, tooltip="GPS Trace").add_to(m)

    # Add triangulated position if available
    if triangulated_position:
        folium.Marker(
            [triangulated_position["latitude"], triangulated_position["longitude"]],
            popup="Triangulated Position",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # Add optimal path if available
    if optimal_path:
        folium.PolyLine(optimal_path, color="green", weight=2.5, tooltip="Optimal Path").add_to(m)

    # Add Dubin's path if available
    if dubins_path:
        folium.PolyLine(dubins_path, color="red", weight=2.5, tooltip="Dubin's Path").add_to(m)

    # Save the map
    m.save(save_path)
    return m




def save_to_json(data, filename):
    """
    Save data to a JSON file.
    Args:
        data: Dictionary or list to save.
        filename: File path for the JSON file.
    """

    def json_serializer(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert pandas.Timestamp to ISO 8601 string
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy.ndarray to list
        if isinstance(obj, np.generic):  # Handle numpy scalar values
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(filename, "w") as f:
        json.dump(data, f, indent=4, default=json_serializer)


