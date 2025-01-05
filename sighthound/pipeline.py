import os
from data_processing import combine_files
from dynamic_filtering import dynamic_kalman_filter
from triangulation import get_cell_tower_data, triangulate_position, validate_parameters
from weather import get_weather
from optimal_path import get_optimal_path
from dubins_path import calculate_dubins_path
from czml_generator import generate_czml, save_czml
from visualization import plot_gps_trace, save_map, save_to_json


def full_pipeline(file_paths, mapbox_api_key, weather_api_key, cell_api_key, lat, lon, radius=1000):
    """
    Full pipeline to process GPS data and save results.

    Args:
        file_paths (list): Paths to GPS data files.
        mapbox_api_key (str): Mapbox API key for optimal path calculations.
        weather_api_key (str): OpenWeather API key for weather data.
        cell_api_key (str): OpenCellID API key for triangulation.
        lat (float): Latitude for initial triangulation.
        lon (float): Longitude for initial triangulation.
        radius (int): Radius for cell tower data search (default is 1000 meters).

    Returns:
        tuple: Processed data and results.
    """
    # Step 1: Combine and smooth GPS data
    combined_data = combine_files(file_paths)
    smoothed_data = dynamic_kalman_filter(combined_data)

    # Step 2: Fetch cell tower data and triangulate position
    # Step 2: Fetch cell tower data and triangulate position
    try:
        cell_tower_params = [
            {
                "lat": lat,
                "lon": lon,
                "mcc": 262,
                "mnc": 1,
                "lac": 7033,
                "cellid": 12345,
                "rating": 50,
                "direction": 90.0,
                "speed": 10.0,
                "act": "LTE",
                "ta": 5
            },
            {
                "lat": lat + 0.001,
                "lon": lon + 0.001,
                "mcc": 262,
                "mnc": 1,
                "lac": 7033,
                "cellid": 12346,
                "rating": 30,
                "direction": 180.0,
                "speed": 15.0,
                "act": "LTE",
                "ta": 3
            }
        ]

        # Validate parameters and fetch triangulation data
        validated_data = [validate_parameters(params) for params in cell_tower_params]
        cell_tower_data = [get_cell_tower_data(cell_api_key, params) for params in validated_data]
        triangulated_position = triangulate_position(cell_tower_data)

    except Exception as e:
        print(f"Triangulation failed: {e}")
        triangulated_position = None

    # Step 3: Fetch weather data
    weather_data = get_weather(weather_api_key, lat, lon)

    # Step 4: Calculate optimal path
    waypoints = smoothed_data[["latitude", "longitude"]].values.tolist()
    optimal_path = get_optimal_path(mapbox_api_key, waypoints)

    # Step 5: Calculate Dubin's path
    start = (waypoints[0][1], waypoints[0][0], 0)  # (x, y, heading)
    end = (waypoints[-1][1], waypoints[-1][0], 0)
    dubins_path = calculate_dubins_path(start, end)

    # Step 6: Generate CZML for visualization
    czml = generate_czml(smoothed_data, triangulated_position)
    save_czml(czml, "../public/merged_activity.czml")

    # Step 7: Save and visualize results
    save_to_json(smoothed_data.to_dict("records"), "../public/merged_data.json")
    plot_gps_trace(smoothed_data, optimal_path, dubins_path, save_path="../public/gps_trace_plot.png")

    return smoothed_data, triangulated_position, weather_data, optimal_path, dubins_path


if __name__ == "__main__":
    # Dynamically resolve the public directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.join(base_dir, "../public")

    # List of activity files in the public directory
    file_paths = [
        os.path.join(public_dir, "activity_8716659574.gpx"),
        os.path.join(public_dir, "activity_8716659574.kml"),
        os.path.join(public_dir, "OutdoorRun20220427174453.gpx"),
        os.path.join(public_dir, "OutdoorRun20220427174453.kml"),
        os.path.join(public_dir, "OutdoorRun20220909181633.tcx")
    ]

    # API Keys and coordinates
    mapbox_api_key = "pk.eyJ1IjoiY2hvbWJvY2hpbm9rb3NvcmFtb3RvIiwiYSI6ImNsYWIzNzN1YzA5M24zdm4xb2txdXZ0YXQifQ.mltBkVjXA6LjUJ1bi7gdRg"
    weather_api_key = "ae9af9bb6224315e875922b1e22261b5"
    cell_api_key = "pk.8edd3b757a4aa195ef5f4adc2aaef381"
    lat, lon = 48.182234736156666, 11.357110125944603

    # Run the pipeline
    results = full_pipeline(
        file_paths, mapbox_api_key, weather_api_key, cell_api_key, lat, lon
    )

    # Print the results
    print("Smoothed Data:", results[0])
    print("Triangulated Position:", results[1])
    print("Weather Data:", results[2])
    print("Optimal Path:", results[3])
    print("Dubin's Path:", results[4])
