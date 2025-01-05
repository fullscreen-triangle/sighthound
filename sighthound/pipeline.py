from data_processing import combine_files
from dynamic_filtering import dynamic_kalman_filter
from triangulation import get_cell_tower_data, triangulate_position
from weather import get_weather
from optimal_path import get_optimal_path
from dubins_path import calculate_dubins_path
from czml_generator import generate_czml, save_czml
from visualization import plot_gps_trace, save_map, save_to_json
import os


def full_pipeline(file_paths, mapbox_api_key, weather_api_key, cell_api_key, lat, lon, radius=1000):
    # Step 1: Combine and smooth GPS data
    combined_data = combine_files(file_paths)
    smoothed_data = dynamic_kalman_filter(combined_data)

    # Step 2: Triangulate position using cell towers
    cell_tower_data = get_cell_tower_data(cell_api_key, lat, lon, radius)
    triangulated_position = triangulate_position(cell_tower_data)

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
    save_czml(czml, "../public/activity.czml")

    # Step 7: Save and visualize results
    save_to_json(smoothed_data.to_dict("records"), "../public/smoothed_data.json")
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
