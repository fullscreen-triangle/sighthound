import json
import os


def convert_to_geojson(input_file, output_file):
    """
    Convert a JSON file to GeoJSON format by removing altitude data.

    Args:
        input_file (str): The name of the input JSON file.
        output_file (str): The name of the output GeoJSON file.
    """
    # Load the JSON data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Create a GeoJSON template
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Process each point in the JSON data
    for entry in data:
        # Extract the necessary fields
        timestamp = entry["timestamp"]
        latitude = entry["latitude"][0]  # Get the first value from the list
        longitude = entry["longitude"][0]  # Get the first value from the list

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "timestamp": timestamp
            },
            "geometry": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            }
        }

        # Add the feature to the GeoJSON
        geojson["features"].append(feature)

    # Save the GeoJSON to a file
    with open(output_file, "w") as f:
        json.dump(geojson, f, indent=4)

    print(f"GeoJSON file created: {output_file}")


if __name__ == "__main__":
    # Automatically find the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Input and output file names
    input_file = os.path.join(current_directory, "merged_data.json")
    output_file = os.path.join(current_directory, "puchheim_triangulated_sighthound.geojson")

    # Convert the JSON file to GeoJSON
    convert_to_geojson(input_file, output_file)
