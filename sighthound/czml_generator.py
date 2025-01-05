import json
import numpy as np


def generate_czml(gps_data, triangulated_position=None):
    """
    Generate CZML for Cesium visualization.
    Args:
        gps_data: DataFrame containing GPS points with timestamp, latitude, longitude, and altitude.
        triangulated_position: Dictionary with triangulated position (optional).
    Returns:
        CZML as a list of dictionaries.
    """
    czml = [
        {
            "id": "document",
            "name": "GPS Activity",
            "version": "1.0"
        }
    ]

    # Add GPS data as a polyline
    czml.append({
        "id": "gps_trace",
        "polyline": {
            "positions": {
                "cartographicDegrees": [
                    coord for row in gps_data.itertuples()
                    for coord in [row.longitude, row.latitude, row.altitude]
                ]
            },
            "material": {
                "solidColor": {
                    "color": {"rgba": [0, 0, 255, 255]}  # Blue line
                }
            },
            "width": 2
        }
    })

    # Add triangulated position if available
    if triangulated_position:
        czml.append({
            "id": "triangulated_position",
            "position": {
                "cartographicDegrees": [
                    float(triangulated_position["longitude"]),
                    float(triangulated_position["latitude"]),
                    0
                ]
            },
            "point": {
                "pixelSize": 10,
                "color": {"rgba": [255, 0, 0, 255]}  # Red point
            }
        })

    return czml


def save_czml(czml, output_file):
    """
    Save CZML to a file.
    Args:
        czml: CZML data.
        output_file: Path to the output file.
    """
    # Ensure all NumPy arrays are converted to lists
    def numpy_to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(output_file, "w") as f:
        json.dump(czml, f, indent=2, default=numpy_to_python)
