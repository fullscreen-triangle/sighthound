import pandas as pd
from typing import Dict, List, Any
import json
from pathlib import Path


def generate_czml(trajectory: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate CZML document from trajectory DataFrame

    Args:
        trajectory: DataFrame with timestamp, lat, lon, altitude columns

    Returns:
        List of CZML packets
    """

    # Create header packet
    document = [{
        "id": "document",
        "name": "Unified Trajectory",
        "version": "1.0",
        "clock": {
            "interval": f"{trajectory['timestamp'].min().isoformat()}Z/{trajectory['timestamp'].max().isoformat()}Z",
            "currentTime": trajectory['timestamp'].min().isoformat() + "Z",
            "multiplier": 1
        }
    }]

    # Create path packet
    path_positions = []
    for _, row in trajectory.iterrows():
        timestamp = row['timestamp'].isoformat() + "Z"
        path_positions.extend([
            timestamp,
            row['longitude'],
            row['latitude'],
            row.get('altitude', 0)  # Default to 0 if no altitude
        ])

    path_packet = {
        "id": "trajectory_path",
        "name": "Trajectory Path",
        "path": {
            "material": {
                "polylineOutline": {
                    "color": {
                        "rgba": [255, 0, 0, 255]
                    },
                    "outlineColor": {
                        "rgba": [255, 255, 255, 255]
                    },
                    "outlineWidth": 2
                }
            },
            "width": 3,
            "leadTime": 0,
            "trailTime": 0,
            "resolution": 5
        },
        "position": {
            "epoch": trajectory['timestamp'].min().isoformat() + "Z",
            "cartographicDegrees": path_positions
        }
    }
    document.append(path_packet)

    # Create point packet for current position
    point_positions = []
    for _, row in trajectory.iterrows():
        timestamp = row['timestamp'].isoformat() + "Z"
        point_positions.extend([
            timestamp,
            row['longitude'],
            row['latitude'],
            row.get('altitude', 0)
        ])

    point_packet = {
        "id": "current_position",
        "name": "Current Position",
        "billboard": {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            "scale": 0.5,
            "pixelOffset": {
                "cartesian2": [0, 0]
            }
        },
        "position": {
            "epoch": trajectory['timestamp'].min().isoformat() + "Z",
            "cartographicDegrees": point_positions
        }
    }
    document.append(point_packet)

    return document


def save_czml(czml_data: List[Dict[str, Any]], output_path: Path):
    """
    Save CZML document to file

    Args:
        czml_data: CZML document as list of packets
        output_path: Path to save CZML file
    """
    with open(output_path, 'w') as f:
        json.dump(czml_data, f, indent=2)
