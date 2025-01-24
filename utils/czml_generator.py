import pandas as pd
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def generate_czml(
    trajectory: pd.DataFrame,
    style_config: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Generate CZML document from trajectory DataFrame with enhanced styling and metadata

    Args:
        trajectory: DataFrame with timestamp, lat, lon, altitude columns
        style_config: Optional configuration for visual styling

    Returns:
        List of CZML packets
    """
    default_style = {
        'path_color': [255, 0, 0, 255],
        'path_width': 3,
        'point_color': [0, 255, 0, 255],
        'point_size': 8,
        'outline_color': [255, 255, 255, 255],
        'outline_width': 2,
        'label_font': '12pt Roboto',
        'label_style': 'FILL_AND_OUTLINE',
        'label_color': [255, 255, 255, 255]
    }
    style = {**default_style, **(style_config or {})}

    # Ensure timestamps are datetime objects
    trajectory['timestamp'] = pd.to_datetime(trajectory['timestamp'])
    if trajectory['timestamp'].dt.tz is None:
        trajectory['timestamp'] = trajectory['timestamp'].dt.tz_localize(timezone.utc)

    # Create header packet
    document = [{
        "id": "document",
        "name": "Unified Trajectory",
        "version": "1.0",
        "clock": {
            "interval": f"{trajectory['timestamp'].min().isoformat()}/{trajectory['timestamp'].max().isoformat()}",
            "currentTime": trajectory['timestamp'].min().isoformat(),
            "multiplier": 1,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER"
        }
    }]

    # Convert timestamps to seconds since epoch for position data
    start_time = trajectory['timestamp'].min()

    # Create path packet with enhanced styling
    path_positions = []
    for _, row in trajectory.iterrows():
        seconds = (row['timestamp'] - start_time).total_seconds()
        path_positions.extend([
            seconds,
            row['longitude'],
            row['latitude'],
            row.get('altitude', 0)
        ])

    path_packet = {
        "id": "trajectory_path",
        "name": "Trajectory Path",
        "path": {
            "material": {
                "polylineOutline": {
                    "color": {
                        "rgba": style['path_color']
                    },
                    "outlineColor": {
                        "rgba": style['outline_color']
                    },
                    "outlineWidth": style['outline_width']
                }
            },
            "width": style['path_width'],
            "leadTime": 0,
            "trailTime": 0,
            "resolution": 5
        },
        "position": {
            "epoch": start_time.isoformat(),
            "cartographicDegrees": path_positions
        }
    }
    document.append(path_packet)

    # Add confidence visualization if available
    if 'confidence' in trajectory.columns:
        confidence_colors = []
        for _, row in trajectory.iterrows():
            seconds = (row['timestamp'] - start_time).total_seconds()
            # Color gradient based on confidence (red to green)
            color = [
                int(255 * (1 - row['confidence'])),  # Red
                int(255 * row['confidence']),        # Green
                0,                                   # Blue
                255                                  # Alpha
            ]
            confidence_colors.extend([seconds] + color)

        path_packet["path"]["material"]["polylineOutline"]["color"] = {
            "epoch": start_time.isoformat(),
            "rgba": confidence_colors
        }

    # Create point packet for current position
    point_positions = []
    for _, row in trajectory.iterrows():
        seconds = (row['timestamp'] - start_time).total_seconds()
        point_positions.extend([
            seconds,
            row['longitude'],
            row['latitude'],
            row.get('altitude', 0)
        ])

    point_packet = {
        "id": "current_position",
        "name": "Current Position",
        "availability": f"{trajectory['timestamp'].min().isoformat()}/{trajectory['timestamp'].max().isoformat()}",
        "billboard": {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            "scale": style['point_size'] / 8,
            "color": {
                "rgba": style['point_color']
            }
        },
        "label": {
            "text": "Current Position",
            "font": style['label_font'],
            "style": style['label_style'],
            "fillColor": {
                "rgba": style['label_color']
            },
            "outlineColor": {
                "rgba": [0, 0, 0, 255]
            },
            "outlineWidth": 2,
            "horizontalOrigin": "LEFT",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {
                "cartesian2": [10, -10]
            }
        },
        "position": {
            "epoch": start_time.isoformat(),
            "cartographicDegrees": point_positions
        }
    }
    document.append(point_packet)

    return document


def save_czml(czml_data: List[Dict[str, Any]], output_path: Path):
    """
    Save CZML document to file with error handling

    Args:
        czml_data: CZML document as list of packets
        output_path: Path to save CZML file
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(czml_data, f, indent=2)
    except Exception as e:
        raise Exception(f"Failed to save CZML file: {str(e)}")
