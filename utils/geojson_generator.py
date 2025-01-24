import pandas as pd
from typing import Dict, Any, Optional, Union
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def generate_geojson(
    trajectory: pd.DataFrame,
    properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate GeoJSON from trajectory DataFrame

    Args:
        trajectory: DataFrame with timestamp, lat, lon, altitude columns
        properties: Optional additional properties for the GeoJSON

    Returns:
        GeoJSON dictionary
    """
    # Ensure required columns exist
    required_columns = ['timestamp', 'latitude', 'longitude']
    if not all(col in trajectory.columns for col in required_columns):
        raise ValueError(f"Trajectory must contain columns: {required_columns}")

    # Create coordinates list
    coordinates = trajectory[['longitude', 'latitude']].values.tolist()
    if 'altitude' in trajectory.columns:
        coordinates = [coord + [alt] for coord, alt in 
                      zip(coordinates, trajectory['altitude'])]

    # Create timestamps list
    timestamps = trajectory['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').tolist()

    # Create feature properties
    feature_properties = {
        'timestamps': timestamps,
        'start_time': timestamps[0],
        'end_time': timestamps[-1],
        'point_count': len(coordinates)
    }

    # Add confidence scores if available
    if 'confidence' in trajectory.columns:
        feature_properties['confidence_scores'] = trajectory['confidence'].tolist()
        feature_properties['average_confidence'] = float(trajectory['confidence'].mean())

    # Add any additional properties
    if properties:
        feature_properties.update(properties)

    # Create GeoJSON structure
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "properties": feature_properties
    }

    return {
        "type": "FeatureCollection",
        "features": [geojson]
    }


def save_geojson(
    geojson_data: Dict[str, Any],
    output_path: Union[str, Path],
    pretty: bool = True
):
    """
    Save GeoJSON to file with error handling

    Args:
        geojson_data: GeoJSON dictionary
        output_path: Path to save GeoJSON file
        pretty: Whether to format JSON with indentation
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(
                geojson_data,
                f,
                indent=2 if pretty else None,
                ensure_ascii=False
            )
    except Exception as e:
        raise Exception(f"Failed to save GeoJSON file: {str(e)}")


def trajectory_to_geojson_points(
    trajectory: pd.DataFrame,
    properties_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Convert trajectory to GeoJSON points with properties

    Args:
        trajectory: DataFrame with trajectory data
        properties_mapping: Optional mapping of DataFrame columns to GeoJSON properties

    Returns:
        GeoJSON FeatureCollection with points
    """
    features = []
    
    for _, row in trajectory.iterrows():
        # Create point geometry
        geometry = {
            "type": "Point",
            "coordinates": [
                row['longitude'],
                row['latitude']
            ]
        }
        if 'altitude' in row:
            geometry["coordinates"].append(row['altitude'])

        # Create properties
        properties = {
            "timestamp": row['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        # Add mapped properties
        if properties_mapping:
            for prop_name, col_name in properties_mapping.items():
                if col_name in row:
                    properties[prop_name] = row[col_name]

        # Create feature
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": properties
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    } 