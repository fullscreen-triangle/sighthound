from czml3 import Document, Packet
from czml3.properties import Position
from czml3.types import IntervalValue, TimeInterval
from czml3.enums import InterpolationAlgorithms
from czml3.properties import Billboard, Color
from datetime import datetime, timedelta
import json
from typing import List, Dict
from statistics import mean

def load_and_fuse_data(track_path: str, coros_path: str) -> List[Dict]:
    """
    Load and fuse data from both Garmin and Coros files.
    Returns unified data sorted by timestamp.
    """
    # Load both files
    with open(track_path, 'r') as f:
        garmin_data = json.load(f)
    with open(coros_path, 'r') as f:
        coros_data = json.load(f)

    # Convert Garmin data to common format
    garmin_points = []
    for feature in garmin_data['features']:
        point = {
            'timestamp': feature['properties']['timestamp'],
            'latitude': feature['geometry']['coordinates'][1],
            'longitude': feature['geometry']['coordinates'][0],
            'altitude': feature['properties']['altitude'],
            'heart_rate': feature['properties']['heart_rate'],
            'source': 'garmin'
        }
        garmin_points.append(point)

    # Convert Coros data to common format
    coros_points = []
    for point in coros_data:
        point['source'] = 'coros'
        coros_points.append(point)

    # Combine and sort all points by timestamp
    all_points = garmin_points + coros_points
    all_points.sort(key=lambda x: x['timestamp'])
    
    return all_points

def generate_czml(data_points, start_time, end_time):
    """
    Generate CZML document from runner track data.
    
    Args:
        data_points: List of data points with coordinates and metrics
        start_time: Start time as datetime object
        end_time: End time as datetime object
        
    Returns:
        CZML document as string
    """
    import json
    from datetime import datetime
    
    # Format times as ISO 8601 strings
    start_time_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
    end_time_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
    
    # Create CZML document
    czml_doc = [
        # Document packet
        {
            "id": "document",
            "name": "Runner Track",
            "version": "1.0",
            "clock": {
                # Fix: Use proper clock structure with start and end
                "interval": f"{start_time_str}/{end_time_str}",
                "currentTime": start_time_str,
                "multiplier": 10
            }
        },
        # Runner packet
        {
            "id": "runner",
            "name": "Runner",
            "availability": f"{start_time_str}/{end_time_str}",
            "description": "Runner path visualization",
            "billboard": {
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAcCAYAAAATFf3WAAAACXBIWXMAAAsTAAALEwEAmpwYAAAGAGlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNi4wLWMwMDYgNzkuMTY0NzUzLCAyMDIxLzAyLzE1LTExOjUyOjEzICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgMjIuMyAoTWFjaW50b3NoKSIgeG1wOkNyZWF0ZURhdGU9IjIwMjEtMDQtMjdUMTQ6NTM6MTErMDc6MDAiIHhtcDpNb2RpZnlEYXRlPSIyMDIxLTA0LTI3VDE0OjU3OjAyKzA3OjAwIiB4bXA6TWV0YWRhdGFEYXRlPSIyMDIxLTA0LTI3VDE0OjU3OjAyKzA3OjAwIiBkYzpmb3JtYXQ9ImltYWdlL3BuZyIgcGhvdG9zaG9wOkNvbG9yTW9kZT0iMyIgcGhvdG9zaG9wOklDQ1Byb2ZpbGU9InNSR0IgSUVDNjE5NjYtMi4xIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOjUzNGE0YmQzLWZkNjEtNDIwNi04ZGVhLWI0OTcxYzBkMjUyOCIgeG1wTU06RG9jdW1lbnRJRD0iYWRvYmU6ZG9jaWQ6cGhvdG9zaG9wOjBiZGFlZWJhLWM5NWQtMTE0NC1hY2M5LWE5MzcyNzMyZWQxYiIgeG1wTU06T3JpZ2luYWxEb2N1bWVudElEPSJ4bXAuZGlkOjdmNDI5OTVlLTY0MDctNGI1Ni05ZmY3LTIyODgwN2QxNWExYSI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249ImNyZWF0ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6N2Y0Mjk5NWUtNjQwNy00YjU2LTlmZjctMjI4ODA3ZDE1YTFhIiBzdEV2dDp3aGVuPSIyMDIxLTA0LTI3VDE0OjUzOjExKzA3OjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjIuMyAoTWFjaW50b3NoKSIvPiA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0ic2F2ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6NTM0YTRiZDMtZmQ2MS00MjA2LThkZWEtYjQ5NzFjMGQyNTI4IiBzdEV2dDp3aGVuPSIyMDIxLTA0LTI3VDE0OjU3OjAyKzA3OjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjIuMyAoTWFjaW50b3NoKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8L3JkZjpTZXE+IDwveG1wTU06SGlzdG9yeT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz40zRuvAAAFEklEQVRYhe2XW2tdVRSGv7XPyaVJrzdj0lTbGo1NbUyboMaijUWrKCrqiwoi6JM/QHwTQfTNJ0F8EJ/EF/XBGwgaq9b0YpvYJmna5GwvNrEmJOn15MQzV9Y+O9Bc9iapBHywH/bMmWuO9c3vG3PsMUW9Xs+RgBvkFcwrLFw1FhMWLFyBBQtrWbFYZJ0xHRQgRAxzQBCUYGQIUQLjSLpmABTDKhkUBBDM+dkAgZAhcshgMmKCzxLwN3EeQkDq6QFnMHH+dgCrH2CWBQhrxM+tG8jaNNjTRHnPXm5/fBdbntzJ5kd30rhpQ1TqJXVfvMCFo59x7q33ufDRcYpTMzi9CSysBgDW7+Cm/Yc5+O4JjImYMGqBWa9z9eVj/PrCEcKZvxGTrGg8a9JgQ0uJht07GS9PMrz3Q0JY3ELl+Mmv+fOlVyn3tTN1foLKkzso7tzM+a8uk/4xSrZ/D+Wj79G0a0eMbcfR9ykd6GPqQXhuxUTLoZfZ9NYxrPOWzX1rMrDx0SPU3YWnBICIKIbA2Kdf8ctLr+DS1iVF/b88Q9dX/dzR1gTK2udRLs5OMP7xn9RHKvNxzJZt7P3gTRpuaq3n+fVnUM7GBVAVwgAmTp9hZVcGZ2j7+nF2dXUApjpUiW3pnUcYe+bA0pxLCFRGk1Xj5G1tbF/QXiuZd1ufb0XCuDmDZkA8UyNV8PL8rRJIkpTeR/bQUd2y1ePK7c8+wcDJIYIpDUkT8bOY55l3Grc+yLZDTy3YYasG1nPwCbrKXQQfAClqDGTmGEgDg7/8RfGx7TQBIIiP8StbT09z4fQJZvyiG9j6yF6YmcLnypJVq4EJxgTlv+9i79H3sC1tS5ZMoX9ymHsvh0Wvam24wdM/METdZ1Ff/AJVVVAFVcGrcP7M36vGrRuoWnXrVUEMdnd2kE2eXRJQ0jz9M9MEYD4Hg4EYTCwixqoL/pqACq0NJjpXW8bItaR0NLXGGsQsOlTMi8VCoY5MXB/Q/oEhbt29k4l0tlZAVSHPA68HmWqopbpWQE10sQw0dzRx8dE9DA8MRQOrLxARAw8+QH9H++odXMUiSdj0pRQpRhGgrbWJtj2NdHe3ggRCJqCx9qoWqwoygWKxuKZCsraDrNDbgx57nMHR8aVrsDJF06NP0tvdRnxDQzCLNaaKeTDa+NvXPIeLGw/QfPhF+vTnmLlq85ZAqD14kO47b6GYJJHVzM/+r4ogBVo2deKfP8Ktv58iQZaKkoQQAlPP9bH76BGa9+9H1CINgWt3MBdCdLK3JxZGFQnzDZgAJkKi8S6kPbvovf1W2urUYA1PgbWKZkGjYBojQrUWY5wkGo4khEAIQjZXYELlPwOuZQyLbT7xcCUZFokCFVVCFmrZO3d/iB4GsxjGkIBZvS5eg8S1AUFjfUUs4kK1E6nO3vLOLYmLAowBXxVAYrFBY7+O3lQv6d+jTGZLu7uuPihxTdEMC/FXYLKrXNR55OgR5/z/OMRWcC4xQ4NAQwFJEzTNomOaZRlJkpC4JNajc9EvdO41vJ5XGW93Fs9hkJQgTVhOUZNgphQKBTRTQghgOVmWoYkgSYGQSwrL/itKktDbm5Cm6VLAqHEpw2F9HfuohYVGN4uZQ6z6WTMnTAWu3LgBExfHZ+/gw/3UplzP3M3/Xnwt+nN1bkKHvn/BcX0d/ANOpBDIGCkKQgAAAABJRU5ErkJggg==",
                "scale": 1.0,
                "eyeOffset": {
                    "cartesian": [0.0, 0.0, 0.0]
                }
            },
            "position": {
                "interpolationAlgorithm": "LINEAR",
                "interpolationDegree": 1,
                "epoch": start_time_str,
                "cartographicDegrees": _generate_position_data(data_points, start_time)
            },
            "path": {
                "material": {
                    "solidColor": {
                        "color": {
                            "rgba": [255, 0, 0, 255]
                        }
                    }
                },
                "width": 3,
                "leadTime": 0,
                "trailTime": 3600,
                "resolution": 1
            },
            "label": {
                "fillColor": {
                    "rgba": [255, 255, 255, 255]
                },
                "font": "12pt Roboto",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {
                    "cartesian2": [10, 0]
                },
                "style": "FILL",
                "text": "Runner",
                "showBackground": True,
                "backgroundColor": {
                    "rgba": [0, 0, 0, 150]
                }
            }
        }
    ]
    
    # Add point properties if available (like heart rate, cadence, etc.)
    if data_points and isinstance(data_points[0], dict) and any(k in data_points[0] for k in ["heart_rate", "speed", "cadence"]):
        _add_point_properties(czml_doc, data_points, start_time)
    
    # Convert to JSON string
    return json.dumps(czml_doc, indent=2)

def _generate_position_data(data_points, start_time):
    """
    Generate position data array for CZML.
    
    Args:
        data_points: List of data points with coordinates
        start_time: Start time as datetime object
        
    Returns:
        Array of position data in CZML format
    """
    from datetime import datetime
    import time
    
    position_data = []
    
    # Convert start_time to timestamp for calculations
    if isinstance(start_time, datetime):
        start_timestamp = time.mktime(start_time.timetuple())
    else:
        # Try to parse the string
        try:
            start_time_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            start_timestamp = time.mktime(start_time_dt.timetuple())
        except:
            start_timestamp = 0
    
    for i, point in enumerate(data_points):
        # Extract coordinates
        lat, lon, altitude = _extract_coordinates(point)
        
        # Calculate time offset (in seconds)
        # For simplicity, we'll assume equal time intervals if not provided
        if 'timestamp' in point and point['timestamp']:
            try:
                point_time = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00'))
                time_offset = time.mktime(point_time.timetuple()) - start_timestamp
            except:
                time_offset = i * 5  # 5 seconds between points as fallback
        else:
            time_offset = i * 5  # 5 seconds between points as fallback
        
        # Add time and position to the array
        position_data.extend([time_offset, lon, lat, altitude])
    
    return position_data

def _extract_coordinates(point):
    """
    Extract coordinates from a data point.
    
    Args:
        point: Data point which may have coordinates in different formats
        
    Returns:
        Tuple of (latitude, longitude, altitude)
    """
    # Default values
    lat, lon, altitude = 0.0, 0.0, 0.0
    
    if isinstance(point, dict):
        # Try different formats
        if 'latitude' in point and 'longitude' in point:
            try:
                lat = float(point['latitude'])
                lon = float(point['longitude'])
                altitude = float(point.get('altitude', 0))
            except (ValueError, TypeError):
                pass
        elif 'lat' in point and 'lon' in point:
            try:
                lat = float(point['lat'])
                lon = float(point['lon'])
                altitude = float(point.get('altitude', 0))
            except (ValueError, TypeError):
                pass
        elif 'coordinates' in point and isinstance(point['coordinates'], list):
            coords = point['coordinates']
            if len(coords) >= 2:
                try:
                    # GeoJSON format is [longitude, latitude]
                    lon = float(coords[0])
                    lat = float(coords[1])
                    if len(coords) >= 3:
                        altitude = float(coords[2])
                except (ValueError, TypeError):
                    pass
        
        # Try elevation as alternative to altitude
        if altitude == 0 and 'elevation' in point:
            try:
                altitude = float(point['elevation'])
            except (ValueError, TypeError):
                pass
    
    return lat, lon, altitude

def _add_point_properties(czml_doc, data_points, start_time):
    """
    Add point properties like heart rate, cadence, etc. to CZML document.
    
    Args:
        czml_doc: CZML document array
        data_points: List of data points with properties
        start_time: Start time as datetime object
    """
    from datetime import datetime
    import time
    
    # Convert start_time to timestamp for calculations
    if isinstance(start_time, datetime):
        start_timestamp = time.mktime(start_time.timetuple())
    else:
        # Try to parse the string
        try:
            start_time_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            start_timestamp = time.mktime(start_time_dt.timetuple())
        except:
            start_timestamp = 0
    
    # Check what properties are available
    available_properties = set()
    for point in data_points:
        if isinstance(point, dict):
            for key in ["heart_rate", "speed", "cadence", "power", "vertical_oscillation"]:
                if key in point and point[key] is not None:
                    available_properties.add(key)
    
    # Add each available property
    for prop in available_properties:
        property_data = []
        
        # Generate property data
        for i, point in enumerate(data_points):
            if isinstance(point, dict) and prop in point and point[prop] is not None:
                # Calculate time offset
                if 'timestamp' in point and point['timestamp']:
                    try:
                        point_time = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00'))
                        time_offset = time.mktime(point_time.timetuple()) - start_timestamp
                    except:
                        time_offset = i * 5  # Fallback
                else:
                    time_offset = i * 5  # Fallback
                
                try:
                    value = float(point[prop])
                    property_data.extend([time_offset, value])
                except (ValueError, TypeError):
                    pass
        
        # If we have data, add it to the runner packet
        if property_data:
            # Find the runner packet
            for packet in czml_doc:
                if packet.get("id") == "runner":
                    # Add the property
                    packet[prop] = {
                        "epoch": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                        "number": property_data
                    }
                    break

if __name__ == "__main__":
    # Load and fuse data from both sources
    fused_data = load_and_fuse_data('public/track.json', 'public/coros.json')
    
    # Get start and end times from the data
    start_time = datetime.fromisoformat(fused_data[0]['timestamp'].replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(fused_data[-1]['timestamp'].replace('Z', '+00:00'))
    
    # Generate CZML
    czml_content = generate_czml(fused_data, start_time, end_time)
    
    # Save to file
    with open("runner_track.czml", "w") as f:
        f.write(czml_content)
