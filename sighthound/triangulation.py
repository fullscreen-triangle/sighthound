import requests
from geopy.distance import geodesic


def triangulate_position(cell_tower_data):
    weighted_lat = 0
    weighted_lon = 0
    total_weight = 0

    for tower in cell_tower_data:
        weight = 1 / (tower["signal_strength"] + 1e-9)
        weighted_lat += tower["lat"] * weight
        weighted_lon += tower["lon"] * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return {"latitude": weighted_lat / total_weight, "longitude": weighted_lon / total_weight}


def get_cell_tower_data(api_key, lat, lon, radius=1000):
    url = f"https://opencellid.org/api/cell/get"
    params = {
        "key": api_key,
        "lat": lat,
        "lon": lon,
        "radius": radius,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching cell tower data: {response.text}")
        return []
