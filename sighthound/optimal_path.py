import requests


def get_optimal_path(api_key, waypoints):
    """
    Fetch the optimal path using the Mapbox Directions API.
    Args:
        api_key: Mapbox API key.
        waypoints: List of (latitude, longitude) tuples.
    Returns:
        List of (latitude, longitude) tuples representing the optimal path.
    """
    base_url = "https://api.mapbox.com/directions/v5/mapbox/walking"
    coordinates = ";".join(f"{lon},{lat}" for lat, lon in waypoints)
    params = {
        "access_token": api_key,
        "geometries": "geojson"
    }
    response = requests.get(f"{base_url}/{coordinates}", params=params)
    if response.status_code == 200:
        data = response.json()
        return [(coord[1], coord[0]) for coord in data["routes"][0]["geometry"]["coordinates"]]
    else:
        print(f"Error fetching optimal path: {response.text}")
        return []
