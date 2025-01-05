import requests


def validate_parameters(params):
    """
    Validate parameters against OpenCellID API constraints and return them if valid.

    Args:
        params (dict): Parameters to validate.
    Returns:
        dict: The validated parameters.
    Raises:
        ValueError: If a parameter is invalid.
    """
    if not (-90.0 < params["lat"] < 90.0):
        raise ValueError("Latitude must be in the range (-90, 90) and not zero.")
    if not (-180.0 < params["lon"] < 180.0):
        raise ValueError("Longitude must be in the range (-180, 180) and not zero.")
    if not (100 <= params["mcc"] <= 999):
        raise ValueError("MCC must be in the range 100-999.")
    if not (0 <= params["mnc"] <= 999):
        raise ValueError("MNC must be in the range 0-999.")
    if not (1 <= params["lac"] <= 65535):
        raise ValueError("LAC must be in the range 1-65535.")
    if not (1 <= params["cellid"] <= 268435455):
        raise ValueError("CellID must be in the range 1-268435455.")
    return params



def get_cell_tower_data(api_key, params):
    """
    Fetch cell tower triangulation data from OpenCellID API.

    Args:
        api_key (str): API key for OpenCellID.
        params (dict): Parameters for the API request.
            Required keys: lat, lon, mcc, mnc, lac, cellid.
    Returns:
        dict: API response data.
    Raises:
        Exception: If the API request fails or parameters are invalid.
    """
    # Validate the parameters
    validate_parameters(params)

    # Construct the URL with parameters
    base_url = "https://opencellid.org/measure/add"
    query_params = {
        "key": api_key,
        "lat": params["lat"],
        "lon": params["lon"],
        "mcc": params["mcc"],
        "mnc": params["mnc"],
        "lac": params["lac"],
        "cellid": params["cellid"],
        "rating": params.get("rating", 10.0),  # Default rating
        "direction": params.get("direction", 0.0),  # Default direction
        "speed": params.get("speed", 0.0),  # Default speed
        "act": params.get("act", "LTE"),  # Default radio type
        "ta": params.get("ta", 0)  # Default timing advance
    }

    # Make the API call
    response = requests.get(base_url, params=query_params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API call failed: {response.status_code} {response.text}")


def triangulate_position(cell_data_list):
    """
    Perform triangulation using cell tower data.

    Args:
        cell_data_list (list of dict): List of cell tower data with validated parameters.
    Returns:
        dict: Estimated position (latitude, longitude).
    """
    weighted_lat = 0
    weighted_lon = 0
    total_weight = 0

    for cell_data in cell_data_list:
        weight = 1 / (cell_data["rating"] + 1e-9)  # Use rating as weight
        weighted_lat += cell_data["lat"] * weight
        weighted_lon += cell_data["lon"] * weight
        total_weight += weight

    if total_weight == 0:
        raise ValueError("Triangulation failed: no valid data points.")
    return {"latitude": weighted_lat / total_weight, "longitude": weighted_lon / total_weight}
