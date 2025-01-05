import pandas as pd
import gpxpy
from fitparse import FitFile
import xml.etree.ElementTree as ET


def parse_gpx(file_path):
    with open(file_path, 'r') as f:
        gpx = gpxpy.parse(f)
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    "timestamp": point.time,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "altitude": point.elevation
                })
    return pd.DataFrame(data)


def parse_fit(file_path):
    fitfile = FitFile(file_path)
    data = []
    for record in fitfile.get_messages("record"):
        entry = {}
        for field in record:
            if field.name in ["timestamp", "position_lat", "position_long", "altitude"]:
                entry[field.name] = field.value
        data.append(entry)
    return pd.DataFrame(data)


def parse_tcx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    data = []

    for activity in root.findall(".//ns:Activity", namespaces):
        for lap in activity.findall(".//ns:Lap", namespaces):
            for trackpoint in lap.findall(".//ns:Trackpoint", namespaces):
                timestamp_elem = trackpoint.find("ns:Time", namespaces)
                latitude_elem = trackpoint.find(".//ns:LatitudeDegrees", namespaces)
                longitude_elem = trackpoint.find(".//ns:LongitudeDegrees", namespaces)
                altitude_elem = trackpoint.find("ns:AltitudeMeters", namespaces)

                entry = {
                    "timestamp": timestamp_elem.text if timestamp_elem is not None else None,
                    "latitude": float(latitude_elem.text) if latitude_elem is not None else None,
                    "longitude": float(longitude_elem.text) if longitude_elem is not None else None,
                    "altitude": float(altitude_elem.text) if altitude_elem is not None else None,
                }

                if entry["latitude"] is not None and entry["longitude"] is not None:
                    data.append(entry)

    return pd.DataFrame(data)


def parse_kml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {"kml": "http://www.opengis.net/kml/2.2"}
    data = []

    for placemark in root.findall(".//kml:Placemark", namespaces):
        for coord in placemark.findall(".//kml:coordinates", namespaces):
            coords = coord.text.strip().split()
            for c in coords:
                lon, lat, alt = map(float, c.split(","))
                data.append({
                    "timestamp": None,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt
                })

    return pd.DataFrame(data)


def combine_files(file_paths):
    combined_data = []
    for file_path in file_paths:
        if file_path.endswith(".gpx"):
            combined_data.append(parse_gpx(file_path))
        elif file_path.endswith(".fit"):
            combined_data.append(parse_fit(file_path))
        elif file_path.endswith(".tcx"):
            combined_data.append(parse_tcx(file_path))
        elif file_path.endswith(".kml"):
            combined_data.append(parse_kml(file_path))

    combined_df = pd.concat(combined_data).sort_values("timestamp").reset_index(drop=True)

    numeric_columns = combined_df.select_dtypes(include=["float", "int"]).columns
    combined_df[numeric_columns] = combined_df[numeric_columns].interpolate(method="linear")

    return combined_df
