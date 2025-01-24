import numpy as np
from typing import Tuple


class CoordinateConverter:
    """Enhanced coordinate conversion with error handling and additional transformations"""

    def __init__(self):
        self.EARTH_RADIUS = 6371000  # meters
        self.WGS84_A = 6378137.0  # WGS84 semi-major axis
        self.WGS84_B = 6356752.314245  # WGS84 semi-minor axis

    def gps_to_ecef(self, lat: float, lon: float, alt: float = 0) -> Tuple[float, float, float]:
        """Convert GPS coordinates to ECEF (Earth-Centered, Earth-Fixed)"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        N = self.WGS84_A / np.sqrt(1 - 0.00669438 * np.sin(lat_rad)**2)
        
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (0.99330562 * N + alt) * np.sin(lat_rad)
        
        return x, y, z

    def ecef_to_gps(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert ECEF coordinates to GPS"""
        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * self.WGS84_A, p * self.WGS84_B)
        
        lat = np.arctan2(
            z + 0.00669438 * self.WGS84_B * np.sin(theta)**3,
            p - 0.00669438 * self.WGS84_A * np.cos(theta)**3
        )
        lon = np.arctan2(y, x)
        alt = p / np.cos(lat) - self.WGS84_A / np.sqrt(1 - 0.00669438 * np.sin(lat)**2)
        
        return np.degrees(lat), np.degrees(lon), alt

    def gps_to_local(
            self,
            lat: float,
            lon: float,
            ref_lat: float,
            ref_lon: float
    ) -> Tuple[float, float]:
        """
        Convert GPS coordinates to local Cartesian coordinates

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            ref_lat: Reference latitude in degrees
            ref_lon: Reference longitude in degrees

        Returns:
            Tuple of (x, y) in meters
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        ref_lat_rad = np.radians(ref_lat)
        ref_lon_rad = np.radians(ref_lon)

        x = self.EARTH_RADIUS * np.cos(ref_lat_rad) * (lon_rad - ref_lon_rad)
        y = self.EARTH_RADIUS * (lat_rad - ref_lat_rad)

        return x, y

    def local_to_gps(
            self,
            x: float,
            y: float,
            ref_lat: float,
            ref_lon: float
    ) -> Tuple[float, float]:
        """
        Convert local Cartesian coordinates to GPS coordinates

        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            ref_lat: Reference latitude in degrees
            ref_lon: Reference longitude in degrees

        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        ref_lat_rad = np.radians(ref_lat)

        lat = y / self.EARTH_RADIUS + ref_lat_rad
        lon = x / (self.EARTH_RADIUS * np.cos(ref_lat_rad)) + np.radians(ref_lon)

        return np.degrees(lat), np.degrees(lon)

    def calculate_distance(
            self,
            lat1: float,
            lon1: float,
            lat2: float,
            lon2: float
    ) -> float:
        """
        Calculate great-circle distance between two points

        Args:
            lat1, lon1: Coordinates of first point in degrees
            lat2, lon2: Coordinates of second point in degrees

        Returns:
            Distance in meters
        """
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (np.sin(dlat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return self.EARTH_RADIUS * c

    def calculate_bearing(
            self,
            lat1: float,
            lon1: float,
            lat2: float,
            lon2: float
    ) -> float:
        """
        Calculate initial bearing between two points

        Args:
            lat1, lon1: Coordinates of first point in degrees
            lat2, lon2: Coordinates of second point in degrees

        Returns:
            Bearing in degrees
        """
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlon = lon2_rad - lon1_rad

        y = np.sin(dlon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) -
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))

        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360
