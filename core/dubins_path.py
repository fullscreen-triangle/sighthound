from typing import List, Tuple, Optional
from dataclasses import dataclass
import dubins

from utils.coordinates import CoordinateConverter


@dataclass
class DubinsConfig:
    """Configuration for Dubins path calculation"""
    turning_radius: float = 30.0  # meters
    step_size: float = 1.0  # meters
    max_path_length: float = 1000.0  # meters


class DubinsPathCalculator:
    """
    Calculate Dubins paths for GPS trajectories
    """

    def __init__(self, config: Optional[DubinsConfig] = None):
        self.config = config or DubinsConfig()
        self.converter = CoordinateConverter()

    def calculate_path(
            self,
            start_point: Tuple[float, float, float],
            end_point: Tuple[float, float, float]
    ) -> List[Tuple[float, float]]:
        """
        Calculate Dubins path between two points

        Args:
            start_point: (latitude, longitude, heading) of start point
            end_point: (latitude, longitude, heading) of end point

        Returns:
            List of (latitude, longitude) points along the path
        """
        # Convert to local coordinates
        start_local = self.converter.gps_to_local(
            start_point[0],
            start_point[1],
            start_point[0],  # reference point
            start_point[1]
        )
        end_local = self.converter.gps_to_local(
            end_point[0],
            end_point[1],
            start_point[0],  # same reference point
            start_point[1]
        )

        # Calculate Dubins path
        start_config = (start_local[0], start_local[1], start_point[2])
        end_config = (end_local[0], end_local[1], end_point[2])

        path = dubins.shortest_path(
            start_config,
            end_config,
            self.config.turning_radius
        )

        # Sample points along the path
        configurations, _ = path.sample_many(self.config.step_size)

        # Convert back to GPS coordinates
        gps_points = [
            self.converter.local_to_gps(
                conf[0],
                conf[1],
                start_point[0],
                start_point[1]
            ) for conf in configurations
        ]

        return gps_points

    def calculate_path_length(
            self,
            start_point: Tuple[float, float, float],
            end_point: Tuple[float, float, float]
    ) -> float:
        """
        Calculate the length of the Dubins path

        Args:
            start_point: (latitude, longitude, heading) of start point
            end_point: (latitude, longitude, heading) of end point

        Returns:
            Path length in meters
        """
        # Convert to local coordinates
        start_local = self.converter.gps_to_local(
            start_point[0],
            start_point[1],
            start_point[0],
            start_point[1]
        )
        end_local = self.converter.gps_to_local(
            end_point[0],
            end_point[1],
            start_point[0],
            start_point[1]
        )

        # Calculate path
        start_config = (start_local[0], start_local[1], start_point[2])
        end_config = (end_local[0], end_local[1], end_point[2])

        path = dubins.shortest_path(
            start_config,
            end_config,
            self.config.turning_radius
        )

        return path.path_length()

    def is_path_feasible(
            self,
            start_point: Tuple[float, float, float],
            end_point: Tuple[float, float, float]
    ) -> bool:
        """
        Check if a Dubins path is feasible

        Args:
            start_point: (latitude, longitude, heading) of start point
            end_point: (latitude, longitude, heading) of end point

        Returns:
            True if path is feasible, False otherwise
        """
        try:
            path_length = self.calculate_path_length(start_point, end_point)
            return path_length <= self.config.max_path_length
        except Exception:
            return False
