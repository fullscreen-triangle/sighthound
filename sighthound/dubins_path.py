import dubins
import numpy as np


def calculate_dubins_path(start, end, turning_radius=1.0):
    """
    Calculate Dubin's path between two points.
    Args:
        start: Tuple (x, y, heading in radians).
        end: Tuple (x, y, heading in radians).
        turning_radius: Minimum turning radius.
    Returns:
        List of (x, y) tuples representing the Dubin's path.
    """
    path = dubins.shortest_path(start, end, turning_radius)
    configurations, _ = path.sample_many(0.1)  # Sample points along the path
    return [(conf[0], conf[1]) for conf in configurations]
