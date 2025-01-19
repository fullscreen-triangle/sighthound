import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class TriangulationConfig:
    """Configuration for triangulation"""
    min_points: int = 3
    max_distance: float = 100  # Using max_distance to match YAML
    confidence_threshold: float = 0.6
    optimization_method: str = 'Nelder-Mead'
    max_iterations: int = 1000


class TrajectoryTriangulator:
    """
    Triangulates position from multiple data sources
    """

    def __init__(self, config: Optional[TriangulationConfig] = None):
        self.config = config or TriangulationConfig()

    def triangulate_positions(
            self,
            trajectories: List[pd.DataFrame],
            weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Triangulate positions from multiple trajectories

        Args:
            trajectories: List of DataFrames with GPS data
            weights: Optional weights for each trajectory

        Returns:
            DataFrame with triangulated positions
        """
        if weights is None:
            weights = [1.0] * len(trajectories)

        # Align timestamps
        aligned_data = self._align_trajectories(trajectories)

        triangulated_positions = []

        for timestamp, group in aligned_data.groupby('timestamp'):
            points = []
            point_weights = []

            for i, traj in enumerate(trajectories):
                mask = (traj['timestamp'] == timestamp)
                if mask.any():
                    points.append([
                        traj.loc[mask, 'latitude'].iloc[0],
                        traj.loc[mask, 'longitude'].iloc[0]
                    ])
                    point_weights.append(weights[i])

            if len(points) >= self.config.min_points:
                position = self._triangulate_single_position(points, point_weights)
                position['timestamp'] = timestamp
                triangulated_positions.append(position)

        return pd.DataFrame(triangulated_positions)

    def _align_trajectories(
            self,
            trajectories: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """Align trajectories to common timestamps"""
        # Combine all timestamps
        all_timestamps = pd.concat([df['timestamp'] for df in trajectories])
        unique_timestamps = all_timestamps.unique()

        aligned_data = pd.DataFrame({'timestamp': unique_timestamps})

        for i, traj in enumerate(trajectories):
            # Interpolate positions for each trajectory
            for col in ['latitude', 'longitude', 'altitude']:
                if col in traj.columns:
                    interpolated = np.interp(
                        aligned_data['timestamp'].astype(np.int64),
                        traj['timestamp'].astype(np.int64),
                        traj[col]
                    )
                    aligned_data[f'{col}_{i}'] = interpolated

        return aligned_data

    def _triangulate_single_position(
            self,
            points: List[List[float]],
            weights: List[float]
    ) -> Dict[str, float]:
        """
        Triangulate single position from multiple points

        Args:
            points: List of [lat, lon] points
            weights: Weight for each point

        Returns:
            Dictionary with triangulated position
        """
        points = np.array(points)
        weights = np.array(weights)

        # Initial guess (weighted average)
        initial_guess = np.average(points, weights=weights, axis=0)

        # Define objective function
        def objective(x):
            return np.sum(
                weights * np.sqrt(
                    np.sum((points - x) ** 2, axis=1)
                )
            )

        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method=self.config.optimization_method,
            options={'maxiter': self.config.max_iterations}
        )

        if result.success:
            confidence = 1.0 / (1.0 + result.fun)
        else:
            confidence = 0.0

        return {
            'latitude': result.x[0],
            'longitude': result.x[1],
            'confidence': confidence
        }

    def calculate_error_metrics(
            self,
            true_positions: pd.DataFrame,
            triangulated_positions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate error metrics for triangulation"""
        merged = pd.merge(
            true_positions,
            triangulated_positions,
            on='timestamp',
            suffixes=('_true', '_triangulated')
        )

        metrics = {
            'mean_distance_error': self._haversine_distance(
                merged['latitude_true'],
                merged['longitude_true'],
                merged['latitude_triangulated'],
                merged['longitude_triangulated']
            ).mean(),
            'max_distance_error': self._haversine_distance(
                merged['latitude_true'],
                merged['longitude_true'],
                merged['latitude_triangulated'],
                merged['longitude_triangulated']
            ).max(),
            'confidence_score': merged['confidence'].mean()
        }

        return metrics

    def _haversine_distance(
            self,
            lat1: np.ndarray,
            lon1: np.ndarray,
            lat2: np.ndarray,
            lon2: np.ndarray
    ) -> np.ndarray:
        """Calculate haversine distance between points"""
        R = 6371000  # Earth radius in meters

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c
