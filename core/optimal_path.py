import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

from core.dubins_path import DubinsPathCalculator
from core.dynamic_filtering import GPSKalmanFilter


@dataclass
class OptimizerConfig:
    """Configuration for path optimization"""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    smoothing_factor: float = 0.5
    n_workers: int = 4


class PathOptimizer:
    """
    Optimize GPS trajectories using various constraints and algorithms
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.kalman_filter = GPSKalmanFilter()
        self.dubins_calculator = DubinsPathCalculator()

    def optimize_trajectory(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize a GPS trajectory

        Args:
            trajectory: DataFrame with timestamp, latitude, longitude columns

        Returns:
            Optimized trajectory DataFrame
        """
        # First apply Kalman filtering
        filtered_traj = self.kalman_filter.filter_trajectory(trajectory)

        # Then optimize path segments
        optimized_points = self._optimize_path_segments(filtered_traj)

        # Create final trajectory
        return self._create_optimized_trajectory(
            original_traj=trajectory,
            optimized_points=optimized_points
        )

    def _optimize_path_segments(
            self,
            trajectory: pd.DataFrame
    ) -> List[Tuple[float, float]]:
        """Optimize path segments using parallel processing"""
        segments = self._split_into_segments(trajectory)

        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            optimized_segments = list(executor.map(
                self._optimize_segment,
                segments
            ))

        return self._merge_segments(optimized_segments)

    def _optimize_segment(
            self,
            segment: pd.DataFrame
    ) -> List[Tuple[float, float]]:
        """Optimize a single path segment"""
        points = segment[['latitude', 'longitude']].values

        def objective(x):
            # Reshape optimization variables
            proposed_points = x.reshape(-1, 2)

            # Calculate total path length
            path_length = np.sum(np.sqrt(
                np.sum((proposed_points[1:] - proposed_points[:-1]) ** 2, axis=1)
            ))

            # Calculate smoothness penalty
            smoothness = np.sum(
                np.sqrt(np.sum((proposed_points[2:] - 2 * proposed_points[1:-1] +
                                proposed_points[:-2]) ** 2, axis=1))
            )

            # Calculate deviation from original points
            deviation = np.sum((proposed_points - points) ** 2)

            return (path_length +
                    self.config.smoothing_factor * smoothness +
                    deviation)

        # Initial guess is the original points
        x0 = points.flatten()

        # Optimize
        result = minimize(
            objective,
            x0,
            method='BFGS',
            options={
                'maxiter': self.config.max_iterations,
                'gtol': self.config.tolerance
            }
        )

        optimized_points = result.x.reshape(-1, 2)
        return [(lat, lon) for lat, lon in optimized_points]

    def _split_into_segments(
            self,
            trajectory: pd.DataFrame,
            segment_size: int = 100
    ) -> List[pd.DataFrame]:
        """Split trajectory into manageable segments"""
        return np.array_split(trajectory, max(1, len(trajectory) // segment_size))

    def _merge_segments(
            self,
            segments: List[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Merge optimized segments back together"""
        merged = []
        for i, segment in enumerate(segments):
            if i > 0:
                # Ensure smooth transition between segments
                overlap_start = segment[0]
                overlap_end = merged[-1]
                merged[-1] = (
                    (overlap_start[0] + overlap_end[0]) / 2,
                    (overlap_start[1] + overlap_end[1]) / 2
                )
            merged.extend(segment)
        return merged

    def _create_optimized_trajectory(
            self,
            original_traj: pd.DataFrame,
            optimized_points: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """Create final optimized trajectory DataFrame"""
        optimized_df = pd.DataFrame(
            optimized_points,
            columns=['latitude', 'longitude']
        )

        # Interpolate timestamps
        optimized_df['timestamp'] = pd.Series(
            np.linspace(
                original_traj.timestamp.iloc[0],
                original_traj.timestamp.iloc[-1],
                len(optimized_df)
            )
        )

        # Calculate additional metrics
        optimized_df['speed'] = self._calculate_speeds(optimized_df)
        optimized_df['heading'] = self._calculate_headings(optimized_df)

        return optimized_df

    def _calculate_speeds(self, df: pd.DataFrame) -> pd.Series:
        """Calculate speeds between consecutive points"""
        coords = df[['latitude', 'longitude']].values
        times = df['timestamp'].values

        distances = np.sqrt(
            np.sum((coords[1:] - coords[:-1]) ** 2, axis=1)
        )
        time_diffs = np.diff(times) / np.timedelta64(1, 's')

        speeds = distances / time_diffs
        return pd.Series(
            np.concatenate([[speeds[0]], speeds]),
            index=df.index
        )

    def _calculate_headings(self, df: pd.DataFrame) -> pd.Series:
        """Calculate headings between consecutive points"""
        coords = df[['latitude', 'longitude']].values

        # Calculate bearing angles
        lat1 = np.radians(coords[:-1, 0])
        lat2 = np.radians(coords[1:, 0])
        delta_lon = np.radians(coords[1:, 1] - coords[:-1, 1])

        y = np.sin(delta_lon) * np.cos(lat2)
        x = (np.cos(lat1) * np.sin(lat2) -
             np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))

        headings = np.degrees(np.arctan2(y, x))

        # Ensure headings are between 0 and 360
        headings = (headings + 360) % 360

        return pd.Series(
            np.concatenate([[headings[0]], headings]),
            index=df.index
        )
