import numpy as np
from typing import Tuple, Optional
import pandas as pd
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass


@dataclass
class KalmanConfig:
    """Configuration for Kalman Filter"""
    process_noise: float = 1e-3
    measurement_noise: float = 1e-2
    initial_state_covariance: float = 100.0
    dt: float = 1.0  # Time step
    confidence_weight: float = 0.5  # Added for confidence weighting
    min_confidence: float = 0.6  # Added to match data fusion


class GPSKalmanFilter:
    """
    Kalman Filter implementation for GPS tracking
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()
        self.kf = self._initialize_filter()

    def _initialize_filter(self) -> KalmanFilter:
        """Initialize the Kalman Filter with appropriate matrices"""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy], Measurement: [x, y]

        # State transition matrix
        kf.F = np.array([
            [1, self.config.dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.config.dt],
            [0, 0, 0, 1]
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        # Measurement noise matrix
        kf.R = np.eye(2) * self.config.measurement_noise

        # Process noise matrix
        q = self.config.process_noise
        kf.Q = np.array([
            [q * self.config.dt ** 4 / 4, q * self.config.dt ** 3 / 2, 0, 0],
            [q * self.config.dt ** 3 / 2, q * self.config.dt ** 2, 0, 0],
            [0, 0, q * self.config.dt ** 4 / 4, q * self.config.dt ** 3 / 2],
            [0, 0, q * self.config.dt ** 3 / 2, q * self.config.dt ** 2]
        ])

        # Initial state covariance
        kf.P *= self.config.initial_state_covariance

        return kf

    def reset(self):
        """Reset the filter state"""
        self.kf = self._initialize_filter()

    def update(self, measurement: np.ndarray):
        """
        Update the filter with a new measurement

        Args:
            measurement: Array of [x, y] position
        """
        self.kf.predict()
        self.kf.update(measurement)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state and covariance

        Returns:
            Tuple of (state, covariance)
        """
        return self.kf.x, self.kf.P

    def filter_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a complete GPS trajectory with confidence handling
        
        Args:
            data: DataFrame with latitude, longitude and optional confidence columns
        """
        self.reset()
        filtered_positions = []
        
        for _, row in data.iterrows():
            # Adjust measurement noise based on confidence if available
            if 'confidence' in row:
                confidence_factor = max(row.confidence, self.config.min_confidence)
                self.kf.R = np.eye(2) * (self.config.measurement_noise / confidence_factor)
            
            measurement = np.array([row.longitude, row.latitude])
            self.update(measurement)
            state = self.get_state()[0]
            
            position = {
                'timestamp': row.timestamp,
                'longitude': state[0],
                'latitude': state[2],
                'velocity_x': state[1],
                'velocity_y': state[3]
            }
            
            # Preserve confidence if it exists
            if 'confidence' in row:
                position['confidence'] = row.confidence
                
            filtered_positions.append(position)
            
        return pd.DataFrame(filtered_positions)

    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate

        Returns:
            Tuple of (velocity_x, velocity_y)
        """
        state = self.kf.x
        return state[1], state[3]

    def get_position_uncertainty(self) -> np.ndarray:
        """
        Get position uncertainty

        Returns:
            2x2 covariance matrix for position
        """
        return np.array([
            [self.kf.P[0, 0], self.kf.P[0, 2]],
            [self.kf.P[2, 0], self.kf.P[2, 2]]
        ])


class BatchKalmanProcessor:
    """
    Process multiple trajectories in parallel using Kalman filtering
    """

    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers

    def process_batch(self, trajectories: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Process multiple trajectories in parallel

        Args:
            trajectories: List of DataFrames containing GPS trajectories

        Returns:
            List of filtered trajectories
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        filtered_trajectories = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_traj = {
                executor.submit(self._process_single_trajectory, traj): i
                for i, traj in enumerate(trajectories)
            }

            for future in as_completed(future_to_traj):
                try:
                    filtered_traj = future.result()
                    filtered_trajectories.append(filtered_traj)
                except Exception as e:
                    print(f"Error processing trajectory: {str(e)}")

        return filtered_trajectories

    def _process_single_trajectory(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """Process a single trajectory"""
        kf = GPSKalmanFilter()
        return kf.filter_trajectory(trajectory)
