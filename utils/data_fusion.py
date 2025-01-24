from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import logging
from datetime import datetime, timezone
import json
from pathlib import Path
from .coordinates import CoordinateConverter
from .cell_triangulation import CellDataTriangulator
from .weather import WeatherDataIntegrator
from .probability import ProbabilityDensityCalculator
from .triangulation import TrajectoryTriangulator

@dataclass
class DataFusionConfig:
    """Configuration for data fusion"""
    min_confidence: float = 0.6
    max_time_gap: float = 30.0  # seconds
    interpolation_method: str = 'cubic'
    outlier_threshold: float = 3.0  # standard deviations
    weight_factors: Dict[str, float] = field(default_factory=lambda: {
        'gps': 0.4,
        'cell': 0.2,
        'probability': 0.2,
        'weather': 0.1,
        'satellite': 0.1
    })
    fusion_window: float = 60.0  # seconds
    min_samples: int = 2  # Reduced from 3 to allow pairwise fusion
    max_speed: float = 100.0  # km/h
    smooth_window: int = 5
    output_format: str = 'pandas'  # 'pandas' or 'json'
    save_intermediate: bool = True
    intermediate_path: str = "intermediate_fusions"

class ActivityDataFuser:
    """Fuses multiple data sources for accurate activity tracking with progressive fusion"""

    def __init__(self, config: Optional[DataFusionConfig] = None):
        self.config = config or DataFusionConfig()
        self.coord_converter = CoordinateConverter()
        self.cell_triangulator = CellDataTriangulator()
        self.weather_integrator = WeatherDataIntegrator()
        self.pdf_calculator = ProbabilityDensityCalculator()
        self.trajectory_triangulator = TrajectoryTriangulator()
        self.logger = logging.getLogger(__name__)
        
        if self.config.save_intermediate:
            Path(self.config.intermediate_path).mkdir(parents=True, exist_ok=True)

    def _normalize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamp format across different sources"""
        if 'timestamp' in df.columns:
            try:
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                elif not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                
                # Ensure timestamps are in UTC
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                elif df['timestamp'].dt.tz != timezone.utc:
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
                    
            except Exception as e:
                self.logger.error(f"Error normalizing timestamps: {str(e)}")
                return None
        return df

    def fuse_activity_data(
        self,
        trajectories: List[pd.DataFrame],
        cell_data: Optional[pd.DataFrame] = None,
        weather_data: Optional[pd.DataFrame] = None,
        satellite_data: Optional[pd.DataFrame] = None
    ) -> Union[pd.DataFrame, Dict]:
        """Progressive data fusion with fallbacks and intermediate results"""
        try:
            # Initialize fusion result
            fused_result = None
            fusion_metadata = {
                'sources_used': [],
                'confidence_scores': {},
                'fusion_steps': []
            }

            # Normalize and validate trajectories
            valid_trajectories = []
            for i, traj in enumerate(trajectories):
                normalized_traj = self._normalize_timestamp(traj)
                if normalized_traj is not None and not normalized_traj.empty:
                    valid_trajectories.append(normalized_traj)
                    fusion_metadata['sources_used'].append(f'trajectory_{i}')

            if not valid_trajectories:
                raise ValueError("No valid trajectory data available")

            # Start with GPS trajectory fusion if multiple trajectories exist
            if len(valid_trajectories) > 1:
                aligned_trajectories = self._align_trajectories(valid_trajectories)
                confidence_scores = self._calculate_confidence_scores(aligned_trajectories)
                fused_result = self._fuse_gps_trajectories(aligned_trajectories, confidence_scores)
                fusion_metadata['confidence_scores']['gps'] = np.mean(confidence_scores)
                
                if self.config.save_intermediate:
                    self._save_intermediate(fused_result, 'gps_fusion')
            else:
                fused_result = valid_trajectories[0].copy()
                fusion_metadata['confidence_scores']['gps'] = self._calculate_confidence_scores([fused_result])[0]

            # Progressive integration of additional data sources
            if cell_data is not None:
                try:
                    cell_data = self._normalize_timestamp(cell_data)
                    if cell_data is not None:
                        fused_result = self._integrate_cell_data(fused_result, cell_data)
                        fusion_metadata['sources_used'].append('cell_data')
                        if self.config.save_intermediate:
                            self._save_intermediate(fused_result, 'cell_fusion')
                except Exception as e:
                    self.logger.warning(f"Cell data integration failed: {str(e)}")

            if weather_data is not None:
                try:
                    weather_data = self._normalize_timestamp(weather_data)
                    if weather_data is not None:
                        fused_result = self._integrate_weather_data(fused_result, weather_data)
                        fusion_metadata['sources_used'].append('weather_data')
                        if self.config.save_intermediate:
                            self._save_intermediate(fused_result, 'weather_fusion')
                except Exception as e:
                    self.logger.warning(f"Weather data integration failed: {str(e)}")

            if satellite_data is not None:
                try:
                    satellite_data = self._normalize_timestamp(satellite_data)
                    if satellite_data is not None:
                        fused_result = self._integrate_satellite_data(fused_result, satellite_data)
                        fusion_metadata['sources_used'].append('satellite_data')
                        if self.config.save_intermediate:
                            self._save_intermediate(fused_result, 'satellite_fusion')
                except Exception as e:
                    self.logger.warning(f"Satellite data integration failed: {str(e)}")

            # Final smoothing and validation
            try:
                fused_result = self._apply_probability_smoothing(fused_result)
                if self.config.save_intermediate:
                    self._save_intermediate(fused_result, 'final_smoothed')
            except Exception as e:
                self.logger.warning(f"Final smoothing failed: {str(e)}")

            # Prepare output
            if self.config.output_format == 'json':
                return self._prepare_json_output(fused_result, fusion_metadata)
            return fused_result

        except Exception as e:
            self.logger.error(f"Error during data fusion: {str(e)}")
            raise

    def _save_intermediate(self, df: pd.DataFrame, step_name: str):
        """Save intermediate fusion results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{timestamp}.parquet"
        path = Path(self.config.intermediate_path) / filename
        df.to_parquet(path)

    def _prepare_json_output(self, df: pd.DataFrame, metadata: Dict) -> Dict:
        """Prepare JSON-serializable output"""
        try:
            # Convert DataFrame to JSON-serializable format
            trajectory_data = df.copy()
            trajectory_data['timestamp'] = trajectory_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            return {
                'metadata': metadata,
                'trajectory': trajectory_data.to_dict(orient='records')
            }
        except Exception as e:
            self.logger.error(f"Error preparing JSON output: {str(e)}")
            return {'error': str(e)}

    def _align_trajectories(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Align trajectories to a common time base"""
        aligned = []
        min_time = max([df['timestamp'].min() for df in trajectories])
        max_time = min([df['timestamp'].max() for df in trajectories])
        
        time_range = pd.date_range(
            start=min_time,
            end=max_time,
            freq=f"{self.config.fusion_window}S"
        )
        
        for traj in trajectories:
            resampled = traj.set_index('timestamp').reindex(
                time_range
            ).interpolate(method=self.config.interpolation_method)
            aligned.append(resampled.reset_index())
            
        return aligned

    def _calculate_confidence_scores(self, trajectories: List[pd.DataFrame]) -> List[float]:
        """Calculate confidence scores for each trajectory"""
        scores = []
        for traj in trajectories:
            # Check data completeness
            completeness = traj[['latitude', 'longitude']].notna().mean().mean()
            
            # Check coordinate validity
            valid_coords = (
                traj['latitude'].between(-90, 90) &
                traj['longitude'].between(-180, 180)
            ).mean()
            
            # Check temporal consistency
            time_diffs = traj['timestamp'].diff().dt.total_seconds()
            temporal_score = 1 - (
                time_diffs.std() / time_diffs.mean() if not time_diffs.empty else 0
            )
            
            # Calculate final score
            score = np.mean([completeness, valid_coords, temporal_score])
            scores.append(max(score, self.config.min_confidence))
            
        return scores

    def _fuse_gps_trajectories(
        self,
        trajectories: List[pd.DataFrame],
        confidence_scores: List[float]
    ) -> pd.DataFrame:
        """Fuse multiple GPS trajectories using weighted averaging"""
        # Normalize confidence scores
        weights = np.array(confidence_scores) / sum(confidence_scores)
        
        # Initialize fused trajectory
        fused = pd.DataFrame()
        
        # Get common timestamps
        timestamps = sorted(set.intersection(*[
            set(traj['timestamp']) for traj in trajectories
        ]))
        
        for timestamp in timestamps:
            points = []
            for traj in trajectories:
                if timestamp in traj['timestamp'].values:
                    points.append(traj[traj['timestamp'] == timestamp])
            
            if len(points) >= self.config.min_samples:
                weighted_point = self._calculate_weighted_point(points, weights)
                fused = pd.concat([fused, weighted_point])
        
        return fused.reset_index(drop=True)

    def _calculate_weighted_point(
        self,
        points: List[pd.DataFrame],
        weights: np.ndarray
    ) -> pd.DataFrame:
        """Calculate weighted average of points"""
        weighted_point = pd.DataFrame()
        
        for col in points[0].columns:
            if col in ['latitude', 'longitude', 'elevation', 'speed']:
                values = np.array([p[col].iloc[0] for p in points])
                weighted_point[col] = [np.average(values, weights=weights)]
        
        weighted_point['timestamp'] = points[0]['timestamp'].iloc[0]
        weighted_point['confidence'] = np.max(weights)
        
        return weighted_point

    def _integrate_cell_data(
        self,
        trajectory: pd.DataFrame,
        cell_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Integrate cell tower positioning data"""
        cell_positions = self.cell_triangulator.triangulate_positions(cell_data)
        
        for idx, row in trajectory.iterrows():
            if cell_positions:
                cell_pos = self._find_nearest_cell_position(
                    row['timestamp'],
                    cell_positions
                )
                if cell_pos:
                    trajectory.loc[idx, ['latitude', 'longitude']] = self._weighted_position(
                        trajectory.loc[idx, ['latitude', 'longitude']].values,
                        cell_pos,
                        self.config.weight_factors['cell']
                    )
        
        return trajectory

    def _integrate_weather_data(
        self,
        trajectory: pd.DataFrame,
        weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Integrate weather data into trajectory"""
        return self.weather_integrator.integrate_weather_data(
            trajectory,
            weather_data
        )

    def _integrate_satellite_data(
        self,
        trajectory: pd.DataFrame,
        satellite_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Integrate satellite positioning data"""
        for idx, row in trajectory.iterrows():
            sat_pos = self._find_nearest_satellite_position(
                row['timestamp'],
                satellite_data
            )
            if sat_pos is not None:
                trajectory.loc[idx, ['latitude', 'longitude']] = self._weighted_position(
                    trajectory.loc[idx, ['latitude', 'longitude']].values,
                    sat_pos,
                    self.config.weight_factors['satellite']
                )
        
        return trajectory

    def _apply_probability_smoothing(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """Apply final probability-based smoothing"""
        smoothed = self.pdf_calculator.calculate_trajectory_pdf(trajectory)
        trajectory['confidence'] = smoothed['probability']
        
        # Apply rolling average smoothing
        for col in ['latitude', 'longitude', 'elevation', 'speed']:
            if col in trajectory.columns:
                trajectory[col] = trajectory[col].rolling(
                    window=self.config.smooth_window,
                    center=True,
                    min_periods=1
                ).mean()
        
        return trajectory

    def _weighted_position(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
        weight: float
    ) -> np.ndarray:
        """Calculate weighted average of two positions"""
        return pos1 * (1 - weight) + pos2 * weight

    def _find_nearest_cell_position(
        self,
        timestamp: pd.Timestamp,
        cell_positions: Dict
    ) -> Optional[np.ndarray]:
        """Find nearest cell position to given timestamp"""
        try:
            # Convert cell_positions timestamps to pandas Timestamps for comparison
            time_diffs = {
                abs((pd.to_datetime(t, utc=True) - timestamp).total_seconds()): pos
                for t, pos in cell_positions.items()
            }
            
            # Find closest timestamp within max_time_gap
            min_diff = min(time_diffs.keys())
            if min_diff <= self.config.max_time_gap:
                pos = time_diffs[min_diff]
                return np.array([pos['latitude'], pos['longitude']])
                
            return None
        except Exception as e:
            self.logger.warning(f"Error finding nearest cell position: {str(e)}")
            return None

    def _find_nearest_satellite_position(
        self,
        timestamp: pd.Timestamp,
        satellite_data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Find nearest satellite position to given timestamp"""
        try:
            if satellite_data.empty:
                return None
                
            # Ensure timestamp comparison is valid
            satellite_data['timestamp'] = pd.to_datetime(satellite_data['timestamp'], utc=True)
            
            # Find closest timestamp
            satellite_data['time_diff'] = abs(
                (satellite_data['timestamp'] - timestamp).dt.total_seconds()
            )
            
            closest = satellite_data.loc[satellite_data['time_diff'].idxmin()]
            
            # Check if within max_time_gap
            if closest['time_diff'] <= self.config.max_time_gap:
                return np.array([closest['latitude'], closest['longitude']])
                
            return None
        except Exception as e:
            self.logger.warning(f"Error finding nearest satellite position: {str(e)}")
            return None 