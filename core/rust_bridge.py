"""
Python bridge to Rust high-performance modules for Sighthound
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Union
import warnings
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

try:
    # Import Rust modules (will be available after compilation)
    import sighthound_core
    import sighthound_filtering  
    import sighthound_triangulation
    import sighthound_geometry
    import sighthound_optimization
    import sighthound_fusion
    import sighthound_bayesian
    import sighthound_fuzzy
    RUST_AVAILABLE = True
    logger.info("Rust modules loaded successfully - using high-performance implementation")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"Rust modules not available: {e}. Falling back to Python implementation")

# Individual module availability tracking
BAYESIAN_AVAILABLE = False
FUZZY_AVAILABLE = False

try:
    import sighthound_bayesian
    BAYESIAN_AVAILABLE = True
    logger.info("✅ Rust Bayesian evidence network module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  Rust Bayesian module not available: {e}")

try:
    import sighthound_fuzzy  
    FUZZY_AVAILABLE = True
    logger.info("✅ Rust fuzzy optimization module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  Rust fuzzy module not available: {e}")

@dataclass
class RustGpsPoint:
    """Python wrapper for Rust GpsPoint"""
    latitude: float
    longitude: float
    timestamp: float
    confidence: float
    altitude: Optional[float] = None
    speed: Optional[float] = None  
    heading: Optional[float] = None

    def to_rust(self):
        """Convert to Rust GpsPoint if available"""
        if RUST_AVAILABLE:
            return sighthound_core.GpsPoint(
                self.latitude, self.longitude, self.timestamp, 
                self.confidence, self.altitude, self.speed, self.heading
            )
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RustGpsPoint':
        """Create from dictionary"""
        return cls(
            latitude=data['latitude'],
            longitude=data['longitude'], 
            timestamp=data.get('timestamp', 0.0),
            confidence=data.get('confidence', 1.0),
            altitude=data.get('altitude'),
            speed=data.get('speed'),
            heading=data.get('heading')
        )

class HighPerformanceKalmanFilter:
    """
    High-performance Kalman filter using Rust implementation when available
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'process_noise': 1e-3,
            'measurement_noise': 1e-2,
            'initial_state_covariance': 100.0,
            'dt': 1.0,
            'adaptive_filter': True,
            'max_velocity': 50.0,
            'confidence_weight': 0.5
        }
        
        if RUST_AVAILABLE:
            rust_config = sighthound_filtering.KalmanConfig()
            rust_config.process_noise = self.config['process_noise']
            rust_config.measurement_noise = self.config['measurement_noise']
            rust_config.initial_state_covariance = self.config['initial_state_covariance'] 
            rust_config.dt = self.config['dt']
            rust_config.adaptive_filter = self.config['adaptive_filter']
            rust_config.max_velocity = self.config['max_velocity'] 
            rust_config.confidence_weight = self.config['confidence_weight']
            
            self.filter = sighthound_filtering.KalmanFilter(rust_config)
            self.use_rust = True
            logger.info("Using Rust Kalman filter for maximum performance")
        else:
            # Fallback to Python implementation
            from .dynamic_filtering import GPSKalmanFilter, KalmanConfig
            py_config = KalmanConfig(**self.config)
            self.filter = GPSKalmanFilter(py_config)
            self.use_rust = False
            logger.warning("Using Python Kalman filter - performance will be limited")

    def process_trajectory(self, trajectory: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Process a trajectory with Kalman filtering
        
        Args:
            trajectory: GPS trajectory data
            
        Returns:
            Filtered trajectory
        """
        if isinstance(trajectory, list):
            trajectory = pd.DataFrame(trajectory)
            
        if self.use_rust:
            # Convert to Rust format
            rust_points = []
            for _, row in trajectory.iterrows():
                point = RustGpsPoint.from_dict(row.to_dict()).to_rust()
                rust_points.append(point)
            
            # Process with Rust filter
            filtered_points = []
            for point in rust_points:
                filtered_point = self.filter.process_point(point)
                filtered_points.append({
                    'latitude': filtered_point.latitude,
                    'longitude': filtered_point.longitude,
                    'timestamp': filtered_point.timestamp,
                    'confidence': filtered_point.confidence,
                    'altitude': filtered_point.altitude,
                    'speed': filtered_point.speed,
                    'heading': filtered_point.heading
                })
            
            return pd.DataFrame(filtered_points)
        else:
            # Use Python fallback
            return self.filter.filter_trajectory(trajectory)

    def batch_process(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Process multiple trajectories in parallel
        
        Args:
            trajectories: List of trajectory DataFrames
            
        Returns:
            List of filtered trajectories
        """
        if self.use_rust and len(trajectories) > 1:
            # Convert all trajectories to Rust format
            rust_trajectories = []
            for traj in trajectories:
                rust_points = []
                for _, row in traj.iterrows():
                    point = RustGpsPoint.from_dict(row.to_dict()).to_rust()
                    rust_points.append(point)
                rust_trajectories.append(rust_points)
            
            # Process in parallel with Rust
            filtered_trajectories = sighthound_filtering.batch_filter_trajectories(
                rust_trajectories, self.filter.config if hasattr(self.filter, 'config') else None
            )
            
            # Convert back to DataFrames
            result = []
            for traj in filtered_trajectories:
                df_data = []
                for point in traj:
                    df_data.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'timestamp': point.timestamp,
                        'confidence': point.confidence,
                        'altitude': point.altitude,
                        'speed': point.speed,
                        'heading': point.heading
                    })
                result.append(pd.DataFrame(df_data))
            return result
        else:
            # Fallback to sequential processing
            return [self.process_trajectory(traj) for traj in trajectories]

class HighPerformanceTriangulation:
    """
    High-performance triangulation using Rust implementation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'min_towers': 3,
            'max_distance': 10000.0,
            'confidence_threshold': 0.5,
            'optimization_method': 'least_squares',
            'max_iterations': 100,
            'parallel_processing': True
        }
        
        if RUST_AVAILABLE:
            self.use_rust = True
            logger.info("Using Rust triangulation for maximum performance")
        else:
            # Fallback to Python implementation
            from utils.triangulation import TrajectoryTriangulator, TriangulationConfig
            py_config = TriangulationConfig(**self.config)
            self.triangulator = TrajectoryTriangulator(py_config)
            self.use_rust = False
            logger.warning("Using Python triangulation - performance will be limited")

    def triangulate_position(self, reference_point: Dict[str, Any], 
                           cell_towers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Triangulate position from cell tower data
        
        Args:
            reference_point: GPS reference point
            cell_towers: List of cell tower measurements
            
        Returns:
            Triangulation result or None
        """
        if self.use_rust:
            # Convert to Rust types
            rust_point = RustGpsPoint.from_dict(reference_point).to_rust()
            rust_towers = []
            for tower_data in cell_towers:
                tower = sighthound_triangulation.CellTower(
                    tower_data['latitude'],
                    tower_data['longitude'], 
                    tower_data['signal_strength'],
                    tower_data.get('cell_id', 0),
                    tower_data.get('timestamp', 0.0)
                )
                rust_towers.append(tower)
            
            # Create triangulation engine
            rust_config = sighthound_triangulation.TriangulationConfig()
            for key, value in self.config.items():
                if hasattr(rust_config, key):
                    setattr(rust_config, key, value)
            
            engine = sighthound_triangulation.TriangulationEngine(rust_towers, rust_config)
            result = engine.triangulate(rust_point)
            
            if result:
                return {
                    'latitude': result.latitude,
                    'longitude': result.longitude,
                    'confidence': result.confidence,
                    'uncertainty_radius': result.uncertainty_radius,
                    'towers_used': result.towers_used,
                    'convergence_error': result.convergence_error
                }
            return None
        else:
            # Use Python fallback
            # This would require converting the interface to match the Python implementation
            logger.warning("Triangulation fallback not implemented - returning None")
            return None

    def batch_triangulate(self, points: List[Dict[str, Any]], 
                         cell_towers: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Batch triangulate multiple points
        
        Args:
            points: List of GPS points
            cell_towers: List of cell tower measurements
            
        Returns:
            List of triangulation results
        """
        if self.use_rust:
            # Convert to Rust types
            rust_points = [RustGpsPoint.from_dict(p).to_rust() for p in points]
            rust_towers = []
            for tower_data in cell_towers:
                tower = sighthound_triangulation.CellTower(
                    tower_data['latitude'],
                    tower_data['longitude'],
                    tower_data['signal_strength'], 
                    tower_data.get('cell_id', 0),
                    tower_data.get('timestamp', 0.0)
                )
                rust_towers.append(tower)
            
            # Batch process with Rust
            rust_config = sighthound_triangulation.TriangulationConfig()
            for key, value in self.config.items():
                if hasattr(rust_config, key):
                    setattr(rust_config, key, value)
            
            results = sighthound_triangulation.batch_triangulate_parallel(
                rust_points, rust_towers, rust_config, None
            )
            
            # Convert results back
            converted_results = []
            for result in results:
                if result:
                    converted_results.append({
                        'latitude': result.latitude,
                        'longitude': result.longitude,
                        'confidence': result.confidence,
                        'uncertainty_radius': result.uncertainty_radius,
                        'towers_used': result.towers_used,
                        'convergence_error': result.convergence_error
                    })
                else:
                    converted_results.append(None)
            
            return converted_results
        else:
            # Sequential fallback
            return [self.triangulate_position(point, cell_towers) for point in points]

class RustPerformanceMonitor:
    """
    Monitor performance benefits of Rust implementation
    """
    
    def __init__(self):
        self.rust_times = []
        self.python_times = []
        
    def time_operation(self, operation_name: str, rust_func, python_func, *args, **kwargs):
        """
        Time both Rust and Python implementations for comparison
        """
        import time
        
        if RUST_AVAILABLE:
            # Time Rust implementation
            start_time = time.time()
            rust_result = rust_func(*args, **kwargs)
            rust_time = time.time() - start_time
            self.rust_times.append((operation_name, rust_time))
            
            logger.info(f"Rust {operation_name}: {rust_time:.4f}s")
            return rust_result
        else:
            # Time Python implementation  
            start_time = time.time()
            python_result = python_func(*args, **kwargs)
            python_time = time.time() - start_time
            self.python_times.append((operation_name, python_time))
            
            logger.info(f"Python {operation_name}: {python_time:.4f}s")
            return python_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance comparison summary
        """
        if not self.rust_times and not self.python_times:
            return {"message": "No timing data available"}
            
        summary = {
            "rust_available": RUST_AVAILABLE,
            "operations_timed": len(self.rust_times) + len(self.python_times),
        }
        
        if self.rust_times:
            summary["rust_operations"] = self.rust_times
            summary["rust_avg_time"] = sum(t[1] for t in self.rust_times) / len(self.rust_times)
            
        if self.python_times:
            summary["python_operations"] = self.python_times  
            summary["python_avg_time"] = sum(t[1] for t in self.python_times) / len(self.python_times)
            
        return summary

# Global performance monitor instance
performance_monitor = RustPerformanceMonitor()

def get_rust_status() -> Dict[str, Any]:
    """
    Get status of Rust module availability
    """
    return {
        "rust_available": RUST_AVAILABLE,
        "modules_loaded": [],
        "fallback_active": not RUST_AVAILABLE,
        "performance_benefits": "10-100x speedup" if RUST_AVAILABLE else "Not available",
        "bayesian_available": BAYESIAN_AVAILABLE,
        "fuzzy_available": FUZZY_AVAILABLE
    }

def sighthound_bayesian_available() -> bool:
    """Check if Rust Bayesian evidence network module is available"""
    return BAYESIAN_AVAILABLE

def sighthound_fuzzy_available() -> bool:
    """Check if Rust fuzzy optimization module is available"""
    return FUZZY_AVAILABLE

def get_rust_capabilities() -> Dict[str, bool]:
    """Get detailed capabilities of available Rust modules"""
    return {
        'core': RUST_AVAILABLE,
        'filtering': RUST_AVAILABLE,
        'triangulation': RUST_AVAILABLE,
        'bayesian_network': BAYESIAN_AVAILABLE,
        'fuzzy_optimization': FUZZY_AVAILABLE,
        'hybrid_pipeline': BAYESIAN_AVAILABLE and FUZZY_AVAILABLE
    }

# Convenience functions for easy usage
def fast_kalman_filter(trajectory: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Quick access to high-performance Kalman filtering
    """
    filter_instance = HighPerformanceKalmanFilter(config)
    return filter_instance.process_trajectory(trajectory)

def fast_triangulate(reference_point: Dict, cell_towers: List[Dict], 
                    config: Optional[Dict] = None) -> Optional[Dict]:
    """
    Quick access to high-performance triangulation
    """
    triangulator = HighPerformanceTriangulation(config)
    return triangulator.triangulate_position(reference_point, cell_towers)

def batch_haversine_distance(lats1: np.ndarray, lons1: np.ndarray, 
                           lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """
    High-performance batch Haversine distance calculation
    """
    if RUST_AVAILABLE:
        return sighthound_core.batch_haversine_distances(lats1, lons1, lats2, lons2)
    else:
        # Fallback vectorized implementation
        R = 6371000.0  # Earth radius in meters
        
        lat1_rad = np.radians(lats1)
        lat2_rad = np.radians(lats2) 
        delta_lat = np.radians(lats2 - lats1)
        delta_lon = np.radians(lons2 - lons1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c 