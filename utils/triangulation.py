import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
import logging
from tqdm import tqdm
from utils.profiler import timer, profile
import rtree
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from utils.parallel import ParallelProcessor, ParallelConfig

logger = logging.getLogger(__name__)

@dataclass
class TriangulationConfig:
    """Configuration parameters for triangulation"""
    min_points: int = 2
    max_distance: float = 100.0  # meters
    confidence_threshold: float = 0.5
    optimization_method: str = "BFGS"
    max_iterations: int = 100
    weight_decay: float = 0.9
    min_confidence: float = 0.2
    chunk_size: int = 1000  # Process trajectories in chunks
    use_parallel: bool = True
    n_workers: int = max(1, multiprocessing.cpu_count() - 1)
    spatial_index: bool = True
    cache_size: int = 10000  # Size of the LRU cache
    progress_reporting: bool = True
    memory_optimization: bool = True
    batch_size: int = 100  # Batch size for parallel processing

# Create a cache for distance calculations
distance_cache = LRUCache(maxsize=10000)

@cached(distance_cache, key=lambda lat1, lon1, lat2, lon2: hashkey(lat1, lon1, lat2, lon2))
def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points in kilometers.
    Cached for performance.
    """
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

# Vectorized version for batch processing
def _batch_haversine_distance(lats1: np.ndarray, lons1: np.ndarray, 
                             lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """
    Calculate the Haversine distance between arrays of points.
    
    Args:
        lats1, lons1: Arrays of latitudes and longitudes for first set of points
        lats2, lons2: Arrays of latitudes and longitudes for second set of points
        
    Returns:
        Array of distances in meters
    """
    R = 6371000  # Earth radius in meters
    
    lats1, lons1, lats2, lons2 = map(np.radians, [lats1, lons1, lats2, lons2])
    
    dlat = lats2 - lats1
    dlon = lons2 - lons1
    
    a = np.sin(dlat/2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

class SpatialIndex:
    """
    Spatial indexing for quick nearest neighbor lookups
    """
    def __init__(self):
        self.idx = rtree.index.Index()
        self.positions = []
        self.count = 0
        
    def insert(self, lat: float, lon: float, data: Any) -> int:
        """
        Insert a position into the spatial index
        
        Args:
            lat: Latitude
            lon: Longitude
            data: Data to associate with the position
            
        Returns:
            Index of the inserted position
        """
        # R-tree uses (minx, miny, maxx, maxy) format
        self.idx.insert(self.count, (lon, lat, lon, lat), obj=data)
        self.positions.append((lat, lon, data))
        idx = self.count
        self.count += 1
        return idx
    
    def nearest(self, lat: float, lon: float, n: int = 5) -> List[Tuple[float, float, Any]]:
        """
        Find the n nearest positions to a point
        
        Args:
            lat: Latitude
            lon: Longitude
            n: Number of nearest neighbors to find
            
        Returns:
            List of (lat, lon, data) tuples
        """
        # Query the R-tree for nearest neighbors
        nearest_idxs = list(self.idx.nearest((lon, lat, lon, lat), n))
        return [self.positions[i] for i in nearest_idxs]
    
    def within_distance(self, lat: float, lon: float, distance_km: float) -> List[Tuple[float, float, Any]]:
        """
        Find all positions within a certain distance of a point
        
        Args:
            lat: Latitude
            lon: Longitude
            distance_km: Distance in kilometers
            
        Returns:
            List of (lat, lon, data) tuples within the distance
        """
        # Convert distance to approximate bounding box
        # Earth circumference ~= 40000 km
        # 1 degree ~= 111 km at the equator
        degrees = distance_km / 111.0
        
        # Create bounding box
        min_lon, max_lon = lon - degrees, lon + degrees
        min_lat, max_lat = lat - degrees, lat + degrees
        
        # Query the R-tree for positions within the bounding box
        within_idxs = list(self.idx.intersection((min_lon, min_lat, max_lon, max_lat)))
        
        # Filter by actual distance
        result = []
        for i in within_idxs:
            p_lat, p_lon, data = self.positions[i]
            if _haversine_distance(lat, lon, p_lat, p_lon) <= distance_km:
                result.append((p_lat, p_lon, data))
        
        return result
    
    def batch_insert(self, positions: List[Tuple[float, float, Any]]) -> None:
        """
        Insert multiple positions in batch
        
        Args:
            positions: List of (lat, lon, data) tuples
        """
        for lat, lon, data in positions:
            self.insert(lat, lon, data)

class TrajectoryTriangulator:
    """
    Enhanced triangulation with confidence scoring, parallel processing and spatial indexing
    """
    def __init__(self, config: Optional[TriangulationConfig] = None):
        """
        Initialize the triangulator with configuration
        
        Args:
            config: Configuration parameters for triangulation
        """
        self.config = config or TriangulationConfig()
        self.spatial_index = SpatialIndex() if self.config.spatial_index else None
        
        # Initialize parallel processor with configuration
        if self.config.use_parallel:
            parallel_config = ParallelConfig(
                n_workers=self.config.n_workers,
                chunk_size=self.config.chunk_size,
                batch_size=self.config.batch_size,
                show_progress=self.config.progress_reporting,
                use_processes=True,  # Use processes for CPU-bound operations
                progress_desc="Triangulating positions"
            )
            self.parallel_processor = ParallelProcessor(parallel_config)
        
    @timer
    def triangulate_positions(self, 
                             trajectories: List[pd.DataFrame], 
                             weights: Optional[List[float]] = None,
                             progress_callback: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
        """
        Triangulate positions from multiple data sources
        
        Args:
            trajectories: List of DataFrames, each containing a trajectory
            weights: Weights for each trajectory (optional)
            progress_callback: Callback function for reporting progress
            
        Returns:
            DataFrame containing triangulated positions
        """
        if not trajectories:
            logger.warning("No trajectories provided for triangulation")
            return pd.DataFrame()
            
        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(trajectories)
        
        # Align trajectories by timestamp
        aligned_data = self._align_trajectories(trajectories)
        
        # Get all unique timestamps
        all_timestamps = sorted(aligned_data.keys())
        
        results = []
        
        # Use optimized parallel processing for large datasets
        if self.config.use_parallel:
            logger.info(f"Processing {len(all_timestamps)} timestamps using parallel processing")
            
            # Process in chunks
            if self.config.memory_optimization:
                chunks = [all_timestamps[i:i + self.config.chunk_size] 
                         for i in range(0, len(all_timestamps), self.config.chunk_size)]
                
                # Define chunk processing function
                def process_chunk(chunk):
                    chunk_results = []
                    for ts in chunk:
                        positions = aligned_data[ts]
                        if len(positions) >= self.config.min_points:
                            src_indices = list(positions.keys())
                            lat, lon, confidence, weight_sum = self._triangulate_single_position(
                                positions, weights, src_indices
                            )
                            
                            chunk_results.append({
                                'timestamp': ts,
                                'latitude': lat,
                                'longitude': lon,
                                'confidence': confidence,
                                'weight_sum': weight_sum,
                                'src_count': len(positions),
                                'src_indices': src_indices
                            })
                    return chunk_results
                
                # Process chunks in parallel
                results = self.parallel_processor.process_chunks(
                    process_chunk, 
                    chunks,
                )
            else:
                # Define timestamp processing function
                def process_timestamp(ts):
                    positions = aligned_data[ts]
                    if len(positions) >= self.config.min_points:
                        src_indices = list(positions.keys())
                        lat, lon, confidence, weight_sum = self._triangulate_single_position(
                            positions, weights, src_indices
                        )
                        
                        return {
                            'timestamp': ts,
                            'latitude': lat,
                            'longitude': lon,
                            'confidence': confidence,
                            'weight_sum': weight_sum,
                            'src_count': len(positions),
                            'src_indices': src_indices
                        }
                    return None
                
                # Process all timestamps in parallel using batches
                batch_results = self.parallel_processor.batch_map(
                    lambda batch: [process_timestamp(ts) for ts in batch],
                    all_timestamps
                )
                
                # Filter out None results
                results = [r for r in batch_results if r is not None]
        else:
            # Sequential processing
            for ts in tqdm(all_timestamps, desc="Triangulating positions", disable=not self.config.progress_reporting):
                positions = aligned_data[ts]
                if len(positions) >= self.config.min_points:
                    src_indices = list(positions.keys())
                    lat, lon, confidence, weight_sum = self._triangulate_single_position(
                        positions, weights, src_indices
                    )
                    
                    results.append({
                        'timestamp': ts,
                        'latitude': lat,
                        'longitude': lon,
                        'confidence': confidence,
                        'weight_sum': weight_sum,
                        'src_count': len(positions),
                        'src_indices': src_indices
                    })
                    
                # Report progress if callback provided
                if progress_callback:
                    progress = (all_timestamps.index(ts) + 1) / len(all_timestamps)
                    progress_callback(progress)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        # Add additional statistics
        if not results_df.empty:
            results_df['confidence_score'] = results_df['weight_sum'] / sum(weights)
            
            # Calculate error metrics
            if self.config.use_parallel:
                # Calculate metrics in parallel
                error_metrics = self.calculate_error_metrics(trajectories, results_df)
                logger.info(f"Triangulation error metrics: {error_metrics}")
            
        return results_df
    
    def _process_chunk(self, 
                       chunk_idx: int, 
                       timestamps: List[Any], 
                       aligned_data: Dict[Any, Dict[int, Tuple[float, float, float]]], 
                       weights: List[float]) -> List[Dict[str, Any]]:
        """
        Process a chunk of timestamps for triangulation
        
        Args:
            chunk_idx: Index of the chunk
            timestamps: List of timestamps in the chunk
            aligned_data: Dictionary of aligned trajectories
            weights: Weights for each trajectory
            
        Returns:
            List of dictionaries containing triangulated positions
        """
        chunk_results = []
        
        for ts in timestamps:
            positions = aligned_data[ts]
            if len(positions) >= self.config.min_points:
                src_indices = list(positions.keys())
                lat, lon, confidence, weight_sum = self._triangulate_single_position(
                    positions, weights, src_indices
                )
                
                chunk_results.append({
                    'timestamp': ts,
                    'latitude': lat,
                    'longitude': lon,
                    'confidence': confidence,
                    'weight_sum': weight_sum,
                    'src_count': len(positions),
                    'src_indices': src_indices
                })
                
        return chunk_results

    def _align_trajectories(self, trajectories: List[pd.DataFrame]) -> Dict[Any, Dict[int, Tuple[float, float, float]]]:
        """
        Align trajectories by timestamp
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Dictionary mapping timestamps to dictionaries of source index to (lat, lon, confidence)
        """
        # Create spatial index if enabled
        if self.config.spatial_index and self.spatial_index is None:
            self.spatial_index = SpatialIndex()
        
        # Use more efficient data structure for large datasets
        aligned_data = {}
        
        # Process each trajectory
        for traj_idx, trajectory in enumerate(trajectories):
            if trajectory.empty:
                continue
                
            # Ensure timestamp column exists and is properly formatted
            if 'timestamp' not in trajectory.columns:
                logger.warning(f"Trajectory {traj_idx} has no timestamp column, skipping")
                continue
                
            # Process in chunks for memory efficiency if dataset is large
            if self.config.memory_optimization and len(trajectory) > self.config.chunk_size:
                chunks = [trajectory.iloc[i:i + self.config.chunk_size] 
                         for i in range(0, len(trajectory), self.config.chunk_size)]
                
                for chunk in chunks:
                    self._process_trajectory_chunk(chunk, traj_idx, aligned_data)
            else:
                # Process entire trajectory at once
                self._process_trajectory_chunk(trajectory, traj_idx, aligned_data)
                
        return aligned_data
    
    def _process_trajectory_chunk(self, 
                                 trajectory_chunk: pd.DataFrame, 
                                 traj_idx: int, 
                                 aligned_data: Dict[Any, Dict[int, Tuple[float, float, float]]]) -> None:
        """
        Process a chunk of a trajectory and add to aligned data
        
        Args:
            trajectory_chunk: Chunk of trajectory data
            traj_idx: Index of the trajectory
            aligned_data: Dictionary to update with aligned data
        """
        for _, row in trajectory_chunk.iterrows():
            ts = row['timestamp']
            lat = row['latitude']
            lon = row['longitude']
            
            # Extract confidence if available, otherwise use default
            confidence = row.get('confidence', 1.0)
            
            # Add to spatial index if enabled
            if self.spatial_index is not None:
                self.spatial_index.insert(lat, lon, (ts, traj_idx, confidence))
            
            # Add to aligned data
            if ts not in aligned_data:
                aligned_data[ts] = {}
                
            aligned_data[ts][traj_idx] = (lat, lon, confidence)

    def _triangulate_single_position(self, 
                                   positions: Dict[int, Tuple[float, float, float]], 
                                   weights: List[float],
                                   source_indices: List[int]) -> Tuple[float, float, float, float]:
        """
        Triangulate a single position from multiple sources
        
        Args:
            positions: Dictionary mapping source index to (lat, lon, confidence)
            weights: Weights for each source
            source_indices: List of source indices
            
        Returns:
            Tuple of (latitude, longitude, confidence, weight_sum)
        """
        # Extract positions and confidences
        lats = []
        lons = []
        confs = []
        src_weights = []
        
        for idx in source_indices:
            lat, lon, conf = positions[idx]
            w = weights[idx] if idx < len(weights) else 1.0
            
            lats.append(lat)
            lons.append(lon)
            confs.append(conf)
            src_weights.append(w)
            
        # Weighted average as initial guess
        combined_weights = np.array(src_weights) * np.array(confs)
        weight_sum = sum(combined_weights)
        
        if weight_sum == 0:
            # Equal weighting if all weights are zero
            weight_sum = len(source_indices)
            weighted_lat = sum(lats) / weight_sum
            weighted_lon = sum(lons) / weight_sum
        else:
            weighted_lat = sum(lat * w for lat, w in zip(lats, combined_weights)) / weight_sum
            weighted_lon = sum(lon * w for lon, w in zip(lons, combined_weights)) / weight_sum
        
        # For simple cases, just use weighted average
        if len(source_indices) <= 2 or weight_sum < self.config.min_confidence:
            return weighted_lat, weighted_lon, weight_sum, weight_sum
            
        # For more complex cases, use optimization
        try:
            # Define objective function for optimization
            def objective(pos):
                opt_lat, opt_lon = pos
                
                # Calculate distances to all source positions
                distances = np.array([
                    _haversine_distance(opt_lat, opt_lon, lat, lon) 
                    for lat, lon in zip(lats, lons)
                ])
                
                # Apply distance-based weighting
                distance_weights = np.exp(-distances / self.config.weight_decay)
                
                # Combine weights
                total_weights = np.array(src_weights) * np.array(confs) * distance_weights
                
                # Calculate weighted error
                weighted_error = np.sum(distances * total_weights) / np.sum(total_weights)
                
                return weighted_error
                
            # Run optimization
            initial_guess = [weighted_lat, weighted_lon]
            result = minimize(
                objective, 
                initial_guess, 
                method=self.config.optimization_method,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                opt_lat, opt_lon = result.x
                # Calculate confidence based on optimization result
                confidence = weight_sum * (1.0 - min(1.0, result.fun / self.config.max_distance))
                return opt_lat, opt_lon, confidence, weight_sum
        except Exception as e:
            logger.warning(f"Optimization failed: {str(e)}, falling back to weighted average")
        
        # Fallback to weighted average
        return weighted_lat, weighted_lon, weight_sum, weight_sum
            
    @timer
    def calculate_error_metrics(self, true_trajectories: List[pd.DataFrame], triangulated: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate error metrics between true trajectories and triangulated positions
        
        Args:
            true_trajectories: List of true trajectories
            triangulated: DataFrame of triangulated positions
            
        Returns:
            Dictionary of error metrics
        """
        if triangulated.empty:
            return {'mean_error': float('nan'), 'median_error': float('nan'), 'max_error': float('nan')}
            
        # Merge all true trajectories for comparison
        all_true = pd.concat(true_trajectories, ignore_index=True)
        all_true = all_true.drop_duplicates('timestamp').set_index('timestamp')
        
        # Set index on triangulated
        triangulated_indexed = triangulated.set_index('timestamp')
        
        # Find common timestamps
        common_timestamps = set(all_true.index).intersection(set(triangulated_indexed.index))
        
        if not common_timestamps:
            return {'mean_error': float('nan'), 'median_error': float('nan'), 'max_error': float('nan')}
            
        # Calculate errors for each common timestamp
        errors = []
        
        # Batch process for large datasets
        if self.config.use_parallel and len(common_timestamps) > 1000:
            # Convert to lists for parallel processing
            true_lats = []
            true_lons = []
            tri_lats = []
            tri_lons = []
            
            for ts in common_timestamps:
                true_lats.append(all_true.loc[ts, 'latitude'])
                true_lons.append(all_true.loc[ts, 'longitude'])
                tri_lats.append(triangulated_indexed.loc[ts, 'latitude'])
                tri_lons.append(triangulated_indexed.loc[ts, 'longitude'])
            
            # Calculate distances in batches
            def calc_batch_distances(batch_indices):
                batch_errors = []
                for i in batch_indices:
                    batch_errors.append(_haversine_distance(
                        true_lats[i], true_lons[i], 
                        tri_lats[i], tri_lons[i]
                    ))
                return batch_errors
            
            # Split into batches
            batch_indices = [list(range(i, min(i + self.config.batch_size, len(common_timestamps)))) 
                           for i in range(0, len(common_timestamps), self.config.batch_size)]
            
            # Process batches in parallel
            batch_results = self.parallel_processor.map(calc_batch_distances, batch_indices)
            
            # Flatten results
            for batch in batch_results:
                errors.extend(batch)
        else:
            # Sequential processing
            for ts in common_timestamps:
                true_lat = all_true.loc[ts, 'latitude']
                true_lon = all_true.loc[ts, 'longitude']
                tri_lat = triangulated_indexed.loc[ts, 'latitude']
                tri_lon = triangulated_indexed.loc[ts, 'longitude']
                
                # Calculate error
                error = _haversine_distance(true_lat, true_lon, tri_lat, tri_lon)
                errors.append(error)
        
        # Calculate metrics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)
        
        return {
            'mean_error': float(mean_error),
            'median_error': float(median_error),
            'max_error': float(max_error),
            'error_points': len(errors)
        }
