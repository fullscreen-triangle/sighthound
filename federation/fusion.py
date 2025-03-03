import json
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .state import FederatedState, DataPoint
from typing import List, Tuple

class DataFusion:
    def __init__(self):
        self.state = FederatedState()
        self.models = {
            'heart_rate': RandomForestRegressor(),
            'speed': RandomForestRegressor(),
            'cadence': RandomForestRegressor()
        }
        
    def load_device_data(self, device_id: str, file_path: str):
        """Load data from JSON file for a specific device."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        data_points = []
        
        if isinstance(data, dict) and 'features' in data:  # GeoJSON format
            features = data['features']
            for feature in features:
                props = feature.get('properties', {})
                geom = feature.get('geometry', {}).get('coordinates', [])
                
                if len(geom) >= 2:
                    point = DataPoint(
                        timestamp=datetime.fromisoformat(props['timestamp'].replace('Z', '+00:00')),
                        latitude=geom[1],
                        longitude=geom[0],
                        altitude=props.get('altitude'),
                        heart_rate=props.get('heart_rate'),
                        cadence=props.get('cadence'),
                        speed=props.get('speed'),
                        stance_time=props.get('stance_time'),
                        stance_time_balance=props.get('stance_time_balance'),
                        step_length=props.get('step_length'),
                        vertical_ratio=props.get('vertical_ratio'),
                        vertical_oscillation=props.get('vertical_oscillation')
                    )
                    data_points.append(point)
                    
        else:  # Regular JSON array format
            for item in data:
                point = DataPoint(
                    timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                    latitude=item['latitude'],
                    longitude=item['longitude'],
                    altitude=item['altitude'],
                    heart_rate=item['heart_rate'],
                    cadence=item['cadence'],
                    speed=item['speed'],
                    stance_time=item['stance_time'],
                    stance_time_balance=item['stance_time_balance'],
                    step_length=item['step_length'],
                    vertical_ratio=item['vertical_ratio'],
                    vertical_oscillation=item['vertical_oscillation'],
                    power=item.get('power'),
                    form_power=item.get('form_power'),
                    accumulated_power=item.get('accumulated_power')
                )
                data_points.append(point)
                
        self.state.add_device_data(device_id, data_points)
        
    def train_local_models(self, device_id: str):
        """Train local models for a specific device."""
        data = self.state.device_data[device_id]
        
        # Extract features and targets
        features = np.array([[
            d.latitude, d.longitude, d.altitude,
            d.stance_time, d.stance_time_balance,
            d.step_length, d.vertical_ratio, d.vertical_oscillation
        ] for d in data])
        
        for metric, model in self.models.items():
            targets = np.array([getattr(d, metric) for d in data])
            model.fit(features, targets)
            
        # Store model weights
        model_weights = [model.feature_importances_ for model in self.models.values()]
        self.state.update_local_model(device_id, np.mean(model_weights, axis=0))
        
    def _align_timestamps(self, ref_data: List[DataPoint], target_data: List[DataPoint]) -> List[Tuple[DataPoint, DataPoint]]:
        """
        Align timestamps between two sets of data points.
        
        Args:
            ref_data: Reference device data points
            target_data: Target device data points
            
        Returns:
            List of tuples containing aligned (reference, target) data points
        """
        # Sort data by timestamp
        ref_data = sorted(ref_data, key=lambda x: x.timestamp)
        target_data = sorted(target_data, key=lambda x: x.timestamp)
        
        # Find overlapping time range
        start_time = max(ref_data[0].timestamp, target_data[0].timestamp)
        end_time = min(ref_data[-1].timestamp, target_data[-1].timestamp)
        
        # Filter data points within the overlapping range
        ref_filtered = [p for p in ref_data if start_time <= p.timestamp <= end_time]
        target_filtered = [p for p in target_data if start_time <= p.timestamp <= end_time]
        
        # Create aligned pairs
        aligned_pairs = []
        
        # Simple approach: find closest timestamp match for each reference point
        for ref_point in ref_filtered:
            # Find closest target point by timestamp
            closest_target = min(target_filtered, 
                               key=lambda x: abs((x.timestamp - ref_point.timestamp).total_seconds()))
            
            # Only include pairs that are within a reasonable time difference (e.g., 5 seconds)
            time_diff = abs((closest_target.timestamp - ref_point.timestamp).total_seconds())
            if time_diff <= 5:  # 5 seconds threshold
                aligned_pairs.append((ref_point, closest_target))
        
        return aligned_pairs

    def fuse_measurements(self, ref_device: str, target_device: str) -> List[DataPoint]:
        """
        Fuse measurements from two devices using the global model.
        
        Args:
            ref_device: Reference device name
            target_device: Target device name
            
        Returns:
            List of fused DataPoint objects
        """
        if ref_device not in self.state.device_data or target_device not in self.state.device_data:
            raise ValueError(f"Devices {ref_device} and {target_device} must be loaded first")
            
        ref_data = self.state.device_data[ref_device]
        target_data = self.state.device_data[target_device]
        
        # Align timestamps between devices
        aligned_data = self._align_timestamps(ref_data, target_data)
        
        # Fuse data points
        fused_points = []
        for ref, target in aligned_data:
            # Convert string values to float where needed
            try:
                ref_stance_time_balance = float(ref.stance_time_balance) if ref.stance_time_balance else 0.0
                target_stance_time_balance = float(target.stance_time_balance) if target.stance_time_balance else 0.0
                
                ref_vertical_ratio = float(ref.vertical_ratio) if ref.vertical_ratio else 0.0
                target_vertical_ratio = float(target.vertical_ratio) if target.vertical_ratio else 0.0
                
                ref_vertical_oscillation = float(ref.vertical_oscillation) if ref.vertical_oscillation else 0.0
                target_vertical_oscillation = float(target.vertical_oscillation) if target.vertical_oscillation else 0.0
                
                ref_step_length = float(ref.step_length) if ref.step_length else 0.0
                target_step_length = float(target.step_length) if target.step_length else 0.0
                
                ref_stance_time = float(ref.stance_time) if ref.stance_time else 0.0
                target_stance_time = float(target.stance_time) if target.stance_time else 0.0
                
                # Create fused data point
                fused_point = DataPoint(
                    timestamp=ref.timestamp,
                    latitude=ref.latitude,
                    longitude=ref.longitude,
                    altitude=(ref.altitude + target.altitude) / 2,
                    heart_rate=(ref.heart_rate + target.heart_rate) / 2,
                    cadence=(ref.cadence + target.cadence) / 2,
                    speed=(ref.speed + target.speed) / 2,
                    stance_time=(ref_stance_time + target_stance_time) / 2,
                    stance_time_balance=(ref_stance_time_balance + target_stance_time_balance) / 2,
                    step_length=(ref_step_length + target_step_length) / 2,
                    vertical_ratio=(ref_vertical_ratio + target_vertical_ratio) / 2,
                    vertical_oscillation=(ref_vertical_oscillation + target_vertical_oscillation) / 2,
                    power=target.power,  # Use target device power if available
                    form_power=target.form_power,
                    accumulated_power=target.accumulated_power
                )
                fused_points.append(fused_point)
            except (ValueError, TypeError) as e:
                # Log the error and skip this data point
                print(f"Error fusing data point: {e}")
                continue
                
        return fused_points
