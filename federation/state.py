from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

@dataclass
class DataPoint:
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float
    heart_rate: float
    cadence: float
    speed: float
    stance_time: float
    stance_time_balance: float
    step_length: float
    vertical_ratio: float
    vertical_oscillation: float
    
    # Optional fields that may only exist in some devices
    power: Optional[float] = None
    form_power: Optional[float] = None
    accumulated_power: Optional[float] = None

class FederatedState:
    def __init__(self):
        self.device_data: Dict[str, List[DataPoint]] = {}
        self.global_model_weights = None
        self.local_models = {}
        
    def add_device_data(self, device_id: str, data_points: List[DataPoint]):
        """Add data from a specific device."""
        self.device_data[device_id] = data_points
        
    def get_time_aligned_data(self, reference_device: str, target_device: str, 
                            time_window: float = 1.0) -> tuple:
        """
        Align data points from two devices based on timestamps.
        time_window: maximum time difference in seconds to consider points aligned
        """
        ref_data = self.device_data[reference_device]
        target_data = self.device_data[target_device]
        
        aligned_ref = []
        aligned_target = []
        
        for ref_point in ref_data:
            # Find closest matching point in target data
            closest_point = min(target_data, 
                              key=lambda x: abs((x.timestamp - ref_point.timestamp).total_seconds()))
            
            time_diff = abs((closest_point.timestamp - ref_point.timestamp).total_seconds())
            if time_diff <= time_window:
                aligned_ref.append(ref_point)
                aligned_target.append(closest_point)
                
        return aligned_ref, aligned_target
    
    def update_local_model(self, device_id: str, model_weights):
        """Update local model weights for a specific device."""
        self.local_models[device_id] = model_weights
        
    def aggregate_global_model(self):
        """Aggregate local models into global model using FedAvg."""
        if not self.local_models:
            return
            
        # Simple averaging of model weights
        all_weights = np.array(list(self.local_models.values()))
        self.global_model_weights = np.mean(all_weights, axis=0)
