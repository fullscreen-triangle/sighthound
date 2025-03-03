import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PowerMetrics:
    mean_power: float
    peak_power: float
    power_zones: Dict[str, Tuple[float, float]]
    power_distribution: Dict[str, float]
    accumulated_power_rate: float
    form_power_efficiency: float
    
class PowerAnalyzer:
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        self.df = pd.DataFrame(self.raw_data)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def analyze_power_metrics(self) -> PowerMetrics:
        """Calculate comprehensive power metrics"""
        power_data = self.df['power'].values
        form_power_data = self.df['form_power'].values
        accumulated_power_data = self.df['accumulated_power'].values
        
        # Basic statistics
        mean_power = np.mean(power_data)
        peak_power = np.max(power_data)
        
        # Power zones (customized for running power)
        power_zones = {
            'recovery': (0, mean_power * 0.8),
            'endurance': (mean_power * 0.8, mean_power * 1.0),
            'tempo': (mean_power * 1.0, mean_power * 1.1),
            'threshold': (mean_power * 1.1, mean_power * 1.2),
            'vo2max': (mean_power * 1.2, float('inf'))
        }
        
        # Calculate time spent in each zone
        total_time = len(power_data)
        power_distribution = {}
        for zone, (min_power, max_power) in power_zones.items():
            zone_time = np.sum((power_data >= min_power) & (power_data < max_power))
            power_distribution[zone] = zone_time / total_time * 100
            
        # Calculate power accumulation rate (watts per second)
        power_acc_rate = np.diff(accumulated_power_data).mean()
        
        # Calculate form power efficiency (lower is better)
        form_power_efficiency = np.mean(form_power_data / power_data)
        
        return PowerMetrics(
            mean_power=mean_power,
            peak_power=peak_power,
            power_zones=power_zones,
            power_distribution=power_distribution,
            accumulated_power_rate=power_acc_rate,
            form_power_efficiency=form_power_efficiency
        )
    
    def detect_activity_patterns(self) -> Dict[str, List[int]]:
        """Detect different activity patterns based on power metrics"""
        power_data = self.df['power'].values
        form_power = self.df['form_power'].values
        
        # Use rolling statistics to identify patterns
        window = 5
        rolling_mean = pd.Series(power_data).rolling(window=window).mean()
        rolling_std = pd.Series(power_data).rolling(window=window).std()
        
        patterns = {
            'steady_state': [],  # Consistent power output
            'surge': [],        # Sudden increase in power
            'recovery': [],     # Lower power periods
            'transition': []    # Changes in power output
        }
        
        for i in range(window, len(power_data)):
            if rolling_std[i] < np.mean(rolling_std) * 0.5:
                patterns['steady_state'].append(i)
            elif power_data[i] > rolling_mean[i] + rolling_std[i]:
                patterns['surge'].append(i)
            elif power_data[i] < rolling_mean[i] - rolling_std[i]:
                patterns['recovery'].append(i)
            else:
                patterns['transition'].append(i)
                
        return patterns
    
    def analyze_curve_power(self, curve_indices: List[int]) -> Dict[str, float]:
        """Analyze power metrics specifically during curved sections"""
        curve_power = self.df.iloc[curve_indices]['power'].values
        curve_form_power = self.df.iloc[curve_indices]['form_power'].values
        
        return {
            'mean_curve_power': np.mean(curve_power),
            'peak_curve_power': np.max(curve_power),
            'curve_form_power_efficiency': np.mean(curve_form_power / curve_power),
            'power_variability': np.std(curve_power)
        }
    
    def is_peak_activity(self, window_minutes: int = 5) -> bool:
        """Determine if this segment represents peak activity"""
        window_size = window_minutes * 60  # Convert to seconds
        power_rolling = pd.Series(self.df['power'].values).rolling(window=window_size).mean()
        current_mean = np.mean(self.df['power'].values)
        
        # Compare current segment to rolling average
        return current_mean > np.percentile(power_rolling.dropna(), 90)
    
    def get_power_likelihood_map(self) -> np.ndarray:
        """Generate likelihood scores for each data point being in a curve"""
        power_scores = stats.zscore(self.df['power'].values)
        form_power_scores = stats.zscore(self.df['form_power'].values)
        
        # Combine metrics to create likelihood scores
        likelihood = (power_scores + form_power_scores) / 2
        return stats.norm.cdf(likelihood)  # Convert to probability scale
