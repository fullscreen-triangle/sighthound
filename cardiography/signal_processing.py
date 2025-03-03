import numpy as np
from scipy import signal
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MLSignalManager:
    """Manage ML models for signal processing and anomaly detection"""
    
    def __init__(self, model_dir: str = 'signal_models'):
        self.model_dir = model_dir
        self.models = {
            'anomaly_detector': self._load_or_create_model('anomaly_detector.joblib', model_type='isolation_forest'),
            'hr_predictor': self._load_or_create_model('hr_predictor.joblib', model_type='random_forest'),
            'hrv_predictor': self._load_or_create_model('hrv_predictor.joblib', model_type='random_forest')
        }
        self.scalers = {
            'hr': self._load_or_create_scaler('hr_scaler.joblib'),
            'hrv': self._load_or_create_scaler('hrv_scaler.joblib')
        }
    
    def _load_or_create_model(self, filename: str, model_type: str):
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        
        if model_type == 'isolation_forest':
            return IsolationForest(contamination=0.1, random_state=42)
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _load_or_create_scaler(self, filename: str) -> StandardScaler:
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return StandardScaler()
    
    def save_models(self):
        """Save all models and scalers"""
        os.makedirs(self.model_dir, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{name}.joblib'))
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(self.model_dir, f'{name}_scaler.joblib'))

class HeartRateProcessor:
    def __init__(self):
        self.sampling_rate: Optional[float] = None
        self.heart_rate_data: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        self.ml_manager = MLSignalManager()
        self.window_size = 10  # Number of samples to consider for features

    def _prepare_features(self, data_window: np.ndarray) -> np.ndarray:
        """Prepare feature vector for ML models"""
        return np.array([
            np.mean(data_window),
            np.std(data_window),
            np.max(data_window),
            np.min(data_window),
            np.median(data_window),
            np.gradient(data_window).mean(),
            signal.detrend(data_window).std()
        ]).reshape(1, -1)

    def load_from_geojson(self, track_data: Dict) -> None:
        """Load heart rate data from GeoJSON track data."""
        features = track_data['features']
        
        # Extract heart rate and timestamps
        heart_rates = []
        timestamps = []
        
        for feature in features:
            props = feature['properties']
            heart_rates.append(float(props['heart_rate']))
            timestamps.append(datetime.strptime(props['timestamp'], '%Y-%m-%d %H:%M:%S%z'))
            
        self.heart_rate_data = np.array(heart_rates)
        self.timestamps = np.array(timestamps)
        
        # Calculate sampling rate (in Hz)
        time_diffs = np.diff(timestamps)
        avg_time_diff = np.mean(time_diffs)
        self.sampling_rate = 1 / avg_time_diff.total_seconds()

    def get_heart_rate_variability(self) -> Dict[str, float]:
        """Calculate HRV metrics using both traditional and ML methods."""
        if self.heart_rate_data is None:
            raise ValueError("Heart rate data not loaded")

        # Traditional HRV calculations
        traditional_hrv = {
            'sdnn': np.std(self.heart_rate_data),
            'rmssd': np.sqrt(np.mean(np.diff(self.heart_rate_data) ** 2)),
            'mean_hr': np.mean(self.heart_rate_data),
            'max_hr': np.max(self.heart_rate_data),
            'min_hr': np.min(self.heart_rate_data)
        }

        # ML-based HRV prediction
        features = np.array([self._prepare_features(
            self.heart_rate_data[i:i+self.window_size]
        ) for i in range(len(self.heart_rate_data) - self.window_size)])
        
        scaled_features = self.ml_manager.scalers['hrv'].transform(features.reshape(features.shape[0], -1))
        ml_hrv = self.ml_manager.models['hrv_predictor'].predict(scaled_features)

        # Combine traditional and ML predictions
        combined_hrv = {
            'sdnn': 0.7 * traditional_hrv['sdnn'] + 0.3 * np.mean(ml_hrv),
            'rmssd': traditional_hrv['rmssd'],  # Keep traditional for now
            'mean_hr': traditional_hrv['mean_hr'],
            'max_hr': traditional_hrv['max_hr'],
            'min_hr': traditional_hrv['min_hr']
        }

        # Update ML models with new data
        self._update_hrv_model(traditional_hrv, features)
        
        return combined_hrv

    def _update_hrv_model(self, traditional_hrv: Dict[str, float], features: np.ndarray):
        """Update HRV prediction model"""
        self.ml_manager.scalers['hrv'].partial_fit(features.reshape(features.shape[0], -1))
        scaled_features = self.ml_manager.scalers['hrv'].transform(features.reshape(features.shape[0], -1))
        
        # Use traditional SDNN as target
        y = np.full(features.shape[0], traditional_hrv['sdnn'])
        self.ml_manager.models['hrv_predictor'].fit(scaled_features, y)
        self.ml_manager.save_models()

    def detect_anomalies(self, threshold: float = 2.0) -> List[int]:
        """Detect anomalies using both statistical and ML methods."""
        if self.heart_rate_data is None:
            raise ValueError("Heart rate data not loaded")

        # Statistical anomaly detection
        z_scores = np.abs(signal.detrend(self.heart_rate_data) / np.std(self.heart_rate_data))
        statistical_anomalies = set(np.where(z_scores > threshold)[0])

        # ML-based anomaly detection
        features = np.array([self._prepare_features(
            self.heart_rate_data[i:i+self.window_size]
        ) for i in range(len(self.heart_rate_data) - self.window_size)])
        
        ml_anomalies = set(np.where(
            self.ml_manager.models['anomaly_detector'].predict(features.reshape(features.shape[0], -1)) == -1
        )[0])

        # Combine both methods (union of anomalies)
        combined_anomalies = statistical_anomalies.union(ml_anomalies)
        
        # Update anomaly detection model
        self._update_anomaly_model(features)
        
        return sorted(list(combined_anomalies))

    def _update_anomaly_model(self, features: np.ndarray):
        """Update anomaly detection model"""
        self.ml_manager.models['anomaly_detector'].fit(features.reshape(features.shape[0], -1))
        self.ml_manager.save_models()

    def get_frequency_analysis(self) -> Dict[str, np.ndarray]:
        """Perform frequency domain analysis with ML-enhanced noise reduction."""
        if self.heart_rate_data is None or self.sampling_rate is None:
            raise ValueError("Heart rate data or sampling rate not available")

        # Traditional frequency analysis
        detrended = signal.detrend(self.heart_rate_data)
        window = signal.hann(len(detrended))
        windowed = detrended * window
        
        # ML-based signal cleaning
        features = np.array([self._prepare_features(
            self.heart_rate_data[i:i+self.window_size]
        ) for i in range(len(self.heart_rate_data) - self.window_size)])
        
        scaled_features = self.ml_manager.scalers['hr'].transform(features.reshape(features.shape[0], -1))
        cleaned_signal = self.ml_manager.models['hr_predictor'].predict(scaled_features)
        
        # Combine original and cleaned signals
        final_signal = 0.7 * windowed[self.window_size:] + 0.3 * cleaned_signal
        
        # Compute FFT on combined signal
        fft = np.fft.fft(final_signal)
        freqs = np.fft.fftfreq(len(final_signal), 1/self.sampling_rate)
        
        # Get positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        power = np.abs(fft[pos_mask])
        
        # Update HR predictor model
        self._update_hr_model(features, windowed[self.window_size:])
        
        return {
            'frequencies': freqs,
            'power': power
        }

    def _update_hr_model(self, features: np.ndarray, target_signal: np.ndarray):
        """Update heart rate prediction model"""
        self.ml_manager.scalers['hr'].partial_fit(features.reshape(features.shape[0], -1))
        scaled_features = self.ml_manager.scalers['hr'].transform(features.reshape(features.shape[0], -1))
        self.ml_manager.models['hr_predictor'].fit(scaled_features, target_signal)
        self.ml_manager.save_models()

    def get_time_domain_features(self) -> Dict[str, float]:
        """Extract time domain features from heart rate data."""
        if self.heart_rate_data is None:
            raise ValueError("Heart rate data not loaded")

        # Calculate first derivative (rate of change)
        hr_gradient = np.gradient(self.heart_rate_data)
        
        features = {
            'mean_gradient': np.mean(np.abs(hr_gradient)),
            'max_gradient': np.max(np.abs(hr_gradient)),
            'variance': np.var(self.heart_rate_data),
            'skewness': float(pd.Series(self.heart_rate_data).skew()),
            'kurtosis': float(pd.Series(self.heart_rate_data).kurtosis())
        }
        return features
