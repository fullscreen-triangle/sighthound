import math
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Standardize logger configuration
logger = logging.getLogger(__name__)

class AtmosphericConditions:
    """Calculate atmospheric conditions at given altitude and location"""
    
    # Constants
    SEA_LEVEL_PRESSURE = 101325  # Pa
    SEA_LEVEL_TEMP = 288.15     # K
    LAPSE_RATE = -0.0065       # K/m
    GAS_CONSTANT = 8.31447     # J/(mol·K)
    GRAVITY = 9.80665          # m/s²
    MOLAR_MASS_AIR = 0.0289644 # kg/mol
    
    def __init__(self, altitude: float, temperature: float = None):
        self.altitude = altitude
        self.temperature = temperature or self._calculate_temperature()
        
    def _calculate_temperature(self) -> float:
        """Calculate temperature at altitude using standard atmosphere model"""
        return self.SEA_LEVEL_TEMP + self.LAPSE_RATE * self.altitude
    
    def atmospheric_pressure(self) -> float:
        """Calculate atmospheric pressure at altitude using barometric formula"""
        exponent = (-self.GRAVITY * self.MOLAR_MASS_AIR * self.altitude) / (self.GAS_CONSTANT * self.SEA_LEVEL_TEMP)
        return self.SEA_LEVEL_PRESSURE * math.exp(exponent)
    
    def water_vapor_pressure(self) -> float:
        """Calculate water vapor pressure using Magnus formula"""
        T = self.temperature - 273.15  # Convert to Celsius
        return 610.94 * math.exp(17.625 * T / (T + 243.04))
    
    def oxygen_partial_pressure(self) -> float:
        """Calculate oxygen partial pressure"""
        dry_air_o2_fraction = 0.2094
        atmospheric_pressure = self.atmospheric_pressure()
        water_vapor = self.water_vapor_pressure()
        return (atmospheric_pressure - water_vapor) * dry_air_o2_fraction

class RespiratorySystem:
    """Model the respiratory system and its parameters"""
    
    def __init__(self, age: int, weight: float, height: float, gender: str):
        self.age = age
        self.weight = weight  # kg
        self.height = height  # cm
        self.gender = gender.lower()
        
    def estimate_lung_volumes(self) -> Dict[str, float]:
        """
        Estimate various lung volumes using predictive equations
        Returns: TLC, VC, RV, FRC in liters
        """
        # Using European Respiratory Society equations
        is_male = self.gender == 'male'
        height_m = self.height / 100
        
        if is_male:
            tlc = 7.99 * height_m - 7.08  # Total Lung Capacity
            vc = 6.10 * height_m - 5.20   # Vital Capacity
        else:
            tlc = 6.60 * height_m - 5.79
            vc = 4.66 * height_m - 3.89
            
        rv = tlc - vc  # Residual Volume
        frc = rv * 1.4  # Functional Residual Capacity
        
        return {
            'total_lung_capacity': tlc,
            'vital_capacity': vc,
            'residual_volume': rv,
            'functional_residual_capacity': frc
        }
    
    def estimate_blood_volume(self) -> float:
        """Estimate blood volume using Nadler's equation"""
        is_male = self.gender == 'male'
        height_m = self.height / 100
        
        if is_male:
            return (0.3669 * (height_m ** 3) + 0.03219 * self.weight + 0.6041) * 1000
        return (0.3561 * (height_m ** 3) + 0.03308 * self.weight + 0.1833) * 1000

class BloodComposition:
    """Model blood composition and oxygen binding characteristics"""
    
    # Constants
    NORMAL_HB = 15.0  # g/dL for males
    O2_BINDING_CAPACITY = 1.34  # mL O2/g Hb
    HILL_COEFFICIENT = 2.7
    P50 = 26.6  # mmHg
    
    def __init__(self, hematocrit: float = None, hemoglobin: float = None):
        self.hematocrit = hematocrit or 0.45  # 45% is typical
        self.hemoglobin = hemoglobin or self.NORMAL_HB
        
    def oxygen_carrying_capacity(self) -> float:
        """Calculate maximum oxygen carrying capacity in mL O2/dL blood"""
        return self.hemoglobin * self.O2_BINDING_CAPACITY
    
    def oxygen_saturation(self, po2: float) -> float:
        """
        Calculate oxygen saturation using Hill equation
        po2: partial pressure of oxygen in mmHg
        """
        return (po2 ** self.HILL_COEFFICIENT) / (po2 ** self.HILL_COEFFICIENT + self.P50 ** self.HILL_COEFFICIENT)

class OrganMetrics:
    """Calculate organ sizes and metabolic parameters"""
    
    def __init__(self, age: int, weight: float, height: float, gender: str):
        self.age = age
        self.weight = weight
        self.height = height
        self.gender = gender.lower()
        self.bsa = self._calculate_bsa()
    
    def _calculate_bsa(self) -> float:
        """Calculate body surface area using DuBois formula"""
        height_m = self.height / 100
        return 0.007184 * (self.weight ** 0.425) * (height_m ** 0.725)
    
    def estimate_organ_volumes(self) -> Dict[str, float]:
        """
        Estimate volumes of major organs involved in respiration
        Returns volumes in mL
        """
        is_male = self.gender == 'male'
        
        # Simplified organ volume estimations based on BSA and gender
        heart_volume = self.bsa * (1000 if is_male else 900)
        lung_volume = self.bsa * (1200 if is_male else 1000)
        liver_volume = self.bsa * (1600 if is_male else 1400)
        
        return {
            'heart': heart_volume,
            'lungs': lung_volume,
            'liver': liver_volume
        }
    
    def estimate_metabolic_rate(self) -> float:
        """Estimate basal metabolic rate using Harris-Benedict equation"""
        is_male = self.gender == 'male'
        
        if is_male:
            return 88.362 + (13.397 * self.weight) + (4.799 * self.height) - (5.677 * self.age)
        return 447.593 + (9.247 * self.weight) + (3.098 * self.height) - (4.330 * self.age)

class MLModelManager:
    """Manage machine learning models for physiological parameter prediction"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.models = {
            'vo2': self._load_or_create_model('vo2_model.joblib'),
            'diffusion': self._load_or_create_model('diffusion_model.joblib'),
            'saturation': self._load_or_create_model('saturation_model.joblib')
        }
        self.scalers = {
            'vo2': self._load_or_create_scaler('vo2_scaler.joblib'),
            'diffusion': self._load_or_create_scaler('diffusion_scaler.joblib'),
            'saturation': self._load_or_create_scaler('saturation_scaler.joblib')
        }
        
    def _load_or_create_model(self, filename: str) -> RandomForestRegressor:
        """Load existing model or create new one if not exists"""
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _load_or_create_scaler(self, filename: str) -> StandardScaler:
        """Load existing scaler or create new one if not exists"""
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        return StandardScaler()
    
    def save_models(self):
        """Save all models and scalers"""
        os.makedirs(self.model_dir, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{name}_model.joblib'))
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(self.model_dir, f'{name}_scaler.joblib'))

class OxygenUptakeCalculator:
    """
    Calculate oxygen uptake rate during running based on physiological models.
    
    This class implements the energy expenditure and oxygen uptake calculations
    based on athlete characteristics and running dynamics.
    """
    
    def __init__(self, mass, height, age, gender):
        """
        Initialize oxygen uptake calculator.
        
        Args:
            mass: Athlete's mass in kg
            height: Athlete's height in meters
            age: Athlete's age in years
            gender: Athlete's gender ('male' or 'female')
        """
        self.mass = mass
        self.height = height
        self.age = age
        self.gender = gender.lower()  # Convert to lowercase for consistency
        
        # Validate parameters
        if self.mass <= 0 or self.height <= 0 or self.age <= 0:
            raise ValueError("Mass, height, and age must be positive values")
        
        if self.gender not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        
        # Calculate body surface area using Du Bois formula
        self.BSA = 0.007184 * (mass ** 0.425) * (height * 100) ** 0.725  # height in cm for formula
        
        # Calculate basal metabolic rate (BMR) using Mifflin-St Jeor equation
        if self.gender == 'male':
            self.BMR = 10 * mass + 6.25 * (height * 100) - 5 * age + 5  # kcal/day
        else:  # female
            self.BMR = 10 * mass + 6.25 * (height * 100) - 5 * age - 161  # kcal/day
            
        # Convert BMR to ml O2/min (1 kcal ≈ 200 ml O2)
        self.resting_vo2 = self.BMR * 200 / 1440  # ml/min (1440 minutes in a day)
        
        # Initialize ML model manager
        self.ml_manager = self._init_ml_manager()
        
        # Initialize other variables
        self.vo2max_estimated = self._estimate_vo2max()  # ml/kg/min
        
        # Try to load models from continuous learning system
        self._load_from_continuous_learning()
        
        logger.info(f"Initialized OxygenUptakeCalculator for {gender} athlete: "
                   f"{mass:.1f}kg, {height:.2f}m, {age} years")
        logger.info(f"Estimated VO2max: {self.vo2max_estimated:.1f} ml/kg/min")
        
    def _init_ml_manager(self):
        """Initialize or load ML model manager."""
        try:
            return MLModelManager()
        except Exception as e:
            logger.error(f"Error initializing ML model manager: {e}")
            return None
            
    def _load_from_continuous_learning(self):
        """Try to load models from continuous learning system."""
        try:
            from federation.continuous_learning import continuous_learning_manager
            
            # Try to get the oxygen uptake model
            model = continuous_learning_manager.model_registry.get_model("oxygen_uptake")
            
            if model is not None:
                logger.info("Loaded oxygen uptake model from continuous learning system")
                if hasattr(self, 'ml_manager') and self.ml_manager is not None:
                    self.ml_manager.models['vo2'] = model
                return True
        except Exception as e:
            logger.warning(f"Could not load model from continuous learning system: {e}")
        
        return False
    
    def _estimate_vo2max(self) -> float:
        """Estimate VO2max using a simple formula"""
        return 2 * self.resting_vo2  # This is a placeholder; actual estimation methods may vary
    
    def calculate_oxygen_gradient(self) -> List[Dict[str, float]]:
        """
        Calculate oxygen pressure gradient from atmosphere to alveoli
        Returns list of pressure points through respiratory system
        """
        gradient_points = []
        
        for feature in self.track_data['features']:
            altitude = feature['properties']['altitude']
            if isinstance(altitude, str):
                altitude = float(altitude)
                
            atm_conditions = AtmosphericConditions(altitude)
            
            # Calculate pressures at different points
            p_atmosphere = atm_conditions.atmospheric_pressure() / 1000  # Convert to kPa
            p_o2_atmosphere = atm_conditions.oxygen_partial_pressure() / 1000
            p_h2o = atm_conditions.water_vapor_pressure() / 1000
            
            # Estimated pressure drops through respiratory system
            p_trachea = p_o2_atmosphere * 0.95  # ~5% drop in trachea
            p_bronchi = p_trachea * 0.90       # ~10% drop in bronchi
            p_alveoli = p_bronchi * 0.87       # ~13% drop in alveoli
            
            gradient_points.append({
                'timestamp': feature['properties']['timestamp'],
                'altitude': altitude,
                'p_atmosphere': p_atmosphere,
                'p_o2_atmosphere': p_o2_atmosphere,
                'p_h2o': p_h2o,
                'p_trachea': p_trachea,
                'p_bronchi': p_bronchi,
                'p_alveoli': p_alveoli
            })
            
        return gradient_points
    
    def _prepare_features(self, feature: Dict) -> np.ndarray:
        """Prepare feature vector for ML models"""
        return np.array([
            feature['properties']['heart_rate'],
            feature['properties']['speed'],
            feature['properties']['cadence'],
            feature['properties']['altitude'],
            self.organ_metrics.age,
            self.organ_metrics.weight,
            self.organ_metrics.height,
            1 if self.organ_metrics.gender == 'male' else 0
        ]).reshape(1, -1)
    
    def calculate_oxygen_consumption(self) -> List[Dict[str, float]]:
        """Calculate oxygen consumption using both traditional and ML methods"""
        consumption_data = []
        
        for feature in self.track_data['features']:
            # Traditional calculation
            traditional_vo2 = self._calculate_traditional_vo2(feature)
            
            # ML-based prediction
            features = self._prepare_features(feature)
            scaled_features = self.ml_manager.scalers['vo2'].transform(features)
            ml_vo2 = self.ml_manager.models['vo2'].predict(scaled_features)[0]
            
            # Combine predictions (weighted average)
            final_vo2 = 0.7 * traditional_vo2 + 0.3 * ml_vo2
            
            consumption_data.append({
                'timestamp': feature['properties']['timestamp'],
                'heart_rate': feature['properties']['heart_rate'],
                'speed': feature['properties']['speed'],
                'vo2_traditional': traditional_vo2,
                'vo2_ml': ml_vo2,
                'vo2_final': final_vo2,
                'oxygen_consumption': final_vo2 * self.weight
            })
            
        # Train model with new data
        self._update_ml_models(consumption_data)
        return consumption_data
    
    def _calculate_traditional_vo2(self, feature: Dict) -> float:
        """Traditional VO2 calculation"""
        heart_rate = feature['properties']['heart_rate']
        vo2_max = 15 * (220 - self.organ_metrics.age) / heart_rate
        return vo2_max * (heart_rate / (220 - self.organ_metrics.age))
    
    def _update_ml_models(self, new_data: List[Dict]):
        """Update ML models with new data"""
        X = np.array([[
            d['heart_rate'],
            d['speed'],
            feature['properties']['cadence'],
            feature['properties']['altitude'],
            self.organ_metrics.age,
            self.organ_metrics.weight,
            self.organ_metrics.height,
            1 if self.organ_metrics.gender == 'male' else 0
        ] for d, feature in zip(new_data, self.track_data['features'])])
        
        y_vo2 = np.array([d['vo2_traditional'] for d in new_data])
        
        # Update VO2 model
        self.ml_manager.scalers['vo2'].partial_fit(X)
        X_scaled = self.ml_manager.scalers['vo2'].transform(X)
        self.ml_manager.models['vo2'].fit(X_scaled, y_vo2)
        
        # Save updated models
        self.ml_manager.save_models()
    
    def calculate_alveolar_exchange(self) -> List[Dict[str, float]]:
        """Calculate alveolar exchange using both traditional and ML methods"""
        exchange_data = []
        gradient_data = self.calculate_oxygen_gradient()
        
        for point, feature in zip(gradient_data, self.track_data['features']):
            # Traditional calculations
            p_alveoli_mmhg = point['p_alveoli'] * 7.50062
            traditional_saturation = self.blood_composition.oxygen_saturation(p_alveoli_mmhg)
            
            # ML-based predictions
            features = self._prepare_features(feature)
            scaled_features = self.ml_manager.scalers['saturation'].transform(features)
            ml_saturation = self.ml_manager.models['saturation'].predict(scaled_features)[0]
            
            # Combine predictions
            final_saturation = 0.8 * traditional_saturation + 0.2 * ml_saturation
            
            exchange_data.append({
                'timestamp': point['timestamp'],
                'p_alveoli': point['p_alveoli'],
                'saturation_traditional': traditional_saturation,
                'saturation_ml': ml_saturation,
                'saturation_final': final_saturation,
                'oxygen_content': final_saturation * self.blood_composition.oxygen_carrying_capacity()
            })
            
        # Update ML models
        self._update_saturation_model(exchange_data)
        return exchange_data

    def _update_saturation_model(self, exchange_data: List[Dict]):
        """Update saturation prediction model"""
        X = np.array([[
            feature['properties']['heart_rate'],
            feature['properties']['speed'],
            feature['properties']['cadence'],
            feature['properties']['altitude'],
            self.organ_metrics.age,
            self.organ_metrics.weight,
            self.organ_metrics.height,
            1 if self.organ_metrics.gender == 'male' else 0
        ] for feature in self.track_data['features']])
        
        y_saturation = np.array([d['saturation_traditional'] for d in exchange_data])
        
        # Update saturation model
        self.ml_manager.scalers['saturation'].partial_fit(X)
        X_scaled = self.ml_manager.scalers['saturation'].transform(X)
        self.ml_manager.models['saturation'].fit(X_scaled, y_saturation)

    def visualize_oxygen_cascade(self, output_file: str = 'oxygen_cascade.png'):
        """Visualize oxygen pressure cascade through respiratory system"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for visualization")
            
        gradient_data = self.calculate_oxygen_gradient()
        timestamps = [datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00')) 
                     for point in gradient_data]
        
        # Create pressure arrays
        p_atmosphere = [point['p_atmosphere'] for point in gradient_data]
        p_o2 = [point['p_o2_atmosphere'] for point in gradient_data]
        p_trachea = [point['p_trachea'] for point in gradient_data]
        p_bronchi = [point['p_bronchi'] for point in gradient_data]
        p_alveoli = [point['p_alveoli'] for point in gradient_data]
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Plot pressure curves
        plt.plot(timestamps, p_atmosphere, label='Atmospheric Pressure', linewidth=2)
        plt.plot(timestamps, p_o2, label='O₂ Partial Pressure', linewidth=2)
        plt.plot(timestamps, p_trachea, label='Tracheal pO₂', linewidth=2)
        plt.plot(timestamps, p_bronchi, label='Bronchial pO₂', linewidth=2)
        plt.plot(timestamps, p_alveoli, label='Alveolar pO₂', linewidth=2)
        
        plt.title('Oxygen Pressure Cascade Through Respiratory System')
        plt.xlabel('Time')
        plt.ylabel('Pressure (kPa)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()
    
    def visualize_oxygen_consumption(self, output_file: str = 'oxygen_consumption.png'):
        """Visualize oxygen consumption and related parameters"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for visualization")
            
        consumption_data = self.calculate_oxygen_consumption()
        exchange_data = self.calculate_alveolar_exchange()
        
        timestamps = [datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00')) 
                     for point in consumption_data]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        sns.set_style("whitegrid")
        
        # Plot oxygen consumption and heart rate
        ax1.plot([point['vo2_final'] for point in consumption_data], 
                 label='VO₂ Final', color='blue')
        ax1_twin = ax1.twinx()
        ax1_twin.plot([point['heart_rate'] for point in consumption_data], 
                     label='Heart Rate', color='red', linestyle='--')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('VO₂ (mL/kg/min)')
        ax1_twin.set_ylabel('Heart Rate (bpm)')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot hemoglobin saturation and oxygen content
        ax2.plot([point['saturation_final'] * 100 for point in exchange_data], 
                 label='Hb Saturation (%)', color='green')
        ax2_twin = ax2.twinx()
        ax2_twin.plot([point['oxygen_content'] for point in exchange_data], 
                     label='O₂ Content', color='purple', linestyle='--')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Saturation (%)')
        ax2_twin.set_ylabel('O₂ Content (mL/dL)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def analyze_track(self, track_data):
        """
        Analyze oxygen uptake rate along a track.
        
        Args:
            track_data: Track data points from JSON file
            
        Returns:
            Dictionary with oxygen uptake analysis results
        """
        try:
            # Use the module-level logger instead of creating a new one
            
            if isinstance(track_data, list):
                logger.info(f"Track data is a list with {len(track_data)} entries")
                if len(track_data) > 0:
                    logger.info(f"First entry sample: {str(track_data[0])[:500]}...")
            else:
                logger.info(f"Track data is not a list: {type(track_data)}")
                logger.info(f"Track data sample: {str(track_data)[:500]}...")
            
            # Process data points based on the track_data structure
            vo2_values = []
            hrmax = 220 - self.age  # Estimated max heart rate
            
            # For continuous learning - collect actual VO2 measurements if available
            actual_vo2_measurements = []
            
            # Extract data points from track_data based on its structure
            data_points = []
            
            if isinstance(track_data, list):
                # It's already a list of data points
                data_points = track_data
            elif isinstance(track_data, dict):
                # Check for common keys that might contain the list of points
                for key in ["features", "coordinates", "points", "data"]:
                    if key in track_data and isinstance(track_data[key], list):
                        data_points = track_data[key]
                        logger.info(f"Found {len(data_points)} data points in '{key}' key")
                        break
                
                # If we still don't have data points, use the track_data itself as a single point
                if not data_points:
                    data_points = [track_data]
            
            # Process each data point
            for point in data_points:
                # Extract relevant metrics from the data point
                metrics = self._extract_metrics_from_point(point)
                
                # Calculate oxygen uptake
                vo2 = self._calculate_vo2(metrics)
                
                # Store the result
                vo2_values.append({
                    "timestamp": metrics.get("timestamp", ""),
                    "vo2": vo2,
                    "percentage_vo2max": (vo2 / self.vo2max_estimated) * 100 if self.vo2max_estimated > 0 else 0,
                    "heart_rate": metrics.get("heart_rate", 0),
                    "percentage_hrmax": (metrics.get("heart_rate", 0) / hrmax) * 100 if hrmax > 0 else 0,
                    "energy_expenditure": vo2 * self.mass * 0.005,  # Simplified: vo2 (ml/kg/min) * mass (kg) * 0.005 kcal/ml
                    "metrics_used": metrics
                })
                
                # Check if this point has actual measured VO2 (for continuous learning)
                if isinstance(point, dict) and "measured_vo2" in point and point["measured_vo2"] is not None:
                    try:
                        measured_vo2 = float(point["measured_vo2"])
                        actual_vo2_measurements.append((metrics, measured_vo2))
                    except (ValueError, TypeError):
                        pass
            
            # Calculate summary statistics
            if vo2_values:
                avg_vo2 = sum(v["vo2"] for v in vo2_values) / len(vo2_values)
                max_vo2 = max(v["vo2"] for v in vo2_values)
                avg_percentage_vo2max = sum(v["percentage_vo2max"] for v in vo2_values) / len(vo2_values)
                max_percentage_vo2max = max(v["percentage_vo2max"] for v in vo2_values)
                total_energy = sum(v["energy_expenditure"] for v in vo2_values)
            else:
                avg_vo2 = 0
                max_vo2 = 0
                avg_percentage_vo2max = 0
                max_percentage_vo2max = 0
                total_energy = 0
            
            # Update continuous learning if we have actual VO2 measurements
            if actual_vo2_measurements:
                self._update_continuous_learning(actual_vo2_measurements)
            
            # Return the results
            return {
                "vo2_values": vo2_values[:20],  # Limit to 20 entries for brevity
                "summary": {
                    "avg_vo2": avg_vo2,
                    "max_vo2": max_vo2,
                    "avg_percentage_vo2max": avg_percentage_vo2max,
                    "max_percentage_vo2max": max_percentage_vo2max,
                    "estimated_vo2max": self.vo2max_estimated,
                    "total_energy_expenditure": total_energy,
                    "total_points_analyzed": len(vo2_values)
                }
            }
            
        except Exception as e:
            logger.error(f"Oxygen uptake track analysis failed: {e}", exc_info=True)
            return {"error": str(e), "vo2_values": [], "summary": {}}
    
    def _extract_metrics_from_point(self, point):
        """Extract relevant metrics from a data point."""
        metrics = {}
        
        try:
            # Check the type of point
            if not isinstance(point, dict):
                # Try to convert to dict if it's a string
                if isinstance(point, str):
                    try:
                        import json
                        point = json.loads(point)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse point as JSON: {point}")
                        return metrics
                else:
                    logger.warning(f"Point is not a dictionary: {type(point)}")
                    return metrics
            
            # Get timestamp
            for ts_key in ['timestamp', 'time', 'datetime', 'date']:
                if ts_key in point and point[ts_key]:
                    metrics['timestamp'] = str(point[ts_key])
                    break
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Get speed (m/s)
            for speed_key in ['speed', 'velocity', 'pace']:
                if speed_key in point and point[speed_key] is not None:
                    try:
                        metrics['speed'] = float(point[speed_key])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get heart rate (bpm)
            for hr_key in ['heart_rate', 'hr', 'heartrate', 'pulse']:
                if hr_key in point and point[hr_key] is not None:
                    try:
                        metrics['heart_rate'] = float(point[hr_key])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get grade/slope (%)
            for grade_key in ['grade', 'slope', 'incline', 'gradient']:
                if grade_key in point and point[grade_key] is not None:
                    try:
                        metrics['grade'] = float(point[grade_key])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get altitude (m)
            for alt_key in ['altitude', 'elevation', 'height']:
                if alt_key in point and point[alt_key] is not None:
                    try:
                        metrics['altitude'] = float(point[alt_key])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Get cadence (steps/min)
            for cadence_key in ['cadence', 'step_rate', 'steps_per_minute', 'spm']:
                if cadence_key in point and point[cadence_key] is not None:
                    try:
                        metrics['cadence'] = float(point[cadence_key])
                        break
                    except (ValueError, TypeError):
                        pass
                        
            # Check for nested properties
            if 'properties' in point and isinstance(point['properties'], dict):
                props = point['properties']
                
                # Check all the same keys in the properties
                for metric_name, keys in {
                    'speed': ['speed', 'velocity', 'pace'],
                    'heart_rate': ['heart_rate', 'hr', 'heartrate', 'pulse'],
                    'grade': ['grade', 'slope', 'incline', 'gradient'],
                    'altitude': ['altitude', 'elevation', 'height'],
                    'cadence': ['cadence', 'step_rate', 'steps_per_minute', 'spm']
                }.items():
                    if metric_name not in metrics or metrics[metric_name] == 0:
                        for key in keys:
                            if key in props and props[key] is not None:
                                try:
                                    metrics[metric_name] = float(props[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                                    
            # Set defaults for missing metrics
            if 'speed' not in metrics or metrics['speed'] == 0:
                metrics['speed'] = 3.0  # Default running speed (m/s)
                
            if 'heart_rate' not in metrics:
                # Estimate heart rate from speed using a simple model
                metrics['heart_rate'] = 80 + 20 * metrics['speed']  # Very rough estimate
                
            if 'grade' not in metrics:
                metrics['grade'] = 0.0  # Flat terrain
                
            if 'altitude' not in metrics:
                metrics['altitude'] = 0.0  # Sea level
                
            if 'cadence' not in metrics:
                # Estimate cadence from speed
                metrics['cadence'] = 150 + 10 * metrics['speed']  # Very rough estimate
            
            # Log the extracted metrics at debug level
            logger.debug(f"Extracted metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics from point: {e}", exc_info=True)
            # Return minimal default metrics
            return {
                'timestamp': datetime.now().isoformat(),
                'speed': 3.0,
                'heart_rate': 140,
                'grade': 0.0,
                'altitude': 0.0,
                'cadence': 170
            }
    
    def _calculate_vo2(self, metrics):
        """
        Calculate VO2 based on exercise metrics.
        
        Args:
            metrics: Dictionary with exercise metrics
            
        Returns:
            VO2 in ml/kg/min
        """
        # Extract metrics with fallbacks
        speed = metrics.get("speed", 0.0)  # m/s
        grade = metrics.get("grade", 0.0)  # decimal
        heart_rate = metrics.get("heart_rate", 0.0)  # bpm
        
        # VO2 using ACSM running equation (if speed is available)
        vo2_speed = 0.0
        if speed > 0:
            # Convert to min/km for ACSM equation
            pace_min_km = 16.67 / speed if speed > 0 else 0
            # ACSM running equation: VO2 (ml/kg/min) = 3.5 + (0.2 * speed) + (0.9 * speed * grade)
            speed_m_min = speed * 60
            vo2_speed = 3.5 + (0.2 * speed_m_min) + (0.9 * speed_m_min * grade / 100)
        
        # VO2 using heart rate (if available)
        vo2_hr = 0.0
        if heart_rate > 0:
            # Estimate VO2 from heart rate using a simplified relationship
            max_hr = 220 - self.age
            hr_reserve = max_hr - self.resting_vo2 / 10  # Rough estimate of resting HR
            vo2_reserve = self.vo2max_estimated - 3.5
            vo2_hr = 3.5 + (heart_rate - self.resting_vo2 / 10) / hr_reserve * vo2_reserve if hr_reserve > 0 else 0
        
        # Use the most reliable estimate available, with preference for speed-based calculation
        if vo2_speed > 0:
            return vo2_speed
        elif vo2_hr > 0:
            return vo2_hr
        else:
            # Default to a reasonable value if no metrics available
            return 10.0  # ml/kg/min (light activity)

    def _update_continuous_learning(self, vo2_measurements):
        """Update continuous learning system with actual VO2 measurements."""
        try:
            from federation.continuous_learning import continuous_learning_manager
            
            # Format the data for model training
            features = []
            targets = []
            
            for metrics, actual_vo2 in vo2_measurements:
                # Extract feature vector from metrics
                feature = [
                    float(metrics.get("speed", 0)),
                    float(metrics.get("heart_rate", 0)),
                    float(metrics.get("grade", 0)),
                    float(metrics.get("altitude", 0)),
                    float(metrics.get("cadence", 0)),
                    float(self.age),
                    float(self.mass),
                    float(self.height),
                    1.0 if self.gender == 'male' else 0.0  # Gender as binary feature
                ]
                
                features.append(feature)
                targets.append(actual_vo2)
            
            # Store the data for future model training
            run_data = {
                "oxygen_training_data": {
                    "features": features,
                    "targets": targets
                }
            }
            
            # Generate a unique ID
            import hashlib
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            data_hash = hashlib.md5(str(features[:3]).encode()).hexdigest()[:8]
            training_id = f"oxygen_training_{timestamp.replace(':', '-')}_{data_hash}"
            
            # Store the training data
            continuous_learning_manager.data_collector.store_run_data(
                training_id, run_data, {"type": "oxygen_training_data"})
            
            logger.info(f"Stored {len(features)} new training samples for oxygen uptake model")
            
            # Schedule model retraining if we have enough new data
            if len(features) > 10:
                continuous_learning_manager.schedule_retraining(
                    "oxygen_uptake", "oxygen", interval_hours=24)
                logger.info("Scheduled oxygen uptake model retraining")
                
        except Exception as e:
            logger.warning(f"Could not update continuous learning system: {e}")

# Example usage and visualization code will be added
