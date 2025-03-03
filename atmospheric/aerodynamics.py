import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import requests
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

# Standardize logger configuration
logger = logging.getLogger(__name__)

class WeatherDataFetcher:
    """Fetch weather data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Verify API key on initialization
        self.verify_api_key()
        
    def verify_api_key(self) -> bool:
        """Verify that the API key is valid by making a test request"""
        try:
            test_params = {
                'lat': 40.7128,  # New York City coordinates for test
                'lon': -74.0060,
                'appid': self.api_key,
                'units': 'metric'  # Use metric units for consistency
            }
            test_url = f"{self.base_url}/weather"
            
            logger.info(f"Verifying OpenWeatherMap API key with test request to {test_url}")
            
            response = requests.get(test_url, params=test_params, timeout=10)
            
            # Check if the request was successful
            if response.status_code == 200:
                logger.info("OpenWeatherMap API key verification successful")
                return True
            elif response.status_code == 401:
                logger.error("OpenWeatherMap API key invalid or unauthorized")
                print(f"ERROR: OpenWeatherMap API key invalid (401 Unauthorized)")
                return False
            else:
                logger.error(f"OpenWeatherMap API returned unexpected status code: {response.status_code}")
                print(f"ERROR: OpenWeatherMap API returned status code {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying OpenWeatherMap API key: {e}", exc_info=True)
            print(f"ERROR connecting to OpenWeatherMap API: {str(e)}")
            return False
    
    def get_weather_data(self, lat: float, lon: float, timestamp: Optional[Union[str, datetime]] = None) -> Dict:
        """
        Fetch weather data for given latitude, longitude and timestamp.
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Timestamp (optional - if None, current weather is fetched)
            
        Returns:
            Dictionary with weather data
        """
        try:
            response = requests.get(
                f"{self.base_url}/weather",
                params={
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'
                },
                timeout=15
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                logger.info("Weather data fetched successfully")
                weather_data = response.json()
                
                # Log a sample of the data for debugging
                logger.debug(f"Sample of weather data: {str(weather_data)[:500]}...")
                
                # Extract relevant weather information and standardize format
                processed_data = self._process_weather_data(weather_data)
                return processed_data
            else:
                logger.error(f"Error fetching weather data: {response.status_code} - {response.text}")
                print(f"ERROR: Failed to fetch weather data. Status code: {response.status_code}")
                
                # Return default values as fallback
                return {
                    'temperature': 15.0,  # 15°C
                    'pressure': 1013.25,  # Standard atmospheric pressure (hPa)
                    'humidity': 50.0,     # 50% humidity
                    'wind_speed': 0.0,    # No wind
                    'wind_direction': 0,  # North
                    'rain': 0.0,          # No rain
                    'error': f"API error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Exception while fetching weather data: {e}", exc_info=True)
            print(f"ERROR: Failed to fetch weather data: {str(e)}")
            
            # Return default values as fallback
            return {
                'temperature': 15.0,
                'pressure': 1013.25,
                'humidity': 50.0,
                'wind_speed': 0.0,
                'wind_direction': 0,
                'rain': 0.0,
                'error': f"Exception: {str(e)}"
            }
            
    def _process_weather_data(self, data: Dict) -> Dict:
        """Process the raw weather data from the API and standardize format"""
        try:
            result = {}
            
            # Regular weather API format
            if 'main' in data:
                # Standard weather API response
                result['temperature'] = data.get('main', {}).get('temp', 15.0)
                result['pressure'] = data.get('main', {}).get('pressure', 1013.25)
                result['humidity'] = data.get('main', {}).get('humidity', 50.0)
                result['wind_speed'] = data.get('wind', {}).get('speed', 0.0)
                result['wind_direction'] = data.get('wind', {}).get('deg', 0)
                result['rain'] = data.get('rain', {}).get('1h', 0.0) if 'rain' in data else 0.0
            elif 'current' in data:
                # OneCall API format
                weather = data['current']
                result['temperature'] = weather.get('temp', 15.0)
                result['pressure'] = weather.get('pressure', 1013.25)
                result['humidity'] = weather.get('humidity', 50.0)
                result['wind_speed'] = weather.get('wind_speed', 0.0)
                result['wind_direction'] = weather.get('wind_deg', 0)
                result['rain'] = weather.get('rain', {}).get('1h', 0.0) if 'rain' in weather else 0.0
            elif 'data' in data and len(data['data']) > 0:
                # Historical format
                weather = data['data'][0]
                result['temperature'] = weather.get('temp', 15.0)
                result['pressure'] = weather.get('pressure', 1013.25)
                result['humidity'] = weather.get('humidity', 50.0)
                result['wind_speed'] = weather.get('wind_speed', 0.0)
                result['wind_direction'] = weather.get('wind_deg', 0)
                result['rain'] = weather.get('rain', 0.0)
            else:
                # Unknown format - log the data structure and use defaults
                logger.warning(f"Unknown API response format. Keys: {list(data.keys())}")
                result['temperature'] = 15.0
                result['pressure'] = 1013.25
                result['humidity'] = 50.0
                result['wind_speed'] = 0.0
                result['wind_direction'] = 0
                result['rain'] = 0.0
                result['error'] = "Unknown API response format"
            
            # Add source data structure info for debugging
            result['data_keys'] = list(data.keys())
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing weather data: {e}", exc_info=True)
            # Return default values
            return {
                'temperature': 15.0,
                'pressure': 1013.25,
                'humidity': 50.0,
                'wind_speed': 0.0,
                'wind_direction': 0,
                'rain': 0.0,
                'error': f"Data processing error: {str(e)}"
            }

class MLAerodynamicsManager:
    """Manage ML models for aerodynamic predictions"""
    
    def __init__(self, model_dir: str = 'aero_models'):
        self.model_dir = model_dir
        self.models = {
            'drag': self._load_or_create_model('drag_model.joblib'),
            'wind_effect': self._load_or_create_model('wind_effect_model.joblib'),
            'power_output': self._load_or_create_model('power_output_model.joblib')
        }
        self.scalers = {
            'drag': self._load_or_create_scaler('drag_scaler.joblib'),
            'wind': self._load_or_create_scaler('wind_scaler.joblib'),
            'power': self._load_or_create_scaler('power_scaler.joblib')
        }
    
    def _load_or_create_model(self, filename: str) -> RandomForestRegressor:
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
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
            joblib.dump(model, os.path.join(self.model_dir, f'{name}_model.joblib'))
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(self.model_dir, f'{name}_scaler.joblib'))

class AerodynamicsCalculator:
    """Calculate aerodynamic effects on runner"""
    
    # Constants
    AIR_DENSITY_SL = 1.225  # kg/m³ at sea level
    GRAVITY = 9.81  # m/s²
    DRAG_COEFFICIENT = 0.9  # Typical for runner
    
    def __init__(self, weight: float, height: float, weather_api_key: str):
        self.weight = weight
        self.height = height
        self.bsa = self._calculate_bsa()
        self.weather_fetcher = WeatherDataFetcher(weather_api_key)
        self.ml_manager = MLAerodynamicsManager()
    
    def _calculate_bsa(self) -> float:
        """Calculate body surface area using DuBois formula"""
        height_m = self.height / 100
        return 0.007184 * (self.weight ** 0.425) * (height_m ** 0.725)
    
    def _prepare_features(self, track_point: Dict, weather_data: Dict) -> np.ndarray:
        """Prepare feature vector for ML models"""
        return np.array([
            track_point['properties']['speed'],
            track_point['properties']['altitude'],
            weather_data['wind']['speed'],
            weather_data['wind']['deg'],
            weather_data['main']['temp'],
            weather_data['main']['pressure'],
            weather_data['main']['humidity'],
            self.weight,
            self.height,
            self.bsa
        ]).reshape(1, -1)
    
    def calculate_air_density(self, pressure: float, temperature: float, humidity: float) -> float:
        """Calculate air density using ideal gas law with humidity correction"""
        # Convert pressure to Pa and temperature to K
        pressure_pa = pressure * 100
        temp_k = temperature + 273.15
        
        # Water vapor pressure using Magnus formula
        vapor_pressure = 610.94 * np.exp(17.625 * temperature / (temperature + 243.04))
        
        # Humidity correction
        actual_vapor_pressure = vapor_pressure * (humidity / 100)
        
        # Dry air density
        dry_air_density = (pressure_pa - actual_vapor_pressure) / (287.05 * temp_k)
        
        # Water vapor density
        vapor_density = actual_vapor_pressure / (461.495 * temp_k)
        
        return dry_air_density + vapor_density
    
    def calculate_drag_force(self, speed: float, air_density: float) -> Tuple[float, float]:
        """
        Calculate the drag force.
        
        Args:
            speed: Current speed in m/s
            air_density: Air density in kg/m³
            
        Returns:
            Tuple of (traditional_drag, machine_learning_adjusted_drag)
        """
        # Check if the scaler is fitted, if not, fit it
        if not hasattr(self.ml_manager, 'scalers') or 'drag' not in self.ml_manager.scalers:
            self._initialize_scalers()
        elif hasattr(self.ml_manager.scalers['drag'], 'n_features_in_'):
            # Scaler exists but need to check if it's fitted
            try:
                # This will raise an exception if not fitted
                self.ml_manager.scalers['drag'].n_features_in_
            except:
                self._initialize_scalers()
        
        # Calculate frontal area
        frontal_area = self.bsa * 0.5  # Approximate frontal area
        
        # Calculate traditional drag force using the drag equation
        # F_d = 0.5 * ρ * v² * A * C_d
        traditional_drag = 0.5 * air_density * (speed ** 2) * frontal_area * self.DRAG_COEFFICIENT
        
        # Use machine learning to adjust drag force based on features
        features = np.array([self.weight, self.height, speed, air_density])
        
        try:
            # Scale the features
            scaled_features = self.ml_manager.scalers['drag'].transform(features.reshape(1, -1))
            
            # Predict drag adjustment
            drag_adjustment = self.ml_manager.models['drag'].predict(scaled_features)[0]
            
            # Apply adjustment to traditional drag (could be positive or negative)
            final_drag = max(0, traditional_drag + drag_adjustment)
        except Exception as e:
            logger.warning(f"ML drag adjustment failed: {e}")
            final_drag = traditional_drag
        
        return traditional_drag, final_drag
    
    def _initialize_scalers(self):
        """Initialize and fit scalers with sample data if they're not already fitted."""
        logger.info("Initializing aerodynamics scalers with sample data")
        
        # Create a StandardScaler for drag features
        from sklearn.preprocessing import StandardScaler
        
        # Create sample data covering a range of possible values
        sample_data = []
        
        # Add variety of athlete sizes
        for mass in [50.0, 60.0, 70.0, 80.0, 90.0]:
            for height in [1.5, 1.65, 1.75, 1.85, 1.95]:
                # Add variety of speeds and air densities
                for speed in [2.0, 4.0, 6.0, 8.0, 10.0]:
                    for density in [1.0, 1.1, 1.2, 1.3, 1.4]:
                        sample_data.append([mass, height, speed, density])
        
        # Convert to numpy array
        sample_data = np.array(sample_data)
        
        # Create and fit the scaler
        scaler = StandardScaler()
        scaler.fit(sample_data)
        
        # Store in the ml_manager
        if not hasattr(self.ml_manager, 'scalers'):
            self.ml_manager.scalers = {}
        self.ml_manager.scalers['drag'] = scaler
        
        logger.info(f"Scaler initialized with {len(sample_data)} sample points")
        
        # If models don't exist, create default ones
        if not hasattr(self.ml_manager, 'models') or 'drag' not in self.ml_manager.models:
            self._initialize_models(sample_data)
            
    def _initialize_models(self, sample_data):
        """Initialize machine learning models if they don't exist."""
        logger.info("Initializing machine learning models with default behavior")
        
        # For simplicity, we'll use a LinearRegression model that returns small values
        from sklearn.linear_model import LinearRegression
        
        # Create target values that are near zero to have minimal impact initially
        # Small random noise around zero
        np.random.seed(42)  # For reproducibility
        y = np.random.normal(0, 0.01, size=len(sample_data))
        
        # Create and fit the model
        model = LinearRegression()
        model.fit(sample_data, y)
        
        # Store the model
        if not hasattr(self.ml_manager, 'models'):
            self.ml_manager.models = {}
        self.ml_manager.models['drag'] = model
        
        logger.info("Default model initialized")
        
    def test_api_connection(self):
        """Test if the weather API connection is working."""
        try:
            # Test with default coordinates
            lat, lon = 40.7128, -74.0060  # New York coordinates
            
            logger.info(f"Testing weather API connection with coordinates: lat={lat}, lon={lon}")
            
            # Use the weather_fetcher instance we already have
            weather_data = self.weather_fetcher.get_weather_data(lat, lon)
            
            # Log the response for debugging
            logger.info(f"Weather API test response: {str(weather_data)[:500]}...")
            
            # If we got data back without an error key, the connection is working
            if 'error' not in weather_data:
                logger.info("Weather API connection test successful")
                print("Weather API connection successful")
                return True
            else:
                logger.error(f"Weather API connection test failed: {weather_data.get('error', 'Unknown error')}")
                print(f"Weather API connection failed: {weather_data.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Weather API connection test failed with exception: {e}", exc_info=True)
            print(f"ERROR: Weather API connection test failed: {str(e)}")
            return False
            
    def fit_models(self, track_data):
        """
        Fit machine learning models with track data.
        
        Args:
            track_data: Track data points
        """
        logger.info("Fitting aerodynamics models with track data")
        
        # Initialize scalers first
        self._initialize_scalers()
        
        # For now, we're using pre-initialized models
        # In a real implementation, you would extract features from track_data
        # and train the models with this data
        
        logger.info("Models initialized with default behavior")
    
    def analyze_track(self, track_data: Dict) -> List[Dict[str, float]]:
        """
        Analyze aerodynamic effects along a track.
        
        Args:
            track_data: Track data points
            
        Returns:
            List of dictionaries with aerodynamic metrics
        """
        try:
            logger.info("Analyzing aerodynamic effects along track")
            
            # Log the structure of track_data to understand its format
            if isinstance(track_data, list):
                logger.info(f"Track data is a list with {len(track_data)} entries")
                if len(track_data) > 0:
                    logger.info(f"First entry sample: {str(track_data[0])[:500]}...")
            else:
                logger.info(f"Track data is not a list: {type(track_data)}")
                logger.info(f"Track data sample: {str(track_data)[:500]}...")
            
            # Initialize the models and scalers
            self._initialize_scalers()
            
            # Try to use continuous learning models if available
            self._load_continuous_learning_models()
            
            # Process the track data based on its structure
            results = []
            actual_drag_values = []  # For error calculation if available
            
            # Check if we have features or coordinates
            if isinstance(track_data, list):
                # List of data points (time series)
                for i, point in enumerate(track_data):
                    try:
                        point_analysis = self.analyze_track_point(point)
                        results.append(point_analysis)
                        
                        # If the point has actual drag values (from wind tunnel or other measurements)
                        if isinstance(point, dict) and "actual_drag" in point:
                            try:
                                actual_drag = float(point["actual_drag"])
                                predicted_drag = point_analysis["drag_force"]
                                actual_drag_values.append((actual_drag, predicted_drag))
                            except (ValueError, TypeError):
                                pass
                    except Exception as e:
                        logger.error(f"Error analyzing point {i}: {e}")
                        # Add default values to maintain sequence integrity
                        results.append({
                            "drag_force": 0.0,
                            "drag_power": 0.0,
                            "air_density": 1.225,  # Standard air density
                            "wind_effect": 0.0,
                            "altitude_effect": 0.0,
                            "atmospheric_power_loss": 0.0,
                            "effective_power": 0.0
                        })
            elif isinstance(track_data, dict):
                # Single data point or different structure
                logger.info("Track data is a dictionary, trying to analyze as single point")
                
                # Try to analyze as a single point
                try:
                    point_analysis = self.analyze_track_point(track_data)
                    results.append(point_analysis)
                except Exception as e:
                    logger.error(f"Error analyzing track as single point: {e}")
                    results.append({
                        "drag_force": 0.0,
                        "drag_power": 0.0,
                        "air_density": 1.225,
                        "wind_effect": 0.0,
                        "altitude_effect": 0.0,
                        "atmospheric_power_loss": 0.0,
                        "effective_power": 0.0
                    })
                    
                # Check if there's a "features" or "coordinates" key with a list
                for key in ["features", "coordinates", "points", "data"]:
                    if key in track_data and isinstance(track_data[key], list):
                        logger.info(f"Found list in key '{key}' with {len(track_data[key])} entries")
                        for i, point in enumerate(track_data[key]):
                            try:
                                point_analysis = self.analyze_track_point(point)
                                results.append(point_analysis)
                                
                                # Check for actual drag values
                                if isinstance(point, dict) and "actual_drag" in point:
                                    try:
                                        actual_drag = float(point["actual_drag"])
                                        predicted_drag = point_analysis["drag_force"]
                                        actual_drag_values.append((actual_drag, predicted_drag))
                                    except (ValueError, TypeError):
                                        pass
                            except Exception as e:
                                logger.error(f"Error analyzing point {i} from '{key}': {e}")
                                results.append({
                                    "drag_force": 0.0,
                                    "drag_power": 0.0,
                                    "air_density": 1.225,
                                    "wind_effect": 0.0,
                                    "altitude_effect": 0.0,
                                    "atmospheric_power_loss": 0.0,
                                    "effective_power": 0.0
                                })
            else:
                logger.error(f"Unsupported track data format: {type(track_data)}")
                
            logger.info(f"Analyzed {len(results)} points in track")
            
            # Calculate error metrics if we have actual values
            if actual_drag_values:
                import numpy as np
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                actuals, preds = zip(*actual_drag_values)
                
                error_metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
                    "mae": float(mean_absolute_error(actuals, preds)),
                    "sample_count": len(actual_drag_values)
                }
                
                # Add error metrics to results
                results_with_metrics = {
                    "points": results,
                    "error_metrics": error_metrics
                }
                
                # Update model with new data for continuous learning
                self._update_continuous_learning(track_data, actual_drag_values)
                
                return results_with_metrics
            else:
                return results
            
        except Exception as e:
            logger.error(f"Aerodynamics track analysis failed: {e}", exc_info=True)
            return []
            
    def analyze_track_point(self, track_point: Dict) -> Dict[str, float]:
        """
        Analyze aerodynamic effects for a single track point.
        
        Args:
            track_point: Data point from track
            
        Returns:
            Dictionary with aerodynamic metrics
        """
        try:
            # Log what we received to understand the structure
            logger.debug(f"Analyzing track point: {track_point}")
            
            # Extract speed from track point or use a default
            speed = 0.0
            if isinstance(track_point, dict):
                # Try different keys that might contain speed
                for speed_key in ['speed', 'velocity', 'pace']:
                    if speed_key in track_point and track_point[speed_key] is not None:
                        try:
                            speed = float(track_point[speed_key])
                            break
                        except (ValueError, TypeError):
                            pass
                            
                # If we couldn't find speed, check properties or nested dictionaries
                if speed == 0.0 and 'properties' in track_point:
                    props = track_point['properties']
                    if isinstance(props, dict):
                        for speed_key in ['speed', 'velocity', 'pace']:
                            if speed_key in props and props[speed_key] is not None:
                                try:
                                    speed = float(props[speed_key])
                                    break
                                except (ValueError, TypeError):
                                    pass
            elif isinstance(track_point, (int, float)):
                # If the track point is just a number, assume it's the speed
                speed = float(track_point)
                
            # Default to a reasonable running speed if we couldn't extract it
            if speed == 0.0:
                speed = 3.0  # m/s (approximately 10.8 km/h)
                logger.warning(f"Couldn't extract speed from track point, using default: {speed} m/s")
            
            # Extract location to determine weather conditions
            latitude, longitude, altitude = 0.0, 0.0, 0.0
            
            if isinstance(track_point, dict):
                # Try to get coordinates
                if 'latitude' in track_point and 'longitude' in track_point:
                    try:
                        latitude = float(track_point['latitude'])
                        longitude = float(track_point['longitude'])
                    except (ValueError, TypeError):
                        pass
                elif 'lat' in track_point and 'lon' in track_point:
                    try:
                        latitude = float(track_point['lat'])
                        longitude = float(track_point['lon'])
                    except (ValueError, TypeError):
                        pass
                elif 'coordinates' in track_point and isinstance(track_point['coordinates'], list):
                    coords = track_point['coordinates']
                    if len(coords) >= 2:
                        try:
                            # GeoJSON format is [longitude, latitude]
                            longitude = float(coords[0])
                            latitude = float(coords[1])
                            if len(coords) >= 3:
                                altitude = float(coords[2])
                        except (ValueError, TypeError):
                            pass
                
                # Get altitude if not already set
                if altitude == 0.0 and 'altitude' in track_point:
                    try:
                        altitude = float(track_point['altitude'])
                    except (ValueError, TypeError):
                        pass
                elif altitude == 0.0 and 'elevation' in track_point:
                    try:
                        altitude = float(track_point['elevation'])
                    except (ValueError, TypeError):
                        pass
            
            # We need air density for drag calculation
            # If we have valid coordinates, we could get this from a weather API
            # For simplicity, we'll estimate based on altitude
            air_density = self.AIR_DENSITY_SL * (1 - 2.25577e-5 * altitude) ** 5.25588 if altitude else self.AIR_DENSITY_SL
            
            # Calculate drag force
            traditional_drag, final_drag = self.calculate_drag_force(speed, air_density)
            
            # Calculate power lost to drag
            drag_power = final_drag * speed
            
            # Calculate wind effect (simplified)
            wind_effect = 0.0  # We'd need wind data from weather API
            
            # Calculate altitude effect on performance
            altitude_effect = -0.01 * altitude / 100  # Simplified model: -1% per 100m
            
            # Calculate total atmospheric power loss
            atmospheric_power_loss = drag_power * (1 + altitude_effect)
            
            # Calculate effective power (what's left after atmospheric losses)
            # This would require knowing the athlete's total power output
            # For now, we'll just estimate based on weight and speed
            total_power = self.weight * 4.0 * speed  # Very rough estimate
            effective_power = max(0, total_power - atmospheric_power_loss)
            
            # Return results
            return {
                "drag_force": float(final_drag),
                "drag_power": float(drag_power),
                "air_density": float(air_density),
                "wind_effect": float(wind_effect),
                "altitude_effect": float(altitude_effect),
                "atmospheric_power_loss": float(atmospheric_power_loss),
                "effective_power": float(effective_power)
            }
            
        except Exception as e:
            logger.error(f"Aerodynamics point analysis failed: {e}", exc_info=True)
            raise

    def _load_continuous_learning_models(self):
        """Try to load models from the continuous learning system."""
        try:
            from federation.continuous_learning import continuous_learning_manager
            
            # Try to get the aerodynamics drag model
            model = continuous_learning_manager.model_registry.get_model("aerodynamics_drag")
            
            if model is not None:
                logger.info("Loaded aerodynamics model from continuous learning system")
                self.ml_manager.models["drag"] = model
                return True
        except Exception as e:
            logger.warning(f"Could not load models from continuous learning system: {e}")
        
        return False
        
    def _update_continuous_learning(self, track_data, drag_values=None):
        """Update the continuous learning system with new data."""
        try:
            from federation.continuous_learning import continuous_learning_manager
            
            # Extract features for model updates
            features = []
            targets = []
            
            # If we already have processed drag values, use those
            if drag_values:
                # These are (actual, predicted) pairs
                for actual_drag, _ in drag_values:
                    # We need to extract the corresponding features
                    # This is a simplified example - would need to match features to actuals
                    targets.append(actual_drag)
            
            # Otherwise, try to extract from track_data
            elif isinstance(track_data, list):
                for point in track_data:
                    if isinstance(point, dict) and "actual_drag" in point and point["actual_drag"] is not None:
                        # Extract feature vector
                        speed = float(point.get("speed", 0) or 0)
                        air_density = float(point.get("air_density", 1.225) or 1.225)
                        altitude = float(point.get("altitude", 0) or 0)
                        wind_speed = float(point.get("wind_speed", 0) or 0)
                        
                        features.append([speed, air_density, altitude, wind_speed])
                        targets.append(float(point["actual_drag"]))
            
            # If we have new training data, store it for future model training
            if features and targets:
                # Store the data for future model training
                run_data = {
                    "aerodynamics_training_data": {
                        "features": features,
                        "targets": targets
                    }
                }
                
                # Generate a unique ID for this training data
                import hashlib
                from datetime import datetime
                timestamp = datetime.now().isoformat()
                data_hash = hashlib.md5(str(features).encode()).hexdigest()[:8]
                training_id = f"aero_training_{timestamp.replace(':', '-')}_{data_hash}"
                
                # Store the training data
                continuous_learning_manager.data_collector.store_run_data(
                    training_id, run_data, {"type": "aerodynamics_training_data"})
                
                logger.info(f"Stored {len(features)} new training samples for aerodynamics model")
                
                # Schedule model retraining if we have enough new data
                if len(features) > 20:  # Only retrain if we have a significant amount of new data
                    continuous_learning_manager.schedule_retraining(
                        "aerodynamics_drag", "aerodynamics", interval_hours=24)
                    logger.info("Scheduled aerodynamics model retraining")
        except Exception as e:
            logger.warning(f"Could not update continuous learning system: {e}")
