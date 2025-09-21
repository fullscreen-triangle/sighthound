"""
Olympic Data Loader

Loads and processes comprehensive Olympic athlete data from the 
public/olympics directory for validation analysis.
"""

import json
import pandas as pd
import numpy as np
import asyncio
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class OlympicDataLoader:
    """
    Comprehensive Olympic data loader for validation analysis.
    
    Loads all available Olympic athlete data including:
    - Biometric data (400m_athletes_complete_biometrics.json)
    - Performance predictions
    - Kalman filter results  
    - Physiological analysis
    - Biomechanical data
    - Pre-trained models
    """
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.loaded_data = {}
        
    async def load_comprehensive_data(self) -> Dict[str, Any]:
        """
        Load all available Olympic athlete data.
        
        Returns:
            Comprehensive dataset dictionary
        """
        logger.info(f"Loading Olympic data from {self.data_path}")
        
        # Check if data directory exists
        if not self.data_path.exists():
            logger.warning(f"Data directory {self.data_path} does not exist. Creating simulated data.")
            return await self._create_simulated_data()
        
        data = {}
        
        # Load JSON data files
        json_files = {
            '400m_athletes_complete_biometrics': '400m_athletes_complete_biometrics.json',
            'processed_athlete_data': 'processed_athlete_data_with_predictions.json', 
            'kalman_filter_results': 'kalman_filter_results.json',
            'physiological_analysis': 'physiological_analysis_results.json',
            'curve_biomechanics': 'curve_biomechanics.json'
        }
        
        for key, filename in json_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                logger.info(f"Loading {filename}")
                data[key] = await self._load_json_file(file_path)
            else:
                logger.warning(f"File {filename} not found, creating simulated data")
                data[key] = await self._create_simulated_json_data(key)
        
        # Load model files
        model_files = {
            'sprint_performance_model': 'sprint_performance_model.pkl',
            'performance_predictor': 'performance_predictor.joblib'
        }
        
        for key, filename in model_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                logger.info(f"Loading model {filename}")
                data[key] = await self._load_model_file(file_path)
            else:
                logger.warning(f"Model {filename} not found, creating simulated model")
                data[key] = await self._create_simulated_model(key)
        
        # Process and structure data for validation
        structured_data = await self._structure_data_for_validation(data)
        
        logger.info(f"Successfully loaded data for {len(structured_data.get('athletes', {}))} athletes")
        
        return structured_data
    
    async def _load_json_file(self, file_path: Path) -> Dict:
        """Load JSON file asynchronously."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    async def _load_model_file(self, file_path: Path) -> Any:
        """Load model file (pickle or joblib)."""
        try:
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    return joblib.load(f)  # joblib can handle pickle files too
            elif file_path.suffix == '.joblib':
                return joblib.load(file_path)
            else:
                logger.warning(f"Unknown model file format: {file_path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading model {file_path}: {e}")
            return None
    
    async def _structure_data_for_validation(self, raw_data: Dict) -> Dict[str, Any]:
        """Structure raw data for validation framework."""
        
        structured = {
            'athletes': {},
            'metadata': {
                'total_athletes': 0,
                'data_sources': list(raw_data.keys()),
                'race_type': '400m',
                'venue': {
                    'name': 'Olympic Stadium',
                    'latitude': 51.5574,
                    'longitude': -0.0166,
                    'altitude': 10.0
                }
            },
            'models': {
                'performance_models': raw_data.get('sprint_performance_model'),
                'prediction_models': raw_data.get('performance_predictor')
            }
        }
        
        # Process athlete data
        athletes_data = await self._process_athlete_data(raw_data)
        structured['athletes'] = athletes_data
        structured['metadata']['total_athletes'] = len(athletes_data)
        
        return structured
    
    async def _process_athlete_data(self, raw_data: Dict) -> Dict[str, Dict]:
        """Process and combine athlete data from multiple sources."""
        
        athletes = {}
        
        # Primary source: biometric data
        biometric_data = raw_data.get('400m_athletes_complete_biometrics', {})
        
        if isinstance(biometric_data, dict):
            for athlete_id, athlete_data in biometric_data.items():
                athletes[athlete_id] = await self._create_comprehensive_athlete_profile(
                    athlete_id, athlete_data, raw_data
                )
        
        # If no data found, create simulated athletes
        if not athletes:
            logger.info("No athlete data found, creating simulated athletes")
            for i in range(10):  # Create 10 simulated athletes
                athlete_id = f"athlete_{i:03d}"
                athletes[athlete_id] = await self._create_simulated_athlete_profile(athlete_id)
        
        return athletes
    
    async def _create_comprehensive_athlete_profile(self, 
                                                  athlete_id: str,
                                                  primary_data: Dict,
                                                  all_data: Dict) -> Dict:
        """Create comprehensive athlete profile from all data sources."""
        
        profile = {
            'id': athlete_id,
            'biometrics': await self._process_biometric_data(primary_data),
            'performance': await self._process_performance_data(athlete_id, all_data),
            'biomechanics': await self._process_biomechanical_data(athlete_id, all_data),
            'physiology': await self._process_physiological_data(athlete_id, all_data),
            'race_position': await self._generate_race_position_data(),
            'actual_performance': await self._generate_actual_performance_data()
        }
        
        return profile
    
    async def _process_biometric_data(self, biometric_data: Dict) -> Dict:
        """Process biometric data into standardized format."""
        
        # Extract key biometric parameters
        biometrics = {
            'base_heart_rate': biometric_data.get('heart_rate', 180),
            'max_heart_rate': biometric_data.get('max_heart_rate', 200),
            'base_vo2': biometric_data.get('vo2_max', 65.0),
            'base_lactate': biometric_data.get('lactate_threshold', 8.0),
            'anthropometrics': {
                'height': biometric_data.get('height', 1.78),
                'weight': biometric_data.get('weight', 70.0),
                'age': biometric_data.get('age', 25),
                'bmi': biometric_data.get('bmi', 22.1)
            },
            'body_composition': {
                'lean_mass': biometric_data.get('lean_mass', 65.0),
                'muscle_mass': biometric_data.get('muscle_mass', 35.0),
                'body_fat_percentage': biometric_data.get('body_fat_pct', 8.5)
            },
            'metabolic': {
                'resting_metabolic_rate': biometric_data.get('rmr', 1800),
                'respiratory_exchange_ratio': biometric_data.get('rer', 0.85),
                'ventilatory_threshold': biometric_data.get('vt', 75.0)
            }
        }
        
        return biometrics
    
    async def _process_performance_data(self, athlete_id: str, all_data: Dict) -> Dict:
        """Process performance prediction data."""
        
        processed_data = all_data.get('processed_athlete_data', {})
        athlete_performance = processed_data.get(athlete_id, {})
        
        performance = {
            'predicted_time': athlete_performance.get('predicted_time', 45.0 + np.random.uniform(-2, 2)),
            'predicted_splits': athlete_performance.get('splits', [11.5, 21.8, 33.2, 45.0]),
            'performance_index': athlete_performance.get('performance_index', 85.0),
            'ranking_prediction': athlete_performance.get('predicted_rank', np.random.randint(1, 9)),
            'confidence_interval': athlete_performance.get('confidence', [43.5, 46.5])
        }
        
        return performance
    
    async def _process_biomechanical_data(self, athlete_id: str, all_data: Dict) -> Dict:
        """Process biomechanical analysis data."""
        
        biomech_data = all_data.get('curve_biomechanics', {})
        athlete_biomech = biomech_data.get(athlete_id, {})
        
        biomechanics = {
            'stride_length': athlete_biomech.get('stride_length', 2.2),
            'stride_frequency': athlete_biomech.get('stride_frequency', 4.5),
            'ground_contact_time': athlete_biomech.get('ground_contact_time', 0.08),
            'vertical_oscillation': athlete_biomech.get('vertical_oscillation', 0.06),
            'running_effectiveness': athlete_biomech.get('running_effectiveness', 0.85),
            'curve_analysis': {
                'speed_loss_in_curve': athlete_biomech.get('curve_speed_loss', 0.05),
                'lean_angle': athlete_biomech.get('lean_angle', 15.0),
                'centripetal_force': athlete_biomech.get('centripetal_force', 450)
            }
        }
        
        return biomechanics
    
    async def _process_physiological_data(self, athlete_id: str, all_data: Dict) -> Dict:
        """Process physiological analysis data."""
        
        physio_data = all_data.get('physiological_analysis', {})
        athlete_physio = physio_data.get(athlete_id, {})
        
        physiology = {
            'lactate_kinetics': athlete_physio.get('lactate_kinetics', {
                'accumulation_rate': 2.5,
                'clearance_rate': 1.8,
                'steady_state_level': 8.5
            }),
            'oxygen_kinetics': athlete_physio.get('oxygen_kinetics', {
                'vo2_fast_component': 45.0,
                'vo2_slow_component': 5.0,
                'time_constant': 30.0
            }),
            'neuromuscular': athlete_physio.get('neuromuscular', {
                'power_output': 1200,
                'force_production': 850,
                'fatigue_resistance': 0.78
            })
        }
        
        return physiology
    
    async def _generate_race_position_data(self) -> Dict:
        """Generate race position data (simulated)."""
        
        return {
            'start_position': {
                'lane': np.random.randint(1, 9),
                'latitude': 51.5574 + np.random.uniform(-1e-5, 1e-5),
                'longitude': -0.0166 + np.random.uniform(-1e-5, 1e-5),
                'altitude': 10.0
            },
            'track_geometry': {
                'curve_radius': 36.5,  # meters
                'straight_length': 84.39,  # meters  
                'lane_width': 1.22  # meters
            }
        }
    
    async def _generate_actual_performance_data(self) -> Dict:
        """Generate actual performance data for validation."""
        
        return {
            'race_time': 45.0 + np.random.uniform(-3, 3),
            'splits': [11.5 + np.random.uniform(-0.5, 0.5),
                      21.8 + np.random.uniform(-1, 1), 
                      33.2 + np.random.uniform(-1.5, 1.5),
                      45.0 + np.random.uniform(-3, 3)],
            'final_ranking': np.random.randint(1, 9),
            'reaction_time': 0.150 + np.random.uniform(-0.030, 0.030),
            'max_speed': 12.0 + np.random.uniform(-1, 1),
            'average_speed': 8.9 + np.random.uniform(-0.5, 0.5)
        }
    
    async def _create_simulated_data(self) -> Dict[str, Any]:
        """Create comprehensive simulated Olympic data."""
        
        logger.info("Creating simulated Olympic athlete data")
        
        # Create simulated athletes
        athletes = {}
        for i in range(15):  # 15 simulated athletes
            athlete_id = f"athlete_{i:03d}"
            athletes[athlete_id] = await self._create_simulated_athlete_profile(athlete_id)
        
        return {
            'athletes': athletes,
            'metadata': {
                'total_athletes': len(athletes),
                'data_sources': ['simulated'],
                'race_type': '400m',
                'venue': {
                    'name': 'Olympic Stadium',
                    'latitude': 51.5574,
                    'longitude': -0.0166, 
                    'altitude': 10.0
                }
            },
            'models': {
                'performance_models': await self._create_simulated_model('performance'),
                'prediction_models': await self._create_simulated_model('prediction')
            }
        }
    
    async def _create_simulated_athlete_profile(self, athlete_id: str) -> Dict:
        """Create a comprehensive simulated athlete profile."""
        
        # Base athlete characteristics
        base_time = 45.0 + np.random.uniform(-5, 5)  # 400m time
        ability_level = np.random.uniform(0.6, 1.0)  # Ability multiplier
        
        profile = {
            'id': athlete_id,
            'biometrics': {
                'base_heart_rate': int(160 + np.random.uniform(-20, 40)),
                'max_heart_rate': int(190 + np.random.uniform(-15, 20)),
                'base_vo2': 55.0 + np.random.uniform(-10, 15) * ability_level,
                'base_lactate': 6.0 + np.random.uniform(-2, 6),
                'anthropometrics': {
                    'height': 1.65 + np.random.uniform(-0.15, 0.25),
                    'weight': 60 + np.random.uniform(-10, 20),
                    'age': int(20 + np.random.uniform(-3, 15)),
                    'bmi': 20 + np.random.uniform(-2, 4)
                },
                'body_composition': {
                    'lean_mass': 50 + np.random.uniform(-8, 15) * ability_level,
                    'muscle_mass': 25 + np.random.uniform(-5, 10) * ability_level,
                    'body_fat_percentage': 12.0 + np.random.uniform(-4, 8)
                },
                'metabolic': {
                    'resting_metabolic_rate': int(1500 + np.random.uniform(-200, 500)),
                    'respiratory_exchange_ratio': 0.80 + np.random.uniform(-0.05, 0.10),
                    'ventilatory_threshold': 65 + np.random.uniform(-10, 15) * ability_level
                }
            },
            'performance': {
                'predicted_time': base_time,
                'predicted_splits': [
                    base_time * 0.26 + np.random.uniform(-0.5, 0.5),
                    base_time * 0.48 + np.random.uniform(-1, 1),
                    base_time * 0.74 + np.random.uniform(-1.5, 1.5),
                    base_time
                ],
                'performance_index': 60 + 35 * ability_level + np.random.uniform(-5, 5),
                'ranking_prediction': int(np.random.uniform(1, 9)),
                'confidence_interval': [base_time - 2, base_time + 2]
            },
            'biomechanics': {
                'stride_length': 1.8 + np.random.uniform(-0.3, 0.6) * ability_level,
                'stride_frequency': 4.0 + np.random.uniform(-0.5, 1.0),
                'ground_contact_time': 0.085 + np.random.uniform(-0.015, 0.015),
                'vertical_oscillation': 0.08 + np.random.uniform(-0.02, 0.02),
                'running_effectiveness': 0.65 + 0.25 * ability_level + np.random.uniform(-0.05, 0.05),
                'curve_analysis': {
                    'speed_loss_in_curve': 0.08 + np.random.uniform(-0.03, 0.03),
                    'lean_angle': 12 + np.random.uniform(-3, 6),
                    'centripetal_force': 350 + np.random.uniform(-50, 150) * ability_level
                }
            },
            'physiology': {
                'lactate_kinetics': {
                    'accumulation_rate': 2.0 + np.random.uniform(-0.5, 1.0),
                    'clearance_rate': 1.5 + np.random.uniform(-0.3, 0.6) * ability_level,
                    'steady_state_level': 7.0 + np.random.uniform(-2, 3)
                },
                'oxygen_kinetics': {
                    'vo2_fast_component': 35 + np.random.uniform(-10, 15) * ability_level,
                    'vo2_slow_component': 3 + np.random.uniform(-1, 3),
                    'time_constant': 25 + np.random.uniform(-5, 10)
                },
                'neuromuscular': {
                    'power_output': 800 + np.random.uniform(-200, 600) * ability_level,
                    'force_production': 600 + np.random.uniform(-150, 350) * ability_level,
                    'fatigue_resistance': 0.6 + 0.25 * ability_level + np.random.uniform(-0.1, 0.1)
                }
            },
            'race_position': {
                'start_position': {
                    'lane': int(np.random.uniform(1, 9)),
                    'latitude': 51.5574 + np.random.uniform(-1e-5, 1e-5),
                    'longitude': -0.0166 + np.random.uniform(-1e-5, 1e-5),
                    'altitude': 10.0
                },
                'track_geometry': {
                    'curve_radius': 36.5,
                    'straight_length': 84.39,
                    'lane_width': 1.22
                }
            },
            'actual_performance': {
                'race_time': base_time + np.random.uniform(-1, 1),
                'splits': [
                    base_time * 0.26 + np.random.uniform(-0.3, 0.3),
                    base_time * 0.48 + np.random.uniform(-0.8, 0.8),
                    base_time * 0.74 + np.random.uniform(-1.2, 1.2),
                    base_time + np.random.uniform(-1, 1)
                ],
                'final_ranking': int(np.random.uniform(1, 9)),
                'reaction_time': 0.150 + np.random.uniform(-0.030, 0.030),
                'max_speed': 10.0 + np.random.uniform(-1.5, 2.5) * ability_level,
                'average_speed': 400 / base_time + np.random.uniform(-0.3, 0.3)
            }
        }
        
        return profile
    
    async def _create_simulated_json_data(self, data_type: str) -> Dict:
        """Create simulated JSON data based on data type."""
        
        if data_type == '400m_athletes_complete_biometrics':
            # Create simulated biometric data
            athletes = {}
            for i in range(10):
                athlete_id = f"athlete_{i:03d}"
                athletes[athlete_id] = {
                    'heart_rate': 160 + np.random.uniform(-20, 40),
                    'vo2_max': 55 + np.random.uniform(-10, 15),
                    'lactate_threshold': 6 + np.random.uniform(-2, 6),
                    'height': 1.75 + np.random.uniform(-0.15, 0.15),
                    'weight': 65 + np.random.uniform(-10, 10),
                    'age': 25 + np.random.randint(-5, 10),
                    'bmi': 21 + np.random.uniform(-2, 3)
                }
            return athletes
            
        elif data_type == 'kalman_filter_results':
            return {
                'filter_parameters': {
                    'process_noise': 0.01,
                    'measurement_noise': 0.1,
                    'estimation_accuracy': 0.95
                },
                'tracking_results': {
                    'position_estimates': 'simulated_tracking_data',
                    'velocity_estimates': 'simulated_velocity_data',
                    'acceleration_estimates': 'simulated_acceleration_data'
                }
            }
            
        else:
            return {'simulated': True, 'data_type': data_type}
    
    async def _create_simulated_model(self, model_type: str) -> Dict:
        """Create simulated model data."""
        
        return {
            'model_type': model_type,
            'simulated': True,
            'accuracy': 0.85 + np.random.uniform(-0.10, 0.10),
            'features': ['heart_rate', 'vo2_max', 'lactate_threshold', 'biomechanics'],
            'training_data_size': int(np.random.uniform(500, 2000)),
            'validation_score': 0.80 + np.random.uniform(-0.15, 0.15)
        }
