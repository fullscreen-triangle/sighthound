import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from biomechanics.dynamics import SprintDynamics
from biomechanics.inverse import InverseDynamicsSolver, HybridDynamicsSolver

# Standardize logger configuration
logger = logging.getLogger(__name__)

@dataclass
class PostureState:
    """Represents a single posture state"""
    timestamp: str
    joint_angles: Dict[str, List[float]]  # Joint angles in [x,y,z] format
    confidence: float                      # Confidence in pose estimation
    ground_contact: bool                   # Whether foot is in contact
    phase: str                            # Gait phase
    mannequin_format: Dict                # Compatible with mannequin.js

@dataclass
class BiomechanicalMetrics:
    """Biomechanical metrics for posture analysis"""
    joint_torques: Dict[str, float]
    ground_reaction_forces: Dict[str, float]
    center_of_mass: List[float]
    stability_score: float
    energy_expenditure: float

class PostureEstimator(nn.Module):
    """Neural network for estimating joint angles from sensor data"""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 21)  # 7 joints x 3 angles
        )
        
    def forward(self, x):
        return self.network(x)

class PostureAnalyzer:
    def __init__(self, athlete_mass: float = 70.0, athlete_height: float = 1.80):
        """
        Initialize the posture analyzer.
        
        Args:
            athlete_mass: Athlete mass in kg
            athlete_height: Athlete height in meters
        """
        self.athlete_mass = athlete_mass
        self.athlete_height = athlete_height
        
        # Initialize joint angle model
        try:
            import torch
            import torch.nn as nn
            
            # Try to load model from continuous learning system first
            self.joint_angle_model = self._load_from_continuous_learning()
            
            # If not available, create a default model
            if self.joint_angle_model is None:
                logger.info("Creating default posture estimation model")
                input_dim = 10  # Default feature vector size
                self.joint_angle_model = PostureEstimator(input_dim)
                
                # Initialize with some default weights
                for param in self.joint_angle_model.parameters():
                    nn.init.normal_(param, mean=0.0, std=0.1)
        except Exception as e:
            logger.error(f"Failed to initialize joint angle model: {e}")
            self.joint_angle_model = None
            
        # Initialize component analyzers
        self.dynamics_analyzer = SprintDynamics(athlete_mass, athlete_height)
        self.inverse_solver = InverseDynamicsSolver(
            link_lengths=[0.4, 0.4],  # Thigh and shank lengths
            link_masses=[7.0, 3.0],   # Segment masses
            moments_of_inertia=[0.15, 0.05]
        )
        self.hybrid_solver = HybridDynamicsSolver(self.inverse_solver)
        
        # Gait phase definitions
        self.gait_phases = ['initial_contact', 'mid_stance', 'propulsion', 'swing']
        
    def _load_from_continuous_learning(self):
        """Try to load model from continuous learning system."""
        try:
            import torch
            from federation.continuous_learning import continuous_learning_manager
            
            # Try to get the posture model
            model = continuous_learning_manager.model_registry.get_model("posture_joint_angles")
            
            if model is not None:
                logger.info("Loaded posture model from continuous learning system")
                return model
        except Exception as e:
            logger.warning(f"Could not load model from continuous learning system: {e}")
        
        return None
            
    def analyze_posture(self, coros_file: str, track_file: str) -> List[PostureState]:
        """
        Analyze posture based on sensor data from COROS and track files.
        
        Args:
            coros_file: Path to COROS JSON file
            track_file: Path to track JSON file
            
        Returns:
            List of PostureState objects representing the posture over time
        """
        try:
            logger.info(f"Analyzing posture from {coros_file} and {track_file}")
            
            # Load data from JSON files
            with open(coros_file, 'r') as f:
                coros_data = json.load(f)
            
            with open(track_file, 'r') as f:
                track_data = json.load(f)
                
            logger.info(f"Successfully loaded data: COROS ({len(coros_data) if isinstance(coros_data, list) else 'dict'} entries), " +
                       f"track ({len(track_data) if isinstance(track_data, list) else 'dict'} entries)")
            
            # Extract relevant features for posture analysis
            # The specific implementation depends on the data structure
            posture_states = []
            
            # Print the first entry of each file to understand structure
            logger.info(f"COROS data sample: {str(coros_data[0] if isinstance(coros_data, list) and len(coros_data) > 0 else coros_data)[:500]}")
            logger.info(f"Track data sample: {str(track_data[0] if isinstance(track_data, list) and len(track_data) > 0 else track_data)[:500]}")
            
            # For continuous learning - store actual angles if available
            actual_angles = []
            
            # Check if we're dealing with a list of data points or a different structure
            if isinstance(coros_data, list) and len(coros_data) > 0:
                # Process list of data points (typical time series)
                for i, coros_point in enumerate(coros_data):
                    # Find corresponding track point (simplified for now)
                    track_point = track_data[i] if isinstance(track_data, list) and i < len(track_data) else None
                    
                    # Create feature vector from both data sources
                    # We'll extract only numeric values to avoid conversion issues
                    features = []
                    
                    # Extract numeric features from COROS data
                    if isinstance(coros_point, dict):
                        for key in ['cadence', 'speed', 'vertical_oscillation', 'stance_time', 
                                    'stance_time_balance', 'step_length', 'power']:
                            if key in coros_point and coros_point[key] is not None:
                                try:
                                    features.append(float(coros_point[key]))
                                except (ValueError, TypeError):
                                    features.append(0.0)  # Default if conversion fails
                            else:
                                features.append(0.0)  # Default if key missing
                    
                    # Extract numeric features from track data
                    if track_point and isinstance(track_point, dict):
                        for key in ['altitude', 'heart_rate', 'speed']:
                            if key in track_point and track_point[key] is not None:
                                try:
                                    features.append(float(track_point[key]))
                                except (ValueError, TypeError):
                                    features.append(0.0)  # Default if conversion fails
                            else:
                                features.append(0.0)  # Default if key missing
                    
                    # Convert to numpy array and ensure all elements are float

                    features = np.array(features, dtype=np.float32)
                    
                    if len(features) > 0:
                        # Estimate joint angles using the feature vector
                        joint_angles = self._estimate_joint_angles(features)
                        
                        # Check if we have actual measured angles for this point (for continuous learning)
                        if isinstance(coros_point, dict) and "measured_angles" in coros_point:
                            measured = coros_point["measured_angles"]
                            if isinstance(measured, dict):
                                actual_angles.append((features, measured))
                        
                        # Create posture state
                        timestamp = coros_point.get('timestamp', f"2022-04-27T15:45:{i:02d}Z")
                        
                        # Create a posture state for this data point
                        posture_state = PostureState(
                            timestamp=timestamp,
                            joint_angles=joint_angles,
                            confidence=self._calculate_confidence(features),
                            ground_contact=True if i % 2 == 0 else False,  # Alternating for demonstration
                            phase="stance" if i % 2 == 0 else "swing",     # Alternating for demonstration
                            mannequin_format=self._create_mannequin_format(joint_angles)
                        )
                        
                        posture_states.append(posture_state)
                        
                        if i == 0:
                            logger.info(f"Created first posture state with features: {features}")
                            logger.info(f"Joint angles: {joint_angles}")
            else:
                # Non-list structure - handle appropriately
                logger.warning("Expected list of data points in COROS file, got different structure")
                # Try to extract relevant data anyway
                features = np.zeros(10, dtype=np.float32)  # Default features
                joint_angles = self._estimate_joint_angles(features)
                
                # Create a single posture state
                posture_state = PostureState(
                    timestamp="2022-04-27T15:45:00Z",
                    joint_angles=joint_angles,
                    confidence=0.5,
                    ground_contact=True,
                    phase="stance",
                    mannequin_format=self._create_mannequin_format(joint_angles)
                )
                
                posture_states.append(posture_state)
            
            logger.info(f"Created {len(posture_states)} posture states")
            
            # Update continuous learning if we have actual angle data
            if actual_angles:
                self._update_continuous_learning(actual_angles)
                
            return posture_states
        
        except Exception as e:
            logger.error(f"Posture analysis failed: {e}", exc_info=True)
            return []
            
    def _update_continuous_learning(self, actual_angles):
        """Update continuous learning system with actual angle data."""
        try:
            from federation.continuous_learning import continuous_learning_manager
            
            # Format the data for model training
            features = []
            targets = []
            
            for feature_vector, angle_dict in actual_angles:
                # Convert angle dictionary to consistent target format
                target = []
                for joint in ['hip_flexion', 'knee_flexion', 'ankle_dorsiflexion', 'hip_abduction', 
                             'elbow_flexion', 'shoulder_flexion', 'trunk_flexion']:
                    target.append(float(angle_dict.get(joint, 0.0)))
                
                features.append(feature_vector.tolist())
                targets.append(target)
            
            # Store the data for future model training
            run_data = {
                "posture_training_data": {
                    "features": features,
                    "targets": targets
                }
            }
            
            # Generate a unique ID
            import hashlib
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            data_hash = hashlib.md5(str(features[:3]).encode()).hexdigest()[:8]
            training_id = f"posture_training_{timestamp.replace(':', '-')}_{data_hash}"
            
            # Store the training data
            continuous_learning_manager.data_collector.store_run_data(
                training_id, run_data, {"type": "posture_training_data"})
            
            logger.info(f"Stored {len(features)} new training samples for posture model")
            
            # Schedule model retraining if we have enough new data
            if len(features) > 20:
                continuous_learning_manager.schedule_retraining(
                    "posture_joint_angles", "posture", interval_hours=24)
                logger.info("Scheduled posture model retraining")
                
        except Exception as e:
            logger.warning(f"Could not update continuous learning system: {e}")

    def _preprocess_sensor_data(self, coros_data: Dict, track_data: Dict) -> np.ndarray:
        """Preprocess and align sensor data"""
        features = []
        for coros_point, track_point in zip(coros_data, track_data['features']):
            feature_vector = [
                coros_point['power'],
                coros_point['form_power'],
                coros_point['cadence'],
                coros_point['speed'],
                coros_point['stance_time'],
                coros_point['vertical_ratio'],
                coros_point['vertical_oscillation'],
                track_point['properties']['stance_time_balance'],
                track_point['properties']['step_length'],
                track_point['properties']['vertical_ratio'],
                track_point['properties']['vertical_oscillation'],
                track_point['properties']['heart_rate']
            ]
            features.append(feature_vector)
        return np.array(features)

    def _estimate_joint_angles(self, features: np.ndarray) -> List[Dict[str, List[float]]]:
        """
        Estimate joint angles using neural network.
        
        Args:
            features: Feature vector for joint angle estimation
        
        Returns:
            Dictionary of joint angles
        """
        try:
            # Log the features we're trying to convert
            logger.info(f"Converting features to tensor: {features}")
            
            # Make sure features are numeric before converting to tensor

            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            elif features.dtype != np.float32:
                features = features.astype(np.float32)
                

            X = torch.FloatTensor(features)
            
            logger.info(f"Successfully converted to tensor with shape: {X.shape}")
            
            # Use the trained model to predict joint angles
            with torch.no_grad():
                # Check for model existence
                if not hasattr(self, 'joint_angle_model') or self.joint_angle_model is None:
                    # Return default joint angles if model isn't available
                    logger.warning("Joint angle model not available, returning defaults")
                    return self._get_default_joint_angles()
                
                angles = self.joint_angle_model(X)
                
            # Convert tensor to numpy for easier handling
            angles_np = angles.numpy()
            
            # Map the outputs to joint names
            joint_angles = {
                'hip_flexion': float(angles_np[0]),
                'knee_flexion': float(angles_np[1]),
                'ankle_dorsiflexion': float(angles_np[2]),
                'hip_abduction': float(angles_np[3]),
                'elbow_flexion': float(angles_np[4]),
                'shoulder_flexion': float(angles_np[5]),
                'trunk_flexion': float(angles_np[6])
            }
            
            return joint_angles
        except Exception as e:
            logger.error(f"Joint angle estimation failed: {e}", exc_info=True)
            # Return default angles as fallback
            return self._get_default_joint_angles()
    
    def _ensure_numeric_features(self, features):
        """
        Ensure all features are numeric types for PyTorch tensor conversion.
        
        Args:
            features: Feature array that might contain string values
            
        Returns:
            Numpy array with only numeric values
        """
        # If features is already a numeric numpy array, return it
        if isinstance(features, np.ndarray) and np.issubdtype(features.dtype, np.number):
            return features
            
        # If it's a numpy array with strings, convert to numeric
        if isinstance(features, np.ndarray):
            # Try to convert strings to float, use defaults for non-convertible values
            numeric_features = np.zeros(features.shape, dtype=np.float32)
            for i, val in enumerate(features):
                try:
                    numeric_features[i] = float(val)
                except (ValueError, TypeError):
                    # Use 0.0 as default for non-numeric values
                    numeric_features[i] = 0.0
            return numeric_features
            
        # If it's a list, convert to numpy array first
        if isinstance(features, list):
            return self._ensure_numeric_features(np.array(features))
            
        # If it's a single value, return a simple array
        return np.array([0.0] * 10, dtype=np.float32)  # Default feature vector size
            
    def _create_mannequin_format(self, joint_angles):
        """Convert joint angles to mannequin format"""
        return {
            "hip": {
                "flexion": joint_angles.get('hip_flexion', 0),
                "abduction": joint_angles.get('hip_abduction', 0),
                "rotation": 0.0
            },
            "knee": {
                "flexion": joint_angles.get('knee_flexion', 0),
                "varus": 0.0,
                "rotation": 0.0
            },
            "ankle": {
                "dorsiflexion": joint_angles.get('ankle_dorsiflexion', 0),
                "inversion": 0.0,
                "rotation": 0.0
            },
            "shoulder": {
                "flexion": joint_angles.get('shoulder_flexion', 0),
                "abduction": 0.0,
                "rotation": 0.0
            },
            "elbow": {
                "flexion": joint_angles.get('elbow_flexion', 0),
                "pronation": 0.0
            },
            "trunk": {
                "flexion": joint_angles.get('trunk_flexion', 0),
                "lateral_flexion": 0.0,
                "rotation": 0.0
            }
        }
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for posture estimation"""
        # Implement confidence calculation based on sensor reliability
        # and biomechanical plausibility
        return np.clip(np.mean(features) / np.std(features), 0, 1)

    def visualize_posture_sequence(self, posture_states: List[PostureState]):
        """Create 3D visualization of posture sequence"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample every nth frame for visualization
        n = max(1, len(posture_states) // 10)
        
        for i, state in enumerate(posture_states[::n]):
            self._plot_skeleton(ax, state.mannequin_format['data'], alpha=0.5)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Posture Sequence Visualization')
        plt.show()

    def _plot_skeleton(self, ax, joint_data: List[List[float]], alpha: float = 1.0):
        """Plot 3D skeleton from joint data"""
        # Convert joint data to 3D coordinates
        joints = np.array(joint_data)
        
        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                  c='b', alpha=alpha)
        
        # Plot connections between joints
        connections = [
            (0, 1), (1, 2), (2, 3),  # Spine
            (1, 4), (4, 5), (5, 6),  # Left arm
            (1, 7), (7, 8), (8, 9),  # Right arm
            (1, 10), (10, 11), (11, 12),  # Left leg
            (1, 13), (13, 14), (14, 15)   # Right leg
        ]
        
        for start, end in connections:
            ax.plot([joints[start, 0], joints[end, 0]],
                   [joints[start, 1], joints[end, 1]],
                   [joints[start, 2], joints[end, 2]],
                   'r-', alpha=alpha)

    def export_to_mannequin(self, posture_states: List[PostureState], output_file: str):
        """Export posture sequence to mannequin.js compatible format"""
        mannequin_sequence = {
            "version": 7,
            "frames": [state.mannequin_format for state in posture_states]
        }
        
        with open(output_file, 'w') as f:
            json.dump(mannequin_sequence, f)

    def analyze_posture_with_processed_data(self, coros_data, track_data):
        """
        Analyze posture using preprocessed data instead of file paths.
        
        Args:
            coros_data: Processed COROS data
            track_data: Processed track data
            
        Returns:
            List of posture states
        """
        try:
            logger.info("Analyzing posture with preprocessed data...")
            
            # Extract relevant features from the data
            # This will vary based on your specific data format
            features = self._extract_features_from_data(coros_data, track_data)
            
            # Estimate joint angles
            joint_angles = self._estimate_joint_angles(features)
            
            # Create posture sequence
            posture_states = self._create_posture_sequence(joint_angles, len(coros_data))
            
            return posture_states
        except Exception as e:
            logger.error(f"Posture analysis with processed data failed: {e}")
            return []
            
    def _extract_features_from_data(self, coros_data, track_data):
        """
        Extract features from processed data objects.
        
        Args:
            coros_data: COROS data
            track_data: Track data
            
        Returns:
            Feature array for posture analysis
        """
        try:
            # This extraction logic will depend on your specific data structure
            # Create a default feature array with zeros

            features = np.zeros(10, dtype=np.float32)  # Default feature vector size
            
            # Example extraction:
            if isinstance(coros_data, list) and len(coros_data) > 0:
                # Extract features from first data point as example
                data_point = coros_data[0]
                
                # Try to extract some useful features (modify based on your data structure)
                if isinstance(data_point, dict):
                    features[0] = float(data_point.get('cadence', 0) or 0)
                    features[1] = float(data_point.get('speed', 0) or 0)
                    features[2] = float(data_point.get('vertical_oscillation', 0) or 0)
                    features[3] = float(data_point.get('stance_time', 0) or 0)
                    features[4] = float(data_point.get('stance_time_balance', 0) or 0)
                    features[5] = float(data_point.get('step_length', 0) or 0)
                    features[6] = float(data_point.get('vertical_ratio', 0) or 0)
                    features[7] = float(data_point.get('power', 0) or 0)
                    features[8] = float(data_point.get('form_power', 0) or 0)
                    features[9] = float(data_point.get('altitude', 0) or 0)
                    
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default feature vector
            return np.zeros(10, dtype=np.float32)
            
    def _create_posture_sequence(self, joint_angles, num_frames=10):
        """
        Create a sequence of posture states based on joint angles.
        
        Args:
            joint_angles: Dictionary of joint angles
            num_frames: Number of frames to generate
            
        Returns:
            List of posture state dictionaries
        """
        posture_states = []
        
        for i in range(num_frames):
            # Create a posture state for each frame
            # We'll use the same joint angles for all frames in this simplified version
            posture_state = {
                "frame": i,
                "timestamp": f"2022-04-27T15:45:{i:02d}Z", 
                "joints": {
                    "hip": {
                        "flexion": joint_angles.get('hip_flexion', 0),
                        "abduction": joint_angles.get('hip_abduction', 0),
                        "rotation": 0.0
                    },
                    "knee": {
                        "flexion": joint_angles.get('knee_flexion', 0),
                        "varus": 0.0,
                        "rotation": 0.0
                    },
                    "ankle": {
                        "dorsiflexion": joint_angles.get('ankle_dorsiflexion', 0),
                        "inversion": 0.0,
                        "rotation": 0.0
                    },
                    "shoulder": {
                        "flexion": joint_angles.get('shoulder_flexion', 0),
                        "abduction": 0.0,
                        "rotation": 0.0
                    },
                    "elbow": {
                        "flexion": joint_angles.get('elbow_flexion', 0),
                        "pronation": 0.0
                    },
                    "trunk": {
                        "flexion": joint_angles.get('trunk_flexion', 0),
                        "lateral_flexion": 0.0,
                        "rotation": 0.0
                    }
                }
            }
            posture_states.append(posture_state)
            
        return posture_states

    def _get_default_joint_angles(self):
        """Return default joint angles for fallback"""
        return {
            'hip_flexion': 20.0,
            'knee_flexion': 40.0,
            'ankle_dorsiflexion': 10.0,
            'hip_abduction': 5.0,
            'elbow_flexion': 80.0,
            'shoulder_flexion': 30.0,
            'trunk_flexion': 10.0
        }


