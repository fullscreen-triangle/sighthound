import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from enum import Enum
from scipy.spatial import distance
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
from datetime import datetime
import math
from sklearn.ensemble import RandomForestClassifier

@dataclass
class MuscleTendonParameters:
    max_isometric_force: float    # Maximum isometric force (N)
    optimal_fiber_length: float   # Optimal fiber length (m)
    tendon_slack_length: float    # Tendon slack length (m)
    pennation_angle: float        # Pennation angle at optimal length (rad)
    max_contraction_velocity: float  # Maximum contraction velocity (m/s)
    activation_time_constant: float  # Activation dynamics time constant (s)
    deactivation_time_constant: float  # Deactivation time constant (s)

@dataclass
class MuscleState:
    activation: float           # Muscle activation level (0-1)
    fiber_length: float        # Current fiber length (m)
    fiber_velocity: float      # Fiber contraction velocity (m/s)
    pennation_angle: float     # Current pennation angle (rad)
    tendon_length: float       # Current tendon length (m)
    tendon_force: float        # Tendon force (N)

class MuscleGroup(Enum):
    # Ankle & Foot Complex
    GASTROCNEMIUS_MEDIALIS = "gastrocnemius_medialis"
    GASTROCNEMIUS_LATERALIS = "gastrocnemius_lateralis"
    SOLEUS = "soleus"
    TIBIALIS_ANTERIOR = "tibialis_anterior"
    PERONEUS_LONGUS = "peroneus_longus"
    PERONEUS_BREVIS = "peroneus_brevis"
    FLEXOR_DIGITORUM_LONGUS = "flexor_digitorum_longus"
    FLEXOR_HALLUCIS_LONGUS = "flexor_hallucis_longus"
    EXTENSOR_DIGITORUM_LONGUS = "extensor_digitorum_longus"
    EXTENSOR_HALLUCIS_LONGUS = "extensor_hallucis_longus"
    
    # Intrinsic Foot Muscles
    FLEXOR_DIGITORUM_BREVIS = "flexor_digitorum_brevis"
    ABDUCTOR_HALLUCIS = "abductor_hallucis"
    FLEXOR_HALLUCIS_BREVIS = "flexor_hallucis_brevis"
    INTRINSIC_PLANTAR_MUSCLES = "intrinsic_plantar_muscles"
    
    # Thigh - Anterior
    RECTUS_FEMORIS = "rectus_femoris"
    VASTUS_LATERALIS = "vastus_lateralis"
    VASTUS_MEDIALIS = "vastus_medialis"
    VASTUS_INTERMEDIUS = "vastus_intermedius"
    
    # Thigh - Posterior
    BICEPS_FEMORIS_LONG = "biceps_femoris_long"
    BICEPS_FEMORIS_SHORT = "biceps_femoris_short"
    SEMITENDINOSUS = "semitendinosus"
    SEMIMEMBRANOSUS = "semimembranosus"
    
    # Hip
    GLUTEUS_MAXIMUS = "gluteus_maximus"
    GLUTEUS_MEDIUS = "gluteus_medius"
    GLUTEUS_MINIMUS = "gluteus_minimus"
    ILIOPSOAS = "iliopsoas"
    TENSOR_FASCIAE_LATAE = "tensor_fasciae_latae"
    
    # Upper Body (Curve Running)
    LATISSIMUS_DORSI = "latissimus_dorsi"
    TRAPEZIUS = "trapezius"
    RHOMBOIDS = "rhomboids"
    DELTOID = "deltoid"
    BICEPS_BRACHII = "biceps_brachii"
    TRICEPS_BRACHII = "triceps_brachii"

class Thelen2003MuscleDynamics:
    def __init__(self, params: MuscleTendonParameters):
        """
        Initialize Thelen 2003 muscle model
        
        Args:
            params: Muscle-tendon parameters
        """
        self.params = params
        self.curve_factor = 10.0  # Shape factor for force-length curve
        self.damping = 0.1       # Passive damping coefficient
        
    def force_length_curve(self, normalized_length: float) -> float:
        """
        Active force-length relationship
        
        Args:
            normalized_length: Fiber length / optimal length
            
        Returns:
            Force scaling factor
        """
        return np.exp(-((normalized_length - 1.0) ** 2) / self.curve_factor)
    
    def force_velocity_curve(self, normalized_velocity: float) -> float:
        """
        Force-velocity relationship (Hill model)
        
        Args:
            normalized_velocity: Fiber velocity / max velocity
            
        Returns:
            Force scaling factor
        """
        a_f = 0.25  # Hill equation parameter
        f_len = 1.4  # Maximum force at lengthening
        
        if normalized_velocity <= 0:  # Concentric
            return (1.0 + normalized_velocity) / (1.0 - normalized_velocity / a_f)
        else:  # Eccentric
            return f_len - (f_len - 1.0) * np.exp(-7.6 * normalized_velocity)

    def passive_force_length_curve(self, normalized_length: float) -> float:
        """
        Passive force-length relationship
        
        Args:
            normalized_length: Fiber length / optimal length
            
        Returns:
            Passive force
        """
        k_pe = 4.0  # Passive exponential strain coefficient
        return np.exp(k_pe * (normalized_length - 1.0)) - 1.0 if normalized_length > 1.0 else 0.0

    def tendon_force_length_curve(self, normalized_length: float) -> float:
        """
        Tendon force-length relationship
        
        Args:
            normalized_length: Tendon length / slack length
            
        Returns:
            Tendon force scaling factor
        """
        k_t = 35.0  # Tendon strain coefficient
        return k_t * (normalized_length - 1.0) if normalized_length > 1.0 else 0.0

    def activation_dynamics(self, 
                          excitation: float, 
                          activation: float, 
                          dt: float) -> float:
        """
        Compute activation dynamics
        
        Args:
            excitation: Neural excitation signal (0-1)
            activation: Current activation level (0-1)
            dt: Time step
            
        Returns:
            New activation level
        """
        tau = self.params.activation_time_constant if excitation > activation \
              else self.params.deactivation_time_constant
        
        dadt = (excitation - activation) / tau
        return np.clip(activation + dadt * dt, 0.0, 1.0)

    def compute_pennation_angle(self, fiber_length: float) -> float:
        """
        Compute pennation angle based on fiber length
        
        Args:
            fiber_length: Current fiber length
            
        Returns:
            Current pennation angle
        """
        h = self.params.optimal_fiber_length * np.sin(self.params.pennation_angle)
        return np.arcsin(h / fiber_length)

    def compute_muscle_dynamics(self, 
                              state: MuscleState, 
                              excitation: float,
                              dt: float) -> MuscleState:
        """
        Compute one time step of muscle dynamics
        
        Args:
            state: Current muscle state
            excitation: Neural excitation signal
            dt: Time step
            
        Returns:
            Updated muscle state
        """
        # Update activation
        new_activation = self.activation_dynamics(excitation, state.activation, dt)
        
        # Normalize lengths and velocity
        norm_length = state.fiber_length / self.params.optimal_fiber_length
        norm_velocity = state.fiber_velocity / self.params.max_contraction_velocity
        
        # Compute force multipliers
        active_force = self.force_length_curve(norm_length) * \
                      self.force_velocity_curve(norm_velocity)
        passive_force = self.passive_force_length_curve(norm_length)
        
        # Total muscle force
        muscle_force = (new_activation * active_force + passive_force) * \
                      self.params.max_isometric_force * np.cos(state.pennation_angle)
        
        # Tendon force
        norm_tendon_length = state.tendon_length / self.params.tendon_slack_length
        tendon_force = self.tendon_force_length_curve(norm_tendon_length) * \
                      self.params.max_isometric_force
        
        # Velocity correction based on force equilibrium
        force_diff = tendon_force - muscle_force
        velocity_correction = force_diff / (self.damping * self.params.max_isometric_force)
        new_velocity = state.fiber_velocity + velocity_correction * dt
        
        # Update lengths
        new_fiber_length = state.fiber_length + new_velocity * dt
        new_pennation = self.compute_pennation_angle(new_fiber_length)
        
        # Update tendon length based on total length constraint
        muscle_tendon_length = state.tendon_length + \
                             state.fiber_length * np.cos(state.pennation_angle)
        new_tendon_length = muscle_tendon_length - \
                           new_fiber_length * np.cos(new_pennation)
        
        return MuscleState(
            activation=new_activation,
            fiber_length=new_fiber_length,
            fiber_velocity=new_velocity,
            pennation_angle=new_pennation,
            tendon_length=new_tendon_length,
            tendon_force=tendon_force
        )

class CurveDetector:
    def __init__(self, min_curve_radius: float = 15.0, 
                 max_curve_radius: float = 50.0,
                 window_size: int = 5):
        """Initialize curve detector with enhanced parameters"""
        self.min_radius = min_curve_radius
        self.max_radius = max_curve_radius
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # Initialize ML model for curve validation
        self.curve_classifier = self._initialize_classifier()
        
    def _initialize_classifier(self) -> RandomForestClassifier:
        """Initialize ML classifier for curve validation"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
    def calculate_curve_metrics(self, 
                              points: List[Tuple[float, float]], 
                              props: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive curve metrics
        
        Args:
            points: List of (longitude, latitude) coordinates
            props: Point properties from track data
            
        Returns:
            Dictionary of curve metrics
        """
        # Basic radius calculation
        radius = self.calculate_curve_radius(points)
        
        # Calculate heading changes
        headings = self.calculate_heading_changes(points)
        
        # Calculate stride pattern metrics
        stride_metrics = self.analyze_stride_pattern(props)
        
        # Calculate centripetal force estimation
        centripetal = (props['speed']**2) / radius if radius != float('inf') else 0
        
        return {
            'radius': radius,
            'heading_change': headings['mean_change'],
            'heading_consistency': headings['consistency'],
            'stride_asymmetry': stride_metrics['asymmetry'],
            'stride_pattern_score': stride_metrics['pattern_score'],
            'centripetal_force': centripetal,
            'vertical_ratio_change': stride_metrics['vertical_ratio_change'],
            'stance_time_variability': stride_metrics['stance_variability']
        }
        
    def calculate_heading_changes(self, 
                                points: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate heading changes between consecutive points"""
        if len(points) < 3:
            return {'mean_change': 0, 'consistency': 0}
            
        headings = []
        for i in range(len(points)-2):
            p1, p2, p3 = points[i:i+3]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            magnitudes = math.sqrt((v1[0]**2 + v1[1]**2) * (v2[0]**2 + v2[1]**2))
            
            angle = math.acos(min(1, max(-1, dot_product/magnitudes)))
            headings.append(math.degrees(angle))
            
        return {
            'mean_change': np.mean(headings),
            'consistency': 1 - np.std(headings)/np.mean(headings) if headings else 0
        }
        
    def analyze_stride_pattern(self, props: Dict) -> Dict[str, float]:
        """Analyze stride patterns for curve indicators"""
        return {
            'asymmetry': abs(props['stance_time_balance'] - 50.0),
            'pattern_score': self._calculate_pattern_score(props),
            'vertical_ratio_change': props['vertical_ratio'],
            'stance_variability': props['stance_time'] / props['cadence']
        }
        
    def _calculate_pattern_score(self, props: Dict) -> float:
        """Calculate stride pattern score for curve running"""
        # Normalize metrics
        normalized_step = props['step_length'] / 1000  # Convert to meters
        normalized_cadence = props['cadence'] / 200  # Normalize to typical range
        
        # Weight factors for curve running patterns
        weights = {
            'step_length': 0.3,
            'cadence': 0.2,
            'vertical_ratio': 0.25,
            'stance_balance': 0.25
        }
        
        score = (
            weights['step_length'] * normalized_step +
            weights['cadence'] * normalized_cadence +
            weights['vertical_ratio'] * (props['vertical_ratio'] / 10) +
            weights['stance_balance'] * (1 - abs(props['stance_time_balance'] - 50) / 50)
        )
        
        return score

    def calculate_curve_radius(self, 
                             points: List[Tuple[float, float]]) -> float:
        """
        Calculate radius of curvature using three consecutive points
        
        Args:
            points: List of (longitude, latitude) coordinates
            
        Returns:
            Radius of curvature in meters
        """
        if len(points) < 3:
            return float('inf')
            
        # Convert to local cartesian coordinates
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        # Calculate differences
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Calculate second derivatives
        d2x = np.diff(dx)
        d2y = np.diff(dy)
        
        # Curvature formula
        curvature = np.abs(d2x * dy[:-1] - dx[:-1] * d2y) / \
                    (dx[:-1]**2 + dy[:-1]**2)**(3/2)
                    
        return 1/curvature if curvature > 0 else float('inf')
    
    def detect_curves(self, track_data: Dict) -> List[Dict]:
        """Enhanced curve detection with multiple criteria"""
        features = track_data['features']
        segments = []
        
        # Extract coordinates and metrics
        coords = [(f['geometry']['coordinates'][0], 
                  f['geometry']['coordinates'][1]) for f in features]
        
        # Calculate comprehensive metrics
        metrics = []
        window = self.window_size
        
        for i in range(len(features)):
            props = features[i]['properties']
            
            # Calculate metrics if enough points
            if i >= window//2 and i < len(coords) - window//2:
                window_coords = coords[i-window//2:i+window//2+1]
                curve_metrics = self.calculate_curve_metrics(window_coords, props)
            else:
                curve_metrics = {
                    'radius': float('inf'),
                    'heading_change': 0,
                    'heading_consistency': 0,
                    'stride_asymmetry': 0,
                    'stride_pattern_score': 0,
                    'centripetal_force': 0,
                    'vertical_ratio_change': props['vertical_ratio'],
                    'stance_time_variability': 0
                }
            
            metrics.append({
                **curve_metrics,
                'timestamp': datetime.fromisoformat(props['timestamp'].replace('Z', '+00:00'))
            })
        
        # Identify curve segments using multiple criteria
        is_curve = []
        for i, metric in enumerate(metrics):
            # Enhanced criteria for curve classification
            radius_check = self.min_radius <= metric['radius'] <= self.max_radius
            heading_check = metric['heading_change'] > 2.0  # degrees
            pattern_check = metric['stride_pattern_score'] > 0.6
            force_check = metric['centripetal_force'] > 0.5  # significant centripetal force
            
            # Combine criteria with weights
            curve_score = (
                0.4 * radius_check +
                0.2 * heading_check +
                0.2 * pattern_check +
                0.2 * force_check
            )
            
            is_curve.append(curve_score > 0.6)  # Threshold for curve classification
        
        # Apply smoothing with Savitzky-Golay filter
        is_curve = savgol_filter(is_curve, window_length=7, polyorder=1) > 0.5
        
        # Create segments with enhanced metrics
        current_segment = None
        
        for i, curve in enumerate(is_curve):
            if curve and current_segment is None:
                current_segment = {
                    'start_idx': i,
                    'start_time': metrics[i]['timestamp'],
                    'type': 'curve',
                    'metrics': [],
                    'curve_characteristics': {
                        'mean_radius': metrics[i]['radius'],
                        'mean_heading_change': metrics[i]['heading_change'],
                        'stride_pattern': metrics[i]['stride_pattern_score']
                    }
                }
            elif not curve and current_segment is not None:
                self._finalize_segment(current_segment, metrics[i-1])
                segments.append(current_segment)
                current_segment = None
            
            if current_segment is not None:
                current_segment['metrics'].append(metrics[i])
        
        # Add final segment if needed
        if current_segment is not None:
            self._finalize_segment(current_segment, metrics[-1])
            segments.append(current_segment)
        
        return segments
        
    def _finalize_segment(self, segment: Dict, final_metric: Dict):
        """Finalize segment with additional metrics"""
        segment['end_idx'] = final_metric['timestamp']
        segment['end_time'] = final_metric['timestamp']
        
        # Calculate aggregate metrics
        metrics = segment['metrics']
        segment['curve_characteristics'].update({
            'mean_radius': np.mean([m['radius'] for m in metrics]),
            'mean_heading_change': np.mean([m['heading_change'] for m in metrics]),
            'stride_pattern': np.mean([m['stride_pattern_score'] for m in metrics]),
            'consistency': np.mean([m['heading_consistency'] for m in metrics]),
            'duration': (segment['end_time'] - segment['start_time']).total_seconds()
        })

class SprintMuscleDynamics:
    """Specialized muscle dynamics for 400m sprint analysis"""
    def __init__(self):
        # Initialize muscle models for key muscle groups with research-based parameters
        self.muscles = {
            # Lower leg muscles
            MuscleGroup.GASTROCNEMIUS_MEDIALIS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=1500,  # N
                    optimal_fiber_length=0.06,  # m
                    tendon_slack_length=0.24,  # m
                    pennation_angle=np.radians(25),
                    max_contraction_velocity=0.8,  # normalized
                    activation_time_constant=0.01,  # s
                    deactivation_time_constant=0.04  # s
                )
            ),
            MuscleGroup.GASTROCNEMIUS_LATERALIS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=1200,
                    optimal_fiber_length=0.065,
                    tendon_slack_length=0.23,
                    pennation_angle=np.radians(12),
                    max_contraction_velocity=0.8,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            MuscleGroup.SOLEUS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=2800,
                    optimal_fiber_length=0.04,
                    tendon_slack_length=0.26,
                    pennation_angle=np.radians(30),
                    max_contraction_velocity=0.6,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            # Quadriceps
            MuscleGroup.VASTUS_LATERALIS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=3000,
                    optimal_fiber_length=0.08,
                    tendon_slack_length=0.15,
                    pennation_angle=np.radians(5),
                    max_contraction_velocity=1.0,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            MuscleGroup.RECTUS_FEMORIS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=1200,
                    optimal_fiber_length=0.075,
                    tendon_slack_length=0.31,
                    pennation_angle=np.radians(5),
                    max_contraction_velocity=1.0,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            # Hamstrings
            MuscleGroup.BICEPS_FEMORIS_LONG: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=1400,
                    optimal_fiber_length=0.109,
                    tendon_slack_length=0.341,
                    pennation_angle=np.radians(0),
                    max_contraction_velocity=1.0,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            # Hip extensors
            MuscleGroup.GLUTEUS_MAXIMUS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=3000,
                    optimal_fiber_length=0.142,
                    tendon_slack_length=0.125,
                    pennation_angle=np.radians(3),
                    max_contraction_velocity=1.0,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            # Ankle & Foot Complex
            MuscleGroup.FLEXOR_HALLUCIS_LONGUS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=500,
                    optimal_fiber_length=0.04,
                    tendon_slack_length=0.38,
                    pennation_angle=np.radians(15),
                    max_contraction_velocity=0.8,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
            MuscleGroup.PERONEUS_LONGUS: Thelen2003MuscleDynamics(
                MuscleTendonParameters(
                    max_isometric_force=825,
                    optimal_fiber_length=0.05,
                    tendon_slack_length=0.345,
                    pennation_angle=np.radians(15),
                    max_contraction_velocity=0.8,
                    activation_time_constant=0.01,
                    deactivation_time_constant=0.04
                )
            ),
        }
        
        self.neural_model = MuscleNeuralNetwork(input_dim=8, hidden_dim=128)
        self.curve_detector = CurveDetector()
        
    def baseline_excitation(self, muscle: MuscleGroup, stance_time: float) -> float:
        """Calculate baseline muscle excitation pattern"""
        phase = stance_time / 0.2  # Normalize to typical stance time
        
        # Phase-specific activation patterns based on research
        if muscle in [MuscleGroup.GASTROCNEMIUS_MEDIALIS, 
                     MuscleGroup.GASTROCNEMIUS_LATERALIS]:
            return np.sin(np.pi * phase)**2  # Peak in late stance
            
        elif muscle == MuscleGroup.SOLEUS:
            return 0.8 * np.sin(np.pi * (phase + 0.1))**2  # Earlier activation
            
        elif muscle in [MuscleGroup.VASTUS_LATERALIS, 
                       MuscleGroup.VASTUS_MEDIALIS,
                       MuscleGroup.RECTUS_FEMORIS]:
            return np.exp(-((phase - 0.2)/0.1)**2)  # Early stance peak
            
        elif muscle in [MuscleGroup.BICEPS_FEMORIS_LONG, 
                       MuscleGroup.BICEPS_FEMORIS_SHORT]:
            return 0.7 * (np.exp(-((phase - 0.1)/0.1)**2) + 
                         0.5 * np.exp(-((phase - 0.8)/0.1)**2))  # Dual peak
            
        elif muscle == MuscleGroup.GLUTEUS_MAXIMUS:
            return np.exp(-((phase - 0.15)/0.15)**2)  # Early stance
            
        elif muscle == MuscleGroup.TENSOR_FASCIAE_LATAE:
            return 0.6 * (1 - np.exp(-5 * phase)) * np.exp(-2 * phase)  # Curve support
            
        return 0.5  # Default pattern

    def estimate_muscle_excitation(self,
                                 muscle: MuscleGroup,
                                 stance_time: float,
                                 centripetal_force: float,
                                 vertical_oscillation: float,
                                 curve_phase: float) -> float:
        """
        Estimate muscle excitation during curve running
        
        Args:
            muscle: Muscle group
            stance_time: Ground contact time
            centripetal_force: Centripetal force magnitude
            vertical_oscillation: Vertical displacement
            curve_phase: Phase of curve (0-1, where 0.5 is apex)
        """
        features = torch.tensor([
            stance_time,
            centripetal_force,
            vertical_oscillation,
            np.sin(2 * np.pi * curve_phase),  # Circular features
            np.cos(2 * np.pi * curve_phase),
            np.sin(2 * np.pi * curve_phase)**2,  # Sin^2(2*pi*curve_phase)
            np.cos(2 * np.pi * curve_phase)**2,  # Cos^2(2*pi*curve_phase)
            np.sin(4 * np.pi * curve_phase)  # Sin(4*pi*curve_phase)
        ])
        
        # Combine traditional and ML-based estimation with curve-specific adjustments
        base_excitation = self.baseline_excitation(muscle, stance_time)
        ml_excitation = self.neural_model(features).item()
        
        # Adjust for curve running based on muscle function
        curve_factor = self.calculate_curve_factor(muscle, curve_phase)
        
        return base_excitation * curve_factor + 0.3 * ml_excitation

    def calculate_curve_factor(self, muscle: MuscleGroup, curve_phase: float) -> float:
        """Calculate curve-specific activation adjustment"""
        # Inside leg muscles during curve
        inside_boost = 1.2 * np.sin(np.pi * curve_phase)**2
        
        # Outside leg muscles during curve
        outside_boost = 1.15 * np.cos(np.pi * curve_phase)**2
        
        # Upper body compensation
        upper_body_factor = 1.1 + 0.2 * np.sin(2 * np.pi * curve_phase)
        
        # Muscle-specific adjustments
        if muscle in [MuscleGroup.GLUTEUS_MEDIUS, 
                     MuscleGroup.TENSOR_FASCIAE_LATAE,
                     MuscleGroup.PERONEUS_LONGUS,
                     MuscleGroup.PERONEUS_BREVIS]:
            return outside_boost  # Lateral stability
            
        elif muscle in [MuscleGroup.GASTROCNEMIUS_LATERALIS,
                       MuscleGroup.FLEXOR_HALLUCIS_LONGUS,
                       MuscleGroup.ABDUCTOR_HALLUCIS]:
            return inside_boost  # Propulsion and toe-off
            
        elif muscle in [MuscleGroup.LATISSIMUS_DORSI,
                       MuscleGroup.TRAPEZIUS,
                       MuscleGroup.DELTOID]:
            return upper_body_factor  # Upper body compensation
            
        return 1.0

    def analyze_curve_running(self, 
                            track_data: Dict,
                            lane_radius: float) -> Dict[str, Dict]:
        """
        Analyze muscle dynamics during curve running
        
        Args:
            track_data: Track sensor data
            lane_radius: Radius of the running lane
            
        Returns:
            Dictionary of muscle states and metrics
        """
        results = {}
        
        for muscle_group in MuscleGroup:
            states = []
            metrics = []
            
            for feature in track_data['features']:
                props = feature['properties']
                
                # Calculate centripetal force effect
                velocity = props['speed']
                centripetal_force = velocity**2 / lane_radius
                
                # Estimate muscle excitation based on gait phase and curve
                excitation = self.estimate_muscle_excitation(
                    muscle_group,
                    props['stance_time'],
                    centripetal_force,
                    props['vertical_oscillation'],
                    props['curve_phase']
                )
                
                # Update muscle state
                if not states:
                    states.append(self.initialize_muscle_state(muscle_group))
                
                new_state = self.muscles[muscle_group].compute_muscle_dynamics(
                    states[-1],
                    excitation,
                    dt=0.01
                )
                states.append(new_state)
                
                # Calculate metrics
                metrics.append(self.calculate_muscle_metrics(
                    new_state,
                    centripetal_force,
                    props['heart_rate']
                ))
            
            results[muscle_group.value] = {
                'states': states,
                'metrics': metrics
            }
            
        return results

    def calculate_muscle_metrics(self,
                               state: MuscleState,
                               centripetal_force: float,
                               heart_rate: float) -> Dict:
        """Calculate muscle performance metrics"""
        return {
            'power': state.tendon_force * state.fiber_velocity,
            'efficiency': self.calculate_efficiency(state, heart_rate),
            'stress': state.tendon_force / (state.fiber_length * 
                     np.sin(state.pennation_angle)),
            'strain_energy': self.calculate_strain_energy(state),
            'centripetal_contribution': self.estimate_centripetal_contribution(
                state, centripetal_force
            )
        }

    def calculate_efficiency(self, state: MuscleState, heart_rate: float) -> float:
        """Calculate muscle efficiency"""
        mechanical_power = state.tendon_force * state.fiber_velocity
        metabolic_power = state.activation * heart_rate / 180.0  # Simplified model
        return mechanical_power / metabolic_power if metabolic_power > 0 else 0.0

    def calculate_strain_energy(self, state: MuscleState) -> float:
        """Calculate muscle-tendon strain energy"""
        return 0.5 * state.tendon_force * (state.tendon_length - 
               self.muscles[MuscleGroup.GASTROCNEMIUS_MEDIALIS].params.tendon_slack_length)

    def estimate_centripetal_contribution(self,
                                        state: MuscleState,
                                        centripetal_force: float) -> float:
        """Estimate muscle contribution to centripetal force"""
        return state.tendon_force * np.sin(state.pennation_angle) / centripetal_force

    def analyze_track_data(self, track_data: Dict) -> Dict:
        """
        Analyze track data including curve detection
        
        Args:
            track_data: Track data dictionary
            
        Returns:
            Analysis results including curve segments
        """
        # Detect curves
        curve_segments = self.curve_detector.detect_curves(track_data)
        
        # Analyze each segment
        results = []
        for segment in curve_segments:
            # Calculate curve phase for each point
            n_points = segment['end_idx'] - segment['start_idx'] + 1
            curve_phases = np.linspace(0, 1, n_points)
            
            # Analyze muscle dynamics for curve
            muscle_dynamics = self.analyze_curve_running(
                {
                    'features': track_data['features'][segment['start_idx']:segment['end_idx']+1]
                },
                self.curve_detector.calculate_curve_radius(
                    [(f['geometry']['coordinates'][0], f['geometry']['coordinates'][1])
                     for f in track_data['features'][segment['start_idx']:segment['end_idx']+1]]
                )
            )
            
            results.append({
                'segment': segment,
                'characteristics': self.curve_detector.analyze_curve_characteristics(segment),
                'muscle_dynamics': muscle_dynamics
            })
            
        return {
            'curve_segments': results,
            'total_curves': len(results),
            'straight_segments': self.identify_straight_segments(track_data, curve_segments)
        }
    
    def identify_straight_segments(self, 
                                 track_data: Dict,
                                 curve_segments: List[Dict]) -> List[Dict]:
        """Identify straight running segments between curves"""
        straight_segments = []
        last_end = 0
        
        for segment in curve_segments:
            if segment['start_idx'] > last_end:
                straight_segments.append({
                    'start_idx': last_end,
                    'end_idx': segment['start_idx'] - 1,
                    'type': 'straight',
                    'metrics': [
                        {
                            'speed': f['properties']['speed'],
                            'timestamp': f['properties']['timestamp']
                        }
                        for f in track_data['features'][last_end:segment['start_idx']]
                    ]
                })
            last_end = segment['end_idx'] + 1
        
        return straight_segments

class MuscleNeuralNetwork(nn.Module):
    """Neural network for muscle excitation prediction"""
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class MuscleDynamicsVisualizer:
    """Visualization tools for muscle dynamics"""
    def plot_muscle_states(self, results: Dict[str, Dict]):
        """Plot muscle states over time"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for muscle_name, data in results.items():
            states = data['states']
            
            # Extract time series
            activations = [s.activation for s in states]
            lengths = [s.fiber_length for s in states]
            velocities = [s.fiber_velocity for s in states]
            forces = [s.tendon_force for s in states]
            
            # Plot
            axes[0, 0].plot(activations, label=muscle_name)
            axes[0, 1].plot(lengths, label=muscle_name)
            axes[1, 0].plot(velocities, label=muscle_name)
            axes[1, 1].plot(forces, label=muscle_name)
            
        # Set labels
        axes[0, 0].set_title('Muscle Activation')
        axes[0, 1].set_title('Fiber Length')
        axes[1, 0].set_title('Contraction Velocity')
        axes[1, 1].set_title('Tendon Force')
        
        for ax in axes.flat:
            ax.grid(True)
            ax.legend()
            
        plt.tight_layout()
        plt.show()

    def plot_curve_analysis(self, results: Dict[str, Dict], lane_radius: float):
        """Plot curve-specific analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for muscle_name, data in results.items():
            metrics = data['metrics']
            
            # Extract metrics
            powers = [m['power'] for m in metrics]
            efficiencies = [m['efficiency'] for m in metrics]
            centripetal = [m['centripetal_contribution'] for m in metrics]
            
            # Plot
            axes[0, 0].plot(powers, label=muscle_name)
            axes[0, 1].plot(efficiencies, label=muscle_name)
            axes[1, 0].plot(centripetal, label=muscle_name)
            
        # Set labels
        axes[0, 0].set_title('Muscle Power')
        axes[0, 1].set_title('Muscle Efficiency')
        axes[1, 0].set_title('Centripetal Force Contribution')
        
        for ax in axes.flat:
            ax.grid(True)
            ax.legend()
            
        plt.tight_layout()
        plt.show()

# Example usage:
"""
# Initialize analyzers
sprint_dynamics = SprintMuscleDynamics()
visualizer = MuscleDynamicsVisualizer()

# Load track data
with open('track.json', 'r') as f:
    track_data = json.load(f)

# Analyze curve running
lane_radius = 37.72  # Lane 1 radius
results = sprint_dynamics.analyze_track_data(track_data)

# Visualize results
visualizer.plot_muscle_states(results['muscle_dynamics'])
visualizer.plot_curve_analysis(results['muscle_dynamics'], lane_radius)
"""
