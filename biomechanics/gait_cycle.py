import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

class GaitPhase(Enum):
    INITIAL_CONTACT = "initial_contact"
    LOADING_RESPONSE = "loading_response"
    MID_STANCE = "mid_stance"
    TERMINAL_STANCE = "terminal_stance"
    PRE_SWING = "pre_swing"
    INITIAL_SWING = "initial_swing"
    MID_SWING = "mid_swing"
    TERMINAL_SWING = "terminal_swing"

@dataclass
class GaitParameters:
    stance_time: float
    swing_time: float
    cycle_time: float
    float_time: float
    step_length: float
    stride_length: float
    cadence: float
    speed: float
    vertical_oscillation: float
    stance_time_balance: float

@dataclass
class PhaseTimings:
    initial_contact: float
    loading_response: float
    mid_stance: float
    terminal_stance: float
    pre_swing: float
    initial_swing: float
    mid_swing: float
    terminal_swing: float

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

@dataclass
class GaitPhase3D:
    position: Vector3D
    velocity: Vector3D
    acceleration: Vector3D
    force: Vector3D
    moment: Vector3D

@dataclass
class JointAngles:
    hip: float
    knee: float
    ankle: float
    pelvis_tilt: float
    pelvis_rotation: float

class GaitPattern(Enum):
    NORMAL = "normal"
    OVERPRONATION = "overpronation"
    SUPINATION = "supination"
    ASYMMETRIC = "asymmetric"
    EFFICIENT = "efficient"
    FATIGUE = "fatigue"

class GaitNeuralNet(nn.Module):
    def __init__(self, input_size: int):
        super(GaitNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(GaitPattern))
        )
    
    def forward(self, x):
        return self.network(x)

class GaitAnalyzer:
    def __init__(self, athlete_height: float = 1.80, athlete_mass: float = 70.0):
        """
        Initialize gait analyzer with athlete parameters
        
        Args:
            athlete_height: Height in meters
            athlete_mass: Mass in kg
        """
        self.height = athlete_height
        self.mass = athlete_mass
        self.g = 9.81  # gravitational acceleration
        self.leg_length = self.estimate_leg_length()
        self.ml_model = self._initialize_ml_model()
        self.neural_net = GaitNeuralNet(input_size=12)
        self.scaler = StandardScaler()
        
    def estimate_leg_length(self) -> float:
        """Calculate estimated leg length using anthropometric ratios"""
        return self.height * 0.53  # Based on anthropometric studies
        
    def calculate_dimensionless_parameters(self, 
                                         speed: float, 
                                         stride_length: float) -> Dict[str, float]:
        """
        Calculate dimensionless gait parameters
        
        Args:
            speed: Forward velocity in m/s
            stride_length: Stride length in meters
            
        Returns:
            Dictionary of dimensionless parameters
        """
        froude_number = speed ** 2 / (self.g * self.leg_length)
        stride_number = stride_length / self.leg_length
        
        return {
            'froude_number': froude_number,
            'stride_number': stride_number,
            'dimensionless_speed': speed / math.sqrt(self.g * self.leg_length)
        }

    def calculate_phase_durations(self, 
                                stance_time: float, 
                                cycle_time: float) -> PhaseTimings:
        """
        Calculate detailed phase timings based on research
        
        Args:
            stance_time: Total stance time in seconds
            cycle_time: Complete gait cycle time in seconds
            
        Returns:
            PhaseTimings object with detailed breakdown
        """
        # Phase timing proportions based on research
        swing_time = cycle_time - stance_time
        
        return PhaseTimings(
            initial_contact=0.02 * cycle_time,
            loading_response=0.10 * cycle_time,
            mid_stance=0.20 * cycle_time,
            terminal_stance=0.20 * cycle_time,
            pre_swing=0.10 * cycle_time,
            initial_swing=0.13 * cycle_time,
            mid_swing=0.12 * cycle_time,
            terminal_swing=0.13 * cycle_time
        )

    def calculate_spatial_parameters(self,
                                  speed: float,
                                  stance_time: float,
                                  step_length: float) -> Dict[str, float]:
        """
        Calculate spatial parameters of gait
        
        Args:
            speed: Forward velocity in m/s
            stance_time: Stance phase duration in seconds
            step_length: Step length in meters
            
        Returns:
            Dictionary of spatial parameters
        """
        stance_distance = speed * stance_time
        stride_length = step_length * 2
        
        return {
            'stance_distance': stance_distance,
            'stride_length': stride_length,
            'step_width': self.estimate_step_width(speed),
            'progression_angle': self.calculate_progression_angle(step_length)
        }

    def estimate_step_width(self, speed: float) -> float:
        """
        Estimate step width based on speed
        
        Args:
            speed: Forward velocity in m/s
            
        Returns:
            Estimated step width in meters
        """
        # Step width typically decreases with speed
        base_width = self.leg_length * 0.15
        speed_factor = max(0.5, 1.0 - (speed / 8.0))
        return base_width * speed_factor

    def calculate_progression_angle(self, step_length: float) -> float:
        """
        Calculate foot progression angle
        
        Args:
            step_length: Step length in meters
            
        Returns:
            Progression angle in degrees
        """
        # Typical progression angle is 5-7 degrees
        base_angle = 6.0
        length_factor = step_length / self.leg_length
        return base_angle * length_factor

    def calculate_stability_metrics(self,
                                 vertical_oscillation: float,
                                 stance_time_balance: float) -> Dict[str, float]:
        """
        Calculate stability-related metrics
        
        Args:
            vertical_oscillation: Vertical displacement in meters
            stance_time_balance: L/R stance time balance
            
        Returns:
            Dictionary of stability metrics
        """
        stability_index = 1.0 - (vertical_oscillation / (self.leg_length * 0.1))
        symmetry_index = abs(50.0 - stance_time_balance) / 50.0
        
        return {
            'stability_index': stability_index,
            'symmetry_index': symmetry_index,
            'vertical_stiffness': self.calculate_vertical_stiffness(vertical_oscillation)
        }

    def calculate_vertical_stiffness(self, vertical_oscillation: float) -> float:
        """
        Calculate vertical stiffness
        
        Args:
            vertical_oscillation: Vertical displacement in meters
            
        Returns:
            Vertical stiffness in N/m
        """
        peak_force = self.mass * self.g * 2.5  # Typical peak force ~2.5 BW
        return peak_force / vertical_oscillation if vertical_oscillation > 0 else float('inf')

    def analyze_gait_cycle(self, track_data: Dict) -> Dict[str, Dict]:
        """
        Perform comprehensive gait analysis from track data
        
        Args:
            track_data: Dictionary containing track data
            
        Returns:
            Dictionary containing detailed gait analysis
        """
        results = {}
        
        for i, feature in enumerate(track_data['features']):
            props = feature['properties']
            
            # Convert units
            speed = props['speed']  # m/s
            stance_time = props['stance_time'] / 1000  # ms to s
            cadence = props['cadence']  # steps/min
            step_length = props['step_length'] / 1000  # mm to m
            vertical_oscillation = props['vertical_oscillation'] / 1000  # mm to m
            
            # Calculate cycle time
            cycle_time = 60.0 / cadence * 2  # seconds per complete cycle
            swing_time = cycle_time - stance_time
            
            # Basic gait parameters
            params = GaitParameters(
                stance_time=stance_time,
                swing_time=swing_time,
                cycle_time=cycle_time,
                float_time=swing_time * 0.2,  # Approximate float time
                step_length=step_length,
                stride_length=step_length * 2,
                cadence=cadence,
                speed=speed,
                vertical_oscillation=vertical_oscillation,
                stance_time_balance=props['stance_time_balance']
            )
            
            # Calculate all metrics
            phase_timings = self.calculate_phase_durations(stance_time, cycle_time)
            spatial_params = self.calculate_spatial_parameters(speed, stance_time, step_length)
            stability_metrics = self.calculate_stability_metrics(
                vertical_oscillation, props['stance_time_balance']
            )
            dimensionless_params = self.calculate_dimensionless_parameters(
                speed, spatial_params['stride_length']
            )
            
            # Store results
            results[i] = {
                'timestamp': props['timestamp'],
                'basic_parameters': vars(params),
                'phase_timings': vars(phase_timings),
                'spatial_parameters': spatial_params,
                'stability_metrics': stability_metrics,
                'dimensionless_parameters': dimensionless_params,
                'metabolic_power': self.estimate_metabolic_power(
                    speed, props['heart_rate']
                ),
                'mechanical_efficiency': self.calculate_mechanical_efficiency(
                    speed, vertical_oscillation, stance_time
                )
            }
            
        return results

    def estimate_metabolic_power(self, speed: float, heart_rate: float) -> float:
        """
        Estimate metabolic power output
        
        Args:
            speed: Forward velocity in m/s
            heart_rate: Heart rate in bpm
            
        Returns:
            Estimated metabolic power in Watts
        """
        # Basic metabolic power estimation
        base_metabolic_rate = 1.0  # W/kg
        speed_component = 0.98 * speed ** 2
        hr_component = (heart_rate / 180.0) ** 1.5
        
        return self.mass * (base_metabolic_rate + speed_component) * hr_component

    def calculate_mechanical_efficiency(self,
                                     speed: float,
                                     vertical_oscillation: float,
                                     stance_time: float) -> float:
        """
        Calculate mechanical efficiency of gait
        
        Args:
            speed: Forward velocity in m/s
            vertical_oscillation: Vertical displacement in meters
            stance_time: Stance time in seconds
            
        Returns:
            Mechanical efficiency (dimensionless)
        """
        # Calculate mechanical work
        vertical_work = self.mass * self.g * vertical_oscillation
        horizontal_work = 0.5 * self.mass * speed ** 2
        
        # Calculate power
        total_power = (vertical_work + horizontal_work) / stance_time
        
        # Estimate metabolic power
        metabolic_power = self.mass * (4.0 + 0.98 * speed ** 2)
        
        return total_power / metabolic_power if metabolic_power > 0 else 0.0

    def calculate_3d_kinematics(self,
                               vertical_oscillation: float,
                               vertical_ratio: float,
                               stance_time_balance: float,
                               speed: float,
                               stance_time: float) -> GaitPhase3D:
        """
        Calculate 3D kinematics of the gait cycle
        
        Args:
            vertical_oscillation: Vertical displacement in meters
            vertical_ratio: Vertical oscillation to step length ratio
            stance_time_balance: L/R stance time balance
            speed: Forward velocity in m/s
            stance_time: Stance phase duration in seconds
            
        Returns:
            GaitPhase3D object with position, velocity, acceleration, force and moment
        """
        # Vertical component calculations
        vertical_velocity = vertical_oscillation / (stance_time / 2)
        vertical_acceleration = vertical_velocity / (stance_time / 4)
        
        # Mediolateral calculations (based on stance_time_balance)
        lateral_displacement = self.leg_length * (stance_time_balance - 50) / 100
        lateral_velocity = lateral_displacement / stance_time
        
        # Create 3D vectors
        position = Vector3D(
            x=speed * stance_time,  # forward
            y=lateral_displacement,  # lateral
            z=vertical_oscillation  # vertical
        )
        
        velocity = Vector3D(
            x=speed,
            y=lateral_velocity,
            z=vertical_velocity
        )
        
        acceleration = Vector3D(
            x=speed / stance_time,
            y=lateral_velocity / stance_time,
            z=vertical_acceleration
        )
        
        # Force calculations in 3D
        force = self.calculate_3d_forces(acceleration)
        moment = self.calculate_3d_moments(force, position)
        
        return GaitPhase3D(position, velocity, acceleration, force, moment)

    def calculate_3d_forces(self, acceleration: Vector3D) -> Vector3D:
        """Calculate 3D force components"""
        return Vector3D(
            x=self.mass * acceleration.x,
            y=self.mass * acceleration.y,
            z=self.mass * (acceleration.z + self.g)
        )

    def calculate_3d_moments(self, force: Vector3D, position: Vector3D) -> Vector3D:
        """Calculate 3D moments around CoM"""
        return Vector3D(
            x=force.y * position.z - force.z * position.y,
            y=force.z * position.x - force.x * position.z,
            z=force.x * position.y - force.y * position.x
        )

    def calculate_joint_angles(self,
                             vertical_oscillation: float,
                             step_length: float) -> JointAngles:
        """Calculate joint angles during gait"""
        # Simplified joint angle calculations
        hip_angle = math.atan2(step_length, self.leg_length)
        knee_angle = hip_angle * 1.5
        ankle_angle = hip_angle * 0.7
        pelvis_tilt = vertical_oscillation / self.leg_length * 180 / math.pi
        pelvis_rotation = step_length / self.leg_length * 15
        
        return JointAngles(
            hip=math.degrees(hip_angle),
            knee=math.degrees(knee_angle),
            ankle=math.degrees(ankle_angle),
            pelvis_tilt=pelvis_tilt,
            pelvis_rotation=pelvis_rotation
        )

    def classify_gait_pattern(self, features: Dict[str, float]) -> GaitPattern:
        """Classify gait pattern using ML model"""
        feature_vector = np.array([
            features['vertical_oscillation'],
            features['vertical_ratio'],
            features['stance_time_balance'],
            features['speed'],
            features['cadence'],
            features['step_length']
        ]).reshape(1, -1)
        
        scaled_features = self.scaler.transform(feature_vector)
        prediction = self.ml_model.predict(scaled_features)
        return GaitPattern(prediction[0])

    def visualize_3d_gait(self, gait_phase: GaitPhase3D):
        """Create 3D visualization of gait cycle"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot3D([0, gait_phase.position.x],
                 [0, gait_phase.position.y],
                 [0, gait_phase.position.z])
        
        # Plot force vectors
        ax.quiver(gait_phase.position.x,
                 gait_phase.position.y,
                 gait_phase.position.z,
                 gait_phase.force.x,
                 gait_phase.force.y,
                 gait_phase.force.z)
        
        ax.set_xlabel('Forward (m)')
        ax.set_ylabel('Lateral (m)')
        ax.set_zlabel('Vertical (m)')
        plt.title('3D Gait Analysis')
        plt.show()

    def visualize_joint_angles(self, angles: JointAngles):
        """Visualize joint angles"""
        angles_dict = vars(angles)
        plt.figure(figsize=(10, 6))
        plt.bar(angles_dict.keys(), angles_dict.values())
        plt.title('Joint Angles During Gait')
        plt.ylabel('Degrees')
        plt.xticks(rotation=45)
        plt.show()

    def plot_gait_cycle_phases(self, phase_timings: PhaseTimings):
        """Create circular plot of gait cycle phases"""
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        # Convert phase timings to angles
        phase_angles = {k: v * 2 * np.pi for k, v in vars(phase_timings).items()}
        
        # Plot phases
        bottom = 0
        for phase, angle in phase_angles.items():
            ax.bar(0, angle, bottom=bottom, width=np.pi/4)
            bottom += angle
            
        plt.title('Gait Cycle Phase Distribution')
        plt.show()
