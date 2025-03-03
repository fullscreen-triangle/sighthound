import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ForceComponents:
    horizontal: float  # Horizontal force in Newtons
    vertical: float   # Vertical force in Newtons
    resultant: float  # Resultant force magnitude
    angle: float      # Force angle in degrees

class SprintDynamics:
    def __init__(self, athlete_mass: float = 70.0, athlete_height: float = 1.80):
        """
        Initialize sprint dynamics calculator
        
        Args:
            athlete_mass: Mass of the athlete in kg
            athlete_height: Height of the athlete in meters
        """
        self.mass = athlete_mass
        self.height = athlete_height
        self.g = 9.81  # gravitational acceleration m/s²
        
        # Typical 400m sprint parameters
        self.drag_coefficient = 0.9  # Approximate drag coefficient for running
        self.air_density = 1.225  # kg/m³ at sea level
        self.frontal_area = self.estimate_frontal_area()
        
    def estimate_frontal_area(self) -> float:
        """
        Estimate athlete's frontal area using Du Bois formula
        Returns:
            Estimated frontal area in m²
        """
        return 0.5 * (0.007184 * (self.height * 100) ** 0.725 * self.mass ** 0.425)

    def calculate_ground_reaction_forces(self, 
                                      velocity: float,
                                      stance_time: float,
                                      vertical_oscillation: float,
                                      phase: str = 'mid_stance') -> ForceComponents:
        """
        Calculate ground reaction forces during stance phase with enhanced modeling
        
        Args:
            velocity: Current velocity in m/s
            stance_time: Ground contact time in seconds
            vertical_oscillation: Vertical displacement in meters
            phase: Stance phase ('initial_contact', 'mid_stance', 'propulsion')
            
        Returns:
            ForceComponents object containing force magnitudes and angle
        """
        # Phase-specific force multipliers based on research
        phase_multipliers = {
            'initial_contact': 1.2,  # Impact peak
            'mid_stance': 1.0,       # Active peak
            'propulsion': 1.5        # Push-off peak
        }
        
        # Vertical force components
        # 1. Weight component
        weight_force = self.mass * self.g
        
        # 2. Impact and acceleration component
        vertical_acceleration = 2 * vertical_oscillation / (stance_time ** 2)
        impact_force = self.mass * vertical_acceleration
        
        # 3. Phase-specific adjustment
        phase_multiplier = phase_multipliers.get(phase, 1.0)
        vertical_force = (weight_force + impact_force) * phase_multiplier
        
        # Horizontal force components
        # 1. Air resistance
        air_resistance = 0.5 * self.drag_coefficient * self.air_density * \
                        self.frontal_area * velocity ** 2
                        
        # 2. Propulsive force (considering stance time and velocity)
        propulsive_force = self.mass * (velocity / stance_time)
        
        # 3. Friction component (approximately 0.2-0.3 times vertical force)
        friction_coefficient = 0.25  # Typical track surface
        friction_force = vertical_force * friction_coefficient
        
        # Total horizontal force
        horizontal_force = propulsive_force + air_resistance
        
        # Ensure horizontal force doesn't exceed friction limit
        horizontal_force = min(horizontal_force, friction_force)
        
        # Calculate resultant force and angle
        resultant = np.sqrt(vertical_force ** 2 + horizontal_force ** 2)
        angle = np.degrees(np.arctan2(vertical_force, horizontal_force))
        
        return ForceComponents(
            horizontal=horizontal_force,
            vertical=vertical_force,
            resultant=resultant,
            angle=angle
        )

    def estimate_stance_phase(self, 
                            stance_time: float, 
                            current_time: float) -> str:
        """
        Estimate the stance phase based on timing
        
        Args:
            stance_time: Total stance duration in seconds
            current_time: Current time within stance in seconds
            
        Returns:
            Stance phase classification
        """
        if current_time < stance_time * 0.3:
            return 'initial_contact'
        elif current_time < stance_time * 0.7:
            return 'mid_stance'
        else:
            return 'propulsion'

    def calculate_power_output(self, force: ForceComponents, velocity: float) -> float:
        """
        Calculate instantaneous power output
        
        Args:
            force: ForceComponents object
            velocity: Current velocity in m/s
            
        Returns:
            Power output in Watts
        """
        return force.horizontal * velocity

    def calculate_mechanical_efficiency(self, 
                                     power_output: float,
                                     heart_rate: float) -> float:
        """
        Estimate mechanical efficiency based on heart rate and power output
        
        Args:
            power_output: Mechanical power output in Watts
            heart_rate: Current heart rate in bpm
            
        Returns:
            Estimated mechanical efficiency (0-1)
        """
        # Estimate metabolic power using heart rate (simplified model)
        max_hr = 220 - 25  # Assuming 25-year-old athlete
        hr_ratio = heart_rate / max_hr
        
        # Approximate metabolic power (rough estimation)
        metabolic_power = power_output * (1 + hr_ratio)
        
        return power_output / metabolic_power if metabolic_power > 0 else 0

    def analyze_track_data(self, track_file_path: str) -> Dict:
        """
        Analyze JSON track data and compute dynamics metrics
        
        Args:
            track_file_path: Path to track JSON file
            
        Returns:
            Dictionary containing analyzed metrics
        """
        with open(track_file_path, 'r') as f:
            data = json.load(f)
            
        metrics = []
        
        for feature in data['features']:
            props = feature['properties']
            
            # Convert speed from km/h to m/s if needed
            velocity = props['speed']
            stance_time = props['stance_time'] / 1000  # Convert to seconds
            vertical_osc = props['vertical_oscillation'] / 1000  # Convert to meters
            
            # Calculate forces
            forces = self.calculate_ground_reaction_forces(
                velocity=velocity,
                stance_time=stance_time,
                vertical_oscillation=vertical_osc
            )
            
            # Calculate power
            power = self.calculate_power_output(forces, velocity)
            
            # Calculate efficiency
            efficiency = self.calculate_mechanical_efficiency(
                power_output=power,
                heart_rate=props['heart_rate']
            )
            
            metrics.append({
                'timestamp': props['timestamp'],
                'forces': {
                    'horizontal': forces.horizontal,
                    'vertical': forces.vertical,
                    'resultant': forces.resultant,
                    'angle': forces.angle
                },
                'power_output': power,
                'mechanical_efficiency': efficiency,
                'velocity': velocity,
                'heart_rate': props['heart_rate']
            })
            
        return {
            'metrics': metrics,
            'summary': self._calculate_summary_statistics(metrics)
        }
    
    def _calculate_summary_statistics(self, metrics: List[Dict]) -> Dict:
        """
        Calculate summary statistics from analyzed metrics
        
        Args:
            metrics: List of calculated metrics
            
        Returns:
            Dictionary containing summary statistics
        """
        powers = [m['power_output'] for m in metrics]
        forces = [m['forces']['resultant'] for m in metrics]
        efficiencies = [m['mechanical_efficiency'] for m in metrics]
        
        return {
            'average_power': np.mean(powers),
            'peak_power': np.max(powers),
            'average_force': np.mean(forces),
            'peak_force': np.max(forces),
            'average_efficiency': np.mean(efficiencies),
            'number_of_samples': len(metrics)
        }
