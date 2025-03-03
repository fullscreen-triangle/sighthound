import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial.transform import Rotation
import pandas as pd

@dataclass
class SlopeVector3D:
    incline: float      # Equivalent incline angle in degrees
    lateral_tilt: float # Lateral tilt angle in degrees
    total_angle: float  # Combined angle magnitude
    
@dataclass
class BiomechanicalLoad:
    vertical_force: float    # Vertical ground reaction force (N)
    propulsive_force: float  # Propulsive force (N)
    joint_load: float        # Combined joint loading (N)
    metabolic_cost: float    # Metabolic cost (J/kg/m)

class EquivalentSlopeAnalyzer:
    def __init__(self, athlete_height: float = 1.80, athlete_mass: float = 70.0):
        """
        Initialize equivalent slope analyzer
        
        Args:
            athlete_height: Height in meters
            athlete_mass: Mass in kg
        """
        self.height = athlete_height
        self.mass = athlete_mass
        self.g = 9.81  # gravitational acceleration m/s²
        self.leg_length = self.height * 0.53  # anthropometric estimation
        
    def calculate_equivalent_slope(self,
                                 speed: float,
                                 vertical_oscillation: float,
                                 stance_time: float,
                                 stance_time_balance: float) -> SlopeVector3D:
        """
        Calculate 3D equivalent slope based on running parameters
        
        Args:
            speed: Forward velocity (m/s)
            vertical_oscillation: Vertical displacement (m)
            stance_time: Ground contact time (s)
            stance_time_balance: L/R balance (%)
            
        Returns:
            SlopeVector3D object containing slope components
        """
        # Forward equivalent slope (based on sprint posture)
        # Using Kram & Taylor's relationship between mechanical work and slope
        mechanical_power = self.mass * self.g * vertical_oscillation / stance_time
        forward_slope = math.degrees(math.asin(mechanical_power / (self.mass * self.g * speed)))
        
        # Lateral slope (based on stance time balance)
        lateral_imbalance = (stance_time_balance - 50.0) / 50.0  # normalized to ±1
        lateral_slope = lateral_imbalance * 5.0  # max 5 degrees lateral tilt
        
        # Calculate total slope magnitude
        total_slope = math.sqrt(forward_slope**2 + lateral_slope**2)
        
        return SlopeVector3D(
            incline=forward_slope,
            lateral_tilt=lateral_slope,
            total_angle=total_slope
        )
    
    def calculate_biomechanical_load(self,
                                   slope: SlopeVector3D,
                                   speed: float) -> BiomechanicalLoad:
        """
        Calculate biomechanical loads based on equivalent slope
        
        Args:
            slope: SlopeVector3D object
            speed: Forward velocity (m/s)
            
        Returns:
            BiomechanicalLoad object
        """
        # Convert angles to radians
        incline_rad = math.radians(slope.incline)
        lateral_rad = math.radians(slope.lateral_tilt)
        
        # Calculate force components
        base_force = self.mass * self.g
        
        # Vertical force (increased with slope)
        vertical_force = base_force * math.cos(incline_rad)
        
        # Propulsive force (accounts for slope)
        propulsive_force = base_force * math.sin(incline_rad)
        
        # Joint loading (increases with slope)
        joint_load = base_force * (1 + 0.2 * math.tan(incline_rad))
        
        # Metabolic cost calculation (Minetti's equation)
        metabolic_cost = 3.6 + 0.35 * math.tan(incline_rad)**2 + 0.07 * speed**2
        
        return BiomechanicalLoad(
            vertical_force=vertical_force,
            propulsive_force=propulsive_force,
            joint_load=joint_load,
            metabolic_cost=metabolic_cost
        )
    
    def analyze_track_data(self, track_data: Dict) -> List[Dict]:
        """
        Analyze track data for equivalent slope metrics
        
        Args:
            track_data: Dictionary containing track data
            
        Returns:
            List of analysis results
        """
        results = []
        
        for feature in track_data['features']:
            props = feature['properties']
            
            # Convert units
            speed = props['speed']  # m/s
            vertical_osc = props['vertical_oscillation'] / 1000  # mm to m
            stance_time = props['stance_time'] / 1000  # ms to s
            
            # Calculate equivalent slope
            slope = self.calculate_equivalent_slope(
                speed=speed,
                vertical_oscillation=vertical_osc,
                stance_time=stance_time,
                stance_time_balance=props['stance_time_balance']
            )
            
            # Calculate biomechanical load
            load = self.calculate_biomechanical_load(slope, speed)
            
            results.append({
                'timestamp': props['timestamp'],
                'slope': vars(slope),
                'biomechanical_load': vars(load),
                'speed': speed,
                'vertical_oscillation': vertical_osc
            })
            
        return results
    
    def visualize_3d_slope(self, slope: SlopeVector3D, speed: float):
        """
        Create 3D visualization of equivalent slope
        
        Args:
            slope: SlopeVector3D object
            speed: Forward velocity (m/s)
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create slope plane
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        
        # Calculate Z coordinates for slope plane
        Z = X * math.tan(math.radians(slope.incline)) + \
            Y * math.tan(math.radians(slope.lateral_tilt))
        
        # Plot slope surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        
        # Plot running direction vector
        ax.quiver(0, 0, 0, 
                 speed/5, 0, speed/5 * math.tan(math.radians(slope.incline)),
                 color='r', arrow_length_ratio=0.2)
        
        # Set labels and title
        ax.set_xlabel('Lateral Distance (m)')
        ax.set_ylabel('Forward Distance (m)')
        ax.set_zlabel('Vertical Distance (m)')
        plt.title(f'Equivalent Slope Analysis\nTotal Angle: {slope.total_angle:.1f}°')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, label='Elevation (m)')
        plt.show()
        
    def plot_slope_profile(self, results: List[Dict]):
        """
        Plot slope profile over time
        
        Args:
            results: List of analysis results
        """
        timestamps = [r['timestamp'] for r in results]
        inclines = [r['slope']['incline'] for r in results]
        lateral_tilts = [r['slope']['lateral_tilt'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot incline
        ax1.plot(timestamps, inclines, 'b-', label='Forward Incline')
        ax1.set_ylabel('Incline (degrees)')
        ax1.set_title('Equivalent Slope Profile')
        ax1.grid(True)
        ax1.legend()
        
        # Plot lateral tilt
        ax2.plot(timestamps, lateral_tilts, 'r-', label='Lateral Tilt')
        ax2.set_ylabel('Lateral Tilt (degrees)')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_biomechanical_loads(self, results: List[Dict]):
        """
        Plot biomechanical load metrics
        
        Args:
            results: List of analysis results
        """
        df = pd.DataFrame([r['biomechanical_load'] for r in results])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df)
        plt.title('Biomechanical Load Distribution')
        plt.ylabel('Force (N) / Cost (J/kg/m)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
