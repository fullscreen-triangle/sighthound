import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import math
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import torch
import torch.nn as nn

@dataclass
class Vector3D:
    x: float
    y: float
    z: float
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)

@dataclass
class LinkageState:
    position: Vector3D
    velocity: Vector3D
    acceleration: Vector3D
    angular_velocity: Vector3D
    angular_acceleration: Vector3D
    time: float

@dataclass
class AdvancedLinkModel:
    """Model for advanced linkage dynamics"""
    mass: float
    length: float
    inertia: float
    damping: float
    stiffness: float
    
    def calculate_energy(self, state: LinkageState) -> Dict[str, float]:
        """Calculate kinetic and potential energy"""
        # Kinetic energy
        linear_ke = 0.5 * self.mass * (state.velocity.magnitude() ** 2)
        angular_ke = 0.5 * self.inertia * (state.angular_velocity.magnitude() ** 2)
        
        # Potential energy (simplified)
        pe = self.mass * 9.81 * state.position.z
        
        # Elastic energy
        elastic_e = 0.5 * self.stiffness * (state.position.magnitude() ** 2)
        
        return {
            "linear_kinetic": linear_ke,
            "angular_kinetic": angular_ke,
            "potential": pe,
            "elastic": elastic_e,
            "total": linear_ke + angular_ke + pe + elastic_e
        }

@dataclass
class JerkMetrics:
    linear_jerk: float      # m/s³
    angular_jerk: float     # rad/s³
    jerk_cost: float        # Objective function value
    smoothness_index: float # Normalized smoothness metric

@dataclass
class EnergyMetrics:
    kinetic: float      # Kinetic energy (J)
    potential: float    # Potential energy (J)
    total: float        # Total mechanical energy (J)
    power: float        # Instantaneous power (W)
    work_done: float    # Work done (J)

@dataclass
class Constraints:
    max_velocity: float     # Maximum allowable velocity
    max_acceleration: float # Maximum allowable acceleration
    joint_limits: Dict      # Joint angle limits
    energy_threshold: float # Maximum energy expenditure

class MinimumJerkAnalyzer:
    def __init__(self, dt: float = 0.01):
        """
        Initialize minimum jerk analyzer
        
        Args:
            dt: Time step for calculations (seconds)
        """
        self.dt = dt
        self.g = 9.81  # gravitational acceleration
        
    def calculate_minimum_jerk_trajectory(self,
                                        x0: float,
                                        xf: float,
                                        y0: float,
                                        yf: float,
                                        duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate minimum jerk trajectory between two points
        
        Args:
            x0, y0: Initial position
            xf, yf: Final position
            duration: Movement duration (seconds)
            
        Returns:
            Tuple of position arrays (x, y)
        """
        t = np.arange(0, duration + self.dt, self.dt)
        tau = t / duration
        
        # Minimum jerk polynomial coefficients
        min_jerk_poly = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        # Calculate positions
        x = x0 + (xf - x0) * min_jerk_poly
        y = y0 + (yf - y0) * min_jerk_poly
        
        return x, y
    
    def calculate_derivatives(self,
                            positions: np.ndarray,
                            duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate velocity, acceleration, and jerk from position data
        
        Args:
            positions: Array of positions
            duration: Movement duration (seconds)
            
        Returns:
            Tuple of (velocity, acceleration, jerk) arrays
        """
        dt = duration / (len(positions) - 1)
        
        # Calculate derivatives using central differences
        velocity = np.gradient(positions, dt)
        acceleration = np.gradient(velocity, dt)
        jerk = np.gradient(acceleration, dt)
        
        return velocity, acceleration, jerk
    
    def calculate_jerk_cost(self, jerk: np.ndarray, duration: float) -> float:
        """
        Calculate the jerk cost function value
        
        Args:
            jerk: Array of jerk values
            duration: Movement duration (seconds)
            
        Returns:
            Jerk cost value
        """
        return 0.5 * np.trapz(jerk**2, dx=self.dt)
    
    class SingleLinkModel:
        def __init__(self, length: float, mass: float):
            """
            Initialize single link pendulum model
            
            Args:
                length: Link length (m)
                mass: Link mass (kg)
            """
            self.L = length
            self.m = mass
            self.I = mass * length**2 / 3  # Moment of inertia
            
        def equations_of_motion(self, state: np.ndarray, t: float) -> np.ndarray:
            """
            Define equations of motion for single pendulum
            
            Args:
                state: Current state [theta, omega]
                t: Time
                
            Returns:
                State derivatives [omega, alpha]
            """
            theta, omega = state
            
            # Angular acceleration
            alpha = -9.81 * np.sin(theta) / self.L
            
            return np.array([omega, alpha])
            
    class DoubleLinkModel:
        def __init__(self, L1: float, L2: float, m1: float, m2: float):
            """
            Initialize double pendulum model
            
            Args:
                L1, L2: Link lengths (m)
                m1, m2: Link masses (kg)
            """
            self.L1 = L1
            self.L2 = L2
            self.m1 = m1
            self.m2 = m2
            
        def equations_of_motion(self, state: np.ndarray, t: float) -> np.ndarray:
            """
            Define equations of motion for double pendulum
            
            Args:
                state: Current state [theta1, omega1, theta2, omega2]
                t: Time
                
            Returns:
                State derivatives [omega1, alpha1, omega2, alpha2]
            """
            theta1, omega1, theta2, omega2 = state
            
            # Complex equations for double pendulum motion
            # Based on Lagrangian mechanics
            c = np.cos(theta1 - theta2)
            s = np.sin(theta1 - theta2)
            
            # Matrix elements for solving angular accelerations
            M11 = (self.m1 + self.m2) * self.L1**2
            M12 = self.m2 * self.L1 * self.L2 * c
            M21 = M12
            M22 = self.m2 * self.L2**2
            
            # Force terms
            f1 = -self.m2 * self.L1 * self.L2 * omega2**2 * s - \
                 (self.m1 + self.m2) * 9.81 * self.L1 * np.sin(theta1)
            f2 = self.m2 * self.L1 * self.L2 * omega1**2 * s - \
                 self.m2 * 9.81 * self.L2 * np.sin(theta2)
            
            # Solve for angular accelerations
            det = M11 * M22 - M12 * M21
            alpha1 = (M22 * f1 - M12 * f2) / det
            alpha2 = (-M21 * f1 + M11 * f2) / det
            
            return np.array([omega1, alpha1, omega2, alpha2])
    
    def visualize_minimum_jerk(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             vx: np.ndarray,
                             vy: np.ndarray,
                             ax: np.ndarray,
                             ay: np.ndarray,
                             jx: np.ndarray,
                             jy: np.ndarray):
        """
        Create comprehensive visualization of minimum jerk trajectory
        
        Args:
            x, y: Position arrays
            vx, vy: Velocity arrays
            ax, ay: Acceleration arrays
            jx, jy: Jerk arrays
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trajectory plot
        ax1.plot(x, y, 'b-', label='Path')
        ax1.set_title('Minimum Jerk Trajectory')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True)
        
        # Velocity profile
        t = np.arange(0, len(x)) * self.dt
        ax2.plot(t, vx, 'r-', label='X Velocity')
        ax2.plot(t, vy, 'b-', label='Y Velocity')
        ax2.set_title('Velocity Profiles')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True)
        
        # Acceleration profile
        ax3.plot(t, ax, 'r-', label='X Acceleration')
        ax3.plot(t, ay, 'b-', label='Y Acceleration')
        ax3.set_title('Acceleration Profiles')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.legend()
        ax3.grid(True)
        
        # Jerk profile
        ax4.plot(t, jx, 'r-', label='X Jerk')
        ax4.plot(t, jy, 'b-', label='Y Jerk')
        ax4.set_title('Jerk Profiles')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Jerk (m/s³)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def animate_linkage(self,
                       states: List[LinkageState],
                       link_length: float,
                       duration: float):
        """
        Create animation of link movement
        
        Args:
            states: List of LinkageState objects
            link_length: Length of the link
            duration: Animation duration
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-link_length * 1.5, link_length * 1.5)
        ax.set_ylim(-link_length * 1.5, link_length * 1.5)
        
        line, = ax.plot([], [], 'bo-', lw=2)
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(i):
            state = states[i]
            x = [0, state.position.x]
            y = [0, state.position.y]
            line.set_data(x, y)
            return line,
        
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(states), interval=self.dt*1000,
                           blit=True)
        
        plt.grid(True)
        plt.title('Link Movement Animation')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.show()

    def integrate_with_advanced_model(self, 
                                    advanced_model: AdvancedLinkModel,
                                    start_state: np.ndarray,
                                    end_state: np.ndarray,
                                    duration: float) -> Dict:
        """
        Integrate minimum jerk analysis with advanced link model
        
        Args:
            advanced_model: AdvancedLinkModel instance
            start_state: Initial state
            end_state: Final state
            duration: Movement duration
            
        Returns:
            Dictionary containing integrated analysis results
        """
        # Calculate minimum jerk trajectory
        x0, y0 = self.forward_kinematics(start_state[:self.n_links])
        xf, yf = self.forward_kinematics(end_state[:self.n_links])
        
        x, y = self.calculate_minimum_jerk_trajectory(x0, xf, y0, yf, duration)
        vx, ax, jx = self.calculate_derivatives(x, duration)
        vy, ay, jy = self.calculate_derivatives(y, duration)
        
        # Create optimal trajectory planner
        planner = OptimalTrajectoryPlanner(advanced_model)
        trajectory = planner.optimize_trajectory(start_state, end_state, duration)
        
        return {
            'minimum_jerk': {
                'positions': (x, y),
                'velocities': (vx, vy),
                'accelerations': (ax, ay),
                'jerks': (jx, jy)
            },
            'optimal_trajectory': trajectory
        }

class OptimalTrajectoryPlanner:
    """Plans optimal trajectories considering energy and constraints"""
    def __init__(self, link_model: AdvancedLinkModel):
        self.model = link_model
        
    # ... (rest of OptimalTrajectoryPlanner methods) ...

class EnhancedVisualizer:
    """Advanced visualization tools for link systems"""
    def __init__(self, link_model: AdvancedLinkModel):
        self.model = link_model
        
    # ... (rest of EnhancedVisualizer methods) ...

# Example usage:
"""
# Create constraints
constraints = Constraints(
    max_velocity=5.0,
    max_acceleration=10.0,
    joint_limits={'joint1': (-np.pi/2, np.pi/2),
                 'joint2': (-np.pi/3, np.pi/3)},
    energy_threshold=100.0
)

# Create models
advanced_model = AdvancedLinkModel(
    lengths=[0.5, 0.5],
    masses=[1.0, 1.0],
    moments_of_inertia=[0.1, 0.1],
    constraints=constraints
)

# Initialize analyzers
jerk_analyzer = MinimumJerkAnalyzer()
visualizer = EnhancedVisualizer(advanced_model)

# Perform integrated analysis
results = jerk_analyzer.integrate_with_advanced_model(
    advanced_model=advanced_model,
    start_state=np.array([0.0, 0.0, 0.0, 0.0]),
    end_state=np.array([np.pi/4, np.pi/4, 0.0, 0.0]),
    duration=2.0
)

# Visualize results
visualizer.animate_trajectory(results['optimal_trajectory'])
visualizer.plot_state_variables(results['optimal_trajectory'])
jerk_analyzer.visualize_minimum_jerk(
    *results['minimum_jerk']['positions'],
    *results['minimum_jerk']['velocities'],
    *results['minimum_jerk']['accelerations'],
    *results['minimum_jerk']['jerks']
)
"""