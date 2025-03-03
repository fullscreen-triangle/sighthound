

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn

from biomechanics.kinematics import LinkageState, MinimumJerkAnalyzer


@dataclass
class JointTorques:
    """Joint torques for single or multi-link systems"""
    values: np.ndarray  # Torque values for each joint
    time: np.ndarray    # Corresponding time points

@dataclass
class DynamicForces:
    """Forces acting on the system"""
    inertial: np.ndarray     # Inertial forces
    centripetal: np.ndarray  # Centripetal forces
    coriolis: np.ndarray     # Coriolis forces
    gravity: np.ndarray      # Gravitational forces
    external: np.ndarray     # External forces

class InverseDynamicsSolver:
    def __init__(self, link_lengths: List[float], link_masses: List[float], 
                 moments_of_inertia: List[float], gravity: float = 9.81):
        """
        Initialize inverse dynamics solver
        
        Args:
            link_lengths: List of segment lengths (m)
            link_masses: List of segment masses (kg)
            moments_of_inertia: List of moments of inertia (kg⋅m²)
            gravity: Gravitational acceleration (m/s²)
        """
        self.L = link_lengths
        self.m = link_masses
        self.I = moments_of_inertia
        self.g = gravity
        self.n_links = len(link_lengths)

    def single_link_dynamics(self, state: LinkageState, external_force: Optional[np.ndarray] = None) -> Tuple[float, DynamicForces]:
        """
        Calculate joint torque and forces for single link system
        
        Args:
            state: Current linkage state
            external_force: Optional external force vector [Fx, Fy]
            
        Returns:
            Tuple of (joint_torque, forces)
        """
        # Inertial forces
        inertial = self.I[0] * state.alpha

        # Centripetal force
        centripetal = self.m[0] * state.omega**2 * self.L[0]/2

        # Gravitational torque
        gravity = self.m[0] * self.g * self.L[0]/2 * np.cos(state.theta)

        # External force contribution
        external = np.zeros(2)
        if external_force is not None:
            external = external_force
            
        # Total joint torque
        torque = inertial + centripetal + gravity + \
                (external[0] * np.cos(state.theta) + external[1] * np.sin(state.theta)) * self.L[0]

        forces = DynamicForces(
            inertial=np.array([inertial]),
            centripetal=np.array([centripetal]),
            coriolis=np.zeros(1),  # No Coriolis force in single link
            gravity=np.array([gravity]),
            external=external
        )

        return torque, forces

    def double_link_dynamics(self, states: List[LinkageState], external_forces: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, DynamicForces]:
        """
        Calculate joint torques and forces for double link system
        
        Args:
            states: List of states for both links
            external_forces: Optional list of external force vectors
            
        Returns:
            Tuple of (joint_torques, forces)
        """
        theta1, theta2 = states[0].theta, states[1].theta
        omega1, omega2 = states[0].omega, states[1].omega
        alpha1, alpha2 = states[0].alpha, states[1].alpha

        # Mass matrices
        M11 = self.I[0] + self.I[1] + self.m[1] * self.L[0]**2
        M12 = self.I[1] + self.m[1] * self.L[0] * self.L[1]/2 * np.cos(theta1 - theta2)
        M21 = M12
        M22 = self.I[1]

        # Centripetal and Coriolis terms
        C1 = -self.m[1] * self.L[0] * self.L[1]/2 * np.sin(theta1 - theta2) * omega2**2
        C2 = self.m[1] * self.L[0] * self.L[1]/2 * np.sin(theta1 - theta2) * omega1**2

        # Gravitational terms
        G1 = (self.m[0] * self.L[0]/2 + self.m[1] * self.L[0]) * self.g * np.cos(theta1)
        G2 = self.m[1] * self.g * self.L[1]/2 * np.cos(theta2)

        # External force contributions
        E1, E2 = 0, 0
        if external_forces is not None:
            for i, force in enumerate(external_forces):
                if i == 0:
                    E1 += force[0] * np.cos(theta1) + force[1] * np.sin(theta1)
                E2 += force[0] * np.cos(theta2) + force[1] * np.sin(theta2)

        # Calculate joint torques
        torques = np.array([
            M11 * alpha1 + M12 * alpha2 + C1 + G1 + E1 * self.L[0],
            M21 * alpha1 + M22 * alpha2 + C2 + G2 + E2 * self.L[1]
        ])

        forces = DynamicForces(
            inertial=np.array([M11 * alpha1, M22 * alpha2]),
            centripetal=np.array([C1, C2]),
            coriolis=np.array([M12 * alpha2, M21 * alpha1]),
            gravity=np.array([G1, G2]),
            external=np.array([E1, E2])
        )

        return torques, forces

class NeuralInverseDynamics(nn.Module):
    """Neural network for learning inverse dynamics"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class HybridDynamicsSolver:
    """Combines traditional inverse dynamics with neural network predictions and minimum jerk analysis"""
    def __init__(self, traditional_solver: InverseDynamicsSolver, hidden_dim: int = 64):
        self.traditional_solver = traditional_solver
        self.hidden_dim = hidden_dim
        self.jerk_analyzer = MinimumJerkAnalyzer()
        
        # Initialize neural model
        input_dim = 9  # [theta, omega, alpha, x, y, vx, vy, ax, ay]
        output_dim = traditional_solver.n_links  # One torque per joint
        self.neural_model = NeuralInverseDynamics(input_dim, hidden_dim, output_dim)
        
        # Training history
        self.training_history = {
            'losses': [],
            'epochs': [],
            'predictions': [],
            'jerk_costs': []
        }
        
    def update_model(self, states: List[LinkageState], 
                    measured_torques: np.ndarray,
                    epochs: int = 100,
                    learning_rate: float = 0.001,
                    batch_size: int = 32):
        """
        Update neural model with new data
        
        Args:
            states: List of observed states
            measured_torques: Corresponding measured torques
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
        """
        # Prepare training data
        X = np.array([[s.theta, s.omega, s.alpha, s.x, s.y, s.vx, s.vy, s.ax, s.ay] 
                      for s in states])
        y = measured_torques
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                y_pred = self.neural_model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Store training history
            self.training_history['losses'].append(epoch_loss / len(loader))
            self.training_history['epochs'].append(epoch)
            
            # Store example predictions
            with torch.no_grad():
                self.training_history['predictions'].append(
                    self.neural_model(X_tensor[:5]).numpy()  # Store first 5 predictions
                )
    
    def predict_torques(self, state: LinkageState) -> np.ndarray:
        """
        Predict torques using hybrid approach
        
        Args:
            state: Current linkage state
            
        Returns:
            Predicted torques
        """
        # Get traditional dynamics prediction
        traditional_torques, _ = self.traditional_solver.single_link_dynamics(state)
        
        # Get neural network prediction
        state_vector = torch.FloatTensor([[state.theta, state.omega, state.alpha,
                                         state.x, state.y, state.vx, state.vy,
                                         state.ax, state.ay]])
        neural_torques = self.neural_model(state_vector).detach().numpy()
        
        # Dynamic weighting based on training history
        if len(self.training_history['losses']) > 0:
            # Use more neural network influence if training loss is low
            neural_weight = np.clip(1.0 - self.training_history['losses'][-1], 0.3, 0.7)
            traditional_weight = 1.0 - neural_weight
        else:
            traditional_weight = 0.7
            neural_weight = 0.3
            
        return traditional_weight * traditional_torques + neural_weight * neural_torques

    def analyze_movement(self, 
                        states: List[LinkageState],
                        measured_torques: np.ndarray,
                        duration: float) -> Dict:
        """
        Comprehensive movement analysis combining minimum jerk and dynamics
        
        Args:
            states: List of observed states
            measured_torques: Corresponding measured torques
            duration: Movement duration
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract initial and final positions
        x0, y0 = states[0].x, states[0].y
        xf, yf = states[-1].x, states[-1].y
        
        # Calculate minimum jerk trajectory
        x_mj, y_mj = self.jerk_analyzer.calculate_minimum_jerk_trajectory(
            x0, xf, y0, yf, duration
        )
        
        # Calculate derivatives
        vx_mj, ax_mj, jx_mj = self.jerk_analyzer.calculate_derivatives(x_mj, duration)
        vy_mj, ay_mj, jy_mj = self.jerk_analyzer.calculate_derivatives(y_mj, duration)
        
        # Calculate jerk cost
        jerk_cost = self.jerk_analyzer.calculate_jerk_cost(
            np.sqrt(jx_mj**2 + jy_mj**2), duration
        )
        
        # Store jerk cost in training history
        self.training_history['jerk_costs'].append(jerk_cost)
        
        # Update neural model with new data
        self.update_model(states, measured_torques)
        
        return {
            'minimum_jerk': {
                'positions': (x_mj, y_mj),
                'velocities': (vx_mj, vy_mj),
                'accelerations': (ax_mj, ay_mj),
                'jerks': (jx_mj, jy_mj),
                'cost': jerk_cost
            },
            'dynamics': {
                'torques': measured_torques,
                'states': states
            },
            'training_loss': self.training_history['losses'][-1]
        }

    def plot_analysis_results(self, analysis_results: Dict):
        """
        Comprehensive visualization of movement analysis
        
        Args:
            analysis_results: Results from analyze_movement
        """
        # Create subplots
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Trajectory comparison
        ax1 = fig.add_subplot(gs[0, 0])
        mj = analysis_results['minimum_jerk']
        ax1.plot(mj['positions'][0], mj['positions'][1], 'b-', label='Minimum Jerk')
        actual_x = [s.x for s in analysis_results['dynamics']['states']]
        actual_y = [s.y for s in analysis_results['dynamics']['states']]
        ax1.plot(actual_x, actual_y, 'r--', label='Actual')
        ax1.set_title('Trajectory Comparison')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend()
        ax1.grid(True)
        
        # Velocity profiles
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(mj['velocities'][0], 'b-', label='MJ X Velocity')
        ax2.plot(mj['velocities'][1], 'r-', label='MJ Y Velocity')
        ax2.set_title('Velocity Profiles')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True)
        
        # Jerk profiles
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(mj['jerks'][0], 'b-', label='X Jerk')
        ax3.plot(mj['jerks'][1], 'r-', label='Y Jerk')
        ax3.set_title('Jerk Profiles')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Jerk (m/s³)')
        ax3.legend()
        ax3.grid(True)
        
        # Training metrics
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.training_history['losses'], 'g-')
        ax4.set_title('Training Loss History')
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
        # Jerk cost history
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.training_history['jerk_costs'], 'b-')
        ax5.set_title('Jerk Cost History')
        ax5.set_xlabel('Movement')
        ax5.set_ylabel('Jerk Cost')
        ax5.grid(True)
        
        # Combined metrics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(self.training_history['losses'], 
                   self.training_history['jerk_costs'],
                   alpha=0.5)
        ax6.set_title('Loss vs Jerk Cost')
        ax6.set_xlabel('Training Loss')
        ax6.set_ylabel('Jerk Cost')
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()

class DynamicsVisualizer:
    """Visualization tools for inverse dynamics results"""
    def __init__(self):
        self.colors = plt.cm.viridis(np.linspace(0, 1, 5))
        
    def plot_torques(self, time: np.ndarray, torques: JointTorques):
        """Plot joint torques over time"""
        plt.figure(figsize=(10, 6))
        for i, torque in enumerate(torques.values.T):
            plt.plot(time, torque, label=f'Joint {i+1}', color=self.colors[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N⋅m)')
        plt.title('Joint Torques')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_forces(self, time: np.ndarray, forces: DynamicForces):
        """Plot breakdown of forces"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        force_types = ['Inertial', 'Centripetal', 'Coriolis', 'Gravity']
        force_data = [forces.inertial, forces.centripetal, 
                     forces.coriolis, forces.gravity]
        
        for ax, force_type, force, color in zip(axes.flat, force_types, 
                                              force_data, self.colors):
            for i, f in enumerate(force.T):
                ax.plot(time, f, label=f'Joint {i+1}', color=color)
            ax.set_title(f'{force_type} Forces')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Force (N)')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        plt.show()

# Example usage:
"""
# Initialize solvers
solver = InverseDynamicsSolver(
    link_lengths=[0.4, 0.4],      # Thigh and shank lengths
    link_masses=[7.0, 3.0],       # Approximate segment masses
    moments_of_inertia=[0.15, 0.05]  # Approximate moments of inertia
)

hybrid_solver = HybridDynamicsSolver(solver)
visualizer = DynamicsVisualizer()

# Analyze movement
states = [LinkageState(...), ...]  # List of observed states
measured_torques = np.array([...])  # Measured torques
duration = 1.0  # Movement duration

# Perform analysis
results = hybrid_solver.analyze_movement(states, measured_torques, duration)

# Visualize results
hybrid_solver.plot_analysis_results(results)
"""


