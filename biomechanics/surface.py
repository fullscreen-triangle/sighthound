import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.signal import welch, coherence
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

@dataclass
class SurfaceParameters:
    stiffness: float          # N/m
    damping: float           # Ns/m
    resonant_frequency: float # Hz
    loss_factor: float       # Dimensionless
    restitution: float       # Coefficient of restitution
    deformation: float       # m

@dataclass
class GaitResonance:
    natural_frequency: float  # Hz
    coupling_factor: float   # Dimensionless
    phase_difference: float  # rad
    resonance_match: float   # 0-1 scale

class SurfaceModel(nn.Module):
    """Neural network for surface dynamics prediction"""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [stiffness, damping, resonant_freq, energy_return]
        )
        
    def forward(self, x):
        return self.network(x)

class SurfaceDynamicsAnalyzer:
    def __init__(self, athlete_mass: float = 70.0, model_path: str = "models"):
        """
        Initialize surface dynamics analyzer with ML capabilities
        
        Args:
            athlete_mass: Mass of the athlete in kg
            model_path: Path to save/load models
        """
        self.mass = athlete_mass
        self.g = 9.81  # m/s²
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize ML components
        self.surface_model = SurfaceModel()
        self.scaler = StandardScaler()
        self.load_model()
        
        # Training history
        self.training_history = []
        
    def load_model(self):
        """Load pretrained model and scaler if available"""
        model_file = self.model_path / "surface_model.pt"
        scaler_file = self.model_path / "scaler.joblib"
        
        if model_file.exists() and scaler_file.exists():
            self.surface_model.load_state_dict(torch.load(model_file))
            self.scaler = joblib.load(scaler_file)
            
    def save_model(self):
        """Save current model and scaler"""
        torch.save(self.surface_model.state_dict(), 
                  self.model_path / "surface_model.pt")
        joblib.dump(self.scaler, self.model_path / "scaler.joblib")

    def prepare_features(self, section_data: List[Dict]) -> torch.Tensor:
        """Prepare features for model input"""
        features = []
        for data in section_data:
            props = data['properties']
            features.append([
                props['vertical_oscillation'],
                props['stance_time'],
                props['step_length'],
                props['speed'],
                props['cadence'],
                props['vertical_ratio'],
                props['altitude'],
                self.mass * self.g  # Weight force
            ])
        
        features = np.array(features)
        scaled_features = self.scaler.fit_transform(features)
        return torch.FloatTensor(scaled_features)

    def train_model(self, 
                   new_data: List[Dict], 
                   epochs: int = 100, 
                   batch_size: int = 32):
        """
        Train model on new data
        
        Args:
            new_data: List of track data points
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Prepare features
        X = self.prepare_features(new_data)
        
        # Calculate traditional estimates for training targets
        y = []
        for data in new_data:
            K_surface = self.calculate_surface_stiffness(
                data['properties']['vertical_oscillation'],
                data['properties']['stance_time'],
                data['properties']['step_length']
            )
            c = self.estimate_surface_damping(
                [d['properties']['vertical_oscillation'] for d in new_data],
                [d['properties']['timestamp'] for d in new_data]
            )
            f_resonant = self.calculate_resonant_frequency(K_surface, c)
            energy_return = self.calculate_energy_return(
                K_surface, c,
                data['properties']['vertical_oscillation'],
                data['properties']['stance_time']
            )
            y.append([K_surface, c, f_resonant, energy_return])
        
        y = torch.FloatTensor(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Training loop
        optimizer = torch.optim.Adam(self.surface_model.parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # Training
            self.surface_model.train()
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.surface_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.surface_model.eval()
            with torch.no_grad():
                val_outputs = self.surface_model(X_val)
                val_loss = criterion(val_outputs, y_val)
            
            self.training_history.append({
                'epoch': epoch,
                'val_loss': val_loss.item()
            })
        
        # Save updated model
        self.save_model()

    def calculate_surface_stiffness(self,
                                  vertical_oscillation: float,
                                  ground_contact_time: float,
                                  step_length: float) -> float:
        """
        Calculate surface stiffness using spring-mass model
        
        K_surface = F_max / (y_c + y_surface)
        where:
        F_max = peak ground reaction force
        y_c = center of mass displacement
        y_surface = surface deformation
        """
        # Calculate peak force using impulse-momentum
        flight_time = 60/step_length - ground_contact_time/1000
        v_vertical = vertical_oscillation / (2 * flight_time)
        F_max = self.mass * (self.g + v_vertical/ground_contact_time)
        
        # Estimate surface deformation (typically 3-8mm for synthetic tracks)
        y_surface = 0.005  # Initial estimate
        
        # Calculate surface stiffness
        K_surface = F_max / (vertical_oscillation/1000 + y_surface)
        
        return K_surface
    
    def estimate_surface_damping(self,
                               vertical_oscillation: List[float],
                               timestamps: List[str]) -> float:
        """
        Estimate surface damping coefficient using logarithmic decrement
        
        ζ = ln(x₁/x₂) / (2π)
        c = 2ζ√(km)
        """
        # Calculate time differences
        time_diffs = np.diff([np.datetime64(t) for t in timestamps])
        dt = np.mean(time_diffs).astype(float) / 1e9  # Convert to seconds
        
        # Calculate logarithmic decrement
        peaks = np.array(vertical_oscillation)
        log_dec = np.log(peaks[:-1]/peaks[1:])
        damping_ratio = np.mean(log_dec) / (2 * np.pi)
        
        # Calculate damping coefficient
        K_surface = self.calculate_surface_stiffness(
            np.mean(vertical_oscillation),
            0.200,  # Typical ground contact time
            2.0     # Typical step length
        )
        
        c = 2 * damping_ratio * np.sqrt(K_surface * self.mass)
        return c
    
    def calculate_resonant_frequency(self,
                                   K_surface: float,
                                   c: float) -> float:
        """
        Calculate natural frequency of surface-runner system
        
        ωₙ = √(k/m) * √(1 - ζ²)
        """
        damping_ratio = c / (2 * np.sqrt(K_surface * self.mass))
        natural_freq = np.sqrt(K_surface/self.mass) * np.sqrt(1 - damping_ratio**2)
        return natural_freq / (2 * np.pi)  # Convert to Hz
    
    def calculate_optimal_stride_frequency(self,
                                        resonant_freq: float,
                                        speed: float,
                                        step_length: float) -> float:
        """
        Calculate optimal stride frequency matching surface resonance
        
        f_stride = v / (2 * step_length)
        """
        return speed / (2 * step_length/1000)  # Convert step_length to meters
    
    def calculate_energy_return(self,
                              K_surface: float,
                              c: float,
                              vertical_oscillation: float,
                              contact_time: float) -> float:
        """
        Calculate energy return from surface
        
        E_return = (1/2)K_surface * y²max * exp(-2ζωₙt)
        """
        damping_ratio = c / (2 * np.sqrt(K_surface * self.mass))
        natural_freq = np.sqrt(K_surface/self.mass)
        
        energy_stored = 0.5 * K_surface * (vertical_oscillation/1000)**2
        energy_return = energy_stored * np.exp(-2 * damping_ratio * natural_freq * contact_time/1000)
        
        return energy_return
    
    def analyze_gait_surface_coupling(self,
                                    cadence: float,
                                    vertical_oscillation: float,
                                    K_surface: float,
                                    c: float) -> GaitResonance:
        """
        Analyze coupling between gait and surface dynamics
        """
        # Calculate natural frequencies
        f_surface = self.calculate_resonant_frequency(K_surface, c)
        f_stride = cadence / 60  # Convert to Hz
        
        # Calculate damping ratio
        damping_ratio = c / (2 * np.sqrt(K_surface * self.mass))
        
        # Calculate coupling factor
        coupling = 1 / (1 - (f_stride/f_surface)**2)
        
        # Calculate phase difference
        phase_diff = np.arctan2(2 * damping_ratio * f_stride/f_surface,
                               1 - (f_stride/f_surface)**2)
        
        # Calculate resonance match (1 = perfect match)
        resonance_match = 1 - abs(f_stride - f_surface)/f_surface
        
        return GaitResonance(
            natural_frequency=f_surface,
            coupling_factor=coupling,
            phase_difference=phase_diff,
            resonance_match=resonance_match
        )
    
    def calculate_surface_deformation(self,
                                    K_surface: float,
                                    force: float) -> float:
        """
        Calculate surface deformation under load
        
        y = F/k
        """
        return force / K_surface
    
    def analyze_track_section(self, section_data: List[Dict]) -> Dict:
        """
        Analyze surface dynamics using hybrid approach
        """
        # Traditional analysis
        traditional_results = self._traditional_analysis(section_data)
        
        # ML model predictions
        ml_predictions = self._ml_predictions(section_data)
        
        # Combine predictions with weighted average
        alpha = 0.7  # Weight for ML predictions
        combined_results = {
            'surface_parameters': SurfaceParameters(
                stiffness=(1-alpha) * traditional_results['surface_parameters'].stiffness +
                         alpha * ml_predictions['stiffness'],
                damping=(1-alpha) * traditional_results['surface_parameters'].damping +
                        alpha * ml_predictions['damping'],
                resonant_frequency=(1-alpha) * traditional_results['surface_parameters'].resonant_frequency +
                                 alpha * ml_predictions['resonant_frequency'],
                loss_factor=traditional_results['surface_parameters'].loss_factor,
                restitution=traditional_results['surface_parameters'].restitution,
                deformation=traditional_results['surface_parameters'].deformation
            ),
            'gait_resonance': traditional_results['gait_resonance'],
            'energy_metrics': traditional_results['energy_metrics'],
            'optimal_stride_frequency': traditional_results['optimal_stride_frequency']
        }
        
        # Train model on new data
        self.train_model(section_data)
        
        return combined_results

    def _traditional_analysis(self, section_data: List[Dict]) -> Dict:
        """Perform traditional physics-based analysis"""
        # Extract relevant metrics
        vertical_oscillations = [d['properties']['vertical_oscillation'] for d in section_data]
        timestamps = [d['properties']['timestamp'] for d in section_data]
        step_lengths = [d['properties']['step_length'] for d in section_data]
        stance_times = [d['properties']['stance_time'] for d in section_data]
        speeds = [d['properties']['speed'] for d in section_data]
        altitudes = [d['properties']['altitude'] for d in section_data]
        
        # Calculate surface parameters
        K_surface = self.calculate_surface_stiffness(
            np.mean(vertical_oscillations),
            np.mean(stance_times),
            np.mean(step_lengths)
        )
        
        c = self.estimate_surface_damping(vertical_oscillations, timestamps)
        
        # Calculate resonant frequency
        f_resonant = self.calculate_resonant_frequency(K_surface, c)
        
        # Calculate optimal stride frequency
        f_optimal = self.calculate_optimal_stride_frequency(
            f_resonant,
            np.mean(speeds),
            np.mean(step_lengths)
        )
        
        # Calculate energy return
        energy_return = self.calculate_energy_return(
            K_surface,
            c,
            np.mean(vertical_oscillations),
            np.mean(stance_times)
        )
        
        # Analyze altitude effects
        altitude_gradient = np.gradient(altitudes)
        work_against_gravity = self.mass * self.g * np.sum(np.maximum(0, altitude_gradient))
        
        return {
            'surface_parameters': SurfaceParameters(
                stiffness=K_surface,
                damping=c,
                resonant_frequency=f_resonant,
                loss_factor=c/(2*np.sqrt(K_surface*self.mass)),
                restitution=np.sqrt(energy_return/(0.5*K_surface*(np.mean(vertical_oscillations)/1000)**2)),
                deformation=self.calculate_surface_deformation(K_surface, self.mass*self.g)
            ),
            'gait_resonance': self.analyze_gait_surface_coupling(
                np.mean([d['properties']['cadence'] for d in section_data]),
                np.mean(vertical_oscillations),
                K_surface,
                c
            ),
            'energy_metrics': {
                'energy_return': energy_return,
                'work_against_gravity': work_against_gravity,
                'total_work': energy_return + work_against_gravity
            },
            'optimal_stride_frequency': f_optimal
        }

    def _ml_predictions(self, section_data: List[Dict]) -> Dict:
        """Get predictions from ML model"""
        features = self.prepare_features(section_data)
        
        self.surface_model.eval()
        with torch.no_grad():
            predictions = self.surface_model(features)
        
        return {
            'stiffness': predictions[:, 0].mean().item(),
            'damping': predictions[:, 1].mean().item(),
            'resonant_frequency': predictions[:, 2].mean().item(),
            'energy_return': predictions[:, 3].mean().item()
        }

    def plot_surface_analysis(self, results: Dict):
        """Plot surface analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Surface stiffness vs deformation
        deformations = np.linspace(0, 0.01, 100)
        forces = results['surface_parameters'].stiffness * deformations
        axes[0, 0].plot(deformations * 1000, forces)
        axes[0, 0].set_xlabel('Deformation (mm)')
        axes[0, 0].set_ylabel('Force (N)')
        axes[0, 0].set_title('Surface Force-Deformation')
        
        # Plot 2: Resonance matching
        axes[0, 1].bar(['Surface', 'Stride'],
                      [results['surface_parameters'].resonant_frequency,
                       results['optimal_stride_frequency']])
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].set_title('Resonance Comparison')
        
        # Plot 3: Energy return
        axes[1, 0].pie([results['energy_metrics']['energy_return'],
                       results['energy_metrics']['work_against_gravity']],
                      labels=['Surface Return', 'Gravitational Work'])
        axes[1, 0].set_title('Energy Distribution')
        
        # Plot 4: Coupling analysis
        axes[1, 1].plot(results['gait_resonance'].resonance_match)
        axes[1, 1].set_ylabel('Resonance Match')
        axes[1, 1].set_title('Gait-Surface Coupling')
        
        plt.tight_layout()
        plt.show()
