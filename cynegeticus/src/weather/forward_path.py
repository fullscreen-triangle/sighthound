"""
Forward Path: GPS → Weather Prediction

Uses trajectory completion to derive weather forecast from GPS trajectories.

CATEGORICAL QUESTION: "Where will this atmospheric partition evolve?"
"""

import json
import numpy as np
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple


class PartitionDynamics:
    """
    Evolves atmospheric partition state through trajectory completion

    Not integration - categorical path finding in partition space
    """

    def __init__(self, atmospheric_states: Dict[str, List[Dict]]):
        self.states = atmospheric_states
        self.initial_partition = self._aggregate_partition_state()

    def _aggregate_partition_state(self) -> Dict:
        """
        Aggregate individual molecular states into atmospheric partition

        Each GPS point gives local S-entropy; aggregate to field
        """
        all_states = []
        all_positions = []

        for watch_name, trajectory in self.states.items():
            for state in trajectory:
                all_states.append([state['S_k'], state['S_t'], state['S_e']])
                all_positions.append(state['position'])

        # Mean partition state (could use more sophisticated interpolation)
        mean_S = np.mean(all_states, axis=0)

        return {
            'S_k_mean': float(mean_S[0]),
            'S_t_mean': float(mean_S[1]),
            'S_e_mean': float(mean_S[2]),
            'S_k_std': float(np.std([s[0] for s in all_states])),
            'S_t_std': float(np.std([s[1] for s in all_states])),
            'S_e_std': float(np.std([s[2] for s in all_states])),
            'num_samples': len(all_states),
            'spatial_extent': self._compute_spatial_extent(all_positions)
        }

    def _compute_spatial_extent(self, positions: List[Tuple]) -> Dict:
        """Compute bounding box of measurements"""
        lats = [p[0] for p in positions]
        lons = [p[1] for p in positions]

        return {
            'lat_min': min(lats),
            'lat_max': max(lats),
            'lon_min': min(lons),
            'lon_max': max(lons),
            'center_lat': np.mean(lats),
            'center_lon': np.mean(lons)
        }

    def evolve_partition(self, days: int = 10) -> List[Dict]:
        """
        TRAJECTORY COMPLETION: Where will this partition evolve?

        Not differential equation integration - categorical trajectory finding
        """
        forecasts = []

        S_k = self.initial_partition['S_k_mean']
        S_t = self.initial_partition['S_t_mean']
        S_e = self.initial_partition['S_e_mean']

        # Initial state
        initial_weather = self._partition_to_weather(S_k, S_t, S_e, day=0)
        forecasts.append(initial_weather)

        # Evolve through partition space
        for day in range(1, days + 1):
            # Partition dynamics evolution
            S_k, S_t, S_e = self._partition_dynamics_step(S_k, S_t, S_e, day)

            # Convert to observable weather
            weather = self._partition_to_weather(S_k, S_t, S_e, day)
            forecasts.append(weather)

        return forecasts

    def _partition_dynamics_step(self, S_k: float, S_t: float, S_e: float,
                                  day: int) -> Tuple[float, float, float]:
        """
        Single step of partition dynamics evolution

        Using proper atmospheric time scales from ober-atmos-scripting.tex
        """
        # Physical time scales (days)
        tau_k = 5.0   # Compositional equilibration ~5 days (typical for humidity)
        tau_t = 3.0   # Momentum damping ~3 days (friction time scale)
        tau_e = 2.0   # Energy equilibration ~2 days (radiative time scale)

        # October in Northern Hemisphere: cooling, decreasing energy
        # Munich average: T goes from 15°C to 10°C over 30 days
        seasonal_trend_e = -0.005  # Cooling ~0.5% per day

        # Diurnal and synoptic variability
        diurnal_period = 1.0  # day
        synoptic_period = 5.0  # weather system passage

        # S_k evolution: Compositional relaxation + weather system forcing
        # Humidity tends to increase in autumn (towards saturation)
        S_k_equilibrium = 0.65  # Higher humidity in autumn
        dS_k = -(S_k - S_k_equilibrium) / tau_k + 0.03 * np.sin(2*np.pi*day/synoptic_period)

        # S_t evolution: Momentum with pressure gradient forcing
        # Coupled to S_k (pressure gradients from composition)
        pressure_gradient_forcing = -0.02 * (S_k - 0.5)  # Pressure responds to composition
        dS_t = pressure_gradient_forcing - S_t / tau_t + 0.02 * np.sin(2*np.pi*day/diurnal_period)

        # S_e evolution: Energy balance with radiative cooling
        # Coupled to S_t (advection) and S_k (latent heat)
        advection_term = -0.01 * S_t * (S_e - 0.5)  # Temperature advection
        latent_heat_term = 0.01 * (S_k - 0.5)  # Condensation releases energy
        radiative_cooling = seasonal_trend_e

        dS_e = advection_term + latent_heat_term + radiative_cooling - (S_e - 0.4) / tau_e

        # Update with realistic step size (1 day)
        S_k = np.clip(S_k + dS_k, 0.0, 1.0)
        S_t = np.clip(S_t + dS_t, 0.0, 1.0)
        S_e = np.clip(S_e + dS_e, 0.0, 1.0)

        return S_k, S_t, S_e

    def _partition_to_weather(self, S_k: float, S_t: float, S_e: float,
                             day: int) -> Dict:
        """
        Convert partition coordinates to observable weather variables

        This is the inverse mapping: (S_k, S_t, S_e) → (T, P, humidity, wind, etc.)
        """
        # Temperature (K)
        # S_e encodes energy → temperature
        # Range: 280K - 295K (typical October in Munich)
        T_min, T_max = 280.0, 295.0
        temperature_K = T_min + S_e * (T_max - T_min)
        temperature_C = temperature_K - 273.15

        # Pressure (hPa)
        # S_k encodes composition/density
        # Range: 990 - 1025 hPa (typical range)
        P_min, P_max = 990.0, 1025.0
        pressure = P_min + S_k * (P_max - P_min)

        # Humidity (%)
        # Coupled to S_e (energy) and S_k (composition)
        # Higher S_k + lower S_e → higher humidity
        humidity = 100 * (0.5 * S_k + 0.3 * (1.0 - S_e) + 0.2)
        humidity = np.clip(humidity, 0, 100)

        # Wind speed (m/s)
        # S_t encodes velocity/momentum
        wind_speed = S_t * 15.0  # Max 15 m/s

        # Precipitation probability (%)
        # High humidity + low temperature → higher precip chance
        precip_prob = 0.0
        if humidity > 70 and temperature_C < 15:
            precip_prob = (humidity - 70) / 30 * 100
        precip_prob = np.clip(precip_prob, 0, 100)

        # Cloud cover (%)
        # Coupled to humidity and S_e
        cloud_cover = np.clip(humidity * 0.8 + 20 * (1 - S_e), 0, 100)

        return {
            'day': day,
            'date': (datetime(2025, 10, 13) + timedelta(days=day)).strftime('%Y-%m-%d'),
            'temperature_K': float(temperature_K),
            'temperature_C': float(temperature_C),
            'pressure_hPa': float(pressure),
            'humidity_percent': float(humidity),
            'wind_speed_ms': float(wind_speed),
            'precipitation_prob_percent': float(precip_prob),
            'cloud_cover_percent': float(cloud_cover),
            'S_k': float(S_k),
            'S_t': float(S_t),
            'S_e': float(S_e)
        }

    def save_forecast_json(self, output_path: str):
        """Save weather forecast to JSON"""
        forecast = self.evolve_partition(days=10)

        output = {
            'metadata': {
                'method': 'partition_dynamics_trajectory_completion',
                'initial_partition': self.initial_partition,
                'forecast_start': '2025-10-13',
                'forecast_days': 10,
                'description': 'Weather forecast derived from GPS trajectories via partition dynamics'
            },
            'forecast': forecast
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Weather forecast saved to {output_path}")

    def save_forecast_csv(self, output_path: str):
        """Save weather forecast to CSV"""
        forecast = self.evolve_partition(days=10)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['day', 'date', 'temperature_C', 'pressure_hPa',
                           'humidity_percent', 'wind_speed_ms',
                           'precipitation_prob_percent', 'cloud_cover_percent',
                           'S_k', 'S_t', 'S_e'])

            for day_forecast in forecast:
                writer.writerow([
                    day_forecast['day'],
                    day_forecast['date'],
                    day_forecast['temperature_C'],
                    day_forecast['pressure_hPa'],
                    day_forecast['humidity_percent'],
                    day_forecast['wind_speed_ms'],
                    day_forecast['precipitation_prob_percent'],
                    day_forecast['cloud_cover_percent'],
                    day_forecast['S_k'],
                    day_forecast['S_t'],
                    day_forecast['S_e']
                ])

        print(f"Weather forecast saved to {output_path}")


if __name__ == "__main__":
    # Load atmospheric state from GPS trajectory
    input_file = "c:/Users/kundai/Documents/geosciences/sighthound/validation/results/atmospheric_state_from_gps.json"

    print("Loading atmospheric state from GPS trajectories...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    atmospheric_states = data['atmospheric_states']

    print("Initializing partition dynamics...")
    dynamics = PartitionDynamics(atmospheric_states)

    print(f"Initial partition state:")
    print(f"  S_k: {dynamics.initial_partition['S_k_mean']:.4f} ± {dynamics.initial_partition['S_k_std']:.4f}")
    print(f"  S_t: {dynamics.initial_partition['S_t_mean']:.4f} ± {dynamics.initial_partition['S_t_std']:.4f}")
    print(f"  S_e: {dynamics.initial_partition['S_e_mean']:.4f} ± {dynamics.initial_partition['S_e_std']:.4f}")

    print("\nEvolving partition through trajectory completion...")

    # Save results
    output_dir = Path("c:/Users/kundai/Documents/geosciences/sighthound/validation/results")
    dynamics.save_forecast_json(str(output_dir / "weather_forecast_from_gps.json"))
    dynamics.save_forecast_csv(str(output_dir / "weather_forecast_from_gps.csv"))

    print("\n✓ Forward path (GPS → Weather) complete!")
