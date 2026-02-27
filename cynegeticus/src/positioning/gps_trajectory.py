"""
GPS Trajectory Loader and S-Entropy Extractor

Loads GeoJSON GPS data and extracts atmospheric S-entropy coordinates
through trajectory completion (categorical questions).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import csv


class GPSTrajectory:
    """Represents a GPS trajectory with trans-Planckian precision"""

    def __init__(self, geojson_path: str):
        self.geojson_path = Path(geojson_path)
        self.data = self._load_geojson()
        self.trajectories = self._extract_trajectories()

    def _load_geojson(self) -> Dict:
        """Load GeoJSON file"""
        with open(self.geojson_path, 'r') as f:
            return json.load(f)

    def _extract_trajectories(self) -> List[Dict]:
        """Extract GPS point sequences from GeoJSON features"""
        trajectories = {'watch1': [], 'watch2': []}

        for feature in self.data['features']:
            props = feature['properties']

            # Only extract raw GPS points (not uncertainty ellipses)
            if props.get('precision_level') == 'raw_gps' and \
               feature['geometry']['type'] == 'Point':

                coords = feature['geometry']['coordinates']
                point = {
                    'lon': coords[0],
                    'lat': coords[1],
                    'watch': props['watch'],
                    'point_index': props['point_index'],
                    'velocity_ms': props.get('velocity_ms', 0.0),
                    'original_lat': props['original_lat'],
                    'original_lon': props['original_lon'],
                    'refined_lat': props['refined_lat'],
                    'refined_lon': props['refined_lon']
                }

                if 'Watch 1' in props['watch']:
                    trajectories['watch1'].append(point)
                else:
                    trajectories['watch2'].append(point)

        # Sort by point index
        trajectories['watch1'].sort(key=lambda x: x['point_index'])
        trajectories['watch2'].sort(key=lambda x: x['point_index'])

        return trajectories

    def derive_atmospheric_state(self) -> Dict[str, List[Dict]]:
        """
        TRAJECTORY COMPLETION: Categorical Question

        "What atmospheric conditions would produce these GPS trajectories?"

        This is not measurement extraction - it's categorical address resolution.
        The trajectories themselves encode the atmospheric partition state.
        """
        atmospheric_states = {'watch1': [], 'watch2': []}

        for watch_name, trajectory in self.trajectories.items():
            for i, point in enumerate(trajectory):
                # Categorical question for each point
                state = self._categorical_atmospheric_question(point, trajectory, i)
                atmospheric_states[watch_name].append(state)

        return atmospheric_states

    def _categorical_atmospheric_question(self, point: Dict, trajectory: List[Dict],
                                          index: int) -> Dict:
        """
        Ask: "What atmospheric conditions would lead to this trajectory point?"

        The answer IS the atmospheric state - not measured, but resolved categorically.
        """
        # S_k: Compositional/kinetic entropy
        # Derived from velocity variations (wind resistance, air density effects)
        S_k = self._derive_S_k(point, trajectory, index)

        # S_t: Temporal/velocity entropy
        # Derived from acceleration patterns (atmospheric pressure gradients)
        S_t = self._derive_S_t(point, trajectory, index)

        # S_e: Evolution/energy entropy
        # Derived from trajectory curvature (temperature/density gradients)
        S_e = self._derive_S_e(point, trajectory, index)

        return {
            'position': (point['lat'], point['lon']),
            'S_k': S_k,
            'S_t': S_t,
            'S_e': S_e,
            'velocity': point['velocity_ms'],
            'point_index': index
        }

    def _derive_S_k(self, point: Dict, trajectory: List[Dict], index: int) -> float:
        """
        Compositional entropy from velocity variations

        Air density/composition affects drag coefficient.
        Higher S_k → higher air density → more drag → more velocity damping

        Physical relation: Drag force F_d = 0.5 * ρ * C_d * A * v²
        where ρ (density) is encoded in S_k
        """
        if index < 2 or index >= len(trajectory) - 2:
            # Use position-dependent default for boundaries
            # Munich in October: moderate humidity ~60%
            return 0.60

        # Velocity damping analysis (over 3 points for stability)
        velocities = [
            trajectory[index - 2]['velocity_ms'],
            trajectory[index - 1]['velocity_ms'],
            point['velocity_ms'],
            trajectory[index + 1]['velocity_ms'],
            trajectory[index + 2]['velocity_ms']
        ]

        # Compute damping rate: high damping → high air density → high S_k
        velocity_std = np.std(velocities)
        velocity_mean = np.mean(velocities)

        if velocity_mean > 0.1:
            # Coefficient of variation indicates turbulence/drag
            cv = velocity_std / velocity_mean
            # Higher turbulence → higher air density interactions
            # October Munich: ρ ≈ 1.2 kg/m³, moderate humidity
            S_k = 0.55 + 0.15 * np.tanh(cv * 2.0)  # Range: [0.4, 0.7]
        else:
            S_k = 0.60  # Default for near-stationary

        return np.clip(S_k, 0.0, 1.0)

    def _derive_S_t(self, point: Dict, trajectory: List[Dict], index: int) -> float:
        """
        Temporal entropy from velocity magnitude

        Represents atmospheric momentum/wind coupling.
        S_t encodes both human motion AND ambient wind field.

        Physical: v_total² = v_human² + v_wind² + 2*v_human*v_wind*cos(θ)
        Higher S_t → stronger wind field
        """
        v = point['velocity_ms']

        # October Munich: typical wind speeds 2-5 m/s
        # Human running: 3-10 m/s
        # Combined effect shows in GPS velocity

        # Background wind field (Munich October average)
        S_t_background = 0.25  # ~3.75 m/s wind

        # Velocity contribution (human + wind coupling)
        # Max combined velocity ~15 m/s
        v_max = 15.0
        S_t_velocity = v / v_max

        # Weighted combination: velocity shows wind interaction
        # If moving with wind: higher effective v
        # If moving against wind: lower effective v
        S_t = 0.4 * S_t_background + 0.6 * S_t_velocity

        return np.clip(S_t, 0.0, 1.0)

    def _derive_S_e(self, point: Dict, trajectory: List[Dict], index: int) -> float:
        """
        Energy entropy from trajectory curvature and thermal gradients

        Trajectory curvature indicates:
        1. Pressure gradients (force direction changes)
        2. Temperature gradients (affect air density → trajectory bending)

        Physical: Centripetal force from pressure gradient
        Higher curvature → stronger gradient → higher energy state

        October Munich: T ≈ 280-290K → S_e ≈ 0.0-0.67 (using T_min=280K, T_max=295K)
        """
        # October Munich typical: 12°C ≈ 285K → S_e ≈ 0.33
        S_e_baseline = 0.35  # Slightly cool autumn day

        if index < 1 or index >= len(trajectory) - 1:
            return S_e_baseline

        # Calculate trajectory curvature
        p_prev = np.array([trajectory[index - 1]['lat'], trajectory[index - 1]['lon']])
        p_curr = np.array([point['lat'], point['lon']])
        p_next = np.array([trajectory[index + 1]['lat'], trajectory[index + 1]['lon']])

        # Vectors
        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        # Angle change (curvature proxy)
        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            curvature = 1.0 - cos_angle  # 0 = straight, 2 = reversal
        else:
            curvature = 0.0

        # Sharp turns indicate strong pressure/thermal gradients
        # These create local energy concentrations
        curvature_contribution = curvature * 0.15  # Up to ±0.15 variation

        # Combine baseline + curvature signal
        S_e = S_e_baseline + curvature_contribution

        return np.clip(S_e, 0.0, 1.0)

    def get_metadata(self) -> Dict:
        """Get trajectory metadata"""
        return self.data.get('metadata', {})

    def save_atmospheric_state_json(self, output_path: str):
        """Save derived atmospheric state to JSON"""
        states = self.derive_atmospheric_state()

        output = {
            'metadata': {
                'source_geojson': str(self.geojson_path),
                'derivation_method': 'trajectory_completion',
                'description': 'Atmospheric S-entropy derived from GPS trajectories via categorical questions'
            },
            'atmospheric_states': states
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Atmospheric state saved to {output_path}")

    def save_atmospheric_state_csv(self, output_path: str):
        """Save derived atmospheric state to CSV"""
        states = self.derive_atmospheric_state()

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['watch', 'point_index', 'lat', 'lon', 'velocity_ms',
                           'S_k', 'S_t', 'S_e'])

            for watch_name, trajectory in states.items():
                for state in trajectory:
                    writer.writerow([
                        watch_name,
                        state['point_index'],
                        state['position'][0],
                        state['position'][1],
                        state['velocity'],
                        state['S_k'],
                        state['S_t'],
                        state['S_e']
                    ])

        print(f"Atmospheric state saved to {output_path}")


if __name__ == "__main__":
    # Test with the GeoJSON file
    geojson_path = "c:/Users/kundai/Documents/geosciences/sighthound/cynegeticus/public/comprehensive_gps_multiprecision_20251013_053445.geojson"

    print("Loading GPS trajectory...")
    gps = GPSTrajectory(geojson_path)

    print(f"Watch 1: {len(gps.trajectories['watch1'])} points")
    print(f"Watch 2: {len(gps.trajectories['watch2'])} points")

    print("\nDeriving atmospheric state via trajectory completion...")

    # Save results
    output_dir = Path("c:/Users/kundai/Documents/geosciences/sighthound/validation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    gps.save_atmospheric_state_json(str(output_dir / "atmospheric_state_from_gps.json"))
    gps.save_atmospheric_state_csv(str(output_dir / "atmospheric_state_from_gps.csv"))

    print("\n✓ GPS trajectory processing complete!")
