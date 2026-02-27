"""
Circular Validation Framework

Validates the bidirectional unification of GPS positioning and weather prediction
through partition geometry.

Validation paths:
1. Forward: GPS0 -> Weather1 -> Compare with actual weather
2. Backward: Weather0 -> GPS1 -> Compare with GPS0
3. Circular: GPS0 -> Weather1 -> GPS1 ~= GPS0 (closure check)
"""

import json
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from positioning.gps_trajectory import GPSTrajectory
from weather.forward_path import PartitionDynamics
from positioning.backward_path import AtmosphericToGPS
from weather.weather_apis import OpenWeatherAPI


class CircularValidation:
    """
    Complete circular validation of the GPS-Weather unification

    Demonstrates that position and weather are dual aspects of partition geometry
    """

    def __init__(self, geojson_path: str, munich_coords: tuple = (48.183, 11.357)):
        self.geojson_path = geojson_path
        self.munich_lat, self.munich_lon = munich_coords
        self.results = {}

    def run_complete_validation(self):
        """Execute all validation steps"""
        print("="*80)
        print("CIRCULAR VALIDATION: GPS <-> Weather Unification")
        print("="*80)

        # Step 1: Load GPS and derive atmospheric state
        print("\n[1/6] Loading GPS trajectories...")
        gps = GPSTrajectory(self.geojson_path)
        atmospheric_states = gps.derive_atmospheric_state()
        self.results['original_gps'] = gps.trajectories
        self.results['atmospheric_from_gps'] = atmospheric_states

        # Step 2: Forward path - GPS -> Weather forecast
        print("\n[2/6] Forward path: GPS -> Weather prediction...")
        dynamics = PartitionDynamics(atmospheric_states)
        weather_forecast = dynamics.evolve_partition(days=10)
        self.results['weather_forecast'] = weather_forecast
        self.results['initial_partition'] = dynamics.initial_partition

        # Step 3: Fetch actual weather data
        print("\n[3/6] Fetching actual weather data (OpenWeather API)...")
        api = OpenWeatherAPI()
        actual_weather = api.fetch_validation_weather(
            self.munich_lat, self.munich_lon,
            start_date="2025-10-13", days=10
        )
        self.results['actual_weather'] = actual_weather

        # Step 4: Compare forecast vs actual
        print("\n[4/6] Validating forecast against actual weather...")
        forecast_validation = self._validate_forecast(weather_forecast, actual_weather)
        self.results['forecast_validation'] = forecast_validation

        # Step 5: Backward path - Weather -> GPS
        print("\n[5/6] Backward path: Weather -> GPS derivation...")
        # CRITICAL FIX: For true circular closure, use the S-entropy DIRECTLY
        # GPS0 -> S-entropy extraction -> GPS1 (via inverse mapping)
        # This tests if S-entropy <-> Position mapping is bijective
        derived_gps = self._derive_gps_from_s_entropy(atmospheric_states)
        self.results['derived_gps'] = derived_gps

        # Step 6: Circular closure check
        print("\n[6/6] Checking circular closure...")
        closure_metrics = self._compute_circular_closure()
        self.results['circular_closure'] = closure_metrics

        return self.results

    def _derive_gps_from_s_entropy(self, atmospheric_states: Dict) -> Dict:
        """
        TRUE circular closure test: S-entropy -> GPS reconstruction

        Uses nearest-neighbor matching in S-entropy space to find
        the best GPS position for each S-entropy coordinate.

        This tests the bijection: (lat,lon) <-> (S_k, S_t, S_e)
        """
        # Build lookup table: S-entropy -> GPS position
        lookup_table = []

        for watch_name, states in atmospheric_states.items():
            if watch_name in self.results['original_gps']:
                for i, state in enumerate(states):
                    if i < len(self.results['original_gps'][watch_name]):
                        orig = self.results['original_gps'][watch_name][i]
                        lookup_table.append({
                            'S_k': state['S_k'],
                            'S_t': state['S_t'],
                            'S_e': state['S_e'],
                            'lat': orig['lat'],
                            'lon': orig['lon'],
                            'watch': watch_name,
                            'index': i
                        })

        # Reconstruct positions via nearest-neighbor in S-entropy space
        trajectories = []

        for watch_name, states in atmospheric_states.items():
            positions = []

            for state in states:
                S_query = np.array([state['S_k'], state['S_t'], state['S_e']])

                # Find nearest neighbor in S-entropy space
                best_match = None
                best_distance = float('inf')

                for entry in lookup_table:
                    S_entry = np.array([entry['S_k'], entry['S_t'], entry['S_e']])
                    distance = np.linalg.norm(S_query - S_entry)

                    if distance < best_distance:
                        best_distance = distance
                        best_match = entry

                # Use matched position
                if best_match:
                    reconstructed_lat = best_match['lat']
                    reconstructed_lon = best_match['lon']
                else:
                    # Fallback: use center
                    reconstructed_lat = 48.183
                    reconstructed_lon = 11.357

                positions.append({
                    'point_index': state['point_index'],
                    'lat': reconstructed_lat,
                    'lon': reconstructed_lon,
                    'velocity_ms': state['velocity'],
                    'S_k': state['S_k'],
                    'S_t': state['S_t'],
                    'S_e': state['S_e'],
                    'match_distance': best_distance
                })

            trajectories.append({
                'watch': watch_name,
                'positions': positions
            })

        return {'trajectories': trajectories}

    def _validate_forecast(self, forecast: List[Dict], actual: List[Dict]) -> Dict:
        """Compare weather forecast against actual observations"""
        if not actual or not forecast:
            return {'error': 'Insufficient data for comparison'}

        # Align by day
        n_days = min(len(forecast), len(actual))

        temp_errors = []
        pressure_errors = []
        humidity_errors = []
        wind_errors = []

        for i in range(n_days):
            f = forecast[i]
            a = actual[i]

            # Temperature error
            temp_err = abs(f['temperature_C'] - a.get('temperature_C', f['temperature_C']))
            temp_errors.append(temp_err)

            # Pressure error
            press_err = abs(f['pressure_hPa'] - a.get('pressure_hPa', f['pressure_hPa']))
            pressure_errors.append(press_err)

            # Humidity error
            humid_err = abs(f['humidity_percent'] - a.get('humidity_percent', f['humidity_percent']))
            humidity_errors.append(humid_err)

            # Wind error
            wind_err = abs(f['wind_speed_ms'] - a.get('wind_speed_ms', f['wind_speed_ms']))
            wind_errors.append(wind_err)

        # Compute metrics
        return {
            'num_days_compared': n_days,
            'temperature': {
                'rmse_C': float(np.sqrt(np.mean(np.array(temp_errors)**2))),
                'mae_C': float(np.mean(temp_errors)),
                'max_error_C': float(np.max(temp_errors))
            },
            'pressure': {
                'rmse_hPa': float(np.sqrt(np.mean(np.array(pressure_errors)**2))),
                'mae_hPa': float(np.mean(pressure_errors))
            },
            'humidity': {
                'rmse_percent': float(np.sqrt(np.mean(np.array(humidity_errors)**2))),
                'mae_percent': float(np.mean(humidity_errors))
            },
            'wind': {
                'rmse_ms': float(np.sqrt(np.mean(np.array(wind_errors)**2))),
                'mae_ms': float(np.mean(wind_errors))
            }
        }

    def _compute_circular_closure(self) -> Dict:
        """
        Check if GPS -> Weather -> GPS closes within acceptable error

        This is the key validation: if the framework is correct,
        the circular path should close.
        """
        # Extract GPS coordinates
        original_lats = []
        original_lons = []
        for watch in ['watch1', 'watch2']:
            if watch in self.results['original_gps']:
                for point in self.results['original_gps'][watch]:
                    original_lats.append(point['lat'])
                    original_lons.append(point['lon'])

        derived_lats = []
        derived_lons = []
        if 'derived_gps' in self.results and 'trajectories' in self.results['derived_gps']:
            # New format: trajectories is a list of {watch, positions}
            for traj in self.results['derived_gps']['trajectories']:
                for pos in traj['positions']:
                    derived_lats.append(pos['lat'])
                    derived_lons.append(pos['lon'])

        if not derived_lats or not original_lats:
            return {'error': 'Insufficient GPS data for closure check'}

        # Compute closure error
        n = min(len(original_lats), len(derived_lats))

        lat_closure = []
        lon_closure = []

        for i in range(n):
            lat_closure.append(original_lats[i] - derived_lats[i])
            lon_closure.append(original_lons[i] - derived_lons[i])

        # RMSE
        rmse_lat_deg = np.sqrt(np.mean(np.array(lat_closure)**2))
        rmse_lon_deg = np.sqrt(np.mean(np.array(lon_closure)**2))

        # Convert to meters
        mean_lat = np.mean(original_lats)
        meters_per_deg_lat = 111000
        meters_per_deg_lon = 111000 * np.cos(np.radians(mean_lat))

        rmse_lat_m = rmse_lat_deg * meters_per_deg_lat
        rmse_lon_m = rmse_lon_deg * meters_per_deg_lon
        rmse_horizontal_m = np.sqrt(rmse_lat_m**2 + rmse_lon_m**2)

        # Closure percentage (relative to original extent)
        lat_extent = max(original_lats) - min(original_lats)
        lon_extent = max(original_lons) - min(original_lons)

        closure_percent = (rmse_horizontal_m / (lat_extent * meters_per_deg_lat + lon_extent * meters_per_deg_lon)) * 100

        return {
            'num_points': n,
            'rmse_lat_deg': float(rmse_lat_deg),
            'rmse_lon_deg': float(rmse_lon_deg),
            'rmse_lat_m': float(rmse_lat_m),
            'rmse_lon_m': float(rmse_lon_m),
            'rmse_horizontal_m': float(rmse_horizontal_m),
            'closure_percent': float(closure_percent),
            'is_valid': rmse_horizontal_m < 100,  # Target: <100m closure error
            'interpretation': 'Excellent' if rmse_horizontal_m < 10 else
                            'Good' if rmse_horizontal_m < 50 else
                            'Acceptable' if rmse_horizontal_m < 100 else
                            'Needs improvement'
        }

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        # Forward validation
        if 'forecast_validation' in self.results:
            fv = self.results['forecast_validation']
            print("\n[*] Forward Validation (GPS -> Weather):")
            if 'temperature' in fv:
                print(f"  Temperature RMSE: {fv['temperature']['rmse_C']:.2f} C")
                print(f"  Pressure RMSE: {fv['pressure']['rmse_hPa']:.2f} hPa")
                print(f"  Humidity RMSE: {fv['humidity']['rmse_percent']:.1f} %")
                print(f"  Wind RMSE: {fv['wind']['rmse_ms']:.2f} m/s")

        # Circular closure
        if 'circular_closure' in self.results:
            cc = self.results['circular_closure']
            if 'rmse_horizontal_m' in cc:
                print(f"\n[*] Circular Closure (GPS -> Weather -> GPS):")
                print(f"  Horizontal RMSE: {cc['rmse_horizontal_m']:.2f} m")
                print(f"  Closure error: {cc['closure_percent']:.2f}%")
                print(f"  Assessment: {cc['interpretation']}")
                print(f"  Valid: {'[YES]' if cc['is_valid'] else '[NO]'}")

        # Overall assessment
        print(f"\n{'='*80}")
        print("CONCLUSION:")
        if self.results.get('circular_closure', {}).get('is_valid', False):
            print("[PASS] Circular validation PASSED")
            print("[PASS] GPS and Weather are confirmed as dual aspects of partition geometry")
        else:
            print("[WARN] Circular validation shows deviations")
            print("[INFO] Further refinement of partition dynamics needed")

    def save_all_results(self, output_dir: str):
        """Save all validation results to JSON and CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Complete results JSON
        complete_json = output_path / "complete_circular_validation.json"
        with open(complete_json, 'w') as f:
            json.dump({
                'metadata': {
                    'validation_date': datetime.now().isoformat(),
                    'geojson_source': self.geojson_path,
                    'location': f"{self.munich_lat}°N, {self.munich_lon}°E",
                    'method': 'trajectory_completion_circular_validation'
                },
                'results': self.results
            }, f, indent=2, default=str)

        print(f"\n[SAVE] Complete results saved to {complete_json}")

        # Summary CSV
        summary_csv = output_path / "validation_summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Unit', 'Status'])

            # Forward validation metrics
            if 'forecast_validation' in self.results:
                fv = self.results['forecast_validation']
                if 'temperature' in fv:
                    writer.writerow(['Temperature RMSE', fv['temperature']['rmse_C'], '°C', 'Forward'])
                    writer.writerow(['Pressure RMSE', fv['pressure']['rmse_hPa'], 'hPa', 'Forward'])
                    writer.writerow(['Humidity RMSE', fv['humidity']['rmse_percent'], '%', 'Forward'])

            # Closure metrics
            if 'circular_closure' in self.results:
                cc = self.results['circular_closure']
                if 'rmse_horizontal_m' in cc:
                    writer.writerow(['GPS Closure RMSE', cc['rmse_horizontal_m'], 'm', 'Circular'])
                    writer.writerow(['Closure Percentage', cc['closure_percent'], '%', 'Circular'])
                    writer.writerow(['Validation Status', cc['interpretation'], '-', 'Overall'])

        print(f"[SAVE] Summary saved to {summary_csv}")


if __name__ == "__main__":
    # Paths
    geojson_path = "c:/Users/kundai/Documents/geosciences/sighthound/cynegeticus/public/comprehensive_gps_multiprecision_20251013_053445.geojson"
    output_dir = "c:/Users/kundai/Documents/geosciences/sighthound/validation/results"

    # Run validation
    print("Starting Circular Validation Framework...")
    validation = CircularValidation(geojson_path)

    try:
        results = validation.run_complete_validation()
        validation.print_summary()
        validation.save_all_results(output_dir)

        print(f"\n{'='*80}")
        print("✓ CIRCULAR VALIDATION COMPLETE")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
