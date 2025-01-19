import sys
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import json
import os

from parsers.fit_parser import FITParser
from parsers.gpx_parser import GPXParser
from parsers.kml_parser import KMLParser
from parsers.tcx_parser import TCXParser
from utils.triangulation import TrajectoryTriangulator, TriangulationConfig
from utils.weather import WeatherDataIntegrator, WeatherConfig
from utils.cell_triangulation import CellDataTriangulator, CellTriangulationConfig
from utils.czml_generator import generate_czml, save_czml


class Pipeline:
    def __init__(self, config_path: str, input_folder: str, output_folder: str):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.config = self._load_config(config_path)
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_components(self):
        self.parsers = {
            '.gpx': GPXParser(),
            '.kml': KMLParser(),
            '.tcx': TCXParser(),
            '.fit': FITParser()
        }

        self.triangulator = TrajectoryTriangulator(
            TriangulationConfig(**self.config.get('triangulation', {}))
        )

        if 'weather' in self.config:
            self.weather_integrator = WeatherDataIntegrator(
                WeatherConfig(**self.config['weather'])
            )
        else:
            self.weather_integrator = None

        if 'cell_triangulation' in self.config:
            self.cell_triangulator = CellDataTriangulator(
                CellTriangulationConfig(**self.config['cell_triangulation'])
            )
        else:
            self.cell_triangulator = None

    def run(self):
        print(f"Starting pipeline execution. Input folder: {self.input_folder}")

        try:
            trajectories = self._load_input_files()
            if not trajectories:
                raise ValueError("No valid trajectory files found in input folder")

            unified_trajectory = self._triangulate_trajectories(trajectories)

            if self.weather_integrator:
                unified_trajectory = self.weather_integrator.integrate_weather_data(
                    unified_trajectory
                )

            self._generate_outputs(unified_trajectory)
            print("Pipeline execution completed successfully")

        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            raise

    def _load_input_files(self) -> List[pd.DataFrame]:
        trajectories = []
        for file_path in self.input_folder.glob('*.*'):
            if file_path.suffix.lower() in self.parsers:
                try:
                    parser = self.parsers[file_path.suffix.lower()]
                    trajectory = parser.parse(file_path)
                    trajectories.append(trajectory)
                    print(f"Successfully parsed {file_path}")
                except Exception as e:
                    print(f"Failed to parse {file_path}: {str(e)}")
        return trajectories

    def _triangulate_trajectories(self, trajectories: List[pd.DataFrame]) -> pd.DataFrame:
        unified_trajectory = self.triangulator.triangulate_positions(trajectories, [1.0] * len(trajectories))

        if self.cell_triangulator:
            try:
                enhanced_trajectory = self._enhance_with_cell_data(unified_trajectory)
                return enhanced_trajectory
            except Exception as e:
                print(f"Cell triangulation failed: {str(e)}")
                return unified_trajectory

        return unified_trajectory

    def _enhance_with_cell_data(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        enhanced_positions = []
        batch_size = 10

        for i in range(0, len(trajectory), batch_size):
            batch = trajectory.iloc[i:i + batch_size]

            for _, row in batch.iterrows():
                lat, lon = row['latitude'], row['longitude']
                radius_km = 5.0
                lat_offset = radius_km / 111.32
                lon_offset = radius_km / (111.32 * np.cos(np.radians(lat)))

                bbox = (
                    lat - lat_offset,
                    lon - lon_offset,
                    lat + lat_offset,
                    lon + lon_offset
                )

                try:
                    towers = self.cell_triangulator._get_area_towers(bbox)

                    if len(towers) >= self.config['cell_triangulation']['min_towers']:
                        tower_positions = np.array([[t.lat, t.lon] for t in towers])
                        tower_weights = np.array([
                            self.cell_triangulator._calculate_weight(
                                t.averageSignalStrength,
                                t.range
                            ) for t in towers
                        ])

                        enhanced_pos = np.average(tower_positions, weights=tower_weights, axis=0)

                        enhanced_row = row.copy()
                        enhanced_row['latitude'] = enhanced_pos[0]
                        enhanced_row['longitude'] = enhanced_pos[1]
                        enhanced_row['position_source'] = 'cell_enhanced'
                        enhanced_row['num_towers'] = len(towers)
                    else:
                        enhanced_row = row.copy()
                        enhanced_row['position_source'] = 'original'
                        enhanced_row['num_towers'] = len(towers)

                    enhanced_positions.append(enhanced_row)

                except Exception as e:
                    print(f"Cell triangulation failed for position {lat},{lon}: {str(e)}")
                    row['position_source'] = 'original'
                    enhanced_positions.append(row)

        enhanced_df = pd.DataFrame(enhanced_positions)
        enhanced_df['position_confidence'] = enhanced_df.apply(
            lambda x: 0.8 if x['position_source'] == 'cell_enhanced' else 0.5,
            axis=1
        )

        return enhanced_df

    def _generate_outputs(self, trajectory: pd.DataFrame):
        # Save as CSV
        trajectory.to_csv(self.output_folder / 'trajectory.csv', index=False)

        # Save as GeoJSON
        self._save_geojson(trajectory, self.output_folder / 'trajectory.geojson')

        # Generate and save CZML
        czml_data = generate_czml(trajectory)
        save_czml(czml_data, self.output_folder / 'trajectory.czml')

    def _save_geojson(self, df: pd.DataFrame, output_path: Path):
        features = []
        for _, row in df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['longitude'], row['latitude']]
                },
                "properties": {
                    key: value for key, value in row.items()
                    if key not in ['latitude', 'longitude']
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FOLDER = os.path.join(base_dir, "../public")
    CONFIG_PATH = os.path.join(base_dir, "../config/sighthound.yaml")
    OUTPUT_FOLDER = os.path.join(base_dir, "../output")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        pipeline = Pipeline(CONFIG_PATH, INPUT_FOLDER, OUTPUT_FOLDER)
        pipeline.run()
        return 0
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
