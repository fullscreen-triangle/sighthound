import sys
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import os
import logging
from datetime import datetime

from parsers.gpx_parser import GPXParser
from parsers.fit_parser import FITParser
from parsers.tcx_parser import TCXParser
from parsers.kml_parser import KMLParser
from utils.data_fusion import ActivityDataFuser, DataFusionConfig
from utils.cell_triangulation import CellDataTriangulator
from utils.czml_generator import generate_czml, save_czml
from utils.geojson_generator import generate_geojson, save_geojson


class Pipeline:
    """
    Progressive data processing pipeline with fallback mechanisms
    """
    def __init__(self, config_path: str, input_folder: str, output_folder: str):
        self.config = self._load_config(config_path)
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize parsers with fallback order
        self.parsers = {
            '.gpx': GPXParser(self.config['parsers']['gpx']),
            '.fit': FITParser(self.config['parsers']['fit']),
            '.tcx': TCXParser(self.config['parsers']['tcx']),
            '.kml': KMLParser(self.config['parsers']['kml'])
        }
        
        # Initialize data fusion components
        fusion_config = DataFusionConfig(**self.config['progressive_fusion'])
        self.data_fuser = ActivityDataFuser(fusion_config)
        
        # Initialize cell triangulation if configured
        self.cell_triangulator = None
        if self.config['cell_triangulation'].get('api_key'):
            self.cell_triangulator = CellDataTriangulator(self.config['cell_triangulation'])
        
        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for output and logging"""
        dirs = [
            self.output_folder,
            Path(self.config['logging']['file']).parent,
            Path(self.config['progressive_fusion']['intermediate_path']),
            Path(self.config['output']['failed_parses_path'])
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure logging with file and console handlers"""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise Exception(f"Failed to load config: {str(e)}")

    def run(self):
        """Execute the pipeline with progressive fusion"""
        try:
            # Load and parse input files with quality scoring
            parsed_data = self._load_input_files()
            
            if not any(parsed_data.values()):
                self.logger.warning("No valid trajectories found")
                return False

            # Progressive fusion based on quality
            fused_trajectory = None
            
            # Start with high quality data
            if parsed_data['high_quality']:
                fused_trajectory = self._fuse_trajectories(
                    [data for data, _ in parsed_data['high_quality']],
                    [score for _, score in parsed_data['high_quality']]
                )
            
            # Add medium quality if needed
            if (fused_trajectory is None or self.config['fusion']['use_all_quality_levels']) and parsed_data['medium_quality']:
                trajectories = [data for data, _ in parsed_data['medium_quality']]
                weights = [score for _, score in parsed_data['medium_quality']]
                if fused_trajectory is not None:
                    trajectories.append(fused_trajectory)
                    weights.append(1.0)
                fused_trajectory = self._fuse_trajectories(trajectories, weights)
            
            # Add low quality as last resort
            if (fused_trajectory is None or self.config['fusion']['use_all_quality_levels']) and parsed_data['low_quality']:
                trajectories = [data for data, _ in parsed_data['low_quality']]
                weights = [score * 0.5 for _, score in parsed_data['low_quality']]  # Reduce weight of low quality data
                if fused_trajectory is not None:
                    trajectories.append(fused_trajectory)
                    weights.append(1.0)
                fused_trajectory = self._fuse_trajectories(trajectories, weights)
            
            if fused_trajectory is None:
                self.logger.error("Fusion failed completely")
                return False

            # Generate outputs
            self._generate_outputs(fused_trajectory)
            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return False

    def _validate_trajectory(self, trajectory: pd.DataFrame, filename: Path) -> bool:
        """Validate trajectory data with detailed logging"""
        parser_config = self.config['parsers'][filename.suffix[1:]]
        required_columns = set(parser_config['required_fields'])
        actual_columns = set(trajectory.columns)
        missing_columns = required_columns - actual_columns

        if missing_columns:
            self.logger.warning(f"File {filename} missing columns: {missing_columns}")
            if self.config['output']['save_failed_parses']:
                self._save_failed_parse(trajectory, filename)
            return False
        return True

    def _save_failed_parse(self, trajectory: pd.DataFrame, filename: Path):
        """Save failed parse for inspection"""
        failed_path = Path(self.config['output']['failed_parses_path'])
        save_path = failed_path / f"failed_{filename.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trajectory.to_csv(save_path, index=False)

    def _load_input_files(self) -> Dict[str, List[Tuple[pd.DataFrame, float]]]:
        """Load and parse input files with quality scoring"""
        parsed_data = {
            'high_quality': [],    # quality_score > 0.8
            'medium_quality': [],  # quality_score > 0.5
            'low_quality': []      # quality_score <= 0.5
        }
        
        parse_results = []
        
        # Try parsing all files
        for file_path in self.input_folder.glob('*.*'):
            parser = self.parsers.get(file_path.suffix.lower())
            if parser:
                result = parser.parse(str(file_path))
                if result.success:
                    parse_results.append((result, file_path))
        
        # Sort by quality score
        parse_results.sort(key=lambda x: x[0].quality_score, reverse=True)
        
        # Calculate combined quality score
        if parse_results:
            total_quality = sum(result.quality_score for result, _ in parse_results)
            self.logger.info(f"Combined parsing quality score: {total_quality:.2f}")
            
            # Categorize results
            for result, file_path in parse_results:
                data_entry = (result.data, result.quality_score)
                if result.quality_score > 0.8:
                    parsed_data['high_quality'].append(data_entry)
                elif result.quality_score > 0.5:
                    parsed_data['medium_quality'].append(data_entry)
                else:
                    parsed_data['low_quality'].append(data_entry)
                
            # Log parsing summary
            self._log_parsing_summary(parse_results)
        
        return parsed_data

    def _log_parsing_summary(self, parse_results: List[Tuple[ParserResult, Path]]):
        """Log detailed parsing summary"""
        summary = {
            'total_files': len(parse_results),
            'successful_parses': sum(1 for r, _ in parse_results if r.success),
            'quality_distribution': {
                'high': sum(1 for r, _ in parse_results if r.quality_score > 0.8),
                'medium': sum(1 for r, _ in parse_results if 0.5 < r.quality_score <= 0.8),
                'low': sum(1 for r, _ in parse_results if r.quality_score <= 0.5)
            }
        }
        
        self.logger.info("Parsing Summary:")
        self.logger.info(f"Total files processed: {summary['total_files']}")
        self.logger.info(f"Successful parses: {summary['successful_parses']}")
        self.logger.info("Quality distribution:")
        self.logger.info(f"  High quality: {summary['quality_distribution']['high']}")
        self.logger.info(f"  Medium quality: {summary['quality_distribution']['medium']}")
        self.logger.info(f"  Low quality: {summary['quality_distribution']['low']}")

    def _standardize_timestamp(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp format"""
        if 'timestamp' in trajectory.columns:
            try:
                trajectory['timestamp'] = pd.to_datetime(trajectory['timestamp'], utc=True)
            except Exception as e:
                self.logger.warning(f"Timestamp standardization failed: {str(e)}")
        return trajectory

    def _fuse_trajectories(self, trajectories: List[pd.DataFrame], weights: List[float]) -> Optional[pd.DataFrame]:
        """Perform fusion with given trajectories and weights"""
        if not trajectories or not weights:
            return None

        try:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Perform weighted fusion
            fused_data = trajectories[0]
            for i in range(1, len(trajectories)):
                fused_data += trajectories[i] * normalized_weights[i]

            # Log fusion results
            self._log_fusion_results(trajectories, fused_data)
            
            return fused_data

        except Exception as e:
            self.logger.error(f"Fusion failed: {str(e)}")
            return None

    def _log_fusion_results(self, trajectories: List[pd.DataFrame], fused_data: pd.DataFrame):
        """Log fusion results and statistics"""
        fusion_report = {
            'timestamp': datetime.now().isoformat(),
            'sources_used': [f"trajectory_{i}" for i in range(len(trajectories))],
            'data_points': len(fused_data),
            'time_range': {
                'start': fused_data['timestamp'].min().isoformat(),
                'end': fused_data['timestamp'].max().isoformat()
            },
            'confidence_stats': {
                'mean': float(fused_data.get('confidence', 1.0).mean()),
                'min': float(fused_data.get('confidence', 1.0).min()),
                'max': float(fused_data.get('confidence', 1.0).max())
            }
        }
        
        report_path = Path(self.config['logging']['data_fusion_report'])
        with open(report_path, 'w') as f:
            json.dump(fusion_report, f, indent=2)

    def _log_data_sources(self, trajectories: List[pd.DataFrame]):
        """Log information about parsed data sources"""
        source_info = []
        for i, traj in enumerate(trajectories):
            source_info.append({
                'source_index': i,
                'points': len(traj),
                'time_range': {
                    'start': traj['timestamp'].min().isoformat(),
                    'end': traj['timestamp'].max().isoformat()
                },
                'columns': list(traj.columns)
            })
        
        self.logger.info(f"Parsed data sources: {json.dumps(source_info, indent=2)}")

    def _generate_outputs(self, trajectory: pd.DataFrame):
        """Generate output files based on configuration"""
        # Save as CSV
        trajectory.to_csv(self.output_folder / 'trajectory.csv', index=False)

        # Generate and save CZML if configured
        if 'czml' in self.config['output']['formats']:
            czml_data = generate_czml(trajectory, self.config['output']['czml_options'])
            save_czml(czml_data, self.output_folder / 'trajectory.czml')

        # Generate and save GeoJSON if configured
        if 'geojson' in self.config['output']['formats']:
            geojson_data = generate_geojson(trajectory, self.config['output']['geojson_options'])
            save_geojson(geojson_data, self.output_folder / 'trajectory.geojson')

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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FOLDER = os.path.join(base_dir, "../public")
    CONFIG_PATH = os.path.join(base_dir, "../config/sighthound.yaml")
    OUTPUT_FOLDER = os.path.join(base_dir, "../output")

    try:
        pipeline = Pipeline(CONFIG_PATH, INPUT_FOLDER, OUTPUT_FOLDER)
        success = pipeline.run()
        return 0 if success else 1
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
