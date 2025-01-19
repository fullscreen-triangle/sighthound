from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import logging

from ..parsers.fit_parser import FitParser
from ..parsers.gpx_parser import GPXParser
from ..parsers.tcx_parser import TCXParser
from ..parsers.kml_parser import KMLParser


class DataLoader:
    """High-performance data loader for GPS activity files"""

    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.parsers = {
            '.fit': FitParser(),
            '.gpx': GPXParser(),
            '.tcx': TCXParser(),
            '.kml': KMLParser()
        }
        self.logger = logging.getLogger(__name__)

    def load_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load multiple GPS files in parallel

        Args:
            file_paths: List of file paths to process

        Returns:
            Combined DataFrame with all GPS data
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_file, file_path): file_path
                for file_path in file_paths
            }

            dataframes = []
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        dataframes.append(df)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")

        if not dataframes:
            raise ValueError("No valid data found in provided files")

        return self._combine_dataframes(dataframes)

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """Load a single GPS file"""
        extension = file_path.lower()[-4:]
        if extension not in self.parsers:
            self.logger.warning(f"Unsupported file type: {extension}")
            return pd.DataFrame()

        return self.parsers[extension].parse(file_path)

    def _combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames with intelligent handling"""
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Sort by timestamp
        combined_df.sort_values('timestamp', inplace=True)

        # Remove duplicates
        combined_df.drop_duplicates(
            subset=['timestamp', 'latitude', 'longitude'],
            keep='first',
            inplace=True
        )

        # Handle missing values
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numeric_columns] = combined_df[numeric_columns].interpolate(
            method='cubic',
            limit_direction='both'
        )

        # Reset index
        return combined_df.reset_index(drop=True)

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate the loaded data"""
        required_columns = ['timestamp', 'latitude', 'longitude']
        if not all(col in df.columns for col in required_columns):
            return False

        # Check for valid coordinate ranges
        if not (
                df['latitude'].between(-90, 90).all() and
                df['longitude'].between(-180, 180).all()
        ):
            return False

        return True
