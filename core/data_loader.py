from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..parsers.fit_parser import FitParser
from ..parsers.gpx_parser import GPXParser
from ..parsers.tcx_parser import TCXParser
from ..parsers.kml_parser import KMLParser


@dataclass
class LoaderConfig:
    """Configuration for data loading"""
    n_workers: int = 4
    chunk_size: int = 1000
    validate_data: bool = True
    interpolation_method: str = 'cubic'
    required_columns: List[str] = field(default_factory=lambda: [
        'timestamp', 'latitude', 'longitude'
    ])
    optional_columns: List[str] = field(default_factory=lambda: [
        'elevation', 'heart_rate', 'cadence', 'speed',
        'power', 'temperature', 'distance'
    ])

class DataLoader:
    """High-performance data loader for GPS activity files"""

    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self.parsers = {
            '.fit': FitParser(),
            '.gpx': GPXParser(),
            '.tcx': TCXParser(),
            '.kml': KMLParser()
        }
        self.logger = logging.getLogger(__name__)

    def load_files(self, file_paths: List[Union[str, Path]]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Load multiple GPS files in parallel and return confidence scores

        Returns:
            Tuple of (combined DataFrame, confidence scores by source)
        """
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_file, Path(file_path)): file_path
                for file_path in file_paths
            }

            dataframes = []
            confidence_scores = {}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df, confidence = future.result()
                    if df is not None and not df.empty:
                        dataframes.append(df)
                        confidence_scores[str(file_path)] = confidence
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")

        if not dataframes:
            raise ValueError("No valid data found in provided files")

        return self._combine_dataframes(dataframes), confidence_scores

    def _load_single_file(self, file_path: Path) -> Tuple[pd.DataFrame, float]:
        """Load a single GPS file and calculate confidence score"""
        extension = file_path.suffix.lower()
        if extension not in self.parsers:
            self.logger.warning(f"Unsupported file type: {extension}")
            return pd.DataFrame(), 0.0

        try:
            df = self.parsers[extension].parse(file_path)
            if df.empty:
                return df, 0.0

            # Validate and standardize data
            df = self._standardize_dataframe(df)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(df)
            
            return df, confidence

        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            return pd.DataFrame(), 0.0

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame columns and formats"""
        # Ensure required columns exist
        missing_cols = [col for col in self.config.required_columns 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Add missing optional columns with NaN
        for col in self.config.optional_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _calculate_confidence_score(self, df: pd.DataFrame) -> float:
        """Calculate confidence score for the data source"""
        scores = []
        
        # Check data completeness
        completeness = df[self.config.required_columns].notna().mean().mean()
        scores.append(completeness)
        
        # Check coordinate validity
        valid_coords = (
            df['latitude'].between(-90, 90) &
            df['longitude'].between(-180, 180)
        ).mean()
        scores.append(valid_coords)
        
        # Check temporal consistency
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        temporal_score = 1 - (time_diffs.std() / time_diffs.mean() 
                            if not time_diffs.empty else 0)
        scores.append(temporal_score)
        
        # Optional metrics presence
        optional_score = df[self.config.optional_columns].notna().mean().mean()
        scores.append(optional_score)
        
        # Weighted average of scores
        weights = [0.4, 0.3, 0.2, 0.1]  # Adjust weights as needed
        return float(np.average(scores, weights=weights))

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
            method=self.config.interpolation_method,
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
