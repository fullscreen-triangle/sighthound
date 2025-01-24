from abc import ABC, abstractmethod
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParserResult:
    """Container for parser results and metadata"""
    success: bool
    data: Optional[pd.DataFrame] = None
    quality_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict = None

class BaseParser(ABC):
    """Base parser with quality scoring and error handling"""
    
    def __init__(self):
        self.required_columns = ['timestamp', 'latitude', 'longitude']
        self.optional_columns = ['elevation', 'speed', 'heading', 'distance']
        self.df = None
        
    def parse(self, file_path) -> ParserResult:
        """Parse file with error handling and quality scoring"""
        try:
            self.file_path = file_path
            self._read_file()
            self.df = self._process_dataframe()
            quality_score = self._calculate_quality_score()
            
            return ParserResult(
                success=True,
                data=self.df,
                quality_score=quality_score,
                metadata=self._extract_metadata()
            )
            
        except Exception as e:
            return ParserResult(
                success=False,
                error_message=str(e),
                quality_score=0.0
            )

    def _calculate_quality_score(self) -> float:
        """Calculate quality score based on data completeness and validity"""
        if self.df is None or len(self.df) == 0:
            return 0.0
            
        score = 0.0
        
        # Required columns check (50% of score)
        required_cols_present = all(col in self.df.columns for col in self.required_columns)
        if required_cols_present:
            score += 0.5
            
        # Optional columns (30% of score)
        optional_cols_score = sum(col in self.df.columns for col in self.optional_columns) / len(self.optional_columns) * 0.3
        score += optional_cols_score
        
        # Data density score (20% of score)
        if 'timestamp' in self.df.columns:
            time_range = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds()
            density = len(self.df) / (time_range + 1)  # points per second
            density_score = min(density / 1.0, 1.0) * 0.2  # normalize to max 1 point/second
            score += density_score
            
        return score

    @abstractmethod
    def _extract_metadata(self) -> Dict:
        """Extract format-specific metadata"""
        return {}

    @abstractmethod
    def _read_file(self):
        pass

    def _process_dataframe(self):
        if self.df is None:
            raise ValueError("DataFrame not initialized. Call _read_file first.")

        # Standardize column names
        column_mapping = {
            'time': 'timestamp',
            'ele': 'elevation'
        }
        self.df = self.df.rename(columns=column_mapping)

        # Ensure required columns exist
        missing_columns = [col for col in self.required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.df = self._calculate_distances()
        return self.df

    def _calculate_distances(self):
        if len(self.df) < 2:
            self.df['distance'] = 0
            return self.df

        distances = [0]

        for i in range(1, len(self.df)):
            lat1, lon1 = self.df.iloc[i - 1]['latitude'], self.df.iloc[i - 1]['longitude']
            lat2, lon2 = self.df.iloc[i]['latitude'], self.df.iloc[i]['longitude']

            R = 6371  # Earth's radius in kilometers

            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c

            distances.append(distances[-1] + distance)

        self.df['distance'] = distances
        return self.df
