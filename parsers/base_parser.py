from abc import ABC, abstractmethod
import pandas as pd
import os
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Optional, Tuple, List, Any, ClassVar
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ParserResult:
    """Container for parser results and metadata"""
    success: bool
    data: Optional[pd.DataFrame] = None
    quality_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_context: Optional[Dict[str, Any]] = None

class ParserError(Exception):
    """Base exception for all parser-related errors"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)

class FileReadError(ParserError):
    """Exception raised when file reading fails"""
    pass

class DataProcessingError(ParserError):
    """Exception raised when data processing fails"""
    pass

class InvalidDataError(ParserError):
    """Exception raised when data validation fails"""
    pass

class BaseParser(ABC):
    """Base parser with quality scoring and error handling"""
    
    # Class variables for file format support
    SUPPORTED_EXTENSIONS: ClassVar[List[str]] = []
    PARSER_NAME: ClassVar[str] = "base"
    
    def __init__(self):
        self.required_columns: List[str] = ['timestamp', 'latitude', 'longitude']
        self.optional_columns: List[str] = ['elevation', 'speed', 'heading', 'distance']
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        
    def parse(self, file_path: str) -> ParserResult:
        """
        Parse file with error handling and quality scoring
        
        Args:
            file_path: Path to the file to be parsed
            
        Returns:
            ParserResult object containing parsing results or error information
            
        Raises:
            No exceptions are raised as they are caught and returned in ParserResult
        """
        # Input validation
        if not file_path:
            return ParserResult(
                success=False,
                error_message="No file path provided",
                error_context={"file_path": file_path}
            )
            
        if not os.path.exists(file_path):
            return ParserResult(
                success=False,
                error_message=f"File not found: {file_path}",
                error_context={"file_path": file_path}
            )
            
        # Extension validation
        if self.SUPPORTED_EXTENSIONS:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                return ParserResult(
                    success=False,
                    error_message=f"Unsupported file extension: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}",
                    error_context={"file_path": file_path, "extension": ext}
                )
        
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
            
        except FileReadError as e:
            return ParserResult(
                success=False,
                error_message=str(e),
                quality_score=0.0,
                error_context={"file_path": file_path, "error_type": "file_read", **e.context}
            )
        except DataProcessingError as e:
            return ParserResult(
                success=False,
                error_message=str(e),
                quality_score=0.0,
                error_context={"file_path": file_path, "error_type": "data_processing", **e.context}
            )
        except InvalidDataError as e:
            return ParserResult(
                success=False,
                error_message=str(e),
                quality_score=0.0,
                error_context={"file_path": file_path, "error_type": "invalid_data", **e.context}
            )
        except Exception as e:
            return ParserResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                quality_score=0.0,
                error_context={"file_path": file_path, "error_type": "unexpected"}
            )

    def _calculate_quality_score(self) -> float:
        """
        Calculate quality score based on data completeness and validity
        
        Returns:
            Quality score between 0.0 and 1.0
        """
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
    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extract format-specific metadata
        
        Returns:
            Dictionary containing metadata
        """
        return {}

    @abstractmethod
    def _read_file(self) -> None:
        """
        Read and parse the file specified in self.file_path
        
        Raises:
            FileReadError: If file cannot be read
        """
        pass

    def _process_dataframe(self) -> pd.DataFrame:
        """
        Process DataFrame to ensure it has standardized columns and values
        
        Returns:
            Processed DataFrame
            
        Raises:
            DataProcessingError: If DataFrame processing fails
            InvalidDataError: If DataFrame is missing required columns
        """
        if self.df is None:
            raise DataProcessingError("DataFrame not initialized. Call _read_file first.", 
                                      {"parser": self.PARSER_NAME})

        # Standardize column names
        column_mapping = {
            'time': 'timestamp',
            'ele': 'elevation'
        }
        self.df = self.df.rename(columns=column_mapping)

        # Ensure required columns exist
        missing_columns = [col for col in self.required_columns if col not in self.df.columns]
        if missing_columns:
            raise InvalidDataError(f"Missing required columns: {missing_columns}", 
                                   {"missing_columns": missing_columns})

        self.df = self._calculate_distances()
        return self.df

    def _calculate_distances(self) -> pd.DataFrame:
        """
        Calculate cumulative distances between points
        
        Returns:
            DataFrame with distance column added
            
        Raises:
            DataProcessingError: If distance calculation fails
        """
        if len(self.df) < 2:
            self.df['distance'] = 0
            return self.df

        try:
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
        except Exception as e:
            raise DataProcessingError(f"Error calculating distances: {str(e)}", 
                                     {"error": str(e)})
