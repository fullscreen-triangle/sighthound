import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PDFConfig:
    """Configuration for PDF calculations"""
    bandwidth_method: str = 'scott'  # or 'silverman'
    kernel_function: str = 'gaussian'
    min_probability: float = 1e-10
    max_probability: float = 1.0


class ProbabilityDensityCalculator:
    """
    Advanced probability density calculation system for GPS trajectories
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        self.config = config or PDFConfig()

    def calculate_trajectory_pdf(
            self,
            trajectory: pd.DataFrame,
            additional_metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate PDF for trajectory considering multiple dimensions

        Args:
            trajectory: DataFrame with GPS and other metrics
            additional_metrics: List of additional columns to consider

        Returns:
            DataFrame with PDF values
        """
        pdf_data = trajectory.copy()

        # Basic position PDF
        position_pdf = self._calculate_spatial_pdf(
            pdf_data['latitude'].values,
            pdf_data['longitude'].values
        )

        # Temporal PDF if timestamp exists
        if 'timestamp' in pdf_data.columns:
            temporal_pdf = self._calculate_temporal_pdf(
                pdf_data['timestamp'].values
            )
            position_pdf *= temporal_pdf

        # Additional metrics PDF
        if additional_metrics:
            metric_pdf = self._calculate_metric_pdf(
                pdf_data[additional_metrics].values
            )
            position_pdf *= metric_pdf

        # Normalize probabilities
        pdf_data['probability'] = self._normalize_probability(position_pdf)

        return pdf_data

    def _calculate_spatial_pdf(
            self,
            latitudes: np.ndarray,
            longitudes: np.ndarray
    ) -> np.ndarray:
        """Calculate spatial PDF using KDE"""
        positions = np.vstack([latitudes, longitudes])
        kde = gaussian_kde(
            positions,
            bw_method=self.config.bandwidth_method
        )
        return kde.evaluate(positions)

    def _calculate_temporal_pdf(
            self,
            timestamps: np.ndarray
    ) -> np.ndarray:
        """Calculate temporal PDF"""
        # Convert timestamps to seconds from start
        times = (timestamps - timestamps[0]).astype('timedelta64[s]').astype(float)
        times = times.reshape(1, -1)
        kde = gaussian_kde(times, bw_method=self.config.bandwidth_method)
        return kde.evaluate(times)[0]

    def _calculate_metric_pdf(
            self,
            metrics: np.ndarray
    ) -> np.ndarray:
        """Calculate PDF for additional metrics"""
        kde = gaussian_kde(
            metrics.T,
            bw_method=self.config.bandwidth_method
        )
        return kde.evaluate(metrics.T)

    def _normalize_probability(
            self,
            probabilities: np.ndarray
    ) -> np.ndarray:
        """Normalize probabilities to specified range"""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability

        normalized = (probabilities - probabilities.min()) / (
                probabilities.max() - probabilities.min()
        )
        return min_prob + normalized * (max_prob - min_prob)

    def combine_pdfs(
            self,
            pdfs: List[np.ndarray],
            weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Combine multiple PDFs with optional weights

        Args:
            pdfs: List of probability arrays
            weights: Optional weights for each PDF

        Returns:
            Combined probability array
        """
        if weights is None:
            weights = [1.0] * len(pdfs)

        weights = np.array(weights) / sum(weights)
        combined = np.zeros_like(pdfs[0])

        for pdf, weight in zip(pdfs, weights):
            combined += pdf * weight

        return self._normalize_probability(combined)


class PathProbabilityFuser:
    """
    Fuse multiple path predictions using probability densities
    """

    def __init__(self):
        self.pdf_calculator = ProbabilityDensityCalculator()

    def fuse_paths(
            self,
            paths: List[pd.DataFrame],
            weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Fuse multiple predicted paths into a single optimal path

        Args:
            paths: List of DataFrames containing different path predictions
            weights: Optional weights for each path

        Returns:
            Fused path DataFrame
        """
        # Calculate PDFs for each path
        pdfs = []
        for path in paths:
            pdf_result = self.pdf_calculator.calculate_trajectory_pdf(
                path,
                additional_metrics=['speed'] if 'speed' in path.columns else None
            )
            pdfs.append(pdf_result['probability'].values)

        # Combine PDFs
        combined_pdf = self.pdf_calculator.combine_pdfs(pdfs, weights)

        # Create fused path by weighted averaging
        fused_path = pd.DataFrame()
        for column in ['latitude', 'longitude', 'speed', 'heading']:
            if all(column in path.columns for path in paths):
                weighted_values = np.zeros_like(paths[0][column].values)
                for path, pdf in zip(paths, pdfs):
                    weighted_values += path[column].values * pdf
                fused_path[column] = weighted_values / sum(pdfs)

        fused_path['probability'] = combined_pdf

        return fused_path
