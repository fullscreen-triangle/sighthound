import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import requests
import time
import logging
from cachetools import TTLCache


@dataclass
class CellTriangulationConfig:
    """Configuration for cell tower triangulation"""
    token: str
    min_towers: int = 3
    max_tower_distance_km: float = 5
    signal_strength_threshold: float = -100
    cache_duration_hours: int = 24
    cache_path: str = "cache/cell_towers"
    weight_factors: Dict[str, float] = field(default_factory=lambda: {
        "signal_strength": 0.6,
        "distance": 0.4
    })
    confidence_threshold: float = 0.6  # Added for confidence scoring
    min_signal_strength: float = -120  # dBm
    max_signal_strength: float = -50   # dBm
    base_url: str = "https://opencellid.org"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 10.0


@dataclass
class CellTowerInfo:
    """Information about cell tower from OpenCellID"""
    lat: float
    lon: float
    mcc: int
    mnc: int
    lac: int
    cellid: int
    radio: str
    range: float
    samples: int
    averageSignalStrength: float


class CellDataTriangulator:
    def __init__(self, config: CellTriangulationConfig):
        self.config = config
        self.cache = TTLCache(maxsize=1000, ttl=self.config.cache_duration_hours * 3600)
        self.session = requests.Session()

    def _haversine_distances(self, lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Calculate haversine distances between a point and array of points

        Args:
            lat1, lon1: Reference point coordinates
            lat2, lon2: Arrays of target point coordinates
        Returns:
            Array of distances in meters
        """
        R = 6371000  # Earth's radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat / 2) ** 2 + \
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def _calculate_weight(self, signal_strength: float, tower_range: float) -> float:
        """
        Calculate weight for a tower based on signal strength and range

        Args:
            signal_strength: Average signal strength in dBm
            tower_range: Reported range of the tower in meters
        Returns:
            Weight value between 0 and 1
        """
        signal_weight = (signal_strength + 120) / 70.0
        signal_weight = np.clip(signal_weight, 0, 1)

        range_weight = 1 - (min(tower_range, 5000) / 5000)

        return 0.7 * signal_weight + 0.3 * range_weight

    def _get_area_towers(self, bbox: tuple, radio: str = None) -> List[CellTowerInfo]:
        """
        Fetch cell towers in a given area from OpenCellID API

        Args:
            bbox: Tuple of (latmin, lonmin, latmax, lonmax)
            radio: Optional radio type filter (GSM, UMTS, LTE, etc.)
        """
        cache_key = f"{bbox}_{radio}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(self.config.max_retries):
            try:
                params = {
                    'key': self.config.token,
                    'BBOX': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    'format': 'json'
                }
                if radio:
                    params['radio'] = radio

                response = self.session.get(
                    f"{self.config.base_url}/cell/getInArea",
                    params=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()

                towers = []
                for cell in data['cells']:
                    tower = CellTowerInfo(
                        lat=cell['lat'],
                        lon=cell['lon'],
                        mcc=cell['mcc'],
                        mnc=cell['mnc'],
                        lac=cell['lac'],
                        cellid=cell['cellid'],
                        radio=cell['radio'],
                        range=cell['range'],
                        samples=cell['samples'],
                        averageSignalStrength=cell.get('averageSignalStrength', -100)
                    )
                    towers.append(tower)

                self.cache[cache_key] = towers
                return towers

            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"Failed to fetch area towers: {str(e)}")
                time.sleep(self.config.retry_delay)

        return []

    def triangulate_position(self, lat: float, lon: float, radius_km: float = 5) -> Optional[Dict[str, Any]]:
        """
        Enhanced triangulation with confidence scoring
        
        Args:
            lat: Approximate latitude
            lon: Approximate longitude
            radius_km: Search radius in kilometers
        """
        bbox = self._calculate_bbox(lat, lon, radius_km)
        
        all_towers = []
        tower_confidences = []
        
        for radio in ['LTE', 'UMTS', 'GSM']:
            towers = self._get_area_towers(bbox, radio)
            for tower in towers:
                confidence = self._calculate_tower_confidence(tower)
                if confidence >= self.config.confidence_threshold:
                    all_towers.append(tower)
                    tower_confidences.append(confidence)

        if len(all_towers) < self.config.min_towers:
            return None

        positions = np.array([[t.lat, t.lon] for t in all_towers])
        weights = np.array([
            self._calculate_weight(t.averageSignalStrength, t.range) * conf
            for t, conf in zip(all_towers, tower_confidences)
        ])

        weights = weights / np.sum(weights)
        weighted_position = np.average(positions, weights=weights, axis=0)
        
        # Calculate position confidence
        distances = self._haversine_distances(
            weighted_position[0],
            weighted_position[1],
            positions[:, 0],
            positions[:, 1]
        )
        accuracy = np.average(distances, weights=weights)
        position_confidence = self._calculate_position_confidence(
            accuracy, 
            len(all_towers),
            np.mean(tower_confidences)
        )

        if position_confidence < self.config.confidence_threshold:
            return None

        return {
            'latitude': float(weighted_position[0]),
            'longitude': float(weighted_position[1]),
            'accuracy': float(accuracy),
            'confidence': float(position_confidence),
            'num_towers': len(all_towers),
            'tower_types': self._get_tower_type_distribution(all_towers)
        }

    def _calculate_tower_confidence(self, tower: CellTowerInfo) -> float:
        """Calculate confidence score for a single tower"""
        # Signal strength confidence
        signal_range = self.config.max_signal_strength - self.config.min_signal_strength
        signal_conf = (tower.averageSignalStrength - self.config.min_signal_strength) / signal_range
        signal_conf = np.clip(signal_conf, 0, 1)
        
        # Sample size confidence
        sample_conf = min(tower.samples / 1000, 1)
        
        # Range confidence
        range_conf = 1 - (min(tower.range, 5000) / 5000)
        
        return 0.4 * signal_conf + 0.3 * sample_conf + 0.3 * range_conf

    def _calculate_position_confidence(
            self, 
            accuracy: float, 
            num_towers: int,
            avg_tower_confidence: float
    ) -> float:
        """Calculate overall position confidence"""
        accuracy_conf = 1 - min(accuracy / (self.config.max_tower_distance_km * 1000), 1)
        tower_count_conf = min(num_towers / 10, 1)  # Max confidence at 10 towers
        
        return 0.4 * accuracy_conf + 0.3 * tower_count_conf + 0.3 * avg_tower_confidence
