import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class WeatherConfig:
    """Configuration for weather data retrieval and analysis"""
    openweather_api_key: str
    update_interval: int = 30
    cache_duration: int = 24
    cache_path: str = "cache/weather"
    min_visibility: float = 100  # meters
    max_visibility: float = 10000  # meters
    impact_weights: Dict[str, float] = field(default_factory=lambda: {
        "visibility": 0.3,
        "precipitation": 0.3,
        "wind": 0.2,
        "temperature": 0.1,
        "pressure": 0.1
    })
    confidence_threshold: float = 0.6
    parameters: List[str] = field(default_factory=lambda: [
        "temperature",
        "feels_like",
        "temp_min",
        "temp_max",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_deg",
        "wind_gust",
        "clouds",
        "rain_1h",
        "rain_3h",
        "snow_1h",
        "snow_3h",
        "visibility",
        "weather_main",
        "weather_description",
        "weather_icon",
    ])
    base_url: str = "https://api.openweathermap.org/data/2.5/weather"


class WeatherDataIntegrator:
    """
    Retrieves and integrates weather data with GPS trajectories
    """

    def __init__(self, config: WeatherConfig):
        self.config = config
        self._cache = {}

    def integrate_weather_data(
            self,
            trajectory: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enhanced weather data integration with impact analysis
        """
        weather_data = []
        weather_impacts = []
        weather_confidences = []

        for _, row in trajectory.iterrows():
            weather = self._get_weather_data(
                row['timestamp'],
                row['latitude'],
                row['longitude']
            )
            
            impact = self._calculate_weather_impact(weather)
            confidence = self._calculate_weather_confidence(weather)
            
            weather_data.append(weather)
            weather_impacts.append(impact)
            weather_confidences.append(confidence)

        # Create weather DataFrame
        weather_df = pd.DataFrame(weather_data)
        weather_df['weather_impact'] = weather_impacts
        weather_df['weather_confidence'] = weather_confidences
        
        # Adjust trajectory confidence based on weather
        result = pd.concat([trajectory, weather_df], axis=1)
        if 'confidence' in result.columns:
            result['confidence'] = result['confidence'] * result['weather_confidence']
            
        return result

    def _calculate_weather_impact(self, weather: Dict[str, Any]) -> float:
        """Calculate weather impact on position accuracy"""
        impacts = {
            "visibility": self._visibility_impact(weather.get('visibility', 10000)),
            "precipitation": self._precipitation_impact(weather),
            "wind": self._wind_impact(weather.get('wind_speed', 0)),
            "temperature": self._temperature_impact(weather.get('temperature', 15)),
            "pressure": self._pressure_impact(weather.get('pressure', 1013))
        }
        
        return sum(
            impact * self.config.impact_weights[factor]
            for factor, impact in impacts.items()
        )

    def _calculate_weather_confidence(self, weather: Dict[str, Any]) -> float:
        """Calculate confidence in weather data"""
        # Base confidence from data completeness
        completeness = sum(
            1 for param in self.config.parameters
            if param in weather
        ) / len(self.config.parameters)
        
        # Adjust for extreme conditions
        extremity = 1 - self._calculate_weather_impact(weather)
        
        return min(completeness * 0.7 + extremity * 0.3, 1.0)

    def _visibility_impact(self, visibility: float) -> float:
        """Calculate visibility impact (0-1, higher is worse)"""
        return 1 - np.clip(
            (visibility - self.config.min_visibility) /
            (self.config.max_visibility - self.config.min_visibility),
            0, 1
        )

    def _get_weather_data(
            self,
            timestamp: datetime,
            lat: float,
            lon: float
    ) -> Dict[str, Any]:
        """Get weather data for specific time and location"""
        cache_key = f"{timestamp.date()}_{round(lat, 3)}_{round(lon, 3)}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.config.openweather_api_key,
            'units': 'metric'
        }

        response = requests.get(self.config.base_url, params=params)
        data = response.json()

        weather = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'weather_main': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description']
        }

        self._cache[cache_key] = weather
        return weather
