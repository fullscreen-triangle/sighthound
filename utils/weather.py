import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class WeatherConfig:
    """Configuration for weather data retrieval"""
    openweather_api_key: str
    update_interval: int = 30
    cache_duration: int = 24
    cache_path: str = "cache/weather"
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
        Add weather data to trajectory DataFrame

        Args:
            trajectory: DataFrame with timestamp, lat, lon

        Returns:
            DataFrame with added weather data
        """
        weather_data = []

        for _, row in trajectory.iterrows():
            weather = self._get_weather_data(
                row['timestamp'],
                row['latitude'],
                row['longitude']
            )
            weather_data.append(weather)

        weather_df = pd.DataFrame(weather_data)
        return pd.concat([trajectory, weather_df], axis=1)

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
