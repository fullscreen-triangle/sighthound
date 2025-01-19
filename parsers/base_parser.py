from abc import ABC, abstractmethod
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


class BaseParser(ABC):
    def parse(self, file_path):
        self.file_path = file_path
        self._read_file()
        self.df = self._process_dataframe()
        return self.df

    @abstractmethod
    def _read_file(self):
        pass

    def _process_dataframe(self):
        if self.df is None:
            raise ValueError("DataFrame not initialized. Call _read_file first.")

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
