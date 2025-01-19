from fitparse import FitFile
import pandas as pd
from .base_parser import BaseParser


class FITParser(BaseParser):
    def _read_file(self):
        fitfile = FitFile(self.file_path)

        data = []
        for record in fitfile.get_messages('record'):
            point = {}
            for data_point in record:
                if data_point.name == 'position_lat':
                    point['latitude'] = data_point.value * 180 / 2 ** 31
                elif data_point.name == 'position_long':
                    point['longitude'] = data_point.value * 180 / 2 ** 31
                elif data_point.name == 'altitude':
                    point['elevation'] = data_point.value
                elif data_point.name == 'timestamp':
                    point['time'] = data_point.value

            if 'latitude' in point and 'longitude' in point:
                data.append(point)

        self.df = pd.DataFrame(data)
