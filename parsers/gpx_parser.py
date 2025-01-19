import gpxpy
import pandas as pd
from .base_parser import BaseParser


class GPXParser(BaseParser):
    def _read_file(self):
        with open(self.file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'time': point.time
                    })

        self.df = pd.DataFrame(data)
