import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
from .base_parser import BaseParser


class TCXParser(BaseParser):
    def _read_file(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

        data = []
        trackpoints = root.findall('.//ns:Trackpoint', ns)

        for point in trackpoints:
            time = point.find('ns:Time', ns).text
            lat = point.find('.//ns:LatitudeDegrees', ns)
            lon = point.find('.//ns:LongitudeDegrees', ns)
            ele = point.find('.//ns:AltitudeMeters', ns)

            if lat is not None and lon is not None:
                data.append({
                    'latitude': float(lat.text),
                    'longitude': float(lon.text),
                    'elevation': float(ele.text) if ele is not None else None,
                    'time': datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
                })

        self.df = pd.DataFrame(data)
