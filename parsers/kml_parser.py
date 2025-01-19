import xml.etree.ElementTree as ET
import pandas as pd
from .base_parser import BaseParser


class KMLParser(BaseParser):
    def _read_file(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        data = []
        coordinates = root.findall('.//kml:coordinates', ns)

        for coord_element in coordinates:
            coord_string = coord_element.text.strip()
            coord_list = coord_string.split()

            for coord in coord_list:
                lon, lat, ele = map(float, coord.split(','))
                data.append({
                    'latitude': lat,
                    'longitude': lon,
                    'elevation': ele,
                    'time': None
                })

        self.df = pd.DataFrame(data)
