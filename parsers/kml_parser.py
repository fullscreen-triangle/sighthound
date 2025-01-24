import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from typing import Dict, List
from .base_parser import BaseParser


class KMLParser(BaseParser):
    def _read_file(self):
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            data = []
            
            # Try to find timestamps in TimeStamp or TimeSpan elements
            timestamps = self._extract_timestamps(root, ns)
            coordinates = root.findall('.//kml:coordinates', ns)

            for i, coord_element in enumerate(coordinates):
                coord_string = coord_element.text.strip()
                coord_list = coord_string.split()

                for j, coord in enumerate(coord_list):
                    lon, lat, ele = map(float, coord.split(','))
                    point_data = {
                        'latitude': lat,
                        'longitude': lon,
                        'elevation': ele,
                        'timestamp': timestamps[i] if i < len(timestamps) else None
                    }
                    
                    # Try to extract additional data
                    extended_data = self._extract_extended_data(coord_element, ns)
                    point_data.update(extended_data)
                    
                    data.append(point_data)

            self.df = pd.DataFrame(data)
            if 'timestamp' not in self.df.columns or self.df['timestamp'].isna().all():
                self.df['timestamp'] = pd.date_range(
                    start=datetime.now(),
                    periods=len(self.df),
                    freq='S'
                )
                
        except Exception as e:
            raise ValueError(f"Failed to parse KML file: {str(e)}")

    def _extract_timestamps(self, root: ET.Element, ns: Dict) -> List[datetime]:
        """Extract timestamps from KML TimeStamp or TimeSpan elements"""
        timestamps = []
        time_elements = root.findall('.//kml:TimeStamp/kml:when', ns)
        time_elements.extend(root.findall('.//kml:TimeSpan/kml:begin', ns))
        
        for time_elem in time_elements:
            try:
                timestamps.append(datetime.strptime(time_elem.text, '%Y-%m-%dT%H:%M:%SZ'))
            except:
                timestamps.append(None)
                
        return timestamps

    def _extract_extended_data(self, element: ET.Element, ns: Dict) -> Dict:
        """Extract additional data from ExtendedData elements"""
        extended_data = {}
        data_elements = element.findall('.//kml:ExtendedData/kml:Data', ns)
        
        for data_elem in data_elements:
            name = data_elem.get('name')
            value = data_elem.find('kml:value', ns)
            if name and value is not None:
                extended_data[name.lower()] = value.text
                
        return extended_data

    def _extract_metadata(self) -> Dict:
        """Extract KML-specific metadata"""
        return {
            'format': 'KML',
            'data_points': len(self.df) if self.df is not None else 0,
            'has_elevation': 'elevation' in (self.df.columns if self.df is not None else []),
            'has_timestamps': 'timestamp' in (self.df.columns if self.df is not None else []),
            'has_extended_data': any(col for col in (self.df.columns if self.df is not None else [])
                                   if col not in ['latitude', 'longitude', 'elevation', 'timestamp'])
        }

