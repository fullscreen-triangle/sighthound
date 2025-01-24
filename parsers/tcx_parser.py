import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
from typing import Dict
from .base_parser import BaseParser


class TCXParser(BaseParser):
    def _read_file(self):
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()

            ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
            data = []
            
            trackpoints = root.findall('.//ns:Trackpoint', ns)
            for point in trackpoints:
                point_data = {}
                
                # Extract required fields
                time_elem = point.find('ns:Time', ns)
                lat_elem = point.find('.//ns:LatitudeDegrees', ns)
                lon_elem = point.find('.//ns:LongitudeDegrees', ns)
                
                if time_elem is not None and lat_elem is not None and lon_elem is not None:
                    point_data.update({
                        'timestamp': datetime.strptime(time_elem.text, '%Y-%m-%dT%H:%M:%S.%fZ'),
                        'latitude': float(lat_elem.text),
                        'longitude': float(lon_elem.text)
                    })
                    
                    # Extract optional fields
                    ele_elem = point.find('.//ns:AltitudeMeters', ns)
                    if ele_elem is not None:
                        point_data['elevation'] = float(ele_elem.text)
                        
                    speed_elem = point.find('.//ns:Speed', ns)
                    if speed_elem is not None:
                        point_data['speed'] = float(speed_elem.text)
                        
                    heading_elem = point.find('.//ns:Heading', ns)
                    if heading_elem is not None:
                        point_data['heading'] = float(heading_elem.text)
                        
                    data.append(point_data)

            self.df = pd.DataFrame(data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse TCX file: {str(e)}")

    def _extract_metadata(self) -> Dict:
        """Extract TCX-specific metadata"""
        metadata = {
            'format': 'TCX',
            'data_points': len(self.df) if self.df is not None else 0,
            'has_elevation': 'elevation' in (self.df.columns if self.df is not None else []),
            'has_speed': 'speed' in (self.df.columns if self.df is not None else []),
            'has_heading': 'heading' in (self.df.columns if self.df is not None else [])
        }
        
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
            
            activity_elem = root.find('.//ns:Activity', ns)
            if activity_elem is not None:
                metadata['activity_type'] = activity_elem.get('Sport', 'Unknown')
                
            creator_elem = root.find('.//ns:Creator', ns)
            if creator_elem is not None:
                metadata['device'] = creator_elem.text
                
        except:
            pass
            
        return metadata
