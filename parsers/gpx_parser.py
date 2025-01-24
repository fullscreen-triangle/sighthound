import gpxpy
import pandas as pd
from datetime import datetime
from typing import Dict
from .base_parser import BaseParser


class GPXParser(BaseParser):
    def _read_file(self):
        try:
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
                            'timestamp': point.time,
                            'speed': point.speed if hasattr(point, 'speed') else None,
                            'heading': point.course if hasattr(point, 'course') else None
                        })

            self.df = pd.DataFrame(data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse GPX file: {str(e)}")

    def _extract_metadata(self) -> Dict:
        """Extract GPX-specific metadata"""
        metadata = {
            'format': 'GPX',
            'data_points': len(self.df) if self.df is not None else 0,
            'tracks': 0,
            'segments': 0,
            'has_elevation': 'elevation' in (self.df.columns if self.df is not None else []),
            'has_timestamps': 'timestamp' in (self.df.columns if self.df is not None else [])
        }
        
        try:
            with open(self.file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
                metadata.update({
                    'tracks': len(gpx.tracks),
                    'segments': sum(len(track.segments) for track in gpx.tracks),
                    'creator': gpx.creator if gpx.creator else 'Unknown'
                })
        except:
            pass
            
        return metadata
