#!/usr/bin/env python3
import os
import json
import csv
import datetime
from typing import Dict, List, Any, Union, Optional


class BaseExporter:
    """Base class for all exporters with common functionality."""
    
    def __init__(self, output_dir: str, filename: str = None):
        """Initialize the exporter with output directory and filename."""
        self.output_dir = output_dir
        self.filename = filename
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self, data: Any, **kwargs) -> str:
        """
        Export data to the specified format.
        
        Args:
            data: The data to export
            **kwargs: Additional arguments for specific export formats
            
        Returns:
            str: Path to the exported file
        """
        raise NotImplementedError("Subclasses must implement the export method")
    
    def _get_output_path(self, extension: str) -> str:
        """
        Get the full output path with the appropriate extension.
        
        Args:
            extension: File extension to use
            
        Returns:
            str: Full path to the output file
        """
        if self.filename:
            basename = os.path.splitext(self.filename)[0]
        else:
            basename = f"sighthound_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return os.path.join(self.output_dir, f"{basename}.{extension}")


class CSVExporter(BaseExporter):
    """Exporter for CSV format."""
    
    def export(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: List of dictionaries with consistent keys
            **kwargs: Additional arguments for CSV export
                - columns: List of column names to include (default: all keys in first dict)
                - dialect: CSV dialect to use (default: 'excel')
                
        Returns:
            str: Path to the exported CSV file
        """
        output_path = self._get_output_path('csv')
        
        # Determine columns to include
        columns = kwargs.get('columns', None)
        if not columns and data:
            columns = list(data[0].keys())
        
        # Get CSV dialect
        dialect = kwargs.get('dialect', 'excel')
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, dialect=dialect)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        return output_path


class GeoJSONExporter(BaseExporter):
    """Exporter for GeoJSON format."""
    
    def export(self, data: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs) -> str:
        """
        Export data to GeoJSON format.
        
        Args:
            data: Either a list of point dictionaries or a pre-formatted GeoJSON object
            **kwargs: Additional arguments for GeoJSON export
                - feature_type: Default geometry type ('Point', 'LineString', etc.)
                - include_properties: Whether to include all dict keys as properties
                
        Returns:
            str: Path to the exported GeoJSON file
        """
        output_path = self._get_output_path('geojson')
        
        # If data is already in GeoJSON format
        if isinstance(data, dict) and 'type' in data and data['type'] in ('FeatureCollection', 'Feature'):
            geojson_data = data
        else:
            # Convert data to GeoJSON
            feature_type = kwargs.get('feature_type', 'Point')
            include_properties = kwargs.get('include_properties', True)
            
            features = []
            for item in data:
                # Basic validation
                if 'latitude' not in item or 'longitude' not in item:
                    continue
                
                # Create feature
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': feature_type,
                        'coordinates': [item['longitude'], item['latitude']]
                    }
                }
                
                # Add properties if requested
                if include_properties:
                    properties = {k: v for k, v in item.items() 
                                 if k not in ('latitude', 'longitude')}
                    if properties:
                        feature['properties'] = properties
                
                features.append(feature)
            
            # Create the GeoJSON object
            geojson_data = {
                'type': 'FeatureCollection',
                'features': features
            }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        return output_path


class CZMLExporter(BaseExporter):
    """Exporter for CZML format (Cesium)."""
    
    def export(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """
        Export data to CZML format for Cesium visualization.
        
        Args:
            data: List of dictionaries with position and timestamp information
            **kwargs: Additional arguments for CZML export
                - name: Name of the track (default: "Sighthound Track")
                - description: Description of the track
                - color: Color in RGBA format (default: red)
                
        Returns:
            str: Path to the exported CZML file
        """
        output_path = self._get_output_path('czml')
        
        # Get parameters
        name = kwargs.get('name', 'Sighthound Track')
        description = kwargs.get('description', 'Exported from Sighthound')
        color = kwargs.get('color', [255, 0, 0, 255])
        
        # Create CZML document
        czml_data = [
            # Document packet
            {
                "id": "document",
                "name": name,
                "version": "1.0"
            },
            # Data packet
            {
                "id": "track",
                "name": name,
                "description": description,
                "availability": None,  # Will be set below
                "path": {
                    "material": {
                        "solidColor": {
                            "color": {
                                "rgba": color
                            }
                        }
                    },
                    "width": 3,
                    "leadTime": 0,
                    "trailTime": 0,
                    "resolution": 5
                },
                "point": {
                    "color": {
                        "rgba": color
                    },
                    "outlineColor": {
                        "rgba": [0, 0, 0, 255]
                    },
                    "outlineWidth": 2,
                    "pixelSize": 8
                },
                "position": {
                    "epoch": None,  # Will be set below
                    "cartographicDegrees": []
                }
            }
        ]
        
        # Process position data
        if data:
            # Extract timestamp information if available
            start_time = None
            end_time = None
            epoch = None
            
            position_data = []
            for point in data:
                if 'latitude' not in point or 'longitude' not in point:
                    continue
                
                # Get altitude or default to 0
                altitude = point.get('altitude', 0)
                
                # Process time information if available
                time_offset = 0
                if 'timestamp' in point:
                    timestamp = point['timestamp']
                    if isinstance(timestamp, (int, float)):
                        # Assume Unix timestamp
                        dt = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
                    elif isinstance(timestamp, str):
                        # Try to parse ISO format
                        try:
                            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except ValueError:
                            dt = None
                    else:
                        dt = None
                    
                    if dt:
                        if start_time is None or dt < start_time:
                            start_time = dt
                        if end_time is None or dt > end_time:
                            end_time = dt
                        
                        if epoch is None:
                            epoch = dt
                        
                        # Calculate time offset in seconds
                        time_offset = (dt - epoch).total_seconds()
                
                # Add position entry
                position_data.extend([time_offset, point['longitude'], point['latitude'], altitude])
            
            # Set availability timespan
            if start_time and end_time:
                start_str = start_time.isoformat().replace('+00:00', 'Z')
                end_str = end_time.isoformat().replace('+00:00', 'Z')
                czml_data[1]["availability"] = f"{start_str}/{end_str}"
            
            # Set epoch and position data
            if epoch:
                czml_data[1]["position"]["epoch"] = epoch.isoformat().replace('+00:00', 'Z')
            czml_data[1]["position"]["cartographicDegrees"] = position_data
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(czml_data, f, indent=2)
        
        return output_path


class ShapefileExporter(BaseExporter):
    """Exporter for ESRI Shapefile format."""
    
    def export(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """
        Export data to ESRI Shapefile format.
        
        Args:
            data: List of dictionaries with position information
            **kwargs: Additional arguments for Shapefile export
                - geometry_type: Type of geometry ('POINT', 'LINESTRING', etc.)
                
        Returns:
            str: Path to the exported Shapefile (without extension)
        """
        try:
            import shapefile
        except ImportError:
            raise ImportError("PyShp (shapefile) package is required for Shapefile export. "
                             "Install it with: pip install pyshp")
        
        output_base = self._get_output_path('shp').replace('.shp', '')
        
        # Get parameters
        geometry_type = kwargs.get('geometry_type', 'POINT')
        
        # Create shapefile writer
        if geometry_type == 'POINT':
            writer = shapefile.Writer(output_base, shapeType=shapefile.POINT)
        elif geometry_type == 'LINESTRING':
            writer = shapefile.Writer(output_base, shapeType=shapefile.POLYLINE)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
        
        # Add fields from first record
        if data:
            for key, value in data[0].items():
                if key not in ('latitude', 'longitude', 'coordinates'):
                    if isinstance(value, (int, bool)):
                        writer.field(key, 'N', 10, 0)
                    elif isinstance(value, float):
                        writer.field(key, 'N', 10, 5)
                    else:
                        writer.field(key, 'C', 50)
        
        # Add records
        if geometry_type == 'POINT':
            for point in data:
                if 'latitude' not in point or 'longitude' not in point:
                    continue
                
                writer.point(point['longitude'], point['latitude'])
                
                # Add attributes (all fields except coordinates)
                record = {k: v for k, v in point.items() 
                         if k not in ('latitude', 'longitude', 'coordinates')}
                writer.record(**record)
        
        elif geometry_type == 'LINESTRING':
            # Create a single line from all points
            line = []
            for point in data:
                if 'latitude' not in point or 'longitude' not in point:
                    continue
                line.append([point['longitude'], point['latitude']])
            
            if line:
                writer.line([line])
                
                # Add attributes from the first point as line attributes
                record = {k: v for k, v in data[0].items() 
                         if k not in ('latitude', 'longitude', 'coordinates')}
                writer.record(**record)
        
        # Close the shapefile
        writer.close()
        
        return output_base


class KMLExporter(BaseExporter):
    """Exporter for KML format."""
    
    def export(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """
        Export data to KML format.
        
        Args:
            data: List of dictionaries with position information
            **kwargs: Additional arguments for KML export
                - name: Name of the track (default: "Sighthound Track")
                - description: Description of the track
                - color: Color in ABGR format (default: red)
                - line_width: Width of the line (default: 3)
                
        Returns:
            str: Path to the exported KML file
        """
        output_path = self._get_output_path('kml')
        
        # Get parameters
        name = kwargs.get('name', 'Sighthound Track')
        description = kwargs.get('description', 'Exported from Sighthound')
        color = kwargs.get('color', 'ff0000ff')  # ABGR format
        line_width = kwargs.get('line_width', 3)
        
        # Create KML content
        kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <description>{description}</description>
    <Style id="lineStyle">
      <LineStyle>
        <color>{color}</color>
        <width>{line_width}</width>
      </LineStyle>
      <PointStyle>
        <color>{color}</color>
        <scale>1.0</scale>
      </PointStyle>
    </Style>
    <Placemark>
      <name>{name}</name>
      <styleUrl>#lineStyle</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>
'''
        
        # Add coordinates
        for point in data:
            if 'latitude' not in point or 'longitude' not in point:
                continue
            
            # Get altitude or default to 0
            altitude = point.get('altitude', 0)
            
            kml_content += f"          {point['longitude']},{point['latitude']},{altitude}\n"
        
        # Close KML document
        kml_content += '''        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(kml_content)
        
        return output_path


class GPXExporter(BaseExporter):
    """Exporter for GPX format."""
    
    def export(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """
        Export data to GPX format.
        
        Args:
            data: List of dictionaries with position information
            **kwargs: Additional arguments for GPX export
                - name: Name of the track (default: "Sighthound Track")
                - description: Description of the track
                - type: Type of the track (default: "activity")
                
        Returns:
            str: Path to the exported GPX file
        """
        output_path = self._get_output_path('gpx')
        
        # Get parameters
        name = kwargs.get('name', 'Sighthound Track')
        description = kwargs.get('description', 'Exported from Sighthound')
        track_type = kwargs.get('type', 'activity')
        
        # Create GPX content
        current_date = datetime.datetime.now().isoformat()
        
        gpx_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Sighthound Exporter"
  xmlns="http://www.topografix.com/GPX/1/1"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
  <metadata>
    <name>{name}</name>
    <desc>{description}</desc>
    <time>{current_date}</time>
  </metadata>
  <trk>
    <name>{name}</name>
    <type>{track_type}</type>
    <trkseg>
'''
        
        # Add track points
        for point in data:
            if 'latitude' not in point or 'longitude' not in point:
                continue
            
            # Get altitude or default to None
            ele_str = ''
            if 'altitude' in point and point['altitude'] is not None:
                ele_str = f'      <ele>{point["altitude"]}</ele>\n'
            
            # Get time if available
            time_str = ''
            if 'timestamp' in point and point['timestamp']:
                # Try to format timestamp properly
                try:
                    if isinstance(point['timestamp'], (int, float)):
                        time_str = datetime.datetime.fromtimestamp(
                            point['timestamp'], datetime.timezone.utc
                        ).isoformat()
                    else:
                        time_str = point['timestamp']
                    
                    time_str = f'      <time>{time_str}</time>\n'
                except:
                    pass
            
            gpx_content += f'''    <trkpt lat="{point['latitude']}" lon="{point['longitude']}">
{ele_str}{time_str}    </trkpt>
'''
        
        # Close GPX document
        gpx_content += '''    </trkseg>
  </trk>
</gpx>'''
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(gpx_content)
        
        return output_path


def get_exporter(format_name: str, output_dir: str, filename: Optional[str] = None) -> BaseExporter:
    """
    Factory function to get the appropriate exporter based on format name.
    
    Args:
        format_name: Format name (csv, geojson, czml, etc.)
        output_dir: Output directory
        filename: Optional filename (without extension)
        
    Returns:
        BaseExporter: An exporter instance for the requested format
        
    Raises:
        ValueError: If the format is not supported
    """
    format_name = format_name.lower()
    
    if format_name == 'csv':
        return CSVExporter(output_dir, filename)
    elif format_name == 'geojson':
        return GeoJSONExporter(output_dir, filename)
    elif format_name == 'czml':
        return CZMLExporter(output_dir, filename)
    elif format_name == 'shp' or format_name == 'shapefile':
        return ShapefileExporter(output_dir, filename)
    elif format_name == 'kml':
        return KMLExporter(output_dir, filename)
    elif format_name == 'gpx':
        return GPXExporter(output_dir, filename)
    else:
        raise ValueError(f"Unsupported export format: {format_name}") 