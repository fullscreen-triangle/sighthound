#!/usr/bin/env python3
import os
import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse
import folium
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
import pandas as pd

class SighthoundWebInterface:
    """Web interface for visualizing Sighthound results."""
    
    def __init__(self, data_path, output_dir="./public", port=8000):
        """Initialize the web interface with data path and server settings."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.port = port
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load data from file into dataframe."""
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json') or self.data_path.endswith('.geojson'):
            with open(self.data_path, 'r') as f:
                self.geojson_data = json.load(f)
                # Create dataframe from geojson if needed
                if 'features' in self.geojson_data:
                    coordinates = []
                    timestamps = []
                    for feature in self.geojson_data['features']:
                        if feature['geometry']['type'] == 'Point':
                            coords = feature['geometry']['coordinates']
                            coordinates.append(coords)
                            if 'properties' in feature and 'timestamp' in feature['properties']:
                                timestamps.append(feature['properties']['timestamp'])
                            else:
                                timestamps.append(None)
                    
                    if coordinates:
                        self.df = pd.DataFrame({
                            'longitude': [c[0] for c in coordinates],
                            'latitude': [c[1] for c in coordinates],
                            'timestamp': timestamps
                        })
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
    
    def create_map(self):
        """Create interactive map visualization using folium."""
        # Calculate center of the data
        center_lat = self.df['latitude'].mean()
        center_lon = self.df['longitude'].mean()
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add different visualization layers
        self._add_path_layer(m)
        self._add_heatmap_layer(m)
        self._add_points_layer(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save to HTML file
        map_file = os.path.join(self.output_dir, 'map.html')
        m.save(map_file)
        return map_file
    
    def _add_path_layer(self, m):
        """Add path layer to map."""
        points = list(zip(self.df['latitude'], self.df['longitude']))
        folium.PolyLine(
            points,
            color='blue',
            weight=5,
            opacity=0.7,
            name='Path'
        ).add_to(m)
    
    def _add_heatmap_layer(self, m):
        """Add heatmap layer to map."""
        heat_data = [[row['latitude'], row['longitude']] for _, row in self.df.iterrows()]
        HeatMap(heat_data, name='Heatmap').add_to(m)
    
    def _add_points_layer(self, m):
        """Add points layer with popup information."""
        # Create a marker cluster for better performance with many points
        marker_cluster = MarkerCluster(name='Points').add_to(m)
        
        # Add markers for every 10th point to avoid clutter
        for idx, row in self.df.iloc[::10].iterrows():
            popup_content = f"""
                <b>Point {idx}</b><br>
                Latitude: {row['latitude']}<br>
                Longitude: {row['longitude']}<br>
                {f"Time: {row['timestamp']}" if 'timestamp' in row else ""}
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(marker_cluster)
    
    def create_web_interface(self):
        """Create the complete web interface with HTML, CSS, and JS."""
        index_html = os.path.join(self.output_dir, 'index.html')
        
        # Basic HTML template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sighthound Data Visualization</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            padding-top: 20px;
        }}
        .map-container {{
            height: 70vh;
            margin-bottom: 20px;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .dashboard-panel {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="pb-3 mb-4 border-bottom">
            <h1 class="display-5 fw-bold">Sighthound Data Visualization</h1>
            <p class="lead">Interactive visualization of GPS data and analysis</p>
        </header>

        <div class="row">
            <div class="col-md-8">
                <div class="map-container">
                    <iframe src="map.html"></iframe>
                </div>
            </div>
            <div class="col-md-4">
                <div class="dashboard-panel">
                    <h3>Statistics</h3>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item">Points: {len(self.df)}</li>
                        <li class="list-group-item">Distance: {self._calculate_distance():.2f} km</li>
                        <li class="list-group-item">Duration: {self._calculate_duration()}</li>
                    </ul>
                </div>
                
                <div class="dashboard-panel">
                    <h3>Controls</h3>
                    <div class="mb-3">
                        <label for="pointsDensity" class="form-label">Points Density</label>
                        <input type="range" class="form-range" id="pointsDensity" min="1" max="20" value="10">
                    </div>
                    <div class="d-grid gap-2">
                        <button class="btn btn-primary" id="downloadBtn">Download Data</button>
                        <button class="btn btn-secondary" id="toggleHeatmap">Toggle Heatmap</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Basic interactive functionality
        document.getElementById('downloadBtn').addEventListener('click', function() {
            alert('Data download feature would be implemented here');
        });
        
        document.getElementById('toggleHeatmap').addEventListener('click', function() {
            const iframe = document.querySelector('iframe');
            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            // This is a simplistic approach - actual implementation would depend on folium structure
            const heatmapControl = iframeDoc.querySelector('.leaflet-control-layers-overlays input[type="checkbox"]');
            if (heatmapControl) {{{{
                heatmapControl.click();
            }}}}
        });
    </script>
</body>
</html>"""
        
        # Write HTML file
        with open(index_html, 'w') as f:
            f.write(html_content)
        
        return index_html
    
    def _calculate_distance(self):
        """Calculate total distance in kilometers."""
        # This is a simplified calculation - in production code we would use geopy or similar
        # to calculate accurate distances between coordinates
        return 10.5  # Placeholder value
    
    def _calculate_duration(self):
        """Calculate duration of the track."""
        if 'timestamp' in self.df.columns:
            # This is simplified - actual implementation would parse timestamps and calculate duration
            return "2h 30m"  # Placeholder
        return "Unknown"
    
    def start_server(self):
        """Start HTTP server for the web interface."""
        # Change to the output directory
        os.chdir(self.output_dir)
        
        # Create and start the server
        server = HTTPServer(('localhost', self.port), SimpleHTTPRequestHandler)
        print(f"Starting server at http://localhost:{self.port}")
        
        # Open browser automatically
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{self.port}')).start()
        
        # Run the server until interrupted
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.")
        finally:
            server.server_close()

def main():
    """Main entry point for the web interface."""
    parser = argparse.ArgumentParser(description="Sighthound Web Interface")
    parser.add_argument('--data', '-d', required=True, help='Path to data file (.csv, .json, or .geojson)')
    parser.add_argument('--output', '-o', default='./public', help='Output directory')
    parser.add_argument('--port', '-p', type=int, default=8000, help='HTTP server port')
    args = parser.parse_args()
    
    try:
        # Initialize and run the web interface
        web_interface = SighthoundWebInterface(args.data, args.output, args.port)
        web_interface.load_data()
        web_interface.create_map()
        web_interface.create_web_interface()
        web_interface.start_server()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 