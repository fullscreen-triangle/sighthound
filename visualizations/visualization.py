import folium
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import branca.colormap as cm


@dataclass
class PlotConfig:
    """Configuration for visualization"""
    map_style: str = 'cartodbpositron'
    line_color: str = 'red'
    line_width: int = 2
    point_size: int = 5
    opacity: float = 0.8
    zoom_start: int = 13
    confidence_colors: Dict[str, str] = field(default_factory=lambda: {
        'high': '#00ff00',
        'medium': '#ffff00',
        'low': '#ff0000'
    })
    timeline_height: int = 150
    show_confidence: bool = True
    show_metrics: bool = True
    show_weather: bool = True
    show_satellites: bool = True


class ActivityVisualizer:
    """Enhanced visualization tools for GPS trajectories and metrics"""

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._setup_colormaps()

    def _setup_colormaps(self):
        """Initialize colormaps for different metrics"""
        self.confidence_colormap = cm.LinearColormap(
            colors=['red', 'yellow', 'green'],
            vmin=0, vmax=1,
            caption='Confidence Score'
        )
        
        self.speed_colormap = cm.LinearColormap(
            colors=['blue', 'yellow', 'red'],
            vmin=0, vmax=30,  # adjust based on activity type
            caption='Speed (km/h)'
        )

    def create_map(
            self,
            trajectory: pd.DataFrame,
            additional_layers: Optional[List[Dict[str, Any]]] = None,
            confidence_scores: Optional[Dict[str, float]] = None
    ) -> folium.Map:
        """Create enhanced interactive map visualization"""
        # Create base map
        center_lat = trajectory['latitude'].mean()
        center_lon = trajectory['longitude'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.config.zoom_start,
            tiles=self.config.map_style
        )

        # Add confidence-colored trajectory
        if self.config.show_confidence and 'confidence' in trajectory.columns:
            self._add_confidence_trajectory(m, trajectory)
        else:
            self._add_simple_trajectory(m, trajectory)

        # Add metric visualizations
        if self.config.show_metrics:
            self._add_metric_layers(m, trajectory)

        # Add weather overlay if available
        if self.config.show_weather and 'weather_condition' in trajectory.columns:
            self._add_weather_layer(m, trajectory)

        # Add satellite positions if available
        if self.config.show_satellites and 'satellite_count' in trajectory.columns:
            self._add_satellite_layer(m, trajectory)

        # Add additional layers
        if additional_layers:
            for layer in additional_layers:
                self._add_layer(m, layer)

        # Add time slider
        if 'timestamp' in trajectory.columns:
            self._add_time_slider(m, trajectory)

        return m

    def _add_confidence_trajectory(self, map_obj: folium.Map, data: pd.DataFrame):
        """Add trajectory colored by confidence scores"""
        for i in range(len(data) - 1):
            confidence = data.iloc[i]['confidence']
            color = self.confidence_colormap(confidence)
            
            points = [
                [data.iloc[i]['latitude'], data.iloc[i]['longitude']],
                [data.iloc[i + 1]['latitude'], data.iloc[i + 1]['longitude']]
            ]
            
            folium.PolyLine(
                points,
                weight=self.config.line_width,
                color=color,
                opacity=self.config.opacity
            ).add_to(map_obj)

    def _add_metric_layers(self, map_obj: folium.Map, data: pd.DataFrame):
        """Add visualization layers for various metrics"""
        metric_layers = folium.FeatureGroup(name='Metrics')
        
        if 'speed' in data.columns:
            speed_points = []
            for _, row in data.iterrows():
                speed_points.append([
                    row['latitude'],
                    row['longitude'],
                    row['speed']
                ])
            
            folium.HeatMap(
                speed_points,
                name='Speed Heatmap',
                min_opacity=0.2,
                radius=15,
                blur=10,
                max_zoom=1,
            ).add_to(metric_layers)

        metric_layers.add_to(map_obj)

    def _add_weather_layer(self, map_obj: folium.Map, data: pd.DataFrame):
        """Add weather condition overlay"""
        weather_layers = folium.FeatureGroup(name='Weather')
        
        # Add weather icons at regular intervals
        step = len(data) // 10  # Show ~10 weather points
        for i in range(0, len(data), step):
            row = data.iloc[i]
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"Weather: {row['weather_condition']}",
                icon=folium.Icon(icon='cloud')
            ).add_to(weather_layers)

        weather_layers.add_to(map_obj)

    def _add_satellite_layer(self, map_obj: folium.Map, data: pd.DataFrame):
        """Add satellite coverage visualization"""
        satellite_layers = folium.FeatureGroup(name='Satellites')
        
        for _, row in data.iterrows():
            folium.Circle(
                [row['latitude'], row['longitude']],
                radius=row['satellite_count'] * 5,
                color='blue',
                fill=True,
                popup=f"Satellites: {row['satellite_count']}"
            ).add_to(satellite_layers)

        satellite_layers.add_to(map_obj)

    def _add_time_slider(self, map_obj: folium.Map, data: pd.DataFrame):
        """Add interactive time slider"""
        times = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        slider_html = self._create_time_slider_html(times)
        map_obj.get_root().html.add_child(folium.Element(slider_html))

    def create_metric_plots(
            self,
            data: pd.DataFrame,
            metrics: List[str]
    ) -> Dict[str, go.Figure]:
        """Create enhanced interactive metric plots"""
        plots = {}

        for metric in metrics:
            if metric in data.columns:
                fig = self._create_single_metric_plot(data, metric)
                plots[metric] = fig

        return plots

    def _create_single_metric_plot(
            self,
            data: pd.DataFrame,
            metric: str
    ) -> go.Figure:
        """Create an enhanced plot for a single metric"""
        fig = go.Figure()

        # Add main metric line
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data[metric],
            name=metric.capitalize(),
            line=dict(color=self.config.line_color)
        ))

        # Add confidence bands if available
        if 'confidence' in data.columns:
            self._add_confidence_bands(fig, data, metric)

        # Add weather annotations if available
        if 'weather_condition' in data.columns:
            self._add_weather_annotations(fig, data)

        # Customize layout
        fig.update_layout(
            title=f'{metric.capitalize()} vs Time',
            xaxis_title='Time',
            yaxis_title=metric.capitalize(),
            hovermode='x unified'
        )

        return fig

    def _add_confidence_bands(
            self,
            fig: go.Figure,
            data: pd.DataFrame,
            metric: str
    ):
        """Add confidence bands to metric plot"""
        std_dev = data[metric].std()
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data[metric] + std_dev * data['confidence'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data[metric] - std_dev * data['confidence'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))

    def _add_weather_annotations(
            self,
            fig: go.Figure,
            data: pd.DataFrame
    ):
        """Add weather condition annotations to plot"""
        step = len(data) // 10  # Show ~10 weather annotations
        
        for i in range(0, len(data), step):
            row = data.iloc[i]
            fig.add_annotation(
                x=row['timestamp'],
                y=fig.data[0].y.max(),
                text=row['weather_condition'],
                showarrow=True,
                arrowhead=1
            )

    @staticmethod
    def _create_time_slider_html(times: List[str]) -> str:
        """Create HTML/JavaScript for time slider"""
        # Implementation of time slider HTML/JS code
        # This would return the HTML string for the time slider
        pass

    def _add_layer(
            self,
            map_obj: folium.Map,
            layer_data: Dict[str, Any]
    ):
        """Add custom layer to map"""
        layer_type = layer_data.get('type', 'polyline')

        if layer_type == 'polyline':
            folium.PolyLine(
                layer_data['points'],
                weight=layer_data.get('width', self.config.line_width),
                color=layer_data.get('color', 'blue'),
                opacity=layer_data.get('opacity', self.config.opacity)
            ).add_to(map_obj)

        elif layer_type == 'marker':
            folium.Marker(
                location=layer_data['location'],
                popup=layer_data.get('popup', ''),
                icon=folium.Icon(color=layer_data.get('color', 'red'))
            ).add_to(map_obj)

    def create_dashboard(
            self,
            data: pd.DataFrame,
            metrics: List[str]
    ) -> go.Figure:
        """
        Create comprehensive dashboard with map and metrics

        Args:
            data: DataFrame with trajectory and metrics
            metrics: List of metrics to include

        Returns:
            Plotly figure with dashboard
        """
        # Create subplot grid
        fig = go.Figure()

        # Add map
        fig.add_trace(
            go.Scattermapbox(
                lat=data['latitude'],
                lon=data['longitude'],
                mode='lines',
                line=dict(
                    width=self.config.line_width,
                    color=self.config.line_color
                ),
                name='Trajectory'
            )
        )

        # Add metric subplots
        for i, metric in enumerate(metrics):
            if metric in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[metric],
                        name=metric,
                        yaxis=f'y{i + 2}'
                    )
                )

        # Update layout
        fig.update_layout(
            mapbox_style=self.config.map_style,
            mapbox=dict(
                center=dict(
                    lat=data['latitude'].mean(),
                    lon=data['longitude'].mean()
                ),
                zoom=self.config.zoom_start
            ),
            height=800,
            grid=dict(
                rows=len(metrics) + 1,
                columns=1,
                pattern='independent'
            )
        )

        return fig
