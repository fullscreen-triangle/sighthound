import folium
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for visualization"""
    map_style: str = 'cartodbpositron'
    line_color: str = 'red'
    line_width: int = 2
    point_size: int = 5
    opacity: float = 0.8
    zoom_start: int = 13


class ActivityVisualizer:
    """
    Visualization tools for GPS trajectories and metrics
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()

    def create_map(
            self,
            trajectory: pd.DataFrame,
            additional_layers: Optional[List[Dict[str, Any]]] = None
    ) -> folium.Map:
        """
        Create interactive map visualization

        Args:
            trajectory: DataFrame with lat/lon coordinates
            additional_layers: List of dicts with additional visualization data

        Returns:
            Folium map object
        """
        # Create base map
        center_lat = trajectory['latitude'].mean()
        center_lon = trajectory['longitude'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.config.zoom_start,
            tiles=self.config.map_style
        )

        # Add main trajectory
        points = list(zip(trajectory['latitude'], trajectory['longitude']))
        folium.PolyLine(
            points,
            weight=self.config.line_width,
            color=self.config.line_color,
            opacity=self.config.opacity
        ).add_to(m)

        # Add probability heatmap if available
        if 'probability' in trajectory.columns:
            self._add_probability_heatmap(m, trajectory)

        # Add additional layers
        if additional_layers:
            for layer in additional_layers:
                self._add_layer(m, layer)

        return m

    def _add_probability_heatmap(
            self,
            map_obj: folium.Map,
            data: pd.DataFrame
    ):
        """Add probability heatmap to map"""
        points = list(zip(
            data['latitude'],
            data['longitude'],
            data['probability']
        ))

        folium.HeatMap(
            points,
            min_opacity=0.2,
            radius=15,
            blur=10,
            max_zoom=1,
        ).add_to(map_obj)

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

    def create_metric_plots(
            self,
            data: pd.DataFrame,
            metrics: List[str]
    ) -> Dict[str, go.Figure]:
        """
        Create interactive plots for various metrics

        Args:
            data: DataFrame with metrics
            metrics: List of metric names to plot

        Returns:
            Dictionary of metric names to plotly figures
        """
        plots = {}

        for metric in metrics:
            if metric in data.columns:
                fig = px.line(
                    data,
                    x='timestamp' if 'timestamp' in data.columns else data.index,
                    y=metric,
                    title=f'{metric.capitalize()} vs Time'
                )

                # Add probability bands if available
                if 'probability' in data.columns:
                    self._add_probability_bands(fig, data, metric)

                plots[metric] = fig

        return plots

    def _add_probability_bands(
            self,
            fig: go.Figure,
            data: pd.DataFrame,
            metric: str
    ):
        """Add probability confidence bands to plot"""
        prob_threshold = 0.68  # 1 sigma

        high_prob_mask = data['probability'] >= prob_threshold

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[metric].where(high_prob_mask),
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name=f'High Probability {metric}'
            )
        )

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
