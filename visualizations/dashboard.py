#!/usr/bin/env python3
import os
import time
import threading
import argparse
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import psutil
import datetime
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sighthound_dashboard")

# Global storage for system metrics history
system_metrics_history = {
    'timestamp': [],
    'cpu_percent': [],
    'memory_percent': [],
    'disk_usage_percent': []
}

# Global storage for job status
jobs_status = {}
processing_history = []

# Dashboard application
app = dash.Dash(
    __name__,
    title="Sighthound Dashboard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
    ]
)

# Dashboard layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Sighthound Dashboard", className="display-4"),
        html.P("Monitor processing status and system performance", className="lead")
    ], className="jumbotron py-4 mb-4"),
    
    # Main content
    html.Div([
        # System metrics section
        html.Div([
            html.H2("System Metrics", className="mb-3"),
            dcc.Graph(id='system-metrics-graph'),
            dcc.Interval(
                id='system-metrics-update',
                interval=5000,  # update every 5 seconds
                n_intervals=0
            )
        ], className="mb-5"),
        
        # Processing status section
        html.Div([
            html.H2("Processing Status", className="mb-3"),
            html.Div(id='active-jobs-container', className="mb-4"),
            dcc.Interval(
                id='jobs-update',
                interval=3000,  # update every 3 seconds
                n_intervals=0
            )
        ], className="mb-5"),
        
        # Processing history section
        html.Div([
            html.H2("Processing History", className="mb-3"),
            html.Div(id='processing-history-container'),
            dcc.Interval(
                id='history-update',
                interval=10000,  # update every 10 seconds
                n_intervals=0
            )
        ])
    ], className="container"),
    
    # Footer
    html.Footer([
        html.P("Sighthound Dashboard © 2023", className="text-center text-muted")
    ], className="mt-5 py-3")
])


@app.callback(
    Output('system-metrics-graph', 'figure'),
    Input('system-metrics-update', 'n_intervals')
)
def update_system_metrics(n):
    """Update system metrics graph."""
    # Get current system metrics
    current_time = datetime.datetime.now()
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_usage_percent = psutil.disk_usage('/').percent
    
    # Add to history
    system_metrics_history['timestamp'].append(current_time)
    system_metrics_history['cpu_percent'].append(cpu_percent)
    system_metrics_history['memory_percent'].append(memory_percent)
    system_metrics_history['disk_usage_percent'].append(disk_usage_percent)
    
    # Keep only last 60 samples (5 minutes at 5-second intervals)
    max_samples = 60
    if len(system_metrics_history['timestamp']) > max_samples:
        for key in system_metrics_history:
            system_metrics_history[key] = system_metrics_history[key][-max_samples:]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=system_metrics_history['timestamp'],
        y=system_metrics_history['cpu_percent'],
        name='CPU (%)',
        line=dict(color='#007bff', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=system_metrics_history['timestamp'],
        y=system_metrics_history['memory_percent'],
        name='Memory (%)',
        line=dict(color='#28a745', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=system_metrics_history['timestamp'],
        y=system_metrics_history['disk_usage_percent'],
        name='Disk (%)',
        line=dict(color='#dc3545', width=2)
    ))
    
    fig.update_layout(
        title='System Resource Usage',
        xaxis_title='Time',
        yaxis_title='Usage (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig


@app.callback(
    Output('active-jobs-container', 'children'),
    Input('jobs-update', 'n_intervals')
)
def update_active_jobs(n):
    """Update active jobs display."""
    # In a real implementation, this would query the API or database
    # For this demo, we'll simulate a few jobs
    
    # Simulate updating job status
    simulate_job_updates()
    
    # Create job cards
    job_cards = []
    
    if not jobs_status:
        return html.Div(html.P("No active jobs", className="text-muted"), className="text-center")
    
    for job_id, job in jobs_status.items():
        status_color = {
            'pending': 'secondary',
            'processing': 'primary',
            'completed': 'success',
            'failed': 'danger'
        }.get(job['status'], 'secondary')
        
        job_card = html.Div([
            html.Div([
                html.H5(f"Job {job_id}", className="card-title"),
                html.H6(job['status'].capitalize(), className=f"card-subtitle mb-2 text-{status_color}"),
                html.P(job.get('message', ''), className="card-text"),
                html.Div([
                    html.Div(
                        html.Div(
                            style={"width": f"{job.get('progress', 0)}%"},
                            className=f"progress-bar bg-{status_color}"
                        ),
                        className="progress"
                    ),
                    html.Small(f"{job.get('progress', 0)}%", className="text-muted")
                ]) if job['status'] == 'processing' else None,
                html.P([
                    html.Small(f"Created: {datetime.datetime.fromtimestamp(job['created_at']).strftime('%Y-%m-%d %H:%M:%S')}"),
                    html.Br(),
                    html.Small(f"Updated: {datetime.datetime.fromtimestamp(job.get('updated_at', job['created_at'])).strftime('%Y-%m-%d %H:%M:%S')}")
                ], className="card-text mt-2 text-muted")
            ], className="card-body")
        ], className="card mb-3")
        
        job_cards.append(job_card)
    
    return html.Div(job_cards)


@app.callback(
    Output('processing-history-container', 'children'),
    Input('history-update', 'n_intervals')
)
def update_processing_history(n):
    """Update processing history display."""
    # Create processing history table
    if not processing_history:
        return html.Div(html.P("No processing history", className="text-muted"), className="text-center")
    
    # Create table header
    header = html.Thead(html.Tr([
        html.Th("Job ID"),
        html.Th("Status"),
        html.Th("Started"),
        html.Th("Completed"),
        html.Th("Duration"),
        html.Th("Input File")
    ]))
    
    # Create table rows
    rows = []
    for job in processing_history:
        status_class = {
            'completed': 'table-success',
            'failed': 'table-danger'
        }.get(job['status'], '')
        
        start_time = datetime.datetime.fromtimestamp(job['created_at'])
        end_time = datetime.datetime.fromtimestamp(job.get('completed_at', time.time()))
        duration = end_time - start_time
        
        row = html.Tr([
            html.Td(job['id']),
            html.Td(job['status'].capitalize()),
            html.Td(start_time.strftime('%Y-%m-%d %H:%M:%S')),
            html.Td(end_time.strftime('%Y-%m-%d %H:%M:%S') if 'completed_at' in job else '-'),
            html.Td(str(duration).split('.')[0]),  # Format as HH:MM:SS
            html.Td(os.path.basename(job.get('input_file', '-')))
        ], className=status_class)
        
        rows.append(row)
    
    table_body = html.Tbody(rows)
    
    return html.Div([
        html.Table([header, table_body], className="table table-striped table-hover")
    ], className="table-responsive")


def simulate_job_updates():
    """Simulate job updates for demonstration."""
    current_time = time.time()
    
    # Simulate creating a new job occasionally
    if len(jobs_status) < 3 and n_intervals % 10 == 0:
        job_id = f"job_{len(jobs_status) + 1}"
        jobs_status[job_id] = {
            'id': job_id,
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing processing...',
            'created_at': current_time,
            'updated_at': current_time,
            'input_file': f'sample_data_{job_id}.gpx'
        }
    
    # Update existing jobs
    for job_id, job in list(jobs_status.items()):
        if job['status'] == 'processing':
            # Update progress
            progress = job.get('progress', 0) + 5  # Increment by 5%
            
            if progress >= 100:
                # Job completed
                job['status'] = 'completed' if job_id != 'job_2' else 'failed'  # Make one job fail for demo
                job['progress'] = 100
                job['message'] = 'Processing completed' if job['status'] == 'completed' else 'Processing failed'
                job['completed_at'] = current_time
                
                # Add to history and remove from active jobs after some time
                processing_history.append(job.copy())
                if len(processing_history) > 10:
                    processing_history.pop(0)
                
                # Remove completed jobs after showing them for a while
                if current_time - job.get('completed_at', 0) > 30:  # 30 seconds
                    del jobs_status[job_id]
            else:
                # Update job progress
                job['progress'] = progress
                job['updated_at'] = current_time
                
                # Update message based on progress
                if progress < 30:
                    job['message'] = 'Loading data...'
                elif progress < 60:
                    job['message'] = 'Processing data...'
                else:
                    job['message'] = 'Generating output...'


def run_dashboard(host: str = '0.0.0.0', port: int = 8050, debug: bool = False) -> None:
    """Run the dashboard server."""
    app.run_server(host=host, port=port, debug=debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sighthound Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Initialize n_intervals for simulation
    n_intervals = 0
    
    # Define a function to increment n_intervals in a separate thread
    def increment_n_intervals():
        global n_intervals
        while True:
            n_intervals += 1
            time.sleep(1)
    
    # Start the thread
    threading.Thread(target=increment_n_intervals, daemon=True).start()
    
    logger.info(f"Starting Sighthound Dashboard on {args.host}:{args.port}")
    run_dashboard(args.host, args.port, args.debug) 