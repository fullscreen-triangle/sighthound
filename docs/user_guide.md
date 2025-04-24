# Sighthound User Guide

This guide provides instructions for using the Sighthound tool, which helps you process, analyze, and visualize GPS tracking data from various sources.

## Table of Contents

1. [Installation](#installation)
2. [Command Line Interface](#command-line-interface)
3. [Web Interface](#web-interface)
4. [API Integration](#api-integration)
5. [Configuration](#configuration)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sighthound.git
   cd sighthound
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Command Line Interface

Sighthound provides a command-line interface (CLI) for quick data processing.

### Basic Usage

```bash
python -m core.cli --input /path/to/data.gpx --output /path/to/output/dir --format geojson
```

### Available Options

- `--input`, `-i`: Path to input file or directory (required)
- `--output`, `-o`: Path to output directory (required)
- `--format`, `-f`: Output format (csv, geojson, czml, kml, gpx, shp)
- `--filter`: Apply Kalman filtering to smooth data
- `--triangulate`: Use cell tower data for triangulation
- `--optimize-path`: Calculate optimal path
- `--dubins-path`: Calculate Dubins path
- `--config`, `-c`: Path to configuration file
- `--verbose`, `-v`: Increase verbosity (can be used multiple times)

### Examples

Process a GPX file with Kalman filtering:
```bash
python -m core.cli -i tracking_data.gpx -o ./output --filter
```

Process multiple files with path optimization:
```bash
python -m core.cli -i ./data_directory -o ./output --optimize-path
```

## Web Interface

Sighthound includes a web interface for visualizing GPS data and analysis results.

### Starting the Web Interface

```bash
python -m visualizations.web_interface --data /path/to/data.geojson --port 8000
```

This will start a local web server and automatically open the visualization in your default browser.

### Dashboard

To monitor system status and processing jobs, use the dashboard:

```bash
python -m visualizations.dashboard --port 8050
```

The dashboard shows:
- System resource usage (CPU, memory, disk)
- Active processing jobs and their status
- Processing history

## API Integration

Sighthound provides a REST API for integration with other systems.

### Starting the API Server

```bash
python -m core.api --port 5000
```

### API Endpoints

- `GET /api/version`: Get API version information
- `POST /api/process`: Start a new processing job
- `GET /api/jobs/<job_id>`: Get job status
- `GET /api/jobs/<job_id>/result`: Get job results
- `GET /api/jobs/<job_id>/download`: Download result file

### Example: Starting a Processing Job

Using curl:
```bash
curl -X POST -F "input_file=@tracking_data.gpx" -H "Content-Type: multipart/form-data" http://localhost:5000/api/process
```

Using Python requests:
```python
import requests

url = "http://localhost:5000/api/process"
files = {"input_file": open("tracking_data.gpx", "rb")}
data = {"output_format": "geojson", "apply_filtering": True}

response = requests.post(url, files=files, json=data)
job_id = response.json().get("job_id")

# Check job status
status_url = f"http://localhost:5000/api/jobs/{job_id}"
status_response = requests.get(status_url)
print(status_response.json())
```

## Configuration

Sighthound can be configured using a YAML configuration file.

### Example Configuration

```yaml
# Input formats supported by the system
input_formats:
  - gpx
  - fit
  - tcx
  - kml

# Output formats supported by the system
output_formats:
  - csv
  - geojson
  - czml

# Parameters for Kalman filtering
filter_params:
  process_noise_scale: 0.01
  measurement_noise_scale: 0.1
  initial_state_covariance: 1.0
  state_transition_damping: 0.95
```

Specify the configuration file using the `--config` option:

```bash
python -m core.cli -i data.gpx -o ./output --config config/sighthound.yaml
```

## Common Use Cases

### Filtering Noisy GPS Data

When dealing with noisy GPS data (common with consumer-grade devices), use the Kalman filtering option:

```bash
python -m core.cli -i noisy_data.gpx -o ./output --filter
```

### Analyzing Sports Activities

For analyzing sports activities with detailed path information:

```bash
python -m core.cli -i activity.fit -o ./output --filter --optimize-path
```

### Combining Multiple Data Sources

To combine data from multiple sources:

```bash
python -m core.cli -i ./data_directory -o ./output --filter
```

### Visualizing Routes on a Map

To view routes on an interactive map:

```bash
python -m visualizations.web_interface --data output/processed_data.geojson
```

## Troubleshooting

### Common Issues

#### Missing Dependencies

If you encounter errors about missing packages, ensure you've installed all requirements:

```bash
pip install -r requirements.txt
```

#### Input File Format Errors

If you receive input format errors, check that your file is in a supported format:

```bash
python -m core.cli -i data.gpx -o ./output --verbose
```

The verbose flag will provide more details about any parsing issues.

#### Memory Issues with Large Files

For very large files, you might encounter memory issues. Try processing with batch mode:

```bash
python -m core.cli -i large_data.gpx -o ./output --batch-size 1000
```

### Getting Help

For more detailed help on any command, use the `-h` or `--help` option:

```bash
python -m core.cli --help
python -m visualizations.web_interface --help
python -m core.api --help
``` 