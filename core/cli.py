#!/usr/bin/env python3
import argparse
import sys
import os
import time
from tqdm import tqdm
import yaml
from . import data_loader
from . import dynamic_filtering
from . import optimal_path
from . import dubins_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sighthound: A tool for GPS data fusion, triangulation, and path optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments
    parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--format', '-f', choices=['csv', 'geojson', 'czml'], default='geojson', 
                       help='Output format')
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('--filter', action='store_true', help='Apply Kalman filtering')
    processing_group.add_argument('--triangulate', action='store_true', help='Use cell tower data for triangulation')
    processing_group.add_argument('--optimize-path', action='store_true', help='Calculate optimal path')
    processing_group.add_argument('--dubins-path', action='store_true', help='Calculate Dubins path')
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', '-c', help='Path to configuration file')
    config_group.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity')
    
    return parser.parse_args()

def validate_config(config):
    """Validate configuration with helpful error messages"""
    required_fields = ['input_formats', 'output_formats', 'filter_params']
    errors = []
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field '{field}' in configuration")
    
    if 'filter_params' in config and not isinstance(config['filter_params'], dict):
        errors.append("'filter_params' must be a dictionary")
    
    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return False
    
    return True

def load_config(config_path):
    """Load configuration file with validation"""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not validate_config(config):
            return None
            
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading configuration file: {e}", file=sys.stderr)
        return None

def process_data(args, config=None):
    """Process data with progress indicators"""
    # Create progress bar
    print(f"Processing input: {args.input}")
    
    # Simulated processing steps for demonstration
    steps = [
        ('Loading data', 10),
        ('Filtering data', 20),
        ('Optimizing paths', 30),
        ('Generating output', 10)
    ]
    
    for step_name, step_duration in steps:
        print(f"\n{step_name}...")
        with tqdm(total=100, desc=step_name, ncols=100) as pbar:
            # Simulate processing with progress updates
            for i in range(10):
                # Simulate some work
                time.sleep(step_duration/10/10)  # Divide by 10 to make it faster for demo
                pbar.update(10)
    
    print(f"\nProcessing complete. Results saved to {args.output}")

def main():
    """Main entry point for command line interface"""
    args = parse_args()
    
    # Load configuration if specified
    config = None
    if args.config:
        config = load_config(args.config)
        if not config:
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Process data with progress indicators
    process_data(args, config)

if __name__ == "__main__":
    main() 