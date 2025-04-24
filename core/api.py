#!/usr/bin/env python3
import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from flask import Flask, request, jsonify, send_file
import logging
import threading
from pathlib import Path
import tempfile

from . import data_loader
from . import dynamic_filtering
from . import optimal_path
from . import dubins_path
from .error_handler import ErrorType, SighthoundError, safe_execute
from .exporters import get_exporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sighthound_api")

# Flask application
app = Flask(__name__)

# In-memory job storage for tracking long-running processes
jobs = {}

# Global configuration
config = {
    "output_dir": os.environ.get("SIGHTHOUND_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "sighthound")),
    "max_jobs": int(os.environ.get("SIGHTHOUND_MAX_JOBS", 10)),
    "job_timeout": int(os.environ.get("SIGHTHOUND_JOB_TIMEOUT", 3600)),  # 1 hour
    "allowed_formats": ["csv", "geojson", "kml", "gpx", "czml", "shp"]
}

# Create output directory if it doesn't exist
os.makedirs(config["output_dir"], exist_ok=True)


class JobStatus:
    """Status values for job tracking."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def get_job_info(job_id: str) -> Dict[str, Any]:
    """Get job information from storage."""
    if job_id not in jobs:
        raise SighthoundError(
            message=f"Job not found: {job_id}",
            error_type=ErrorType.VALIDATION_ERROR
        )
    return jobs[job_id]


def clean_old_jobs() -> None:
    """Remove old jobs from storage."""
    current_time = time.time()
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        if job.get("created_at", 0) + config["job_timeout"] < current_time:
            if job["status"] == JobStatus.PROCESSING:
                job["status"] = JobStatus.FAILED
                job["error"] = "Job timed out"
            # Keep metadata but remove result data to save memory
            if "result_data" in job:
                del job["result_data"]


def process_data_async(job_id: str, input_file: str, options: Dict[str, Any]) -> None:
    """Process data asynchronously and update job status."""
    job = jobs[job_id]
    
    try:
        # Update job status
        job["status"] = JobStatus.PROCESSING
        job["progress"] = 0
        
        # Update progress for monitoring
        job["progress"] = 10
        job["status_message"] = "Loading data..."
        
        # Load data (simulated)
        time.sleep(1)  # Simulate loading time
        
        # Apply Kalman filtering if requested
        if options.get("apply_filtering", False):
            job["progress"] = 30
            job["status_message"] = "Applying Kalman filtering..."
            time.sleep(1)  # Simulate processing time
        
        # Calculate optimal path if requested
        if options.get("optimize_path", False):
            job["progress"] = 50
            job["status_message"] = "Calculating optimal path..."
            time.sleep(1)  # Simulate processing time
        
        # Calculate Dubins path if requested
        if options.get("calculate_dubins", False):
            job["progress"] = 70
            job["status_message"] = "Calculating Dubins path..."
            time.sleep(1)  # Simulate processing time
        
        # Export data to requested format
        job["progress"] = 90
        job["status_message"] = "Exporting results..."
        
        # Determine output format
        output_format = options.get("output_format", "geojson").lower()
        if output_format not in config["allowed_formats"]:
            raise SighthoundError(
                message=f"Unsupported output format: {output_format}",
                error_type=ErrorType.VALIDATION_ERROR
            )
        
        # Generate a unique output filename
        output_filename = f"sighthound_{job_id}"
        
        # In a real implementation, we would process the data here
        # For demo purposes, we'll create a simple output file
        sample_data = [
            {"latitude": 37.7749, "longitude": -122.4194, "timestamp": time.time(), "accuracy": 10},
            {"latitude": 37.7750, "longitude": -122.4195, "timestamp": time.time() + 10, "accuracy": 12},
            {"latitude": 37.7751, "longitude": -122.4196, "timestamp": time.time() + 20, "accuracy": 8}
        ]
        
        # Get exporter for the requested format
        exporter = get_exporter(output_format, config["output_dir"], output_filename)
        
        # Export data
        output_path = exporter.export(sample_data, name="API Generated Track")
        
        # Update job status with success information
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 100
        job["status_message"] = "Processing complete"
        job["output_file"] = output_path
        job["completed_at"] = time.time()
        
    except Exception as e:
        # Update job status with error information
        logger.exception(f"Error processing job {job_id}")
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        job["completed_at"] = time.time()


@app.route('/api/version', methods=['GET'])
def get_version() -> Dict[str, str]:
    """Get API version information."""
    return jsonify({
        "name": "Sighthound API",
        "version": "1.0.0",
        "status": "operational"
    })


@app.route('/api/process', methods=['POST'])
def start_processing() -> Dict[str, Any]:
    """Start a new processing job."""
    # Clean old jobs
    clean_old_jobs()
    
    # Check if we've reached the job limit
    if len(jobs) >= config["max_jobs"]:
        raise SighthoundError(
            message="Maximum number of concurrent jobs reached",
            error_type=ErrorType.SYSTEM_ERROR,
            troubleshooting_tips=[
                "Wait for existing jobs to complete",
                "Increase the SIGHTHOUND_MAX_JOBS environment variable"
            ]
        )
    
    try:
        # Get options from request
        options = request.json or {}
        
        # Validate required fields
        if 'input_file' not in request.files:
            return jsonify({
                "error": "No input file provided",
                "status": "error"
            }), 400
        
        input_file = request.files['input_file']
        
        # Generate a unique job ID
        job_id = f"job_{int(time.time())}_{len(jobs) + 1}"
        
        # Save the input file
        input_file_path = os.path.join(config["output_dir"], f"{job_id}_input{os.path.splitext(input_file.filename)[1]}")
        input_file.save(input_file_path)
        
        # Create a new job
        job = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0,
            "status_message": "Job created, waiting to start",
            "created_at": time.time(),
            "input_file": input_file_path,
            "options": options
        }
        jobs[job_id] = job
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=process_data_async,
            args=(job_id, input_file_path, options)
        )
        thread.daemon = True
        thread.start()
        
        # Return job information
        return jsonify({
            "job_id": job_id,
            "status": "created",
            "message": "Processing started"
        })
        
    except Exception as e:
        logger.exception("Error starting job")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a job."""
    try:
        job = get_job_info(job_id)
        
        # Return job information without internal details
        return jsonify({
            "job_id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "message": job.get("status_message", ""),
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "error": job.get("error")
        })
        
    except Exception as e:
        logger.exception(f"Error getting job status for {job_id}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 404


@app.route('/api/jobs/<job_id>/result', methods=['GET'])
def get_job_result(job_id: str) -> Union[Dict[str, Any], Any]:
    """Get the result of a completed job."""
    try:
        job = get_job_info(job_id)
        
        # Check if job is completed
        if job["status"] != JobStatus.COMPLETED:
            return jsonify({
                "error": f"Job is not completed (status: {job['status']})",
                "status": "error"
            }), 400
        
        # Check if output file exists
        if "output_file" not in job or not os.path.exists(job["output_file"]):
            return jsonify({
                "error": "Output file not found",
                "status": "error"
            }), 404
        
        # Determine response format
        format_param = request.args.get('format', 'file')
        
        if format_param == 'file':
            # Send the file
            return send_file(
                job["output_file"],
                as_attachment=True,
                download_name=os.path.basename(job["output_file"])
            )
        else:
            # Return metadata about the file
            return jsonify({
                "job_id": job["id"],
                "status": "completed",
                "output_file": os.path.basename(job["output_file"]),
                "completed_at": job["completed_at"],
                "file_size": os.path.getsize(job["output_file"]),
                "file_url": f"/api/jobs/{job_id}/download"
            })
        
    except Exception as e:
        logger.exception(f"Error getting job result for {job_id}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route('/api/jobs/<job_id>/download', methods=['GET'])
def download_result(job_id: str) -> Any:
    """Download the result file directly."""
    try:
        job = get_job_info(job_id)
        
        # Check if job is completed
        if job["status"] != JobStatus.COMPLETED:
            return jsonify({
                "error": f"Job is not completed (status: {job['status']})",
                "status": "error"
            }), 400
        
        # Check if output file exists
        if "output_file" not in job or not os.path.exists(job["output_file"]):
            return jsonify({
                "error": "Output file not found",
                "status": "error"
            }), 404
        
        # Send the file
        return send_file(
            job["output_file"],
            as_attachment=True,
            download_name=os.path.basename(job["output_file"])
        )
        
    except Exception as e:
        logger.exception(f"Error downloading result for {job_id}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


def run_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
    """Run the API server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sighthound API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    logger.info(f"Starting Sighthound API on {args.host}:{args.port}")
    run_api(args.host, args.port, args.debug) 