#!/usr/bin/env python3

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
import threading
import time

# Configure logging - this will be the central configuration point for all modules
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("running_analysis.log"),
                       logging.StreamHandler()
                   ])

# Create module-level logger
logger = logging.getLogger(__name__)

# Export this logger for other modules to import
__all__ = ['logger', 'continuous_learning_manager']

class ModelRegistry:
    """
    Model registry for tracking, versioning and persisting models.
    
    The registry keeps track of all registered models, their versions,
    and metadata. It also handles saving and loading models to/from disk.
    """
    
    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory for storing the model registry and models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / "registry.json"
        
        # Create registry directory if it doesn't exist
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Load registry if exists, otherwise create new
        if self.registry_file.exists():
            self.registry = self._load_registry()
        else:
            self.registry = self._create_new_registry()
            self._save_registry()
            
        logger.info(f"Initialized model registry at {registry_dir}")
        logger.info(f"Found {len(self.registry['models'])} registered models")
        
    def _load_registry(self) -> Dict:
        """Load model registry from file."""
        try:
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
            logger.info(f"Loaded model registry from {self.registry_file}")
            return registry
        except Exception as e:
            logger.error(f"Error loading model registry: {e}", exc_info=True)
            # If load fails, create a new registry
            logger.info("Creating new registry due to load failure")
            return self._create_new_registry()
            
    def _create_new_registry(self) -> Dict:
        """Create a new empty registry."""
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "models": {}
        }
        
    def _save_registry(self, registry: Dict = None) -> None:
        """Save the model registry to file."""
        registry = registry or self.registry
        registry["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Saved model registry to {self.registry_file}")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}", exc_info=True)
            
    def register_model(self, model_name: str, model_type: str, model_obj: Any, 
                      metadata: Dict = None, version: str = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model (e.g., "posture", "aerodynamics")
            model_obj: The model object to register
            metadata: Dictionary with model metadata
            version: Version string (if None, a new version will be generated)
            
        Returns:
            The version string for the registered model
        """
        # Generate version if not provided
        if not version:
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        # Create model entry in registry if it doesn't exist
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {
                "name": model_name,
                "type": model_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "versions": {},
                "current_version": version
            }
            
        # Update model metadata
        model_entry = self.registry["models"][model_name]
        model_entry["updated_at"] = datetime.now().isoformat()
        model_entry["type"] = model_type  # Update type in case it changed
        
        # Create the model directory
        model_dir = self.registry_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save the model to disk using appropriate format
        model_path = model_dir / f"{version}.joblib"
        
        try:
            # Determine the appropriate serialization method
            if hasattr(model_obj, 'state_dict') and callable(getattr(model_obj, 'state_dict')):
                # PyTorch model
                import torch
                torch.save(model_obj.state_dict(), model_path)
                serialization_method = "pytorch"
            elif hasattr(model_obj, 'get_weights') and callable(getattr(model_obj, 'get_weights')):
                # TensorFlow/Keras model
                try:
                    model_obj.save(model_path)
                    serialization_method = "tensorflow"
                except Exception:
                    # Fallback to joblib if tf.save fails
                    import joblib
                    joblib.dump(model_obj, model_path)
                    serialization_method = "joblib"
            else:
                # Default to joblib for scikit-learn and other models
                import joblib
                joblib.dump(model_obj, model_path)
                serialization_method = "joblib"
                
            logger.info(f"Saved model {model_name} version {version} to {model_path}")
            print(f"Saved model {model_name} version {version} to {model_path}")
            
            # Copy the model to output directory for easy access
            try:
                output_dir = Path("output/models")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_model_path = output_dir / f"{model_name}_{version}.joblib"
                
                if serialization_method == "joblib":
                    import shutil
                    shutil.copy2(model_path, output_model_path)
                    logger.info(f"Copied model to output directory: {output_model_path}")
                else:
                    # For non-joblib formats, save a copy using joblib if possible
                    try:
                        import joblib
                        joblib.dump(model_obj, output_model_path)
                        logger.info(f"Saved joblib copy to output directory: {output_model_path}")
                    except Exception as e:
                        logger.warning(f"Could not save joblib copy: {e}")
            except Exception as e:
                logger.warning(f"Error copying model to output directory: {e}")
                
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}", exc_info=True)
            serialization_method = "failed"
            
        # Add version to registry
        model_entry["versions"][version] = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "path": str(model_path),
            "metadata": metadata or {},
            "serialization_method": serialization_method
        }
        
        # Update current version
        model_entry["current_version"] = version
        
        # Save the registry
        self._save_registry()
        
        return version
        
    def get_model(self, model_name: str, version: str = None) -> Any:
        """
        Get a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version to get (if None, gets the current version)
            
        Returns:
            The model object, or None if not found
        """
        try:
            # Check if model exists
            if model_name not in self.registry["models"]:
                logger.warning(f"Model {model_name} not found in registry")
                return None
                
            model_entry = self.registry["models"][model_name]
            
            # Determine version to load
            version_to_load = version or model_entry.get("current_version")
            
            if not version_to_load or version_to_load not in model_entry["versions"]:
                logger.warning(f"Version {version_to_load} not found for model {model_name}")
                return None
                
            version_entry = model_entry["versions"][version_to_load]
            model_path = version_entry["path"]
            
            # Load the model using the appropriate method
            serialization_method = version_entry.get("serialization_method", "joblib")
            
            if serialization_method == "pytorch":
                # Try to load as PyTorch model
                try:
                    import torch
                    # Need to instantiate the model first - this is tricky without knowing the class
                    # This is a placeholder that would need customization for each model type
                    if model_entry["type"] == "posture":
                        from actigraphy.posture import PostureEstimator
                        model = PostureEstimator(input_dim=10)
                        model.load_state_dict(torch.load(model_path))
                    else:
                        # Default to joblib for unknown types
                        import joblib
                        model = joblib.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading PyTorch model: {e}", exc_info=True)
                    # Try joblib as fallback
                    import joblib
                    model = joblib.load(model_path)
            elif serialization_method == "tensorflow":
                # Try to load as TensorFlow model
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                except Exception as e:
                    logger.error(f"Error loading TensorFlow model: {e}", exc_info=True)
                    # Try joblib as fallback
                    import joblib
                    model = joblib.load(model_path)
            else:
                # Default to joblib
                import joblib
                model = joblib.load(model_path)
                
            logger.info(f"Loaded model {model_name} version {version_to_load}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            return None
            
    def list_models(self) -> List[Dict]:
        """
        List all models in the registry.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for name, model in self.registry["models"].items():
            models.append({
                "name": name,
                "type": model.get("type", "unknown"),
                "current_version": model.get("current_version"),
                "versions": len(model["versions"]),
                "created_at": model.get("created_at"),
                "updated_at": model.get("updated_at")
            })
        return models
        
    def get_model_history(self, model_name: str) -> List[Dict]:
        """Get version history for a model."""
        if model_name not in self.registry["models"]:
            return []
            
        model = self.registry["models"][model_name]
        return [v for _, v in sorted(model["versions"].items())]
        
    def set_current_version(self, model_name: str, version: str) -> bool:
        """
        Set the current version for a model.
        
        Args:
            model_name: Name of the model
            version: Version to set as current
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model exists
            if model_name not in self.registry["models"]:
                logger.warning(f"Model {model_name} not found in registry")
                return False
                
            model_entry = self.registry["models"][model_name]
            
            # Check if version exists
            if version not in model_entry["versions"]:
                logger.warning(f"Version {version} not found for model {model_name}")
                return False
                
            # Set current version
            model_entry["current_version"] = version
            model_entry["updated_at"] = datetime.now().isoformat()
            
            # Save the registry
            self._save_registry()
            
            logger.info(f"Set current version of {model_name} to {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting current version: {e}", exc_info=True)
            return False
            
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model exists
            if model_name not in self.registry["models"]:
                logger.warning(f"Model {model_name} not found in registry")
                return False
                
            model_entry = self.registry["models"][model_name]
            
            # Check if version exists
            if version not in model_entry["versions"]:
                logger.warning(f"Version {version} not found for model {model_name}")
                return False
                
            # Check if version is current
            if model_entry.get("current_version") == version:
                logger.warning(f"Cannot delete current version {version} of model {model_name}")
                return False
                
            # Get version path
            version_entry = model_entry["versions"][version]
            model_path = version_entry["path"]
            
            # Delete model file
            try:
                os.remove(model_path)
                logger.info(f"Deleted model file: {model_path}")
            except Exception as e:
                logger.warning(f"Error deleting model file: {e}")
                
            # Remove version from registry
            del model_entry["versions"][version]
            model_entry["updated_at"] = datetime.now().isoformat()
            
            # Save the registry
            self._save_registry()
            
            logger.info(f"Deleted version {version} of model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting version: {e}", exc_info=True)
            return False

class PerformanceTracker:
    """
    Track and analyze model performance over time.
    """
    
    def __init__(self, storage_dir: str = "models/performance"):
        """
        Initialize the performance tracker.
        
        Args:
            storage_dir: Directory to store performance metrics
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def log_metrics(self, model_name: str, version: str, metrics: Dict, 
                   metadata: Dict = None) -> None:
        """
        Log performance metrics for a model.
        
        Args:
            model_name: Name of the model
            version: Model version
            metrics: Dictionary of performance metrics
            metadata: Additional metadata about the evaluation
        """
        model_dir = self.storage_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        metric_file = model_dir / f"{version}_metrics.json"
        
        # Prepare data to save
        data = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        # Save to file
        with open(metric_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Logged metrics for {model_name} version {version}")
        
    def get_metrics(self, model_name: str, version: str = None) -> Dict:
        """
        Get metrics for a specific model version.
        
        Args:
            model_name: Name of the model
            version: Model version (if None, returns metrics for all versions)
            
        Returns:
            Dictionary of metrics
        """
        model_dir = self.storage_dir / model_name
        if not model_dir.exists():
            return {}
            
        if version:
            metric_file = model_dir / f"{version}_metrics.json"
            if not metric_file.exists():
                return {}
                
            with open(metric_file, 'r') as f:
                return json.load(f)
        else:
            # Return metrics for all versions
            all_metrics = []
            for file in model_dir.glob("*_metrics.json"):
                with open(file, 'r') as f:
                    all_metrics.append(json.load(f))
            return all_metrics
            
    def get_performance_history(self, model_name: str, metric_name: str) -> List[Dict]:
        """
        Get historical performance for a specific metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the metric to track
            
        Returns:
            List of {version, timestamp, value} entries
        """
        model_dir = self.storage_dir / model_name
        if not model_dir.exists():
            return []
            
        history = []
        for file in model_dir.glob("*_metrics.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                
            if metric_name in data["metrics"]:
                history.append({
                    "version": data["version"],
                    "timestamp": data["timestamp"],
                    "value": data["metrics"][metric_name]
                })
                
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        return history
        
    def should_retrain(self, model_name: str, new_metrics: Dict, 
                      threshold: float = 0.1) -> bool:
        """
        Determine if a model should be retrained based on performance drop.
        
        Args:
            model_name: Name of the model
            new_metrics: New performance metrics
            threshold: Threshold for performance decline (e.g., 0.1 = 10% worse)
            
        Returns:
            Boolean indicating whether retraining is recommended
        """
        # Get historical performance
        history = self.get_performance_history(model_name, "rmse")
        if not history:
            return False
            
        # Get best historical performance
        best_performance = min(history, key=lambda x: x["value"])
        
        # Compare with new metrics
        if "rmse" in new_metrics:
            current_rmse = new_metrics["rmse"]
            best_rmse = best_performance["value"]
            
            # Calculate relative decline
            decline = (current_rmse - best_rmse) / best_rmse
            
            # Recommend retraining if decline exceeds threshold
            return decline > threshold
            
        return False
        
        
class DataCollector:
    """
    Collect and prepare data for model training and evaluation.
    """
    
    def __init__(self, data_dir: str = "data/training"):
        """
        Initialize the data collector.
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def store_run_data(self, run_id: str, data: Dict, metadata: Dict = None) -> None:
        """
        Store data from a running session.
        
        Args:
            run_id: Unique identifier for the run
            data: Run data to store
            metadata: Additional metadata about the run
        """
        # Create run directory
        run_dir = self.data_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Store metadata
        if metadata:
            with open(run_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # Store data
        data_file = run_dir / "run_data.json"
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Stored data for run {run_id}")
        
    def get_recent_runs(self, days: int = 30) -> List[str]:
        """
        Get list of recent run IDs.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of run IDs
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_runs = []
        for run_dir in self.data_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    if "timestamp" in metadata:
                        run_date = datetime.fromisoformat(metadata["timestamp"])
                        if run_date >= cutoff_date:
                            recent_runs.append(run_dir.name)
                except Exception as e:
                    logger.error(f"Error reading metadata for run {run_dir.name}: {e}")
            
        return recent_runs
        
    def load_run_data(self, run_id: str) -> Dict:
        """
        Load data for a specific run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run data
        """
        data_file = self.data_dir / run_id / "run_data.json"
        if not data_file.exists():
            return {}
            
        with open(data_file, 'r') as f:
            return json.load(f)
            
    def extract_training_data(self, model_type: str, days: int = 30, 
                            min_runs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract training data for a specific model type.
        
        Args:
            model_type: Type of model for which to extract data
            days: Number of days to look back
            min_runs: Minimum number of runs to include
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        recent_runs = self.get_recent_runs(days)
        
        if len(recent_runs) < min_runs:
            logger.warning(f"Not enough recent runs ({len(recent_runs)}/{min_runs})")
            # Get more runs if needed
            all_runs = [run_dir.name for run_dir in self.data_dir.iterdir() if run_dir.is_dir()]
            all_runs.sort(reverse=True)  # Sort newest first
            recent_runs = all_runs[:min_runs] if len(all_runs) >= min_runs else all_runs
            
        # Extract features and targets based on model type
        features = []
        targets = []
        
        for run_id in recent_runs:
            run_data = self.load_run_data(run_id)
            
            if not run_data:
                continue
                
            # Different extraction logic based on model type
            if model_type == "aerodynamics":
                features_targets = self._extract_aerodynamics_data(run_data)
            elif model_type == "posture":
                features_targets = self._extract_posture_data(run_data)
            elif model_type == "oxygen":
                features_targets = self._extract_oxygen_data(run_data)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                features_targets = [], []
                
            X, y = features_targets
            features.extend(X)
            targets.extend(y)
                
        return np.array(features), np.array(targets)
        
    def _extract_aerodynamics_data(self, run_data: Dict) -> Tuple[List, List]:
        """Extract features and targets for aerodynamics models."""
        X, y = [], []
        
        # Check for required data
        if "aerodynamics_analysis" not in run_data:
            return X, y
            
        aero_data = run_data["aerodynamics_analysis"]
        
        # Convert to data points
        for point in aero_data:
            if isinstance(point, dict) and "speed" in point and "drag_force" in point:
                features = [
                    float(point.get("speed", 0)),
                    float(point.get("air_density", 1.225)),
                    float(point.get("altitude", 0)),
                    float(point.get("wind_speed", 0))
                ]
                
                target = float(point.get("drag_force", 0))
                
                X.append(features)
                y.append(target)
                
        return X, y
        
    def _extract_posture_data(self, run_data: Dict) -> Tuple[List, List]:
        """Extract features and targets for posture models."""
        X, y = [], []
        
        # Check for required data
        if "posture_sequence" not in run_data:
            return X, y
            
        posture_data = run_data["posture_sequence"]
        
        # Convert to data points
        for point in posture_data:
            if isinstance(point, dict) and "metrics" in point and "joints" in point:
                metrics = point["metrics"]
                joints = point["joints"]
                
                if isinstance(metrics, dict) and isinstance(joints, dict):
                    features = [
                        float(metrics.get("cadence", 0)),
                        float(metrics.get("speed", 0)),
                        float(metrics.get("vertical_oscillation", 0)),
                        float(metrics.get("stance_time", 0)),
                        float(metrics.get("stance_time_balance", 0)),
                        float(metrics.get("step_length", 0)),
                        float(metrics.get("power", 0))
                    ]
                    
                    # Flatten joint angles for target
                    target = []
                    for joint, angles in joints.items():
                        if isinstance(angles, dict):
                            for angle_type, value in angles.items():
                                target.append(float(value))
                    
                    X.append(features)
                    y.append(target)
                
        return X, y
        
    def _extract_oxygen_data(self, run_data: Dict) -> Tuple[List, List]:
        """Extract features and targets for oxygen uptake models."""
        X, y = [], []
        
        # Check for required data
        if "oxygen_analysis" not in run_data:
            return X, y
            
        oxygen_data = run_data["oxygen_analysis"]
        
        # Get VO2 values
        if "vo2_values" in oxygen_data and isinstance(oxygen_data["vo2_values"], list):
            for point in oxygen_data["vo2_values"]:
                if isinstance(point, dict) and "metrics_used" in point and "vo2" in point:
                    metrics = point["metrics_used"]
                    
                    features = [
                        float(metrics.get("speed", 0)),
                        float(metrics.get("heart_rate", 0)),
                        float(metrics.get("grade", 0)),
                        float(metrics.get("altitude", 0)),
                        float(metrics.get("cadence", 0))
                    ]
                    
                    target = float(point.get("vo2", 0))
                    
                    X.append(features)
                    y.append(target)
                
        return X, y
        

class ContinuousLearningManager:
    """
    Manage continuous learning across all models in the running analysis pipeline.
    """
    
    def __init__(self, 
                model_registry: ModelRegistry = None,
                performance_tracker: PerformanceTracker = None,
                data_collector: DataCollector = None,
                base_dir: str = "models"):
        """
        Initialize the continuous learning manager.
        
        Args:
            model_registry: Model registry object
            performance_tracker: Performance tracker object
            data_collector: Data collector object
            base_dir: Base directory for all model-related storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components if not provided
        self.model_registry = model_registry or ModelRegistry(
            registry_dir=str(self.base_dir / "registry"))
        self.performance_tracker = performance_tracker or PerformanceTracker(
            storage_dir=str(self.base_dir / "performance"))
        self.data_collector = data_collector or DataCollector(
            data_dir=str(self.base_dir / "training_data"))
            
        # Retraining schedules
        self.retraining_schedules = {}
        self.retraining_threads = {}
        
        # Training handlers for different model types
        self.training_handlers = {}
        
        logger.info("Initialized Continuous Learning Manager")
        
    def register_training_handler(self, model_type: str, 
                                handler: Callable[[np.ndarray, np.ndarray], Any]) -> None:
        """
        Register a training handler function for a specific model type.
        
        Args:
            model_type: Type of model
            handler: Function that takes (X, y) and returns a trained model
        """
        self.training_handlers[model_type] = handler
        logger.info(f"Registered training handler for model type: {model_type}")
        
    def store_run_results(self, run_id: str, results: Dict, 
                        metadata: Dict = None) -> None:
        """
        Store results from a running analysis session.
        
        Args:
            run_id: Unique identifier for the run
            results: Analysis results
            metadata: Additional metadata about the run
        """
        # Add timestamp if not present
        if metadata is None:
            metadata = {}
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
            
        # Store the data
        self.data_collector.store_run_data(run_id, results, metadata)
        
        # Evaluate if we need to retrain models
        self._evaluate_retraining_needs(results)
        
    def _evaluate_retraining_needs(self, results: Dict) -> None:
        """
        Evaluate if any models need retraining based on new results.
        
        Args:
            results: New analysis results
        """
        # Example: check aerodynamics model performance
        if "aerodynamics_analysis" in results:
            aero_results = results["aerodynamics_analysis"]
            if isinstance(aero_results, dict) and "error_metrics" in aero_results:
                metrics = aero_results["error_metrics"]
                should_retrain = self.performance_tracker.should_retrain(
                    "aerodynamics_drag", metrics)
                
                if should_retrain:
                    logger.info("Scheduling aerodynamics model retraining due to performance drop")
                    self.schedule_retraining("aerodynamics_drag", "aerodynamics")
        
    def train_model(self, model_name: str, model_type: str) -> str:
        """
        Train a model with recent data.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            Version of newly trained model, or None if training failed
        """
        logger.info(f"Training model {model_name} of type {model_type}")
        
        # Check if we have a training handler for this model type
        if model_type not in self.training_handlers:
            logger.error(f"No training handler found for model type: {model_type}")
            return None
            
        # Extract training data
        X, y = self.data_collector.extract_training_data(model_type)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"No training data available for {model_name}")
            return None
            
        logger.info(f"Extracted {len(X)} training samples for {model_name}")
        
        # Train the model
        try:
            model = self.training_handlers[model_type](X, y)
            
            # Register the new model version
            metadata = {
                "training_samples": len(X),
                "training_date": datetime.now().isoformat()
            }
            
            version = self.model_registry.register_model(
                model_name, model_type, model, metadata)
                
            logger.info(f"Successfully trained and registered {model_name} version {version}")
            return version
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return None
            
    def evaluate_model(self, model_name: str, model_type: str, 
                      version: str = None) -> Dict:
        """
        Evaluate a model on recent data.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            version: Model version to evaluate (if None, uses current version)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model {model_name} version {version or 'current'}")
        
        # Get the model
        model = self.model_registry.get_model(model_name, version)
        if model is None:
            logger.error(f"Could not load model {model_name} version {version or 'current'}")
            return {}
            
        # Get evaluation data
        X, y = self.data_collector.extract_training_data(model_type, days=14)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"No evaluation data available for {model_name}")
            return {}
            
        # Make predictions
        try:
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(mean_absolute_error(y, y_pred)),
                "sample_count": len(X)
            }
            
            # Log metrics
            self.performance_tracker.log_metrics(
                model_name, version or self.model_registry.registry["models"][model_name]["current_version"],
                metrics)
                
            logger.info(f"Evaluated {model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}
            
    def schedule_retraining(self, model_name: str, model_type: str, 
                          interval_hours: float = 24.0) -> None:
        """
        Schedule regular retraining for a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            interval_hours: Interval between retraining in hours
        """
        # Cancel existing schedule if any
        if model_name in self.retraining_schedules:
            self.retraining_schedules[model_name] = False
            
        # Set up new schedule
        self.retraining_schedules[model_name] = True
        
        # Start retraining thread
        thread = threading.Thread(
            target=self._retraining_worker,
            args=(model_name, model_type, interval_hours),
            daemon=True
        )
        
        self.retraining_threads[model_name] = thread
        thread.start()
        
        logger.info(f"Scheduled retraining for {model_name} every {interval_hours} hours")
        
    def _retraining_worker(self, model_name: str, model_type: str, 
                         interval_hours: float) -> None:
        """
        Worker function for scheduled retraining.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            interval_hours: Interval between retraining in hours
        """
        while self.retraining_schedules.get(model_name, False):
            # Train the model
            new_version = self.train_model(model_name, model_type)
            
            if new_version:
                # Evaluate the new model
                metrics = self.evaluate_model(model_name, model_type, new_version)
                
                # Log retraining event
                logger.info(f"Scheduled retraining of {model_name} complete: version={new_version}, "
                          f"rmse={metrics.get('rmse', 'N/A')}")
            else:
                logger.warning(f"Scheduled retraining of {model_name} failed")
                
            # Sleep until next retraining
            interval_seconds = interval_hours * 3600
            logger.info(f"Next retraining of {model_name} scheduled in {interval_hours} hours")
            
            # Sleep in smaller chunks to allow for graceful shutdown
            chunks = 36  # Check every 100 seconds for a 1-hour interval
            chunk_time = interval_seconds / chunks
            
            for _ in range(chunks):
                if not self.retraining_schedules.get(model_name, False):
                    break
                time.sleep(chunk_time)
                
    def cancel_retraining(self, model_name: str) -> None:
        """
        Cancel scheduled retraining for a model.
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.retraining_schedules:
            self.retraining_schedules[model_name] = False
            logger.info(f"Cancelled retraining schedule for {model_name}")
        else:
            logger.warning(f"No retraining schedule found for {model_name}")
            
    def get_retraining_status(self) -> Dict[str, bool]:
        """Get status of all retraining schedules."""
        return {model: active for model, active in self.retraining_schedules.items()}
        
    def get_model_performances(self) -> Dict[str, Dict]:
        """Get performance summary for all models."""
        result = {}
        
        for model_info in self.model_registry.list_models():
            model_name = model_info["name"]
            metrics = self.performance_tracker.get_metrics(model_name)
            
            if metrics:
                # Get latest metrics
                latest = metrics[-1] if isinstance(metrics, list) else metrics
                
                # Get performance trend
                history = self.performance_tracker.get_performance_history(model_name, "rmse")
                trend = "stable"
                
                if len(history) >= 2:
                    first = history[0]["value"]
                    last = history[-1]["value"]
                    change = (last - first) / first if first > 0 else 0
                    
                    if change < -0.05:
                        trend = "improving"
                    elif change > 0.05:
                        trend = "declining"
                        
                result[model_name] = {
                    "latest_metrics": latest.get("metrics", {}),
                    "trend": trend,
                    "version": latest.get("version", "unknown"),
                    "last_updated": latest.get("timestamp", "unknown")
                }
                
        return result
                

# Example training handlers for different model types
def train_aerodynamics_model(X, y):
    """Train an aerodynamics model with provided data."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a random forest regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Create a pipeline with scaler and model
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        logger.info(f"Trained aerodynamics model with {len(X)} samples")
        
        # Save the model to the output directory directly for immediate use
        import joblib
        output_dir = Path("output/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"aerodynamics_latest.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved latest aerodynamics model to {model_path}")
        
        return pipeline
    except Exception as e:
        logger.error(f"Error training aerodynamics model: {e}", exc_info=True)
        return None

def train_posture_model(X, y):
    """Train a posture model with provided data."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = y.shape[1] if len(y.shape) > 1 else 1
        
        class PostureModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_dim)
                )
                
            def forward(self, x):
                return self.net(x)
        
        model = PostureModel(input_dim, output_dim)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch+1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
        logger.info(f"Trained posture model with {len(X)} samples")
        
        # Save the model for immediate use
        output_dir = Path("output/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "posture_latest.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved latest posture model to {model_path}")
        
        # Also save as joblib for compatibility
        try:
            import joblib
            joblib_path = output_dir / "posture_latest.joblib"
            joblib.dump(model, joblib_path)
            logger.info(f"Saved joblib version to {joblib_path}")
        except Exception as e:
            logger.warning(f"Could not save joblib version: {e}")
        
        return model
    except Exception as e:
        logger.error(f"Error training posture model: {e}", exc_info=True)
        return None

def train_oxygen_model(X, y):
    """Train an oxygen uptake model with provided data."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a gradient boosting regressor
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Create a pipeline with scaler and model
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        logger.info(f"Trained oxygen uptake model with {len(X)} samples")
        
        # Save the model for immediate use
        import joblib
        output_dir = Path("output/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "oxygen_latest.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved latest oxygen model to {model_path}")
        
        return pipeline
    except Exception as e:
        logger.error(f"Error training oxygen model: {e}", exc_info=True)
        return None


# Initialize a global instance for easy import
continuous_learning_manager = ContinuousLearningManager()

# Register default training handlers
continuous_learning_manager.register_training_handler("aerodynamics", train_aerodynamics_model)
continuous_learning_manager.register_training_handler("posture", train_posture_model)
continuous_learning_manager.register_training_handler("oxygen", train_oxygen_model)

if __name__ == "__main__":
    # Example usage
    print("Model Registry Status:")
    for model in continuous_learning_manager.model_registry.list_models():
        print(f"- {model['name']} ({model['type']}): version {model['current_version']}")
    
    print("\nRetraining Status:")
    for model, active in continuous_learning_manager.get_retraining_status().items():
        print(f"- {model}: {'Active' if active else 'Inactive'}") 