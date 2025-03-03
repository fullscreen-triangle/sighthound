#!/usr/bin/env python3

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from acquisition.czgeneration import generate_czml
from actigraphy.posture import PostureAnalyzer
from actigraphy.power_curve import PowerAnalyzer
from atmospheric.aerodynamics import AerodynamicsCalculator
from atmospheric.oxygen_uptake_rate import OxygenUptakeCalculator
from biomechanics.dynamics import SprintDynamics
from biomechanics.surface import SurfaceDynamicsAnalyzer
from cardiography.signal_processing import HeartRateProcessor, MLSignalManager
from federation.fusion import DataFusion

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RunningAnalysisPipeline:
    """Main orchestration class for the running analysis pipeline."""
    
    def __init__(self, 
                 athlete_mass: float,
                 athlete_height: float,
                 athlete_age: int,
                 athlete_gender: str,
                 output_dir: str = "output",
                 weather_api_key: Optional[str] = None,
                 continuous_learning: bool = True):
        """
        Initialize the running analysis pipeline.
        
        Args:
            athlete_mass: Athlete mass in kg
            athlete_height: Athlete height in meters
            athlete_age: Athlete age in years
            athlete_gender: Athlete gender ('male' or 'female')
            output_dir: Directory to store output files
            weather_api_key: OpenWeatherMap API key (optional)
            continuous_learning: Enable continuous learning (default: True)
        """
        self.athlete_mass = athlete_mass
        self.athlete_height = athlete_height
        self.athlete_age = athlete_age
        self.athlete_gender = athlete_gender
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weather_api_key = weather_api_key
        self.continuous_learning = continuous_learning
        
        # Initialize analyzers
        logger.info("Initializing analysis components...")
        
        # Cardiography analyzers
        try:
            self.heart_rate_processor = HeartRateProcessor()
            self.ml_signal_manager = MLSignalManager(model_dir=str(self.output_dir / "models"))
        except Exception as e:
            logger.error(f"Failed to initialize cardiography analyzers: {e}", exc_info=True)
            print(f"ERROR initializing cardiography analyzers: {str(e)}")
        
        # Actigraphy analyzers
        try:
            self.posture_analyzer = PostureAnalyzer(athlete_mass, athlete_height)
        except Exception as e:
            logger.error(f"Failed to initialize posture analyzer: {e}", exc_info=True)
            print(f"ERROR initializing posture analyzer: {str(e)}")
            self.posture_analyzer = None
        
        # Atmospheric analyzers
        if weather_api_key:
            try:
                self.aero_calculator = AerodynamicsCalculator(
                    athlete_mass, athlete_height, weather_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize aerodynamics calculator: {e}", exc_info=True)
                print(f"ERROR initializing aerodynamics calculator: {str(e)}")
                self.aero_calculator = None
        else:
            logger.warning("No weather API key provided. Aerodynamic analysis will be limited.")
            print("WARNING: No weather API key provided. Aerodynamic analysis will be limited.")
            self.aero_calculator = None
        
        # Biomechanics analyzers
        try:
            self.sprint_dynamics = SprintDynamics(athlete_mass, athlete_height)
            self.surface_analyzer = SurfaceDynamicsAnalyzer(athlete_mass)
        except Exception as e:
            logger.error(f"Failed to initialize biomechanics analyzers: {e}", exc_info=True)
            print(f"ERROR initializing biomechanics analyzers: {str(e)}")
        
        # Federation components
        try:
            self.data_fusion = DataFusion()
        except Exception as e:
            logger.error(f"Failed to initialize data fusion: {e}", exc_info=True)
            print(f"ERROR initializing data fusion: {str(e)}")
        
        # Initialize continuous learning if enabled
        if continuous_learning:
            try:
                from federation.continuous_learning import continuous_learning_manager
                logger.info("Continuous learning enabled")
            except Exception as e:
                logger.error(f"Failed to initialize continuous learning: {e}", exc_info=True)
                print(f"ERROR initializing continuous learning: {str(e)}")
                self.continuous_learning = False
        
        # Data storage
        self.fused_data = None
        self.fused_datapoints = None  # Store actual DataPoint objects
        self.posture_states = None
        self.power_metrics = None
        self.aero_results = None
        self.oxygen_results = None
        self.cardio_results = None
        
    def load_and_fuse_data(self, track_file: str, coros_file: str) -> List[Dict]:
        """
        Load and fuse data from track and coros files using federation components.
        
        Args:
            track_file: Path to track.json file (Garmin)
            coros_file: Path to coros.json file
            
        Returns:
            List of fused data points
        """
        try:
            logger.info(f"Loading and fusing data from {track_file} and {coros_file}")
            
            # Load data into federation system
            self.data_fusion.load_device_data("garmin", track_file)
            self.data_fusion.load_device_data("coros", coros_file)
            
            # Train local models for each device
            logger.info("Training local models for each device")
            self.data_fusion.train_local_models("garmin")
            self.data_fusion.train_local_models("coros")
            
            # Aggregate global model
            logger.info("Aggregating global model")
            self.data_fusion.state.aggregate_global_model()
            
            # Fuse measurements from both devices
            logger.info("Fusing measurements from both devices")
            self.fused_datapoints = self.data_fusion.fuse_measurements("garmin", "coros")
            
            # If no fused datapoints were created, fall back to using just Garmin data
            if not self.fused_datapoints:
                logger.warning("No fused datapoints created. Falling back to Garmin data only.")
                self.fused_datapoints = self.data_fusion.devices["garmin"]
            
            # Convert DataPoint objects to dictionaries for JSON serialization
            self.fused_data = []
            for point in self.fused_datapoints:
                self.fused_data.append({
                    'timestamp': point.timestamp.isoformat(),
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'altitude': point.altitude,
                    'heart_rate': point.heart_rate,
                    'cadence': point.cadence,
                    'speed': point.speed,
                    'stance_time': point.stance_time,
                    'stance_time_balance': point.stance_time_balance,
                    'step_length': point.step_length,
                    'vertical_ratio': point.vertical_ratio,
                    'vertical_oscillation': point.vertical_oscillation,
                    'power': point.power,
                    'form_power': point.form_power,
                    'accumulated_power': point.accumulated_power
                })
            
            # Save fused data
            fused_data_file = self.output_dir / "fused_data.json"
            with open(fused_data_file, "w") as f:
                json.dump(self.fused_data, f, indent=2)
            logger.info(f"Saved fused data to {fused_data_file}")
            
            return self.fused_data
        except Exception as e:
            logger.error(f"Error in data fusion: {e}")
            # Load raw data as fallback
            try:
                with open(track_file, 'r') as f:
                    self.fused_data = json.load(f)
                logger.info("Using Garmin data as fallback")
                return self.fused_data
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return []
    
    def analyze_heart_rate(self, track_file: str) -> Dict:
        """
        Analyze heart rate data using the HeartRateProcessor.
        
        Args:
            track_file: Path to track.json file
            
        Returns:
            Dictionary with heart rate analysis results
        """
        try:
            logger.info("Analyzing heart rate data...")
            
            # Load track data
            with open(track_file, 'r') as f:
                track_data = json.load(f)
            
            # Load heart rate data
            self.heart_rate_processor.load_from_geojson(track_data)
            
            # Perform heart rate analysis
            hrv_metrics = {}
            anomalies = []
            freq_analysis = {"frequencies": [], "power": []}
            time_features = {}
            
            try:
                hrv_metrics = self.heart_rate_processor.get_heart_rate_variability()
            except Exception as e:
                logger.warning(f"HRV analysis failed: {e}")
                
            try:
                anomalies = self.heart_rate_processor.detect_anomalies()
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
                
            try:
                freq_analysis = self.heart_rate_processor.get_frequency_analysis()
                # Convert numpy arrays to lists for JSON serialization
                freq_analysis = {
                    "frequencies": freq_analysis["frequencies"].tolist() if hasattr(freq_analysis["frequencies"], "tolist") else [],
                    "power": freq_analysis["power"].tolist() if hasattr(freq_analysis["power"], "tolist") else []
                }
            except Exception as e:
                logger.warning(f"Frequency analysis failed: {e}")
                
            try:
                time_features = self.heart_rate_processor.get_time_domain_features()
            except Exception as e:
                logger.warning(f"Time domain analysis failed: {e}")
            
            # Combine results
            self.cardio_results = {
                "hrv_metrics": hrv_metrics,
                "anomalies": anomalies,
                "frequency_analysis": freq_analysis,
                "time_features": time_features
            }
            
            # Save cardio analysis results
            cardio_file = self.output_dir / "cardio_analysis.json"
            with open(cardio_file, "w") as f:
                json.dump(self.cardio_results, f, indent=2)
            logger.info(f"Saved cardio analysis to {cardio_file}")
            
            return self.cardio_results
        except Exception as e:
            logger.error(f"Heart rate analysis failed: {e}")
            return {}
    
    def analyze_posture(self, track_file: str, coros_file: str):
        """
        Analyze posture using the PostureAnalyzer.
        
        Args:
            track_file: Path to track.json file
            coros_file: Path to coros.json file
        """
        try:
            logger.info("Analyzing posture...")
            
            # Verify files exist and are readable
            if not os.path.exists(track_file):
                print(f"FILE MISSING: {track_file} does not exist")
                logger.error(f"File does not exist: {track_file}")
                return {}
                
            if not os.path.exists(coros_file):
                print(f"FILE MISSING: {coros_file} does not exist")
                logger.error(f"File does not exist: {coros_file}")
                return {}
            
            # Try to load content to check format
            try:
                with open(track_file, 'r') as f:
                    track_data = json.load(f)
                with open(coros_file, 'r') as f:
                    coros_data = json.load(f)
                print(f"Successfully loaded track file with {len(track_data) if isinstance(track_data, list) else 'non-list'} entries")
                print(f"Successfully loaded coros file with {len(coros_data) if isinstance(coros_data, list) else 'non-list'} entries")
            except json.JSONDecodeError:
                print(f"ERROR: One of the input files contains invalid JSON")
                logger.error("Input files contain invalid JSON")
                return {}
                
            # Check if the posture analyzer was initialized
            if not hasattr(self, 'posture_analyzer') or self.posture_analyzer is None:
                print("ERROR: Posture analyzer not initialized")
                logger.error("Posture analyzer not initialized")
                return {}
            
            # Analyze posture
            print("Analyzing posture using track and coros data...")
            self.posture_states = self.posture_analyzer.analyze_posture(coros_file, track_file)
            
            # Convert to serializable format
            posture_results = []
            if self.posture_states:
                for state in self.posture_states:
                    posture_results.append({
                        "timestamp": state.timestamp,
                        "joint_angles": state.joint_angles,
                        "confidence": state.confidence,
                        "ground_contact": state.ground_contact,
                        "phase": state.phase
                    })
            
            # Save posture analysis results
            posture_file = self.output_dir / "posture_analysis.json"
            with open(posture_file, "w") as f:
                json.dump(posture_results, f, indent=2)
            logger.info(f"Saved posture analysis to {posture_file}")
            
            # Check if continuous learning models were used or saved
            models_dir = self.output_dir / "models"
            if models_dir.exists():
                model_files = list(models_dir.glob("posture_*.pt")) + list(models_dir.glob("posture_*.joblib"))
                if model_files:
                    logger.info(f"Found {len(model_files)} posture model files")
                    print(f"Posture models: {[f.name for f in model_files]}")
                else:
                    logger.info("No posture model files found")
            else:
                logger.info("Models directory does not exist yet")
            
            return posture_results
            
        except Exception as e:
            print(f"POSTURE ANALYSIS ERROR: {str(e)}")
            logger.error(f"Posture analysis failed: {e}", exc_info=True)
            return {}
    
    def analyze_power(self, coros_file: str):
        """
        Analyze power metrics using the PowerAnalyzer.
        
        Args:
            coros_file: Path to coros.json file
        """
        try:
            logger.info("Analyzing power metrics...")
            power_analyzer = PowerAnalyzer(coros_file)
            self.power_metrics = power_analyzer.analyze_power_metrics()
            
            # Detect activity patterns
            activity_patterns = {}
            try:
                activity_patterns = power_analyzer.detect_activity_patterns()
            except Exception as e:
                logger.warning(f"Activity pattern detection failed: {e}", exc_info=True)
            
            # Save power analysis results
            power_results = {
                "mean_power": getattr(self.power_metrics, 'mean_power', 0),
                "peak_power": getattr(self.power_metrics, 'peak_power', 0),
                "power_zones": {k: list(v) for k, v in getattr(self.power_metrics, 'power_zones', {}).items()},
                "power_distribution": getattr(self.power_metrics, 'power_distribution', {}),
                "accumulated_power_rate": float(getattr(self.power_metrics, 'accumulated_power_rate', 0)),
                "form_power_efficiency": getattr(self.power_metrics, 'form_power_efficiency', 0),
                "activity_patterns": {k: len(v) for k, v in activity_patterns.items()}
            }
            
            power_file = self.output_dir / "power_analysis.json"
            with open(power_file, "w") as f:
                json.dump(power_results, f, indent=2)
            logger.info(f"Saved power analysis to {power_file}")
            
            return power_results
        except Exception as e:
            logger.error(f"Power analysis failed: {e}", exc_info=True)
            return {}
    
    def analyze_aerodynamics(self, track_file: str):
        """
        Analyze aerodynamic effects using the AerodynamicsCalculator.
        
        Args:
            track_file: Path to track.json file
        """
        try:
            logger.info("Analyzing aerodynamic effects...")
            
            # Verify file exists
            if not os.path.exists(track_file):
                print(f"FILE MISSING: {track_file} does not exist")
                logger.error(f"File does not exist: {track_file}")
                return {}
                
            # Try to load content to check format 
            try:
                with open(track_file, 'r') as f:
                    track_data = json.load(f)
                print(f"Successfully loaded track file with {len(track_data) if isinstance(track_data, list) else 'non-list'} entries")
            except json.JSONDecodeError:
                print(f"ERROR: Track file {track_file} contains invalid JSON")
                logger.error(f"Track file contains invalid JSON: {track_file}")
                return {}
            
            # Check if the aero calculator was initialized
            if not hasattr(self, 'aero_calculator') or self.aero_calculator is None:
                print("ERROR: Aerodynamics calculator not initialized")
                logger.error("Aerodynamics calculator not initialized")
                return {}
            
            # Test weather API connection
            print("Testing weather API connection...")
            if not self.aero_calculator.test_api_connection():
                print("WARNING: Weather API connection failed. Using fallback weather data.")
                logger.warning("Weather API connection failed. Using fallback data.")
            
            # Analyze the track
            self.aero_results = self.aero_calculator.analyze_track(track_data)
            
            # Save aerodynamics analysis results
            aero_file = self.output_dir / "aerodynamics_analysis.json"
            with open(aero_file, "w") as f:
                json.dump(self.aero_results, f, indent=2)
            logger.info(f"Saved aerodynamics analysis to {aero_file}")
            
            # Check if continuous learning models were used or saved
            models_dir = self.output_dir / "models"
            if models_dir.exists():
                model_files = list(models_dir.glob("aerodynamics_*.joblib"))
                if model_files:
                    logger.info(f"Found {len(model_files)} aerodynamics model files")
                    print(f"Aerodynamics models: {[f.name for f in model_files]}")
                else:
                    logger.info("No aerodynamics model files found")
            else:
                logger.info("Models directory does not exist yet")
            
            return self.aero_results
            
        except Exception as e:
            print(f"AERODYNAMICS ANALYSIS ERROR: {str(e)}")
            logger.error(f"Aerodynamics analysis failed: {e}", exc_info=True)
            return {}
    
    def analyze_oxygen_uptake(self, track_file: str):
        """
        Analyze oxygen uptake using the OxygenUptakeCalculator.
        
        Args:
            track_file: Path to track.json file
        """
        try:
            logger.info("Analyzing oxygen uptake...")
            
            # Verify file exists and is readable
            if not os.path.exists(track_file):
                print(f"FILE MISSING: {track_file} does not exist")
                logger.error(f"File does not exist: {track_file}")
                return {}
                
            # Try to load content to check format
            try:
                with open(track_file, 'r') as f:
                    track_data = json.load(f)
                print(f"Successfully loaded track file with {len(track_data) if isinstance(track_data, list) else 'non-list'} entries")
            except json.JSONDecodeError:
                print(f"ERROR: Track file {track_file} contains invalid JSON")
                logger.error(f"Track file contains invalid JSON: {track_file}")
                return {}
            
            # Check if the athlete parameters are valid
            if not all([self.athlete_mass > 0, self.athlete_height > 0, self.athlete_age > 0, 
                      self.athlete_gender in ['male', 'female']]):
                print(f"INVALID ATHLETE PARAMETERS: mass={self.athlete_mass}, height={self.athlete_height}, "
                      f"age={self.athlete_age}, gender={self.athlete_gender}")
                logger.error("Invalid athlete parameters for oxygen uptake calculation")
                return {}
            
            # Initialize oxygen uptake calculator
            print("Initializing OxygenUptakeCalculator...")
            try:
                print(f"Creating OxygenUptakeCalculator with parameters: mass={self.athlete_mass}, height={self.athlete_height}, "
                      f"age={self.athlete_age}, gender={self.athlete_gender}")
                
                # Create the calculator with proper parameters
                oxygen_calculator = OxygenUptakeCalculator(
                    mass=self.athlete_mass, 
                    height=self.athlete_height,
                    age=self.athlete_age,
                    gender=self.athlete_gender
                )
                
                print("OxygenUptakeCalculator initialized successfully")
            except Exception as init_error:
                print(f"OXYGEN CALCULATOR INITIALIZATION ERROR: {str(init_error)}")
                logger.error(f"Oxygen calculator initialization failed: {init_error}", exc_info=True)
                return {}
            
            # Now attempt to analyze oxygen uptake
            print("Attempting to analyze oxygen uptake...")
            self.oxygen_results = oxygen_calculator.analyze_track(track_data)
            
            # Save oxygen analysis results
            oxygen_file = self.output_dir / "oxygen_analysis.json"
            with open(oxygen_file, "w") as f:
                json.dump(self.oxygen_results, f, indent=2)
            logger.info(f"Saved oxygen analysis to {oxygen_file}")
            
            # Check if continuous learning models were used or saved
            models_dir = self.output_dir / "models"
            if models_dir.exists():
                model_files = list(models_dir.glob("oxygen_*.joblib"))
                if model_files:
                    logger.info(f"Found {len(model_files)} oxygen uptake model files")
                    print(f"Oxygen uptake models: {[f.name for f in model_files]}")
                else:
                    logger.info("No oxygen uptake model files found")
            else:
                logger.info("Models directory does not exist yet")
            
            return self.oxygen_results
        except ImportError as e:
            print(f"IMPORT ERROR: Missing required package for oxygen uptake analysis: {e}")
            logger.error(f"Missing required package for oxygen uptake analysis: {e}", exc_info=True)
            return {}
        except Exception as e:
            print(f"OXYGEN UPTAKE ANALYSIS ERROR: {str(e)}")
            logger.error(f"Oxygen uptake analysis failed: {e}", exc_info=True)
            return {}
    
    def generate_visualizations(self):
        """
        Generate visualizations from analysis results.
        
        Returns:
            Dictionary with visualization file paths
        """
        try:
            logger.info("Generating visualizations...")
            
            # Create visualizations directory
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            results = {}
            
            # Generate CZML if we have fused data
            if self.fused_data:
                try:
                    start_time = datetime.fromisoformat(self.fused_data[0]['timestamp'].replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(self.fused_data[-1]['timestamp'].replace('Z', '+00:00'))
                    
                    czml_content = generate_czml(self.fused_data, start_time, end_time)
                    czml_file = viz_dir / "runner_track.czml"
                    
                    with open(czml_file, "w") as f:
                        f.write(czml_content)
                    logger.info(f"Saved CZML visualization to {czml_file}")
                    
                    results["czml_file"] = str(czml_file)
                except Exception as e:
                    logger.warning(f"CZML generation failed: {e}")
            
            # If posture analysis was done, visualize posture sequence
            if self.posture_states:
                try:
                    self.posture_analyzer.visualize_posture_sequence(self.posture_states)
                    logger.info("Generated posture sequence visualization")
                    results["posture_sequence"] = str(self.output_dir / "posture_sequence.json")
                except Exception as e:
                    logger.warning(f"Posture visualization failed: {e}")
            
            return results
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {}
    
    def run_pipeline(self, track_file: str, coros_file: str) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            track_file: Path to track.json file (Garmin)
            coros_file: Path to coros.json file
            
        Returns:
            Dictionary with analysis results
        """
        # Create output directory structure
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique run ID based on timestamp and file names
        import hashlib
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        files_hash = hashlib.md5((track_file + coros_file).encode()).hexdigest()[:8]
        run_id = f"run_{timestamp.replace(':', '-')}_{files_hash}"
        
        # Print header
        print("\n" + "="*50)
        print(f"Starting analysis run: {run_id}")
        print(f"Output directory: {self.output_dir}")
        print("="*50 + "\n")
        
        # Step 1: Load and fuse data
        self.load_and_fuse_data(track_file, coros_file)
        
        # Step 2: Analyze heart rate data
        print("\n[1/6] Running heart rate analysis...")
        cardio_results = self.analyze_heart_rate(track_file)
        print(f"Heart rate analysis {'completed successfully' if cardio_results else 'failed'}")
        
        # Step 3: Analyze posture
        print("\n[2/6] Running posture analysis...")
        posture_results = self.analyze_posture(track_file, coros_file)
        print(f"Posture analysis {'completed successfully' if posture_results else 'failed'}")
        
        # Step 4: Analyze power metrics
        print("\n[3/6] Running power analysis...")
        power_results = self.analyze_power(coros_file)
        print(f"Power analysis {'completed successfully' if power_results else 'failed'}")
        
        # Step 5: Analyze aerodynamics if weather API key is available
        print("\n[4/6] Running aerodynamics analysis...")
        if self.weather_api_key:
            aero_results = self.analyze_aerodynamics(track_file)
            print(f"Aerodynamics analysis {'completed successfully' if aero_results else 'failed'}")
        else:
            print("Skipping aerodynamics analysis: No weather API key provided")
            aero_results = None
        
        # Step 6: Analyze oxygen uptake
        print("\n[5/6] Running oxygen uptake analysis...")
        oxygen_results = self.analyze_oxygen_uptake(track_file)
        print(f"Oxygen uptake analysis {'completed successfully' if oxygen_results else 'failed'}")
        
        # Step 7: Generate visualizations
        print("\n[6/6] Generating visualizations...")
        viz_results = self.generate_visualizations()
        print(f"Visualization generation {'completed successfully' if viz_results else 'failed'}")
        
        # Compile complete results
        complete_results = {
            "output_dir": str(self.output_dir),
            "cardio_analysis": cardio_results,
            "posture_analysis": posture_results,
            "power_analysis": power_results,
            "aerodynamics_analysis": aero_results,
            "oxygen_analysis": oxygen_results,
            "fused_data": self.fused_data
        }
        
        # Save complete results to a single JSON file
        complete_results_file = self.output_dir / "complete_analysis.json"
        with open(complete_results_file, "w") as f:
            json.dump(complete_results, f, indent=2)
        logger.info(f"Saved complete analysis results to {complete_results_file}")
        
        # Store results for continuous learning
        if hasattr(self, 'continuous_learning') and self.continuous_learning:
            try:
                print("\nStoring data for continuous learning...")
                from federation.continuous_learning import continuous_learning_manager
                
                # Prepare metadata for the run
                metadata = {
                    "timestamp": timestamp,
                    "athlete_mass": self.athlete_mass,
                    "athlete_height": self.athlete_height,
                    "athlete_age": self.athlete_age,
                    "athlete_gender": self.athlete_gender,
                    "track_file": track_file,
                    "coros_file": coros_file
                }
                
                # Store the run data for future model training
                continuous_learning_manager.store_run_results(run_id, complete_results, metadata)
                logger.info(f"Stored run data for continuous learning with ID: {run_id}")
                print(f"Stored run data with ID: {run_id}")
                
                # Check if any models should be trained immediately
                if any([
                    bool(posture_results), 
                    bool(aero_results), 
                    bool(oxygen_results)
                ]):
                    print("Checking if models need immediate retraining...")
                    learning_manager = continuous_learning_manager
                    
                    # Get the number of runs we have stored
                    recent_runs = learning_manager.data_collector.get_recent_runs()
                    runs_count = len(recent_runs)
                    
                    # Decide to train initial models if we have enough data
                    if runs_count >= 3:
                        print(f"Found {runs_count} runs, considering immediate model training...")
                        
                        # Check which models to train
                        if bool(posture_results) and not list(models_dir.glob("posture_*")):
                            print("Training initial posture model...")
                            try:
                                learning_manager.train_model("posture_joint_angles", "posture")
                                print("Initial posture model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training initial posture model: {e}", exc_info=True)
                                print(f"Error training posture model: {str(e)}")
                                
                        if bool(aero_results) and not list(models_dir.glob("aerodynamics_*")):
                            print("Training initial aerodynamics model...")
                            try:
                                learning_manager.train_model("aerodynamics_drag", "aerodynamics")
                                print("Initial aerodynamics model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training initial aerodynamics model: {e}", exc_info=True)
                                print(f"Error training aerodynamics model: {str(e)}")
                                
                        if bool(oxygen_results) and not list(models_dir.glob("oxygen_*")):
                            print("Training initial oxygen uptake model...")
                            try:
                                learning_manager.train_model("oxygen_uptake", "oxygen")
                                print("Initial oxygen uptake model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training initial oxygen model: {e}", exc_info=True)
                                print(f"Error training oxygen model: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to store data for continuous learning: {e}", exc_info=True)
                print(f"Error storing data for continuous learning: {str(e)}")
        
        # Return summary of results
        return {
            "output_dir": str(self.output_dir),
            "cardio_analysis": bool(cardio_results),
            "posture_analysis": bool(posture_results),
            "power_analysis": bool(power_results),
            "aerodynamics_analysis": bool(aero_results),
            "oxygen_analysis": bool(oxygen_results),
            "visualizations": viz_results,
            "run_id": run_id
        }

def main():
    """Main entry point for the running analysis pipeline."""
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent
    
    # Hardcoded configuration values with proper paths
    track_file = project_root / "public" / "track.json"
    coros_file = project_root / "public" / "coros.json"
    athlete_mass = 79.0  # kg
    athlete_height = 1.80  # meters
    athlete_age = 33  # years
    athlete_gender = "male"  # 'male' or 'female'
    output_dir = project_root / "output"
    weather_api_key = "ae9af9bb6224315e875922b1e22261b5"
    continuous_learning = True  # Enable continuous learning
    
    # Check if required files exist
    if not track_file.exists():
        logger.error(f"Track file not found: {track_file}")
        print(f"ERROR: Track file not found: {track_file}")
        return
    
    if not coros_file.exists():
        logger.error(f"Coros file not found: {coros_file}")
        print(f"ERROR: Coros file not found: {coros_file}")
        return
    
    # Initialize pipeline
    pipeline = RunningAnalysisPipeline(
        athlete_mass=athlete_mass,
        athlete_height=athlete_height,
        athlete_age=athlete_age,
        athlete_gender=athlete_gender,
        output_dir=str(output_dir),
        weather_api_key=weather_api_key,
        continuous_learning=continuous_learning
    )
    
    # Run pipeline
    print("\nRunning analysis pipeline...")
    results = pipeline.run_pipeline(str(track_file), str(coros_file))
    
    # Print summary
    print("\n=== Running Analysis Complete ===")
    print(f"Results saved to: {results['output_dir']}")
    print(f"Run ID: {results.get('run_id', 'N/A')}")
    print("\nAnalyses performed:")
    print(f"- Cardio analysis: {'✓' if results['cardio_analysis'] else '✗'}")
    print(f"- Posture analysis: {'✓' if results['posture_analysis'] else '✗'}")
    print(f"- Power analysis: {'✓' if results['power_analysis'] else '✗'}")
    print(f"- Aerodynamics analysis: {'✓' if results['aerodynamics_analysis'] else '✗'}")
    print(f"- Oxygen uptake analysis: {'✓' if results['oxygen_analysis'] else '✗'}")
    
    if results.get('visualizations'):
        print("\nVisualizations:")
        for viz_type, path in results['visualizations'].items():
            print(f"- {viz_type}: {path}")
    
    # Report on failures
    if not all([results['cardio_analysis'], results['posture_analysis'], results['power_analysis'], 
                results['aerodynamics_analysis'], results['oxygen_analysis']]):
        print("\nSome analyses failed. Check the log file for details.")
        print("Common issues:")
        print("1. Missing or invalid input files (track.json, coros.json)")
        print("2. Required Python packages not installed")
        print("3. Weather API key invalid or expired")
        print("4. Input data format issues")
        
    # Report on continuous learning if enabled
    if continuous_learning:
        try:
            from federation.continuous_learning import continuous_learning_manager
            
            # Check for model files
            models_dir = Path(output_dir) / "models"
            model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pt"))
            
            if model_files:
                print(f"\nFound {len(model_files)} model files:")
                for model_file in model_files:
                    print(f"- {model_file.name}")
            else:
                print("\nNo model files found. Initial models will be created after collecting more runs.")
            
            # Get model performance info
            model_performances = continuous_learning_manager.get_model_performances()
            
            if model_performances:
                print("\nModel Performance Summary:")
                for model_name, perf in model_performances.items():
                    print(f"- {model_name}: {perf['trend']} trend, "
                          f"RMSE: {perf['latest_metrics'].get('rmse', 'N/A')}")
            
            # Get retraining status
            retraining_status = continuous_learning_manager.get_retraining_status()
            
            if retraining_status:
                print("\nModel Retraining Status:")
                for model, active in retraining_status.items():
                    print(f"- {model}: {'Active' if active else 'Inactive'}")
        except Exception as e:
            print(f"\nError retrieving continuous learning status: {str(e)}")
            
    # List required packages
    print("\nRequired packages:")
    print("- numpy, pandas: For data manipulation")
    print("- scikit-learn: For machine learning models")
    print("- matplotlib, seaborn: For visualizations")
    print("- torch: For deep learning models (posture analyzer)")
    print("- requests: For OpenWeatherMap API access")
    print("- joblib: For model serialization")
    print("- scipy: For signal processing")

if __name__ == "__main__":
    main()