"""
Bayesian Evidence Network Analysis Pipeline with Fuzzy Logic

This module transforms traditional GPS trajectory analysis into a Bayesian evidence 
network where each analysis becomes an optimization of an objective function updated 
using fuzzy evidence.

Key Concepts:
- Each GPS point becomes fuzzy evidence in a Bayesian network
- Analysis steps (filtering, triangulation, etc.) become network nodes
- Confidence scores become fuzzy membership functions
- The entire analysis optimizes a multi-objective function
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from .rust_bridge import (
        sighthound_bayesian_available, 
        sighthound_fuzzy_available,
        get_rust_capabilities
    )
    if sighthound_bayesian_available():
        import sighthound_bayesian as rust_bayesian
    if sighthound_fuzzy_available():
        import sighthound_fuzzy as rust_fuzzy
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    rust_bayesian = None
    rust_fuzzy = None

from .data_loader import GpsDataLoader
from .dynamic_filtering import KalmanFilter
from ..utils.triangulation import CellTowerTriangulation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisEvidence:
    """Represents evidence from an analysis step"""
    source: str
    variable_name: str
    crisp_value: float
    confidence: float
    fuzzy_confidence: Dict[str, float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BayesianAnalysisResult:
    """Result from Bayesian evidence network analysis"""
    objective_value: float
    node_beliefs: Dict[str, Dict[str, float]]
    evidence_summary: Dict[str, Any]
    optimization_stats: Dict[str, float]
    fuzzy_assessment: Dict[str, float]
    quality_metrics: Dict[str, float]

class FuzzyMembershipFunction:
    """Python implementation of fuzzy membership functions"""
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function"""
        if x <= a or x >= d:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a)
        elif x <= c:
            return 1.0
        else:
            return (d - x) / (d - c)
    
    @staticmethod
    def gaussian(x: float, mean: float, sigma: float) -> float:
        """Gaussian membership function"""
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

class PythonFuzzySet:
    """Python fallback implementation of fuzzy sets"""
    
    def __init__(self, label: str, membership_func: str, params: Dict[str, float]):
        self.label = label
        self.membership_func = membership_func
        self.params = params
    
    def membership_degree(self, value: float) -> float:
        """Calculate membership degree for a value"""
        if self.membership_func == "triangular":
            return FuzzyMembershipFunction.triangular(
                value, self.params['a'], self.params['b'], self.params['c']
            )
        elif self.membership_func == "trapezoidal":
            return FuzzyMembershipFunction.trapezoidal(
                value, self.params['a'], self.params['b'], 
                self.params['c'], self.params['d']
            )
        elif self.membership_func == "gaussian":
            return FuzzyMembershipFunction.gaussian(
                value, self.params['mean'], self.params['sigma']
            )
        else:
            return 0.0

class PythonBayesianNode:
    """Python fallback implementation of Bayesian network nodes"""
    
    def __init__(self, node_id: str, node_type: str, variable_name: str):
        self.id = node_id
        self.node_type = node_type
        self.variable_name = variable_name
        self.current_belief = {"low": 0.33, "medium": 0.34, "high": 0.33}
        self.evidence_buffer = []
        self.last_update = 0.0
    
    def add_evidence(self, evidence: AnalysisEvidence):
        """Add evidence to the node"""
        self.evidence_buffer.append(evidence)
        self.last_update = datetime.now().timestamp()
    
    def update_belief(self):
        """Update belief based on accumulated evidence"""
        if not self.evidence_buffer:
            return
        
        # Simple belief update using evidence aggregation
        total_confidence = sum(ev.confidence for ev in self.evidence_buffer)
        if total_confidence > 0:
            # Weight evidence by confidence
            weighted_beliefs = {"low": 0.0, "medium": 0.0, "high": 0.0}
            
            for evidence in self.evidence_buffer:
                weight = evidence.confidence / total_confidence
                for state, fuzzy_conf in evidence.fuzzy_confidence.items():
                    if state in weighted_beliefs:
                        weighted_beliefs[state] += weight * fuzzy_conf
            
            # Normalize
            total = sum(weighted_beliefs.values())
            if total > 0:
                for state in weighted_beliefs:
                    weighted_beliefs[state] /= total
                self.current_belief = weighted_beliefs
        
        self.evidence_buffer.clear()
    
    def belief_entropy(self) -> float:
        """Calculate entropy of current belief"""
        entropy = 0.0
        for prob in self.current_belief.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def map_state(self) -> Tuple[str, float]:
        """Get maximum a posteriori state"""
        best_state = max(self.current_belief.items(), key=lambda x: x[1])
        return best_state

class BayesianAnalysisPipeline:
    """
    Transforms GPS trajectory analysis into a Bayesian Evidence Network
    with fuzzy logic optimization
    """
    
    def __init__(self, 
                 use_rust: bool = True,
                 optimization_iterations: int = 100,
                 population_size: int = 30):
        """
        Initialize the Bayesian Analysis Pipeline
        
        Args:
            use_rust: Whether to use Rust implementations when available
            optimization_iterations: Number of optimization iterations
            population_size: Population size for genetic algorithm
        """
        self.use_rust = use_rust and RUST_AVAILABLE
        self.optimization_iterations = optimization_iterations
        self.population_size = population_size
        
        # Initialize components
        self.gps_loader = GpsDataLoader()
        self.kalman_filter = KalmanFilter()
        self.triangulation = CellTowerTriangulation()
        
        # Network components
        self.nodes = {}
        self.evidence_history = []
        self.objective_weights = {
            'trajectory_smoothness': 0.25,
            'evidence_consistency': 0.25,
            'confidence_maximization': 0.20,
            'uncertainty_minimization': 0.15,
            'temporal_coherence': 0.15
        }
        
        # Fuzzy sets for confidence assessment
        self.confidence_fuzzy_sets = self._create_confidence_fuzzy_sets()
        
        logger.info(f"Bayesian Analysis Pipeline initialized (Rust: {self.use_rust})")
    
    def _create_confidence_fuzzy_sets(self) -> Dict[str, PythonFuzzySet]:
        """Create fuzzy sets for confidence assessment"""
        return {
            'very_low': PythonFuzzySet('very_low', 'trapezoidal', 
                                     {'a': 0.0, 'b': 0.0, 'c': 0.2, 'd': 0.4}),
            'low': PythonFuzzySet('low', 'triangular', 
                                {'a': 0.2, 'b': 0.4, 'c': 0.6}),
            'medium': PythonFuzzySet('medium', 'triangular', 
                                   {'a': 0.4, 'b': 0.6, 'c': 0.8}),
            'high': PythonFuzzySet('high', 'triangular', 
                                 {'a': 0.6, 'b': 0.8, 'c': 1.0}),
            'very_high': PythonFuzzySet('very_high', 'trapezoidal', 
                                      {'a': 0.8, 'b': 0.9, 'c': 1.0, 'd': 1.0}),
        }
    
    def create_fuzzy_evidence(self, 
                            source: str, 
                            variable_name: str, 
                            crisp_value: float, 
                            confidence: float,
                            timestamp: Optional[float] = None) -> AnalysisEvidence:
        """Create fuzzy evidence from crisp analysis results"""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # Calculate fuzzy confidence using membership functions
        fuzzy_confidence = {}
        for label, fuzzy_set in self.confidence_fuzzy_sets.items():
            fuzzy_confidence[label] = fuzzy_set.membership_degree(confidence)
        
        return AnalysisEvidence(
            source=source,
            variable_name=variable_name,
            crisp_value=crisp_value,
            confidence=confidence,
            fuzzy_confidence=fuzzy_confidence,
            timestamp=timestamp
        )
    
    def add_analysis_node(self, 
                         node_id: str, 
                         node_type: str, 
                         variable_name: str) -> str:
        """Add a node to the Bayesian network"""
        if self.use_rust and rust_bayesian:
            # Use Rust implementation
            node = rust_bayesian.BayesianNode(node_id, node_type, variable_name)
        else:
            # Use Python fallback
            node = PythonBayesianNode(node_id, node_type, variable_name)
        
        self.nodes[node_id] = node
        logger.debug(f"Added analysis node: {node_id} ({variable_name})")
        return node_id
    
    def process_gps_trajectory(self, trajectory_data: np.ndarray) -> BayesianAnalysisResult:
        """
        Process GPS trajectory through Bayesian Evidence Network
        
        Args:
            trajectory_data: GPS trajectory as numpy array [lat, lon, timestamp, confidence]
            
        Returns:
            BayesianAnalysisResult with optimization results
        """
        logger.info(f"Processing trajectory with {len(trajectory_data)} points")
        
        # Ensure required nodes exist
        self._ensure_analysis_nodes()
        
        # Convert trajectory to evidence
        evidence_list = self._trajectory_to_evidence(trajectory_data)
        
        # Distribute evidence to nodes
        self._distribute_evidence(evidence_list)
        
        # Perform belief propagation
        self._belief_propagation()
        
        # Optimize objective function
        optimization_result = self._optimize_objective()
        
        # Collect results
        result = BayesianAnalysisResult(
            objective_value=optimization_result.get('final_objective', 0.0),
            node_beliefs=self._collect_node_beliefs(),
            evidence_summary=self._summarize_evidence(evidence_list),
            optimization_stats=optimization_result,
            fuzzy_assessment=self._fuzzy_quality_assessment(),
            quality_metrics=self._calculate_quality_metrics()
        )
        
        logger.info(f"Analysis complete. Objective value: {result.objective_value:.4f}")
        return result
    
    def _ensure_analysis_nodes(self):
        """Ensure all required analysis nodes exist"""
        required_nodes = [
            ('position_reliability', 'latent', 'position_reliability'),
            ('trajectory_smoothness', 'latent', 'trajectory_smoothness'),
            ('gps_confidence', 'observed', 'gps_confidence'),
            ('kalman_estimate', 'latent', 'kalman_estimate'),
            ('triangulation_result', 'latent', 'triangulation_result'),
            ('velocity_consistency', 'latent', 'velocity_consistency'),
            ('temporal_coherence', 'latent', 'temporal_coherence'),
        ]
        
        for node_id, node_type, variable_name in required_nodes:
            if node_id not in self.nodes:
                self.add_analysis_node(node_id, node_type, variable_name)
    
    def _trajectory_to_evidence(self, trajectory_data: np.ndarray) -> List[AnalysisEvidence]:
        """Convert GPS trajectory points to fuzzy evidence"""
        evidence_list = []
        
        for i, point in enumerate(trajectory_data):
            lat, lon, timestamp, confidence = point[:4]
            
            # GPS position evidence
            pos_evidence = self.create_fuzzy_evidence(
                source='gps',
                variable_name='position',
                crisp_value=np.sqrt(lat**2 + lon**2),  # Simple position magnitude
                confidence=confidence,
                timestamp=timestamp
            )
            evidence_list.append(pos_evidence)
            
            # Velocity evidence (if not first point)
            if i > 0:
                prev_point = trajectory_data[i-1]
                dt = timestamp - prev_point[2]
                if dt > 0:
                    dx = lat - prev_point[0]
                    dy = lon - prev_point[1]
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    
                    vel_evidence = self.create_fuzzy_evidence(
                        source='gps_derived',
                        variable_name='velocity',
                        crisp_value=velocity,
                        confidence=confidence * 0.8,  # Derived values have lower confidence
                        timestamp=timestamp
                    )
                    evidence_list.append(vel_evidence)
            
            # Apply Kalman filtering for improved estimates
            if hasattr(self.kalman_filter, 'predict_and_update'):
                kalman_state = self.kalman_filter.predict_and_update([lat, lon])
                kalman_evidence = self.create_fuzzy_evidence(
                    source='kalman_filter',
                    variable_name='filtered_position',
                    crisp_value=np.linalg.norm(kalman_state[:2]),
                    confidence=min(confidence * 1.2, 1.0),  # Kalman improves confidence
                    timestamp=timestamp
                )
                evidence_list.append(kalman_evidence)
        
        return evidence_list
    
    def _distribute_evidence(self, evidence_list: List[AnalysisEvidence]):
        """Distribute evidence to appropriate network nodes"""
        for evidence in evidence_list:
            # Route evidence to appropriate nodes based on variable name and source
            target_nodes = self._get_target_nodes(evidence)
            
            for node_id in target_nodes:
                if node_id in self.nodes:
                    if hasattr(self.nodes[node_id], 'add_evidence'):
                        self.nodes[node_id].add_evidence(evidence)
        
        self.evidence_history.extend(evidence_list)
    
    def _get_target_nodes(self, evidence: AnalysisEvidence) -> List[str]:
        """Determine which nodes should receive this evidence"""
        target_nodes = []
        
        if 'position' in evidence.variable_name:
            target_nodes.extend(['position_reliability', 'trajectory_smoothness'])
        
        if 'velocity' in evidence.variable_name:
            target_nodes.append('velocity_consistency')
        
        if evidence.source == 'gps':
            target_nodes.append('gps_confidence')
        
        if evidence.source == 'kalman_filter':
            target_nodes.append('kalman_estimate')
        
        return target_nodes
    
    def _belief_propagation(self):
        """Perform belief propagation through the network"""
        # Update all nodes with their accumulated evidence
        for node in self.nodes.values():
            if hasattr(node, 'update_belief'):
                node.update_belief()
        
        # Simple message passing (in practice would be more sophisticated)
        # For now, just ensure all nodes have been updated
        logger.debug("Belief propagation completed")
    
    def _optimize_objective(self) -> Dict[str, float]:
        """Optimize the objective function"""
        if self.use_rust and rust_fuzzy:
            # Use Rust fuzzy optimization
            try:
                trajectory_points = self._convert_evidence_to_gps_points()
                optimization_params = {
                    'population_size': float(self.population_size),
                    'max_generations': float(self.optimization_iterations)
                }
                return rust_fuzzy.optimize_trajectory_fuzzy_bayesian(
                    trajectory_points, optimization_params
                )
            except Exception as e:
                logger.warning(f"Rust optimization failed: {e}. Using Python fallback.")
        
        # Python fallback optimization
        return self._python_objective_optimization()
    
    def _python_objective_optimization(self) -> Dict[str, float]:
        """Python fallback for objective function optimization"""
        initial_value = self._calculate_objective_value()
        
        # Simple hill-climbing optimization
        best_value = initial_value
        best_config = self.objective_weights.copy()
        
        for iteration in range(self.optimization_iterations):
            # Small perturbation to weights
            current_config = best_config.copy()
            for key in current_config:
                perturbation = (np.random.random() - 0.5) * 0.1
                current_config[key] = max(0.0, min(1.0, current_config[key] + perturbation))
            
            # Normalize weights
            total_weight = sum(current_config.values())
            if total_weight > 0:
                for key in current_config:
                    current_config[key] /= total_weight
            
            # Evaluate with new configuration
            self.objective_weights = current_config
            current_value = self._calculate_objective_value()
            
            # Accept if better
            if current_value > best_value:
                best_value = current_value
                best_config = current_config.copy()
        
        # Restore best configuration
        self.objective_weights = best_config
        
        return {
            'initial_objective': initial_value,
            'final_objective': best_value,
            'improvement': best_value - initial_value,
            'iterations': self.optimization_iterations
        }
    
    def _calculate_objective_value(self) -> float:
        """Calculate current objective function value"""
        scores = {}
        
        # Trajectory smoothness
        smoothness_scores = []
        for node_id in ['trajectory_smoothness', 'velocity_consistency']:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if hasattr(node, 'belief_entropy'):
                    entropy = node.belief_entropy()
                    smoothness = 1.0 / (1.0 + entropy)
                    smoothness_scores.append(smoothness)
        scores['trajectory_smoothness'] = np.mean(smoothness_scores) if smoothness_scores else 0.5
        
        # Evidence consistency
        consistency_scores = []
        for node in self.nodes.values():
            if hasattr(node, 'map_state'):
                _, confidence = node.map_state()
                consistency_scores.append(confidence)
        scores['evidence_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.5
        
        # Confidence maximization
        confidence_scores = []
        for evidence in self.evidence_history[-50:]:  # Recent evidence
            confidence_scores.append(evidence.confidence)
        scores['confidence_maximization'] = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Uncertainty minimization
        entropy_scores = []
        for node in self.nodes.values():
            if hasattr(node, 'belief_entropy'):
                entropy = node.belief_entropy()
                uncertainty_score = 1.0 / (1.0 + entropy)
                entropy_scores.append(uncertainty_score)
        scores['uncertainty_minimization'] = np.mean(entropy_scores) if entropy_scores else 0.5
        
        # Temporal coherence
        scores['temporal_coherence'] = 0.7  # Simplified for now
        
        # Weighted combination
        total_score = sum(
            scores.get(component, 0.0) * weight 
            for component, weight in self.objective_weights.items()
        )
        
        return total_score
    
    def _convert_evidence_to_gps_points(self) -> List:
        """Convert evidence back to GPS point format for Rust functions"""
        gps_points = []
        
        position_evidence = [e for e in self.evidence_history if 'position' in e.variable_name]
        
        for evidence in position_evidence:
            # This is a simplified conversion - in practice would be more sophisticated
            gps_point = {
                'latitude': evidence.crisp_value * 0.1,  # Simplified
                'longitude': evidence.crisp_value * 0.1,  # Simplified  
                'timestamp': evidence.timestamp,
                'confidence': evidence.confidence
            }
            gps_points.append(gps_point)
        
        return gps_points
    
    def _collect_node_beliefs(self) -> Dict[str, Dict[str, float]]:
        """Collect current beliefs from all nodes"""
        beliefs = {}
        
        for node_id, node in self.nodes.items():
            if hasattr(node, 'current_belief'):
                beliefs[node_id] = node.current_belief.copy()
            elif hasattr(node, 'map_state'):
                state, confidence = node.map_state()
                beliefs[node_id] = {state: confidence}
        
        return beliefs
    
    def _summarize_evidence(self, evidence_list: List[AnalysisEvidence]) -> Dict[str, Any]:
        """Summarize evidence for reporting"""
        return {
            'total_evidence_count': len(evidence_list),
            'evidence_by_source': {
                source: len([e for e in evidence_list if e.source == source])
                for source in set(e.source for e in evidence_list)
            },
            'average_confidence': np.mean([e.confidence for e in evidence_list]),
            'confidence_std': np.std([e.confidence for e in evidence_list]),
            'time_span': max(e.timestamp for e in evidence_list) - min(e.timestamp for e in evidence_list) if evidence_list else 0
        }
    
    def _fuzzy_quality_assessment(self) -> Dict[str, float]:
        """Perform fuzzy assessment of overall analysis quality"""
        # Aggregate fuzzy confidence across all recent evidence
        recent_evidence = self.evidence_history[-100:]  # Last 100 evidence points
        
        if not recent_evidence:
            return {'overall_quality': 0.5}
        
        # Aggregate fuzzy memberships
        aggregated_fuzzy = {label: 0.0 for label in self.confidence_fuzzy_sets.keys()}
        
        for evidence in recent_evidence:
            for label, membership in evidence.fuzzy_confidence.items():
                # Use algebraic sum for aggregation
                current = aggregated_fuzzy.get(label, 0.0)
                aggregated_fuzzy[label] = current + membership - (current * membership)
        
        # Normalize
        total = sum(aggregated_fuzzy.values())
        if total > 0:
            for label in aggregated_fuzzy:
                aggregated_fuzzy[label] /= total
        
        # Calculate overall quality as weighted sum
        quality_weights = {'very_low': 0.1, 'low': 0.3, 'medium': 0.5, 'high': 0.8, 'very_high': 1.0}
        overall_quality = sum(
            aggregated_fuzzy.get(label, 0.0) * quality_weights.get(label, 0.5)
            for label in quality_weights
        )
        
        result = aggregated_fuzzy.copy()
        result['overall_quality'] = overall_quality
        return result
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate additional quality metrics"""
        metrics = {}
        
        # Network connectivity
        metrics['network_connectivity'] = len(self.nodes) / 10.0  # Normalized by expected nodes
        
        # Evidence diversity
        sources = set(e.source for e in self.evidence_history)
        metrics['evidence_diversity'] = len(sources) / 5.0  # Normalized by expected sources
        
        # Temporal consistency
        if len(self.evidence_history) > 1:
            timestamps = [e.timestamp for e in self.evidence_history]
            time_gaps = np.diff(sorted(timestamps))
            avg_gap = np.mean(time_gaps)
            max_gap = np.max(time_gaps)
            metrics['temporal_consistency'] = 1.0 / (1.0 + max_gap / avg_gap) if avg_gap > 0 else 1.0
        else:
            metrics['temporal_consistency'] = 1.0
        
        # Belief convergence
        entropies = []
        for node in self.nodes.values():
            if hasattr(node, 'belief_entropy'):
                entropies.append(node.belief_entropy())
        
        if entropies:
            avg_entropy = np.mean(entropies)
            metrics['belief_convergence'] = 1.0 / (1.0 + avg_entropy)
        else:
            metrics['belief_convergence'] = 0.5
        
        return metrics
    
    def analyze_trajectory_file(self, filename: str) -> BayesianAnalysisResult:
        """Analyze trajectory from file using Bayesian Evidence Network"""
        logger.info(f"Analyzing trajectory file: {filename}")
        
        # Load trajectory data
        trajectory_data = self.gps_loader.load_trajectory(filename)
        
        if trajectory_data is None or len(trajectory_data) == 0:
            raise ValueError(f"Could not load trajectory data from {filename}")
        
        # Convert to numpy array format expected by the pipeline
        if isinstance(trajectory_data, pd.DataFrame):
            # Assume columns: lat, lon, timestamp, confidence (optional)
            required_cols = ['latitude', 'longitude', 'timestamp']
            missing_cols = [col for col in required_cols if col not in trajectory_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add confidence if not present
            if 'confidence' not in trajectory_data.columns:
                trajectory_data['confidence'] = 0.8  # Default confidence
            
            # Convert to numpy array
            np_array = trajectory_data[['latitude', 'longitude', 'timestamp', 'confidence']].values
        else:
            np_array = np.array(trajectory_data)
        
        return self.process_gps_trajectory(np_array)
    
    def batch_analyze_trajectories(self, 
                                 filenames: List[str], 
                                 parallel: bool = True) -> List[BayesianAnalysisResult]:
        """Analyze multiple trajectories in batch"""
        logger.info(f"Batch analyzing {len(filenames)} trajectories")
        
        if self.use_rust and rust_fuzzy and parallel:
            # Use Rust batch processing
            try:
                trajectories = []
                for filename in filenames:
                    trajectory_data = self.gps_loader.load_trajectory(filename)
                    if trajectory_data is not None:
                        # Convert to expected format
                        if isinstance(trajectory_data, pd.DataFrame):
                            np_array = trajectory_data[['latitude', 'longitude', 'timestamp', 'confidence']].values
                        else:
                            np_array = np.array(trajectory_data)
                        
                        gps_points = self._numpy_to_gps_points(np_array)
                        trajectories.append(gps_points)
                
                optimization_params = {
                    'population_size': float(self.population_size),
                    'max_generations': float(self.optimization_iterations)
                }
                
                rust_results = rust_fuzzy.batch_optimize_trajectories_fuzzy(
                    trajectories, optimization_params
                )
                
                # Convert Rust results to BayesianAnalysisResult format
                results = []
                for i, rust_result in enumerate(rust_results):
                    result = BayesianAnalysisResult(
                        objective_value=rust_result.get('final_objective', 0.0),
                        node_beliefs={},  # Would be populated from Rust
                        evidence_summary={'filename': filenames[i]},
                        optimization_stats=rust_result,
                        fuzzy_assessment={},
                        quality_metrics={}
                    )
                    results.append(result)
                
                return results
                
            except Exception as e:
                logger.warning(f"Rust batch processing failed: {e}. Using Python fallback.")
        
        # Python fallback
        results = []
        for filename in filenames:
            try:
                result = self.analyze_trajectory_file(filename)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {filename}: {e}")
                # Add placeholder result
                placeholder = BayesianAnalysisResult(
                    objective_value=0.0,
                    node_beliefs={},
                    evidence_summary={'filename': filename, 'error': str(e)},
                    optimization_stats={},
                    fuzzy_assessment={},
                    quality_metrics={}
                )
                results.append(placeholder)
        
        return results
    
    def _numpy_to_gps_points(self, np_array: np.ndarray) -> List:
        """Convert numpy array to GPS points for Rust"""
        gps_points = []
        for row in np_array:
            gps_point = {
                'latitude': float(row[0]),
                'longitude': float(row[1]),
                'timestamp': float(row[2]) if len(row) > 2 else 0.0,
                'confidence': float(row[3]) if len(row) > 3 else 0.8
            }
            gps_points.append(gps_point)
        return gps_points
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary of current network state"""
        return {
            'nodes': {
                node_id: {
                    'type': getattr(node, 'node_type', 'unknown'),
                    'variable': getattr(node, 'variable_name', 'unknown'),
                    'belief_entropy': node.belief_entropy() if hasattr(node, 'belief_entropy') else 0.0,
                    'map_state': node.map_state() if hasattr(node, 'map_state') else ('unknown', 0.0)
                }
                for node_id, node in self.nodes.items()
            },
            'evidence_stats': {
                'total_count': len(self.evidence_history),
                'recent_count': len([e for e in self.evidence_history if e.timestamp > (datetime.now().timestamp() - 3600)]),
                'sources': list(set(e.source for e in self.evidence_history))
            },
            'objective_weights': self.objective_weights.copy(),
            'system_info': {
                'rust_available': self.use_rust,
                'optimization_iterations': self.optimization_iterations,
                'population_size': self.population_size
            }
        }

# Convenience functions for common use cases

def analyze_trajectory_bayesian(filename: str, 
                              use_rust: bool = True,
                              optimization_iterations: int = 100) -> BayesianAnalysisResult:
    """
    Convenient function to analyze a single trajectory file using Bayesian Evidence Network
    
    Args:
        filename: Path to trajectory file
        use_rust: Whether to use Rust implementations
        optimization_iterations: Number of optimization iterations
        
    Returns:
        BayesianAnalysisResult with analysis results
    """
    pipeline = BayesianAnalysisPipeline(
        use_rust=use_rust,
        optimization_iterations=optimization_iterations
    )
    return pipeline.analyze_trajectory_file(filename)

def batch_analyze_trajectories_bayesian(filenames: List[str],
                                       use_rust: bool = True,
                                       optimization_iterations: int = 50,
                                       parallel: bool = True) -> List[BayesianAnalysisResult]:
    """
    Convenient function to analyze multiple trajectory files in batch
    
    Args:
        filenames: List of trajectory file paths
        use_rust: Whether to use Rust implementations
        optimization_iterations: Number of optimization iterations
        parallel: Whether to use parallel processing
        
    Returns:
        List of BayesianAnalysisResult objects
    """
    pipeline = BayesianAnalysisPipeline(
        use_rust=use_rust,
        optimization_iterations=optimization_iterations
    )
    return pipeline.batch_analyze_trajectories(filenames, parallel=parallel)

def create_custom_analysis_pipeline(**kwargs) -> BayesianAnalysisPipeline:
    """
    Create a customized Bayesian Analysis Pipeline
    
    Args:
        **kwargs: Keyword arguments for BayesianAnalysisPipeline constructor
        
    Returns:
        Configured BayesianAnalysisPipeline instance
    """
    return BayesianAnalysisPipeline(**kwargs) 