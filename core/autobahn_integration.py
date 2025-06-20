"""
Autobahn Integration Layer for Sighthound
Connects GPS trajectory analysis with consciousness-aware probabilistic reasoning

This module integrates Sighthound's Bayesian Evidence Network with the Autobahn
Oscillatory Bio-Metabolic RAG system for advanced probabilistic reasoning tasks.
All complex probabilistic reasoning is delegated to Autobahn's consciousness-
aware biological intelligence architecture.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import aiohttp
import subprocess
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AutobahnQuery:
    """Query structure for Autobahn probabilistic reasoning engine"""
    query_text: str
    context: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    reasoning_type: str  # 'bayesian', 'fuzzy', 'consciousness', 'biological'
    hierarchy_level: str  # 'planck', 'quantum', 'molecular', 'cellular', 'biological', 'cognitive', 'social', 'cosmic'
    metabolic_mode: str  # 'flight', 'cold_blooded', 'mammalian', 'anaerobic'
    consciousness_threshold: float = 0.7
    atp_budget: float = 150.0
    coherence_threshold: float = 0.85
    target_entropy: float = 2.2
    immune_sensitivity: float = 0.8

@dataclass
class AutobahnResponse:
    """Response structure from Autobahn system"""
    response_content: str
    quality_score: float
    consciousness_level: float
    atp_consumption: float
    membrane_coherence: float
    entropy_optimization: float
    phi_value: float  # Integrated Information Theory consciousness measure
    threat_analysis: Dict[str, Any]
    oscillatory_efficiency: float
    biological_intelligence_score: float
    fire_circle_communication_score: float
    dual_proximity_signaling: Dict[str, float]
    credibility_assessment: Dict[str, float]
    temporal_determinism_confidence: float
    categorical_predeterminism_score: float
    bmf_frame_selection: List[str]  # Biological Maxwell's Demon frame selection
    metadata: Dict[str, Any] = field(default_factory=dict)

class AutobahnClient:
    """Client for communicating with Autobahn probabilistic reasoning engine"""
    
    def __init__(self, 
                 autobahn_endpoint: Optional[str] = None,
                 autobahn_path: Optional[str] = None,
                 use_local_binary: bool = True):
        """
        Initialize Autobahn client
        
        Args:
            autobahn_endpoint: HTTP endpoint for Autobahn service
            autobahn_path: Path to local Autobahn binary
            use_local_binary: Whether to use local binary or HTTP service
        """
        self.autobahn_endpoint = autobahn_endpoint or "http://localhost:8080/api/v1"
        self.autobahn_path = autobahn_path or self._find_autobahn_binary()
        self.use_local_binary = use_local_binary
        self.session = None
        
        logger.info(f"Autobahn client initialized (local: {use_local_binary})")
        if self.use_local_binary and self.autobahn_path:
            logger.info(f"Using Autobahn binary: {self.autobahn_path}")
        elif not self.use_local_binary:
            logger.info(f"Using Autobahn endpoint: {self.autobahn_endpoint}")
    
    def _find_autobahn_binary(self) -> Optional[str]:
        """Find Autobahn binary in common locations"""
        possible_paths = [
            "./autobahn/target/release/autobahn",
            "../autobahn/target/release/autobahn",
            "~/autobahn/target/release/autobahn",
            "/usr/local/bin/autobahn",
            "autobahn"  # In PATH
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path) and os.access(expanded_path, os.X_OK):
                return expanded_path
        
        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'autobahn'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass
        
        logger.warning("Autobahn binary not found. HTTP endpoint will be required.")
        return None
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.use_local_binary:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def query_autobahn(self, query: AutobahnQuery) -> AutobahnResponse:
        """
        Send query to Autobahn for probabilistic reasoning
        
        Args:
            query: AutobahnQuery with reasoning request
            
        Returns:
            AutobahnResponse with consciousness-aware analysis
        """
        if self.use_local_binary and self.autobahn_path:
            return await self._query_local_binary(query)
        else:
            return await self._query_http_endpoint(query)
    
    async def _query_local_binary(self, query: AutobahnQuery) -> AutobahnResponse:
        """Query local Autobahn binary"""
        try:
            # Create input JSON
            input_data = {
                "query": query.query_text,
                "context": query.context,
                "evidence": query.evidence,
                "configuration": {
                    "reasoning_type": query.reasoning_type,
                    "hierarchy_level": query.hierarchy_level,
                    "metabolic_mode": query.metabolic_mode,
                    "consciousness_threshold": query.consciousness_threshold,
                    "atp_budget": query.atp_budget,
                    "coherence_threshold": query.coherence_threshold,
                    "target_entropy": query.target_entropy,
                    "immune_sensitivity": query.immune_sensitivity
                }
            }
            
            # Execute Autobahn binary
            process = await asyncio.create_subprocess_exec(
                self.autobahn_path,
                "--mode", "probabilistic-reasoning",
                "--input", "stdin",
                "--output", "json",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            input_json = json.dumps(input_data).encode('utf-8')
            stdout, stderr = await process.communicate(input=input_json)
            
            if process.returncode != 0:
                logger.error(f"Autobahn binary error: {stderr.decode()}")
                return self._create_fallback_response(query, "Binary execution failed")
            
            # Parse response
            response_data = json.loads(stdout.decode())
            return self._parse_autobahn_response(response_data)
            
        except Exception as e:
            logger.error(f"Error querying local Autobahn binary: {e}")
            return self._create_fallback_response(query, str(e))
    
    async def _query_http_endpoint(self, query: AutobahnQuery) -> AutobahnResponse:
        """Query Autobahn HTTP endpoint"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                "query": query.query_text,
                "context": query.context,
                "evidence": query.evidence,
                "configuration": {
                    "reasoning_type": query.reasoning_type,
                    "hierarchy_level": query.hierarchy_level,
                    "metabolic_mode": query.metabolic_mode,
                    "consciousness_threshold": query.consciousness_threshold,
                    "atp_budget": query.atp_budget,
                    "coherence_threshold": query.coherence_threshold,
                    "target_entropy": query.target_entropy,
                    "immune_sensitivity": query.immune_sensitivity
                }
            }
            
            async with self.session.post(
                f"{self.autobahn_endpoint}/reasoning",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return self._parse_autobahn_response(response_data)
                else:
                    error_text = await response.text()
                    logger.error(f"Autobahn HTTP error {response.status}: {error_text}")
                    return self._create_fallback_response(query, f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Error querying Autobahn HTTP endpoint: {e}")
            return self._create_fallback_response(query, str(e))
    
    def _parse_autobahn_response(self, response_data: Dict[str, Any]) -> AutobahnResponse:
        """Parse response from Autobahn into structured format"""
        return AutobahnResponse(
            response_content=response_data.get("response", {}).get("content", ""),
            quality_score=response_data.get("quality_score", 0.0),
            consciousness_level=response_data.get("consciousness_level", 0.0),
            atp_consumption=response_data.get("atp_consumption", 0.0),
            membrane_coherence=response_data.get("membrane_coherence", 0.0),
            entropy_optimization=response_data.get("entropy_optimization", 0.0),
            phi_value=response_data.get("phi_value", 0.0),
            threat_analysis=response_data.get("threat_analysis", {}),
            oscillatory_efficiency=response_data.get("oscillatory_efficiency", 0.0),
            biological_intelligence_score=response_data.get("biological_intelligence_score", 0.0),
            fire_circle_communication_score=response_data.get("fire_circle_communication_score", 0.0),
            dual_proximity_signaling=response_data.get("dual_proximity_signaling", {}),
            credibility_assessment=response_data.get("credibility_assessment", {}),
            temporal_determinism_confidence=response_data.get("temporal_determinism_confidence", 0.0),
            categorical_predeterminism_score=response_data.get("categorical_predeterminism_score", 0.0),
            bmf_frame_selection=response_data.get("bmf_frame_selection", []),
            metadata=response_data.get("metadata", {})
        )
    
    def _create_fallback_response(self, query: AutobahnQuery, error: str) -> AutobahnResponse:
        """Create fallback response when Autobahn is unavailable"""
        logger.warning(f"Creating fallback response due to: {error}")
        
        return AutobahnResponse(
            response_content=f"Fallback response: Unable to connect to Autobahn reasoning engine. Error: {error}",
            quality_score=0.3,  # Low quality for fallback
            consciousness_level=0.0,
            atp_consumption=0.0,
            membrane_coherence=0.0,
            entropy_optimization=0.0,
            phi_value=0.0,
            threat_analysis={"status": "unavailable"},
            oscillatory_efficiency=0.0,
            biological_intelligence_score=0.0,
            fire_circle_communication_score=0.0,
            dual_proximity_signaling={"death_proximity": 0.0, "life_proximity": 0.0},
            credibility_assessment={"beauty_credibility_efficiency": 0.0},
            temporal_determinism_confidence=0.0,
            categorical_predeterminism_score=0.0,
            bmf_frame_selection=[],
            metadata={"fallback": True, "error": error}
        )

class AutobahnIntegratedBayesianPipeline:
    """
    Bayesian Analysis Pipeline integrated with Autobahn consciousness-aware reasoning
    
    This pipeline delegates all complex probabilistic reasoning to Autobahn's
    bio-metabolic consciousness architecture while handling GPS trajectory analysis
    locally in Sighthound.
    """
    
    def __init__(self, 
                 autobahn_client: Optional[AutobahnClient] = None,
                 default_metabolic_mode: str = "mammalian",
                 default_hierarchy_level: str = "biological",
                 consciousness_threshold: float = 0.7):
        """
        Initialize integrated pipeline
        
        Args:
            autobahn_client: Client for Autobahn system
            default_metabolic_mode: Default ATP metabolic mode
            default_hierarchy_level: Default oscillatory hierarchy level
            consciousness_threshold: Minimum consciousness level for reasoning
        """
        from .bayesian_analysis_pipeline import BayesianAnalysisPipeline
        
        self.bayesian_pipeline = BayesianAnalysisPipeline(use_rust=True)
        self.autobahn_client = autobahn_client or AutobahnClient()
        self.default_metabolic_mode = default_metabolic_mode
        self.default_hierarchy_level = default_hierarchy_level
        self.consciousness_threshold = consciousness_threshold
        
        # Reasoning task categories that should be sent to Autobahn
        self.autobahn_reasoning_tasks = {
            'probabilistic_inference',
            'uncertainty_quantification', 
            'belief_propagation',
            'evidence_fusion',
            'causal_reasoning',
            'temporal_reasoning',
            'consciousness_assessment',
            'credibility_evaluation',
            'threat_assessment',
            'optimization_guidance',
            'pattern_recognition',
            'anomaly_detection',
            'predictive_modeling',
            'decision_making',
            'knowledge_integration'
        }
        
        logger.info("Autobahn-integrated Bayesian pipeline initialized")
    
    async def analyze_trajectory_with_consciousness(self,
                                                  trajectory_data: np.ndarray,
                                                  reasoning_tasks: Optional[List[str]] = None,
                                                  metabolic_mode: Optional[str] = None,
                                                  hierarchy_level: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze GPS trajectory with consciousness-aware probabilistic reasoning
        
        Args:
            trajectory_data: GPS trajectory as numpy array
            reasoning_tasks: Specific reasoning tasks to perform
            metabolic_mode: ATP metabolic mode for Autobahn
            hierarchy_level: Oscillatory hierarchy level
            
        Returns:
            Comprehensive analysis results combining Sighthound and Autobahn
        """
        logger.info(f"Starting consciousness-aware trajectory analysis ({len(trajectory_data)} points)")
        
        # Step 1: Perform initial Sighthound Bayesian analysis
        bayesian_result = self.bayesian_pipeline.process_gps_trajectory(trajectory_data)
        
        # Step 2: Extract evidence and context for Autobahn
        evidence_for_autobahn = self._extract_evidence_for_autobahn(
            trajectory_data, bayesian_result
        )
        
        # Step 3: Determine reasoning tasks
        if reasoning_tasks is None:
            reasoning_tasks = self._determine_reasoning_tasks(bayesian_result)
        
        # Step 4: Query Autobahn for consciousness-aware reasoning
        autobahn_results = {}
        
        async with self.autobahn_client:
            for task in reasoning_tasks:
                if task in self.autobahn_reasoning_tasks:
                    query = self._create_autobahn_query(
                        task, evidence_for_autobahn, bayesian_result,
                        metabolic_mode or self.default_metabolic_mode,
                        hierarchy_level or self.default_hierarchy_level
                    )
                    
                    autobahn_response = await self.autobahn_client.query_autobahn(query)
                    autobahn_results[task] = autobahn_response
        
        # Step 5: Integrate results
        integrated_result = self._integrate_results(
            bayesian_result, autobahn_results, trajectory_data
        )
        
        logger.info(f"Consciousness-aware analysis complete. "
                   f"Consciousness level: {integrated_result.get('overall_consciousness_level', 0.0):.3f}")
        
        return integrated_result
    
    def _extract_evidence_for_autobahn(self, 
                                     trajectory_data: np.ndarray,
                                     bayesian_result) -> List[Dict[str, Any]]:
        """Extract evidence from Sighthound analysis for Autobahn reasoning"""
        evidence = []
        
        # GPS trajectory evidence
        for i, point in enumerate(trajectory_data):
            lat, lon, timestamp, confidence = point[:4]
            evidence.append({
                "type": "gps_observation",
                "latitude": float(lat),
                "longitude": float(lon), 
                "timestamp": float(timestamp),
                "confidence": float(confidence),
                "sequence_index": i
            })
        
        # Bayesian network node beliefs
        for node_id, beliefs in bayesian_result.node_beliefs.items():
            evidence.append({
                "type": "bayesian_belief",
                "node_id": node_id,
                "beliefs": beliefs,
                "entropy": sum(-p * np.log2(p) for p in beliefs.values() if p > 0)
            })
        
        # Quality metrics as evidence
        for metric_name, value in bayesian_result.quality_metrics.items():
            evidence.append({
                "type": "quality_metric",
                "metric_name": metric_name,
                "value": float(value)
            })
        
        # Fuzzy assessment as evidence
        for assessment_type, score in bayesian_result.fuzzy_assessment.items():
            evidence.append({
                "type": "fuzzy_assessment",
                "assessment_type": assessment_type,
                "score": float(score)
            })
        
        return evidence
    
    def _determine_reasoning_tasks(self, bayesian_result) -> List[str]:
        """Determine which reasoning tasks should be performed based on analysis"""
        tasks = ['probabilistic_inference', 'consciousness_assessment']
        
        # Add tasks based on analysis quality
        if bayesian_result.objective_value < 0.5:
            tasks.extend(['uncertainty_quantification', 'anomaly_detection'])
        
        if len(bayesian_result.node_beliefs) > 5:
            tasks.append('belief_propagation')
        
        if bayesian_result.quality_metrics.get('belief_convergence', 0) < 0.7:
            tasks.append('evidence_fusion')
        
        # Always include credibility and threat assessment
        tasks.extend(['credibility_evaluation', 'threat_assessment'])
        
        return list(set(tasks))  # Remove duplicates
    
    def _create_autobahn_query(self,
                             task: str,
                             evidence: List[Dict[str, Any]],
                             bayesian_result,
                             metabolic_mode: str,
                             hierarchy_level: str) -> AutobahnQuery:
        """Create Autobahn query for specific reasoning task"""
        
        task_queries = {
            'probabilistic_inference': (
                "Analyze the probabilistic structure of this GPS trajectory data. "
                "What are the most likely true positions given the evidence and uncertainty? "
                "Consider temporal coherence, spatial consistency, and confidence propagation."
            ),
            'consciousness_assessment': (
                "Assess the consciousness-level integration of this trajectory analysis. "
                "What is the Φ (phi) value for the integrated information? "
                "How does the biological membrane coherence affect the analysis quality?"
            ),
            'uncertainty_quantification': (
                "Quantify the uncertainty in this trajectory analysis. "
                "What are the primary sources of uncertainty and how do they propagate? "
                "Provide uncertainty bounds for position estimates."
            ),
            'belief_propagation': (
                "Analyze the belief propagation through the Bayesian network. "
                "Are beliefs converging appropriately? What evidence is most influential? "
                "Identify potential belief propagation issues."
            ),
            'evidence_fusion': (
                "Evaluate the evidence fusion process in this trajectory analysis. "
                "How well are different evidence sources being integrated? "
                "What is the optimal fusion strategy for this data?"
            ),
            'credibility_evaluation': (
                "Assess the credibility of this trajectory analysis using beauty-credibility "
                "efficiency models. How does the analysis quality relate to credibility? "
                "What factors affect the perceived reliability of results?"
            ),
            'threat_assessment': (
                "Perform biological immune system threat assessment on this analysis. "
                "Are there any adversarial patterns or coherence threats? "
                "What is the immune system response recommendation?"
            ),
            'anomaly_detection': (
                "Detect anomalies in this GPS trajectory using biological intelligence. "
                "What patterns deviate from expected biological movement patterns? "
                "Identify potential data quality issues or unusual behaviors."
            )
        }
        
        query_text = task_queries.get(task, f"Perform {task} on the provided trajectory evidence.")
        
        context = {
            "task_type": task,
            "trajectory_length": len([e for e in evidence if e["type"] == "gps_observation"]),
            "bayesian_objective_value": bayesian_result.objective_value,
            "analysis_timestamp": datetime.now().isoformat(),
            "sighthound_version": "hybrid_bayesian_fuzzy",
            "evidence_summary": {
                "total_evidence_points": len(evidence),
                "evidence_types": list(set(e["type"] for e in evidence))
            }
        }
        
        return AutobahnQuery(
            query_text=query_text,
            context=context,
            evidence=evidence,
            reasoning_type=self._map_task_to_reasoning_type(task),
            hierarchy_level=hierarchy_level,
            metabolic_mode=metabolic_mode,
            consciousness_threshold=self.consciousness_threshold
        )
    
    def _map_task_to_reasoning_type(self, task: str) -> str:
        """Map reasoning task to Autobahn reasoning type"""
        mapping = {
            'probabilistic_inference': 'bayesian',
            'uncertainty_quantification': 'bayesian',
            'belief_propagation': 'bayesian',
            'evidence_fusion': 'fuzzy',
            'consciousness_assessment': 'consciousness',
            'credibility_evaluation': 'biological',
            'threat_assessment': 'biological',
            'anomaly_detection': 'biological'
        }
        return mapping.get(task, 'biological')
    
    def _integrate_results(self,
                         bayesian_result,
                         autobahn_results: Dict[str, AutobahnResponse],
                         trajectory_data: np.ndarray) -> Dict[str, Any]:
        """Integrate Sighthound Bayesian results with Autobahn consciousness reasoning"""
        
        # Calculate overall consciousness level
        consciousness_levels = [r.consciousness_level for r in autobahn_results.values()]
        overall_consciousness = np.mean(consciousness_levels) if consciousness_levels else 0.0
        
        # Calculate overall quality score
        quality_scores = [r.quality_score for r in autobahn_results.values()]
        autobahn_quality = np.mean(quality_scores) if quality_scores else 0.0
        combined_quality = 0.6 * bayesian_result.objective_value + 0.4 * autobahn_quality
        
        # Aggregate ATP consumption
        total_atp_consumption = sum(r.atp_consumption for r in autobahn_results.values())
        
        # Aggregate threat analysis
        threat_analyses = [r.threat_analysis for r in autobahn_results.values()]
        combined_threat_level = self._aggregate_threat_analysis(threat_analyses)
        
        # Extract consciousness insights
        consciousness_insights = {}
        for task, response in autobahn_results.items():
            if response.phi_value > 0:
                consciousness_insights[task] = {
                    "phi_value": response.phi_value,
                    "consciousness_level": response.consciousness_level,
                    "membrane_coherence": response.membrane_coherence,
                    "biological_intelligence": response.biological_intelligence_score
                }
        
        # Compile integrated result
        integrated_result = {
            # Original Sighthound results
            "sighthound_analysis": {
                "objective_value": bayesian_result.objective_value,
                "node_beliefs": bayesian_result.node_beliefs,
                "evidence_summary": bayesian_result.evidence_summary,
                "optimization_stats": bayesian_result.optimization_stats,
                "fuzzy_assessment": bayesian_result.fuzzy_assessment,
                "quality_metrics": bayesian_result.quality_metrics
            },
            
            # Autobahn consciousness analysis
            "autobahn_analysis": {
                "reasoning_tasks_performed": list(autobahn_results.keys()),
                "overall_consciousness_level": overall_consciousness,
                "total_atp_consumption": total_atp_consumption,
                "consciousness_insights": consciousness_insights,
                "threat_assessment": combined_threat_level,
                "task_responses": {
                    task: {
                        "response": response.response_content,
                        "quality_score": response.quality_score,
                        "consciousness_level": response.consciousness_level,
                        "phi_value": response.phi_value,
                        "biological_intelligence": response.biological_intelligence_score,
                        "fire_circle_communication": response.fire_circle_communication_score,
                        "dual_proximity_signaling": response.dual_proximity_signaling,
                        "credibility_assessment": response.credibility_assessment,
                        "temporal_determinism": response.temporal_determinism_confidence,
                        "categorical_predeterminism": response.categorical_predeterminism_score,
                        "bmf_frame_selection": response.bmf_frame_selection
                    }
                    for task, response in autobahn_results.items()
                }
            },
            
            # Integrated metrics
            "integrated_metrics": {
                "combined_quality_score": combined_quality,
                "overall_consciousness_level": overall_consciousness,
                "consciousness_emergence_detected": overall_consciousness > self.consciousness_threshold,
                "biological_intelligence_active": any(
                    r.biological_intelligence_score > 0.5 for r in autobahn_results.values()
                ),
                "fire_circle_communication_detected": any(
                    r.fire_circle_communication_score > 0.5 for r in autobahn_results.values()
                ),
                "dual_proximity_signaling_active": any(
                    max(r.dual_proximity_signaling.values()) > 0.5 
                    for r in autobahn_results.values() 
                    if r.dual_proximity_signaling
                ),
                "threat_level": combined_threat_level,
                "analysis_reliability": self._calculate_analysis_reliability(
                    bayesian_result, autobahn_results
                )
            },
            
            # Trajectory metadata
            "trajectory_metadata": {
                "total_points": len(trajectory_data),
                "time_span": float(trajectory_data[-1][2] - trajectory_data[0][2]) if len(trajectory_data) > 1 else 0.0,
                "spatial_extent": self._calculate_spatial_extent(trajectory_data),
                "average_confidence": float(np.mean(trajectory_data[:, 3])) if trajectory_data.shape[1] > 3 else 0.8
            },
            
            # System information
            "system_info": {
                "analysis_timestamp": datetime.now().isoformat(),
                "sighthound_version": "hybrid_bayesian_fuzzy_autobahn",
                "autobahn_integration": "active",
                "consciousness_aware": True,
                "biological_intelligence": True,
                "oscillatory_dynamics": True,
                "fire_circle_communication": True
            }
        }
        
        return integrated_result
    
    def _aggregate_threat_analysis(self, threat_analyses: List[Dict[str, Any]]) -> str:
        """Aggregate threat analyses from multiple Autobahn responses"""
        if not threat_analyses:
            return "unknown"
        
        threat_levels = []
        for analysis in threat_analyses:
            if isinstance(analysis, dict) and "threat_level" in analysis:
                threat_levels.append(analysis["threat_level"])
        
        if not threat_levels:
            return "safe"
        
        # Return highest threat level
        threat_hierarchy = ["safe", "low", "medium", "high", "critical"]
        max_threat = "safe"
        for level in threat_levels:
            if level in threat_hierarchy:
                if threat_hierarchy.index(level) > threat_hierarchy.index(max_threat):
                    max_threat = level
        
        return max_threat
    
    def _calculate_analysis_reliability(self,
                                     bayesian_result,
                                     autobahn_results: Dict[str, AutobahnResponse]) -> float:
        """Calculate overall analysis reliability combining Sighthound and Autobahn"""
        sighthound_reliability = bayesian_result.objective_value
        
        if not autobahn_results:
            return sighthound_reliability * 0.7  # Penalize for missing consciousness analysis
        
        # Calculate Autobahn reliability based on consciousness and quality
        autobahn_reliability_scores = []
        for response in autobahn_results.values():
            reliability = (
                0.3 * response.quality_score +
                0.3 * response.consciousness_level +
                0.2 * response.biological_intelligence_score +
                0.2 * (1.0 - max(response.dual_proximity_signaling.values()) 
                       if response.dual_proximity_signaling else 0.5)
            )
            autobahn_reliability_scores.append(reliability)
        
        autobahn_reliability = np.mean(autobahn_reliability_scores)
        
        # Combine reliabilities
        combined_reliability = 0.6 * sighthound_reliability + 0.4 * autobahn_reliability
        
        return combined_reliability
    
    def _calculate_spatial_extent(self, trajectory_data: np.ndarray) -> float:
        """Calculate spatial extent of trajectory"""
        if len(trajectory_data) < 2:
            return 0.0
        
        lats = trajectory_data[:, 0]
        lons = trajectory_data[:, 1]
        
        lat_range = np.max(lats) - np.min(lats)
        lon_range = np.max(lons) - np.min(lons)
        
        # Approximate distance in meters (rough calculation)
        lat_meters = lat_range * 111000  # 1 degree lat ≈ 111km
        lon_meters = lon_range * 111000 * np.cos(np.radians(np.mean(lats)))
        
        return np.sqrt(lat_meters**2 + lon_meters**2)

# Convenience functions for easy usage

async def analyze_trajectory_with_autobahn(trajectory_data: np.ndarray,
                                         autobahn_endpoint: Optional[str] = None,
                                         autobahn_path: Optional[str] = None,
                                         reasoning_tasks: Optional[List[str]] = None,
                                         metabolic_mode: str = "mammalian",
                                         hierarchy_level: str = "biological",
                                         consciousness_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Convenient function to analyze trajectory with Autobahn consciousness reasoning
    
    Args:
        trajectory_data: GPS trajectory as numpy array
        autobahn_endpoint: Autobahn HTTP endpoint
        autobahn_path: Path to Autobahn binary
        reasoning_tasks: Specific reasoning tasks
        metabolic_mode: ATP metabolic mode
        hierarchy_level: Oscillatory hierarchy level
        consciousness_threshold: Minimum consciousness level
        
    Returns:
        Integrated analysis results
    """
    client = AutobahnClient(autobahn_endpoint, autobahn_path)
    pipeline = AutobahnIntegratedBayesianPipeline(
        client, metabolic_mode, hierarchy_level, consciousness_threshold
    )
    
    return await pipeline.analyze_trajectory_with_consciousness(
        trajectory_data, reasoning_tasks, metabolic_mode, hierarchy_level
    )

async def batch_analyze_trajectories_with_autobahn(trajectories: List[np.ndarray],
                                                 autobahn_endpoint: Optional[str] = None,
                                                 autobahn_path: Optional[str] = None,
                                                 parallel: bool = True) -> List[Dict[str, Any]]:
    """
    Batch analyze multiple trajectories with Autobahn consciousness reasoning
    
    Args:
        trajectories: List of GPS trajectory arrays
        autobahn_endpoint: Autobahn HTTP endpoint
        autobahn_path: Path to Autobahn binary
        parallel: Whether to process in parallel
        
    Returns:
        List of integrated analysis results
    """
    client = AutobahnClient(autobahn_endpoint, autobahn_path)
    pipeline = AutobahnIntegratedBayesianPipeline(client)
    
    if parallel:
        tasks = [
            pipeline.analyze_trajectory_with_consciousness(traj)
            for traj in trajectories
        ]
        return await asyncio.gather(*tasks)
    else:
        results = []
        for traj in trajectories:
            result = await pipeline.analyze_trajectory_with_consciousness(traj)
            results.append(result)
        return results

def create_autobahn_integrated_pipeline(**kwargs) -> AutobahnIntegratedBayesianPipeline:
    """
    Create an Autobahn-integrated Bayesian analysis pipeline
    
    Args:
        **kwargs: Keyword arguments for AutobahnIntegratedBayesianPipeline
        
    Returns:
        Configured pipeline instance
    """
    return AutobahnIntegratedBayesianPipeline(**kwargs) 