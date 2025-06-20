use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tokio::process::Command as TokioCommand;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use reqwest::Client;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use futures::future::join_all;
use crossbeam::channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use once_cell::sync::Lazy;

use sighthound_core::GpsPoint;
use sighthound_bayesian::{BayesianEvidenceNetwork, FuzzyEvidence};
use sighthound_fuzzy::FuzzyBayesianOptimizer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Global Autobahn client pool for connection reuse
static AUTOBAHN_POOL: Lazy<Arc<AutobahnConnectionPool>> = Lazy::new(|| {
    Arc::new(AutobahnConnectionPool::new())
});

/// Autobahn reasoning query structure optimized for Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnQuery {
    pub query_id: String,
    pub query_text: String,
    pub context: AutobahnContext,
    pub evidence: Vec<AutobahnEvidence>,
    pub reasoning_config: AutobahnReasoningConfig,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnContext {
    pub task_type: String,
    pub trajectory_length: usize,
    pub bayesian_objective_value: f64,
    pub sighthound_version: String,
    pub evidence_summary: EvidenceSummary,
    pub spatial_bounds: SpatialBounds,
    pub temporal_bounds: TemporalBounds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnEvidence {
    pub evidence_type: String,
    pub variable_name: String,
    pub crisp_value: f64,
    pub confidence: f64,
    pub fuzzy_memberships: HashMap<String, f64>,
    pub timestamp: f64,
    pub source: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnReasoningConfig {
    pub reasoning_type: String,
    pub hierarchy_level: String,
    pub metabolic_mode: String,
    pub consciousness_threshold: f64,
    pub atp_budget: f64,
    pub coherence_threshold: f64,
    pub target_entropy: f64,
    pub immune_sensitivity: f64,
    pub fire_circle_communication: bool,
    pub dual_proximity_signaling: bool,
    pub biological_membrane_optimization: bool,
    pub temporal_determinism: bool,
    pub categorical_predeterminism: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    pub total_evidence_points: usize,
    pub evidence_types: Vec<String>,
    pub confidence_distribution: ConfidenceDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub quartiles: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialBounds {
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
    pub spatial_extent_meters: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBounds {
    pub start_timestamp: f64,
    pub end_timestamp: f64,
    pub duration_seconds: f64,
    pub sampling_rate_hz: f64,
}

/// Autobahn response structure with full consciousness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnResponse {
    pub query_id: String,
    pub response_content: String,
    pub quality_score: f64,
    pub consciousness_metrics: ConsciousnessMetrics,
    pub biological_intelligence: BiologicalIntelligence,
    pub fire_circle_analysis: FireCircleAnalysis,
    pub dual_proximity_signals: DualProximitySignals,
    pub credibility_assessment: CredibilityAssessment,
    pub temporal_analysis: TemporalAnalysis,
    pub threat_assessment: ThreatAssessment,
    pub performance_metrics: PerformanceMetrics,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    pub phi_value: f64,  // Integrated Information Theory
    pub consciousness_level: f64,
    pub global_workspace_activation: f64,
    pub self_awareness_score: f64,
    pub metacognition_level: f64,
    pub qualia_generation_active: bool,
    pub agency_illusion_strength: f64,
    pub persistence_illusion_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalIntelligence {
    pub membrane_coherence: f64,
    pub ion_channel_optimization: f64,
    pub atp_consumption: f64,
    pub metabolic_efficiency: f64,
    pub biological_processing_score: f64,
    pub environment_assisted_transport: f64,
    pub fire_light_coupling_650nm: f64,
    pub temperature_optimization_310k: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireCircleAnalysis {
    pub communication_complexity_score: f64,
    pub temporal_coordination_detected: bool,
    pub sedentary_period_optimization: f64,
    pub non_action_communication_patterns: Vec<String>,
    pub abstract_conceptualization_level: f64,
    pub seventy_nine_fold_amplification: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualProximitySignals {
    pub death_proximity_signaling: f64,
    pub life_proximity_signaling: f64,
    pub mortality_risk_assessment: f64,
    pub vitality_detection_score: f64,
    pub leadership_hierarchy_indicators: Vec<String>,
    pub reproductive_strategy_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredibilityAssessment {
    pub beauty_credibility_efficiency: f64,
    pub social_function_optimization: f64,
    pub contextual_expectation_alignment: f64,
    pub truth_inversion_detection: bool,
    pub credibility_paradox_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub predetermined_event_confidence: f64,
    pub categorical_completion_progress: f64,
    pub mathematical_structure_navigation: f64,
    pub recognition_space_position: Vec<f64>,
    pub temporal_determinism_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAssessment {
    pub threat_level: String,
    pub immune_system_activation: bool,
    pub t_cell_response: f64,
    pub b_cell_response: f64,
    pub memory_cell_activation: f64,
    pub coherence_interference_detected: bool,
    pub adversarial_patterns: Vec<String>,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub processing_time_ms: u64,
    pub atp_efficiency: f64,
    pub oscillatory_efficiency: f64,
    pub entropy_optimization: f64,
    pub network_efficiency: f64,
    pub biological_optimization: f64,
}

/// High-performance connection pool for Autobahn instances
pub struct AutobahnConnectionPool {
    connections: DashMap<String, Arc<AutobahnConnection>>,
    config: Arc<RwLock<AutobahnConfig>>,
    performance_stats: Arc<RwLock<PerformanceStats>>,
}

#[derive(Debug, Clone)]
pub struct AutobahnConnection {
    pub connection_id: String,
    pub connection_type: AutobahnConnectionType,
    pub endpoint: Option<String>,
    pub binary_path: Option<String>,
    pub client: Option<Client>,
    pub last_used: Instant,
    pub performance_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
}

#[derive(Debug, Clone)]
pub enum AutobahnConnectionType {
    LocalBinary,
    HttpEndpoint,
    DirectRustLibrary,  // Future: direct Rust library integration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnConfig {
    pub endpoint: String,
    pub binary_path: String,
    pub use_local_binary: bool,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub connection_pool_size: usize,
    pub default_metabolic_mode: String,
    pub default_hierarchy_level: String,
    pub consciousness_threshold: f64,
    pub performance_monitoring: bool,
}

#[derive(Debug, Default)]
pub struct PerformanceStats {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_response_time_ms: f64,
    pub consciousness_emergence_count: u64,
    pub biological_intelligence_activations: u64,
    pub fire_circle_detections: u64,
    pub threat_detections: u64,
}

impl AutobahnConnectionPool {
    pub fn new() -> Self {
        let config = AutobahnConfig {
            endpoint: "http://localhost:8080/api/v1".to_string(),
            binary_path: "../autobahn/target/release/autobahn".to_string(),
            use_local_binary: true,
            timeout_seconds: 30,
            max_retries: 3,
            connection_pool_size: 4,
            default_metabolic_mode: "mammalian".to_string(),
            default_hierarchy_level: "biological".to_string(),
            consciousness_threshold: 0.7,
            performance_monitoring: true,
        };

        Self {
            connections: DashMap::new(),
            config: Arc::new(RwLock::new(config)),
            performance_stats: Arc::new(RwLock::new(PerformanceStats::default())),
        }
    }

    pub async fn get_connection(&self) -> Result<Arc<AutobahnConnection>> {
        let config = self.config.read().unwrap().clone();
        
        // Try to find an available connection
        for entry in self.connections.iter() {
            let connection = entry.value();
            if connection.last_used.elapsed() < Duration::from_secs(300) { // 5 min timeout
                return Ok(connection.clone());
            }
        }

        // Create new connection
        let connection_id = Uuid::new_v4().to_string();
        let connection = if config.use_local_binary {
            self.create_binary_connection(&connection_id, &config).await?
        } else {
            self.create_http_connection(&connection_id, &config).await?
        };

        let connection = Arc::new(connection);
        self.connections.insert(connection_id.clone(), connection.clone());
        
        Ok(connection)
    }

    async fn create_binary_connection(&self, id: &str, config: &AutobahnConfig) -> Result<AutobahnConnection> {
        // Verify binary exists and is executable
        if !std::path::Path::new(&config.binary_path).exists() {
            return Err(anyhow::anyhow!("Autobahn binary not found at: {}", config.binary_path));
        }

        Ok(AutobahnConnection {
            connection_id: id.to_string(),
            connection_type: AutobahnConnectionType::LocalBinary,
            endpoint: None,
            binary_path: Some(config.binary_path.clone()),
            client: None,
            last_used: Instant::now(),
            performance_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    async fn create_http_connection(&self, id: &str, config: &AutobahnConfig) -> Result<AutobahnConnection> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;

        // Test connection
        let response = client.get(&format!("{}/health", config.endpoint))
            .send()
            .await;

        match response {
            Ok(_) => {
                Ok(AutobahnConnection {
                    connection_id: id.to_string(),
                    connection_type: AutobahnConnectionType::HttpEndpoint,
                    endpoint: Some(config.endpoint.clone()),
                    binary_path: None,
                    client: Some(client),
                    last_used: Instant::now(),
                    performance_history: Arc::new(RwLock::new(Vec::new())),
                })
            },
            Err(e) => Err(anyhow::anyhow!("Failed to connect to Autobahn endpoint: {}", e))
        }
    }
}

/// High-performance Autobahn client for consciousness-aware reasoning
#[pyclass]
pub struct AutobahnClient {
    pool: Arc<AutobahnConnectionPool>,
    query_cache: Arc<DashMap<String, (AutobahnQuery, AutobahnResponse)>>,
    performance_monitor: Arc<RwLock<PerformanceStats>>,
}

#[pymethods]
impl AutobahnClient {
    #[new]
    pub fn new() -> Self {
        Self {
            pool: AUTOBAHN_POOL.clone(),
            query_cache: Arc::new(DashMap::new()),
            performance_monitor: Arc::new(RwLock::new(PerformanceStats::default())),
        }
    }

    /// Query Autobahn for consciousness-aware probabilistic reasoning
    pub fn query_consciousness_reasoning(&self, 
                                       py: Python,
                                       trajectory: Vec<GpsPoint>,
                                       reasoning_tasks: Vec<String>,
                                       metabolic_mode: Option<String>,
                                       hierarchy_level: Option<String>) -> PyResult<HashMap<String, serde_json::Value>> {
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            self.async_query_consciousness_reasoning(
                trajectory, 
                reasoning_tasks, 
                metabolic_mode, 
                hierarchy_level
            ).await
        });

        match result {
            Ok(response) => {
                let serialized = serde_json::to_value(&response)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                
                if let serde_json::Value::Object(map) = serialized {
                    let mut result_map = HashMap::new();
                    for (k, v) in map {
                        result_map.insert(k, v);
                    }
                    Ok(result_map)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid response format"))
                }
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }
    }

    /// Batch process multiple trajectories with consciousness reasoning
    pub fn batch_consciousness_analysis(&self,
                                      py: Python,
                                      trajectories: Vec<Vec<GpsPoint>>,
                                      reasoning_tasks: Vec<String>,
                                      parallel: Option<bool>) -> PyResult<Vec<HashMap<String, serde_json::Value>>> {
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        let use_parallel = parallel.unwrap_or(true);
        
        let result = rt.block_on(async {
            if use_parallel {
                self.parallel_batch_analysis(trajectories, reasoning_tasks).await
            } else {
                self.sequential_batch_analysis(trajectories, reasoning_tasks).await
            }
        });

        match result {
            Ok(responses) => {
                let mut result_vec = Vec::new();
                for response in responses {
                    let serialized = serde_json::to_value(&response)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                    
                    if let serde_json::Value::Object(map) = serialized {
                        let mut result_map = HashMap::new();
                        for (k, v) in map {
                            result_map.insert(k, v);
                        }
                        result_vec.push(result_map);
                    }
                }
                Ok(result_vec)
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, f64> {
        let stats = self.performance_monitor.read().unwrap();
        let mut result = HashMap::new();
        
        result.insert("total_queries".to_string(), stats.total_queries as f64);
        result.insert("successful_queries".to_string(), stats.successful_queries as f64);
        result.insert("failed_queries".to_string(), stats.failed_queries as f64);
        result.insert("success_rate".to_string(), 
                     if stats.total_queries > 0 { 
                         stats.successful_queries as f64 / stats.total_queries as f64 
                     } else { 0.0 });
        result.insert("average_response_time_ms".to_string(), stats.average_response_time_ms);
        result.insert("consciousness_emergence_count".to_string(), stats.consciousness_emergence_count as f64);
        result.insert("biological_intelligence_activations".to_string(), stats.biological_intelligence_activations as f64);
        result.insert("fire_circle_detections".to_string(), stats.fire_circle_detections as f64);
        result.insert("threat_detections".to_string(), stats.threat_detections as f64);
        
        result
    }
}

impl AutobahnClient {
    async fn async_query_consciousness_reasoning(&self,
                                               trajectory: Vec<GpsPoint>,
                                               reasoning_tasks: Vec<String>,
                                               metabolic_mode: Option<String>,
                                               hierarchy_level: Option<String>) -> Result<AutobahnResponse> {
        
        let start_time = Instant::now();
        
        // Create Autobahn query from trajectory
        let query = self.create_autobahn_query(
            trajectory, 
            reasoning_tasks, 
            metabolic_mode.unwrap_or_else(|| "mammalian".to_string()),
            hierarchy_level.unwrap_or_else(|| "biological".to_string())
        ).await?;

        // Get connection and execute query
        let connection = self.pool.get_connection().await?;
        let response = self.execute_query(&connection, &query).await?;

        // Update performance statistics
        self.update_performance_stats(&response, start_time.elapsed()).await;

        Ok(response)
    }

    async fn parallel_batch_analysis(&self,
                                   trajectories: Vec<Vec<GpsPoint>>,
                                   reasoning_tasks: Vec<String>) -> Result<Vec<AutobahnResponse>> {
        
        let futures: Vec<_> = trajectories.into_iter().map(|trajectory| {
            let tasks = reasoning_tasks.clone();
            async move {
                self.async_query_consciousness_reasoning(
                    trajectory, 
                    tasks, 
                    Some("mammalian".to_string()), 
                    Some("biological".to_string())
                ).await
            }
        }).collect();

        let results = join_all(futures).await;
        
        let mut responses = Vec::new();
        for result in results {
            match result {
                Ok(response) => responses.push(response),
                Err(e) => {
                    tracing::error!("Batch analysis failed for trajectory: {}", e);
                    // Create fallback response
                    responses.push(self.create_fallback_response(&e.to_string()));
                }
            }
        }

        Ok(responses)
    }

    async fn sequential_batch_analysis(&self,
                                     trajectories: Vec<Vec<GpsPoint>>,
                                     reasoning_tasks: Vec<String>) -> Result<Vec<AutobahnResponse>> {
        
        let mut responses = Vec::new();
        
        for trajectory in trajectories {
            match self.async_query_consciousness_reasoning(
                trajectory, 
                reasoning_tasks.clone(), 
                Some("cold_blooded".to_string()),  // Energy efficient for sequential
                Some("biological".to_string())
            ).await {
                Ok(response) => responses.push(response),
                Err(e) => {
                    tracing::error!("Sequential analysis failed for trajectory: {}", e);
                    responses.push(self.create_fallback_response(&e.to_string()));
                }
            }
        }

        Ok(responses)
    }

    async fn create_autobahn_query(&self,
                                 trajectory: Vec<GpsPoint>,
                                 reasoning_tasks: Vec<String>,
                                 metabolic_mode: String,
                                 hierarchy_level: String) -> Result<AutobahnQuery> {
        
        let query_id = Uuid::new_v4().to_string();
        
        // Extract evidence from trajectory
        let evidence = self.extract_evidence_from_trajectory(&trajectory);
        
        // Create context
        let context = self.create_context(&trajectory, &evidence);
        
        // Create reasoning config
        let reasoning_config = AutobahnReasoningConfig {
            reasoning_type: self.determine_primary_reasoning_type(&reasoning_tasks),
            hierarchy_level,
            metabolic_mode,
            consciousness_threshold: 0.7,
            atp_budget: self.calculate_atp_budget(&reasoning_tasks),
            coherence_threshold: 0.85,
            target_entropy: 2.2,
            immune_sensitivity: 0.8,
            fire_circle_communication: reasoning_tasks.contains(&"fire_circle_analysis".to_string()),
            dual_proximity_signaling: reasoning_tasks.contains(&"dual_proximity_assessment".to_string()),
            biological_membrane_optimization: true,
            temporal_determinism: reasoning_tasks.contains(&"temporal_reasoning".to_string()),
            categorical_predeterminism: reasoning_tasks.contains(&"categorical_analysis".to_string()),
        };

        // Create query text
        let query_text = self.generate_query_text(&reasoning_tasks, &trajectory);

        Ok(AutobahnQuery {
            query_id,
            query_text,
            context,
            evidence,
            reasoning_config,
            timestamp: Utc::now(),
        })
    }

    fn extract_evidence_from_trajectory(&self, trajectory: &[GpsPoint]) -> Vec<AutobahnEvidence> {
        let mut evidence = Vec::new();

        for (i, point) in trajectory.iter().enumerate() {
            // GPS position evidence
            evidence.push(AutobahnEvidence {
                evidence_type: "gps_observation".to_string(),
                variable_name: "position".to_string(),
                crisp_value: (point.latitude.powi(2) + point.longitude.powi(2)).sqrt(),
                confidence: point.confidence,
                fuzzy_memberships: self.calculate_fuzzy_memberships(point.confidence),
                timestamp: point.timestamp,
                source: "gps_sensor".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("latitude".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(point.latitude).unwrap()));
                    meta.insert("longitude".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(point.longitude).unwrap()));
                    meta.insert("sequence_index".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
                    meta
                },
            });

            // Velocity evidence (if not first point)
            if i > 0 {
                let prev_point = &trajectory[i-1];
                let dt = point.timestamp - prev_point.timestamp;
                if dt > 0.0 {
                    let distance = point.distance_to(prev_point);
                    let velocity = distance / dt;
                    
                    evidence.push(AutobahnEvidence {
                        evidence_type: "derived_velocity".to_string(),
                        variable_name: "velocity".to_string(),
                        crisp_value: velocity,
                        confidence: point.confidence * 0.8, // Derived values have lower confidence
                        fuzzy_memberships: self.calculate_fuzzy_memberships(point.confidence * 0.8),
                        timestamp: point.timestamp,
                        source: "gps_derived".to_string(),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("distance_meters".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(distance).unwrap()));
                            meta.insert("time_delta_seconds".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(dt).unwrap()));
                            meta
                        },
                    });
                }
            }
        }

        evidence
    }

    fn calculate_fuzzy_memberships(&self, confidence: f64) -> HashMap<String, f64> {
        let mut memberships = HashMap::new();
        
        // Confidence fuzzy sets
        memberships.insert("very_low".to_string(), 
                          if confidence <= 0.4 { (0.4 - confidence) / 0.4 } else { 0.0 });
        memberships.insert("low".to_string(), 
                          if confidence >= 0.2 && confidence <= 0.6 { 
                              if confidence <= 0.4 { (confidence - 0.2) / 0.2 } else { (0.6 - confidence) / 0.2 }
                          } else { 0.0 });
        memberships.insert("medium".to_string(), 
                          if confidence >= 0.4 && confidence <= 0.8 { 
                              if confidence <= 0.6 { (confidence - 0.4) / 0.2 } else { (0.8 - confidence) / 0.2 }
                          } else { 0.0 });
        memberships.insert("high".to_string(), 
                          if confidence >= 0.6 && confidence <= 1.0 { 
                              if confidence <= 0.8 { (confidence - 0.6) / 0.2 } else { (1.0 - confidence) / 0.2 }
                          } else { 0.0 });
        memberships.insert("very_high".to_string(), 
                          if confidence >= 0.8 { (confidence - 0.8) / 0.2 } else { 0.0 });

        memberships
    }

    fn create_context(&self, trajectory: &[GpsPoint], evidence: &[AutobahnEvidence]) -> AutobahnContext {
        let lats: Vec<f64> = trajectory.iter().map(|p| p.latitude).collect();
        let lons: Vec<f64> = trajectory.iter().map(|p| p.longitude).collect();
        let confidences: Vec<f64> = trajectory.iter().map(|p| p.confidence).collect();
        let timestamps: Vec<f64> = trajectory.iter().map(|p| p.timestamp).collect();

        let spatial_bounds = SpatialBounds {
            lat_min: lats.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            lat_max: lats.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            lon_min: lons.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            lon_max: lons.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            spatial_extent_meters: self.calculate_spatial_extent(trajectory),
        };

        let temporal_bounds = TemporalBounds {
            start_timestamp: timestamps.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            end_timestamp: timestamps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            duration_seconds: timestamps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - 
                            timestamps.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            sampling_rate_hz: if trajectory.len() > 1 { 
                (trajectory.len() - 1) as f64 / (timestamps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - 
                                                timestamps.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
            } else { 0.0 },
        };

        let confidence_mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let confidence_variance = confidences.iter()
            .map(|x| (x - confidence_mean).powi(2))
            .sum::<f64>() / confidences.len() as f64;
        let confidence_std = confidence_variance.sqrt();

        let mut sorted_confidences = confidences.clone();
        sorted_confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = sorted_confidences[sorted_confidences.len() / 4];
        let q2 = sorted_confidences[sorted_confidences.len() / 2];
        let q3 = sorted_confidences[3 * sorted_confidences.len() / 4];

        AutobahnContext {
            task_type: "trajectory_consciousness_analysis".to_string(),
            trajectory_length: trajectory.len(),
            bayesian_objective_value: 0.75, // Will be updated by Bayesian analysis
            sighthound_version: "hybrid_rust_autobahn_v1.0".to_string(),
            evidence_summary: EvidenceSummary {
                total_evidence_points: evidence.len(),
                evidence_types: evidence.iter().map(|e| e.evidence_type.clone()).collect::<std::collections::HashSet<_>>().into_iter().collect(),
                confidence_distribution: ConfidenceDistribution {
                    mean: confidence_mean,
                    std_dev: confidence_std,
                    min: sorted_confidences[0],
                    max: sorted_confidences[sorted_confidences.len() - 1],
                    quartiles: [q1, q2, q3],
                },
            },
            spatial_bounds,
            temporal_bounds,
        }
    }

    fn calculate_spatial_extent(&self, trajectory: &[GpsPoint]) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }

        let mut max_distance = 0.0;
        for i in 0..trajectory.len() {
            for j in i+1..trajectory.len() {
                let distance = trajectory[i].distance_to(&trajectory[j]);
                if distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        max_distance
    }

    fn determine_primary_reasoning_type(&self, reasoning_tasks: &[String]) -> String {
        if reasoning_tasks.contains(&"consciousness_assessment".to_string()) {
            "consciousness".to_string()
        } else if reasoning_tasks.contains(&"probabilistic_inference".to_string()) {
            "bayesian".to_string()
        } else if reasoning_tasks.contains(&"evidence_fusion".to_string()) {
            "fuzzy".to_string()
        } else {
            "biological".to_string()
        }
    }

    fn calculate_atp_budget(&self, reasoning_tasks: &[String]) -> f64 {
        let base_budget = 150.0;
        let per_task_cost = 50.0;
        let consciousness_bonus = if reasoning_tasks.contains(&"consciousness_assessment".to_string()) { 100.0 } else { 0.0 };
        
        base_budget + (reasoning_tasks.len() as f64 * per_task_cost) + consciousness_bonus
    }

    fn generate_query_text(&self, reasoning_tasks: &[String], trajectory: &[GpsPoint]) -> String {
        let task_descriptions = reasoning_tasks.iter().map(|task| {
            match task.as_str() {
                "consciousness_assessment" => "Assess consciousness emergence and Φ (phi) integration",
                "probabilistic_inference" => "Perform probabilistic inference on trajectory patterns",
                "fire_circle_analysis" => "Analyze fire circle communication complexity patterns",
                "dual_proximity_assessment" => "Evaluate death/life proximity signaling",
                "threat_assessment" => "Biological immune system threat detection",
                "temporal_reasoning" => "Temporal determinism and categorical predeterminism analysis",
                _ => "General consciousness-aware analysis"
            }
        }).collect::<Vec<_>>().join(", ");

        format!(
            "Perform consciousness-aware analysis of GPS trajectory with {} points. Tasks: {}. \
            Apply biological intelligence, fire circle communication analysis, dual-proximity signaling, \
            membrane coherence optimization, and integrated information theory. \
            Assess consciousness emergence, biological immune threats, and temporal determinism patterns.",
            trajectory.len(),
            task_descriptions
        )
    }

    async fn execute_query(&self, connection: &AutobahnConnection, query: &AutobahnQuery) -> Result<AutobahnResponse> {
        match connection.connection_type {
            AutobahnConnectionType::LocalBinary => {
                self.execute_binary_query(connection, query).await
            },
            AutobahnConnectionType::HttpEndpoint => {
                self.execute_http_query(connection, query).await
            },
            AutobahnConnectionType::DirectRustLibrary => {
                // Future: direct Rust library integration
                Err(anyhow::anyhow!("Direct Rust library integration not yet implemented"))
            }
        }
    }

    async fn execute_binary_query(&self, connection: &AutobahnConnection, query: &AutobahnQuery) -> Result<AutobahnResponse> {
        let binary_path = connection.binary_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Binary path not set"))?;

        let input_json = serde_json::to_string(query)?;

        let mut child = TokioCommand::new(binary_path)
            .arg("--mode")
            .arg("consciousness-reasoning")
            .arg("--input")
            .arg("stdin")
            .arg("--output")
            .arg("json")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Write input
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(input_json.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        // Read output
        let output = child.wait_with_output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Autobahn binary failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let response: AutobahnResponse = serde_json::from_str(&stdout)?;

        Ok(response)
    }

    async fn execute_http_query(&self, connection: &AutobahnConnection, query: &AutobahnQuery) -> Result<AutobahnResponse> {
        let client = connection.client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("HTTP client not set"))?;
        
        let endpoint = connection.endpoint.as_ref()
            .ok_or_else(|| anyhow::anyhow!("HTTP endpoint not set"))?;

        let response = client
            .post(&format!("{}/consciousness-reasoning", endpoint))
            .json(query)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("HTTP request failed: {}", error_text));
        }

        let autobahn_response: AutobahnResponse = response.json().await?;
        Ok(autobahn_response)
    }

    async fn update_performance_stats(&self, response: &AutobahnResponse, duration: Duration) {
        let mut stats = self.performance_monitor.write().unwrap();
        
        stats.total_queries += 1;
        stats.successful_queries += 1;
        
        // Update average response time
        let new_time = duration.as_millis() as f64;
        stats.average_response_time_ms = 
            (stats.average_response_time_ms * (stats.total_queries - 1) as f64 + new_time) / stats.total_queries as f64;

        // Update consciousness metrics
        if response.consciousness_metrics.consciousness_level > 0.7 {
            stats.consciousness_emergence_count += 1;
        }
        
        if response.biological_intelligence.biological_processing_score > 0.5 {
            stats.biological_intelligence_activations += 1;
        }
        
        if response.fire_circle_analysis.communication_complexity_score > 0.5 {
            stats.fire_circle_detections += 1;
        }
        
        if response.threat_assessment.threat_level != "safe" {
            stats.threat_detections += 1;
        }
    }

    fn create_fallback_response(&self, error: &str) -> AutobahnResponse {
        AutobahnResponse {
            query_id: Uuid::new_v4().to_string(),
            response_content: format!("Fallback response due to error: {}", error),
            quality_score: 0.3,
            consciousness_metrics: ConsciousnessMetrics {
                phi_value: 0.0,
                consciousness_level: 0.0,
                global_workspace_activation: 0.0,
                self_awareness_score: 0.0,
                metacognition_level: 0.0,
                qualia_generation_active: false,
                agency_illusion_strength: 0.0,
                persistence_illusion_strength: 0.0,
            },
            biological_intelligence: BiologicalIntelligence {
                membrane_coherence: 0.0,
                ion_channel_optimization: 0.0,
                atp_consumption: 0.0,
                metabolic_efficiency: 0.0,
                biological_processing_score: 0.0,
                environment_assisted_transport: 0.0,
                fire_light_coupling_650nm: 0.0,
                temperature_optimization_310k: 0.0,
            },
            fire_circle_analysis: FireCircleAnalysis {
                communication_complexity_score: 0.0,
                temporal_coordination_detected: false,
                sedentary_period_optimization: 0.0,
                non_action_communication_patterns: vec![],
                abstract_conceptualization_level: 0.0,
                seventy_nine_fold_amplification: 0.0,
            },
            dual_proximity_signals: DualProximitySignals {
                death_proximity_signaling: 0.0,
                life_proximity_signaling: 0.0,
                mortality_risk_assessment: 0.0,
                vitality_detection_score: 0.0,
                leadership_hierarchy_indicators: vec![],
                reproductive_strategy_patterns: vec![],
            },
            credibility_assessment: CredibilityAssessment {
                beauty_credibility_efficiency: 0.0,
                social_function_optimization: 0.0,
                contextual_expectation_alignment: 0.0,
                truth_inversion_detection: false,
                credibility_paradox_score: 0.0,
            },
            temporal_analysis: TemporalAnalysis {
                predetermined_event_confidence: 0.0,
                categorical_completion_progress: 0.0,
                mathematical_structure_navigation: 0.0,
                recognition_space_position: vec![],
                temporal_determinism_strength: 0.0,
            },
            threat_assessment: ThreatAssessment {
                threat_level: "unknown".to_string(),
                immune_system_activation: false,
                t_cell_response: 0.0,
                b_cell_response: 0.0,
                memory_cell_activation: 0.0,
                coherence_interference_detected: false,
                adversarial_patterns: vec![],
                recommended_action: "fallback_mode".to_string(),
            },
            performance_metrics: PerformanceMetrics {
                processing_time_ms: 0,
                atp_efficiency: 0.0,
                oscillatory_efficiency: 0.0,
                entropy_optimization: 0.0,
                network_efficiency: 0.0,
                biological_optimization: 0.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("fallback".to_string(), serde_json::Value::Bool(true));
                meta.insert("error".to_string(), serde_json::Value::String(error.to_string()));
                meta
            },
        }
    }
}

/// High-level functions for consciousness-aware trajectory analysis
#[pyfunction]
pub fn analyze_trajectory_consciousness_rust(trajectory: Vec<GpsPoint>,
                                           reasoning_tasks: Option<Vec<String>>,
                                           metabolic_mode: Option<String>,
                                           hierarchy_level: Option<String>) -> PyResult<HashMap<String, serde_json::Value>> {
    
    let client = AutobahnClient::new();
    let tasks = reasoning_tasks.unwrap_or_else(|| vec![
        "consciousness_assessment".to_string(),
        "probabilistic_inference".to_string(),
        "biological_intelligence".to_string()
    ]);
    
    Python::with_gil(|py| {
        client.query_consciousness_reasoning(py, trajectory, tasks, metabolic_mode, hierarchy_level)
    })
}

#[pyfunction]
pub fn batch_analyze_consciousness_rust(trajectories: Vec<Vec<GpsPoint>>,
                                      reasoning_tasks: Option<Vec<String>>,
                                      parallel: Option<bool>) -> PyResult<Vec<HashMap<String, serde_json::Value>>> {
    
    let client = AutobahnClient::new();
    let tasks = reasoning_tasks.unwrap_or_else(|| vec![
        "consciousness_assessment".to_string(),
        "threat_assessment".to_string(),
        "biological_intelligence".to_string()
    ]);
    
    Python::with_gil(|py| {
        client.batch_consciousness_analysis(py, trajectories, tasks, parallel)
    })
}

#[pyfunction]
pub fn get_autobahn_performance_stats() -> HashMap<String, f64> {
    let client = AutobahnClient::new();
    client.get_performance_stats()
}

/// Python module definition
#[pymodule]
fn sighthound_autobahn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AutobahnClient>()?;
    m.add_function(wrap_pyfunction!(analyze_trajectory_consciousness_rust, m)?)?;
    m.add_function(wrap_pyfunction!(batch_analyze_consciousness_rust, m)?)?;
    m.add_function(wrap_pyfunction!(get_autobahn_performance_stats, m)?)?;
    Ok(())
} 