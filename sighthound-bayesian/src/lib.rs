use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use nalgebra::{DMatrix, DVector, Vector2, Vector3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use dashmap::DashMap;
use petgraph::{Graph, Directed, Direction};
use petgraph::graph::{NodeIndex, EdgeIndex};
use sighthound_core::GpsPoint;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Fuzzy membership function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FuzzyMembership {
    Triangular { a: f64, b: f64, c: f64 },
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    Gaussian { mean: f64, std_dev: f64 },
    Sigmoid { center: f64, slope: f64 },
    Linear { min: f64, max: f64 },
}

impl FuzzyMembership {
    /// Calculate membership value for given input
    pub fn membership(&self, x: f64) -> f64 {
        match self {
            FuzzyMembership::Triangular { a, b, c } => {
                if x <= *a || x >= *c {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else {
                    (c - x) / (c - b)
                }
            }
            FuzzyMembership::Trapezoidal { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else if x <= *c {
                    1.0
                } else {
                    (d - x) / (d - c)
                }
            }
            FuzzyMembership::Gaussian { mean, std_dev } => {
                let exp_arg = -0.5 * ((x - mean) / std_dev).powi(2);
                exp_arg.exp()
            }
            FuzzyMembership::Sigmoid { center, slope } => {
                1.0 / (1.0 + (-slope * (x - center)).exp())
            }
            FuzzyMembership::Linear { min, max } => {
                if x <= *min {
                    0.0
                } else if x >= *max {
                    1.0
                } else {
                    (x - min) / (max - min)
                }
            }
        }
    }
}

/// Fuzzy set with linguistic label and membership function
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySet {
    #[pyo3(get, set)]
    pub label: String,
    pub membership_function: FuzzyMembership,
    #[pyo3(get, set)]
    pub support_min: f64,
    #[pyo3(get, set)]
    pub support_max: f64,
}

#[pymethods]
impl FuzzySet {
    #[new]
    pub fn new(label: String, membership_function: FuzzyMembership, support_min: f64, support_max: f64) -> Self {
        Self {
            label,
            membership_function,
            support_min,
            support_max,
        }
    }

    /// Calculate membership degree for a value
    pub fn membership_degree(&self, value: f64) -> f64 {
        if value < self.support_min || value > self.support_max {
            0.0
        } else {
            self.membership_function.membership(value)
        }
    }

    /// Create fuzzy set for GPS confidence levels
    #[staticmethod]
    pub fn gps_confidence_sets() -> Vec<FuzzySet> {
        vec![
            FuzzySet::new(
                "very_low".to_string(),
                FuzzyMembership::Trapezoidal { a: 0.0, b: 0.0, c: 0.2, d: 0.4 },
                0.0, 1.0
            ),
            FuzzySet::new(
                "low".to_string(),
                FuzzyMembership::Triangular { a: 0.2, b: 0.4, c: 0.6 },
                0.0, 1.0
            ),
            FuzzySet::new(
                "medium".to_string(),
                FuzzyMembership::Triangular { a: 0.4, b: 0.6, c: 0.8 },
                0.0, 1.0
            ),
            FuzzySet::new(
                "high".to_string(),
                FuzzyMembership::Triangular { a: 0.6, b: 0.8, c: 1.0 },
                0.0, 1.0
            ),
            FuzzySet::new(
                "very_high".to_string(),
                FuzzyMembership::Trapezoidal { a: 0.8, b: 0.9, c: 1.0, d: 1.0 },
                0.0, 1.0
            ),
        ]
    }

    /// Create fuzzy sets for signal strength
    #[staticmethod]
    pub fn signal_strength_sets() -> Vec<FuzzySet> {
        vec![
            FuzzySet::new(
                "very_weak".to_string(),
                FuzzyMembership::Trapezoidal { a: -120.0, b: -120.0, c: -110.0, d: -100.0 },
                -120.0, -30.0
            ),
            FuzzySet::new(
                "weak".to_string(),
                FuzzyMembership::Triangular { a: -110.0, b: -95.0, c: -80.0 },
                -120.0, -30.0
            ),
            FuzzySet::new(
                "moderate".to_string(),
                FuzzyMembership::Triangular { a: -90.0, b: -75.0, c: -60.0 },
                -120.0, -30.0
            ),
            FuzzySet::new(
                "strong".to_string(),
                FuzzyMembership::Triangular { a: -70.0, b: -55.0, c: -40.0 },
                -120.0, -30.0
            ),
            FuzzySet::new(
                "very_strong".to_string(),
                FuzzyMembership::Trapezoidal { a: -50.0, b: -40.0, c: -30.0, d: -30.0 },
                -120.0, -30.0
            ),
        ]
    }
}

/// Fuzzy evidence that combines crisp values with fuzzy confidence
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyEvidence {
    #[pyo3(get, set)]
    pub variable_name: String,
    #[pyo3(get, set)]
    pub crisp_value: f64,
    #[pyo3(get, set)]
    pub confidence: f64,
    pub fuzzy_confidence: HashMap<String, f64>, // Linguistic confidence levels
    #[pyo3(get, set)]
    pub timestamp: f64,
    #[pyo3(get, set)]
    pub source: String,
}

#[pymethods]
impl FuzzyEvidence {
    #[new]
    pub fn new(
        variable_name: String,
        crisp_value: f64,
        confidence: f64,
        timestamp: f64,
        source: String,
    ) -> Self {
        let confidence_sets = FuzzySet::gps_confidence_sets();
        let mut fuzzy_confidence = HashMap::new();
        
        for set in confidence_sets {
            fuzzy_confidence.insert(set.label.clone(), set.membership_degree(confidence));
        }

        Self {
            variable_name,
            crisp_value,
            confidence,
            fuzzy_confidence,
            timestamp,
            source,
        }
    }

    /// Create GPS position evidence
    #[staticmethod]
    pub fn from_gps_point(point: &GpsPoint, variable_type: String) -> FuzzyEvidence {
        let value = match variable_type.as_str() {
            "latitude" => point.latitude,
            "longitude" => point.longitude,
            _ => 0.0,
        };

        FuzzyEvidence::new(
            variable_type,
            value,
            point.confidence,
            point.timestamp,
            "gps".to_string(),
        )
    }

    /// Aggregate fuzzy confidence using max operator
    pub fn max_confidence(&self) -> f64 {
        self.fuzzy_confidence.values().fold(0.0, |acc, &x| acc.max(x))
    }

    /// Get dominant linguistic confidence label
    pub fn dominant_confidence_label(&self) -> String {
        self.fuzzy_confidence
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(label, _)| label.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// Node in the Bayesian Evidence Network
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianNode {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub node_type: String, // "observed", "latent", "decision"
    #[pyo3(get, set)]
    pub variable_name: String,
    pub current_belief: DashMap<String, f64>, // Belief distribution over states
    pub prior_belief: HashMap<String, f64>,
    pub evidence_buffer: Vec<FuzzyEvidence>,
    #[pyo3(get, set)]
    pub last_update: f64,
}

#[pymethods]
impl BayesianNode {
    #[new]
    pub fn new(id: String, node_type: String, variable_name: String) -> Self {
        let mut prior_belief = HashMap::new();
        prior_belief.insert("low".to_string(), 0.33);
        prior_belief.insert("medium".to_string(), 0.34);
        prior_belief.insert("high".to_string(), 0.33);

        let current_belief = DashMap::new();
        for (state, prob) in &prior_belief {
            current_belief.insert(state.clone(), *prob);
        }

        Self {
            id,
            node_type,
            variable_name,
            current_belief,
            prior_belief,
            evidence_buffer: Vec::new(),
            last_update: 0.0,
        }
    }

    /// Add fuzzy evidence to the node
    pub fn add_evidence(&mut self, evidence: FuzzyEvidence) {
        self.evidence_buffer.push(evidence);
        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
    }

    /// Update belief using fuzzy evidence aggregation
    pub fn update_belief(&mut self) {
        if self.evidence_buffer.is_empty() {
            return;
        }

        // Aggregate evidence using fuzzy operators
        let mut aggregated_confidence = HashMap::new();
        for evidence in &self.evidence_buffer {
            for (label, confidence) in &evidence.fuzzy_confidence {
                let current = aggregated_confidence.get(label).unwrap_or(&0.0);
                // Use algebraic sum for aggregation: a ⊕ b = a + b - ab
                let new_conf = current + confidence - (current * confidence);
                aggregated_confidence.insert(label.clone(), new_conf);
            }
        }

        // Update beliefs based on aggregated evidence
        let total_evidence: f64 = aggregated_confidence.values().sum();
        if total_evidence > 0.0 {
            for (state, evidence_strength) in aggregated_confidence {
                let normalized_evidence = evidence_strength / total_evidence;
                let prior = self.prior_belief.get(&state).unwrap_or(&0.0);
                
                // Bayesian update: P(H|E) ∝ P(E|H) * P(H)
                let likelihood = normalized_evidence;
                let posterior = likelihood * prior;
                
                self.current_belief.insert(state, posterior);
            }
            
            // Normalize beliefs
            self.normalize_beliefs();
        }

        // Clear processed evidence
        self.evidence_buffer.clear();
    }

    fn normalize_beliefs(&self) {
        let total: f64 = self.current_belief.iter().map(|entry| *entry.value()).sum();
        if total > 0.0 {
            for mut entry in self.current_belief.iter_mut() {
                *entry.value_mut() /= total;
            }
        }
    }

    /// Get belief entropy (measure of uncertainty)
    pub fn belief_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for entry in self.current_belief.iter() {
            let p = *entry.value();
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Get maximum a posteriori (MAP) state
    pub fn map_state(&self) -> (String, f64) {
        self.current_belief
            .iter()
            .max_by(|a, b| a.value().partial_cmp(b.value()).unwrap())
            .map(|entry| (entry.key().clone(), *entry.value()))
            .unwrap_or_else(|| ("unknown".to_string(), 0.0))
    }
}

/// Edge in the Bayesian Network representing conditional dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianEdge {
    pub from_node: String,
    pub to_node: String,
    pub conditional_probability_table: HashMap<(String, String), f64>, // P(child|parent)
    pub edge_weight: f64,
    pub edge_type: String, // "causal", "evidential", "temporal"
}

impl BayesianEdge {
    pub fn new(from_node: String, to_node: String, edge_type: String) -> Self {
        Self {
            from_node,
            to_node,
            conditional_probability_table: HashMap::new(),
            edge_weight: 1.0,
            edge_type,
        }
    }

    /// Set conditional probability P(child_state | parent_state)
    pub fn set_conditional_probability(&mut self, parent_state: String, child_state: String, probability: f64) {
        self.conditional_probability_table.insert((parent_state, child_state), probability);
    }
}

/// Bayesian Evidence Network for trajectory analysis
#[pyclass]
pub struct BayesianEvidenceNetwork {
    graph: Graph<BayesianNode, BayesianEdge, Directed>,
    node_indices: HashMap<String, NodeIndex>,
    objective_function: ObjectiveFunction,
    fuzzy_rule_base: FuzzyRuleBase,
}

/// Objective function for optimization
#[pyclass]
#[derive(Debug, Clone)]
pub struct ObjectiveFunction {
    #[pyo3(get, set)]
    pub name: String,
    weights: HashMap<String, f64>,
    #[pyo3(get, set)]
    pub current_value: f64,
    #[pyo3(get, set)]
    pub target_value: f64,
}

#[pymethods]
impl ObjectiveFunction {
    #[new]
    pub fn new(name: String) -> Self {
        let mut weights = HashMap::new();
        weights.insert("trajectory_smoothness".to_string(), 0.3);
        weights.insert("evidence_consistency".to_string(), 0.25);
        weights.insert("confidence_maximization".to_string(), 0.2);
        weights.insert("uncertainty_minimization".to_string(), 0.15);
        weights.insert("temporal_coherence".to_string(), 0.1);

        Self {
            name,
            weights,
            current_value: 0.0,
            target_value: 1.0,
        }
    }

    /// Calculate objective function value based on network state
    pub fn evaluate(&mut self, network: &BayesianEvidenceNetwork) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        // Trajectory smoothness: penalize abrupt changes
        if let Some(smoothness_score) = self.calculate_trajectory_smoothness(network) {
            total_score += smoothness_score * self.weights.get("trajectory_smoothness").unwrap_or(&0.0);
            total_weight += self.weights.get("trajectory_smoothness").unwrap_or(&0.0);
        }

        // Evidence consistency: reward consistent evidence
        let consistency_score = self.calculate_evidence_consistency(network);
        total_score += consistency_score * self.weights.get("evidence_consistency").unwrap_or(&0.0);
        total_weight += self.weights.get("evidence_consistency").unwrap_or(&0.0);

        // Confidence maximization: prefer high-confidence solutions
        let confidence_score = self.calculate_confidence_score(network);
        total_score += confidence_score * self.weights.get("confidence_maximization").unwrap_or(&0.0);
        total_weight += self.weights.get("confidence_maximization").unwrap_or(&0.0);

        // Uncertainty minimization: prefer low-entropy beliefs
        let uncertainty_score = self.calculate_uncertainty_score(network);
        total_score += uncertainty_score * self.weights.get("uncertainty_minimization").unwrap_or(&0.0);
        total_weight += self.weights.get("uncertainty_minimization").unwrap_or(&0.0);

        self.current_value = if total_weight > 0.0 { total_score / total_weight } else { 0.0 };
        self.current_value
    }

    fn calculate_trajectory_smoothness(&self, network: &BayesianEvidenceNetwork) -> Option<f64> {
        // Calculate smoothness based on trajectory nodes
        let mut smoothness_scores = Vec::new();
        
        for node_idx in network.graph.node_indices() {
            let node = &network.graph[node_idx];
            if node.variable_name.contains("position") {
                let entropy = node.belief_entropy();
                let smoothness = 1.0 / (1.0 + entropy); // Higher entropy = less smooth
                smoothness_scores.push(smoothness);
            }
        }

        if smoothness_scores.is_empty() {
            None
        } else {
            Some(smoothness_scores.iter().sum::<f64>() / smoothness_scores.len() as f64)
        }
    }

    fn calculate_evidence_consistency(&self, network: &BayesianEvidenceNetwork) -> f64 {
        // Calculate how consistent evidence is across nodes
        let mut consistency_scores = Vec::new();
        
        for node_idx in network.graph.node_indices() {
            let node = &network.graph[node_idx];
            let (_, max_belief) = node.map_state();
            consistency_scores.push(max_belief);
        }

        if consistency_scores.is_empty() {
            0.0
        } else {
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        }
    }

    fn calculate_confidence_score(&self, network: &BayesianEvidenceNetwork) -> f64 {
        // Average confidence across all nodes
        let mut confidence_sum = 0.0;
        let mut node_count = 0;

        for node_idx in network.graph.node_indices() {
            let node = &network.graph[node_idx];
            let (_, confidence) = node.map_state();
            confidence_sum += confidence;
            node_count += 1;
        }

        if node_count > 0 {
            confidence_sum / node_count as f64
        } else {
            0.0
        }
    }

    fn calculate_uncertainty_score(&self, network: &BayesianEvidenceNetwork) -> f64 {
        // Inverse of average entropy (lower entropy = higher score)
        let mut entropy_sum = 0.0;
        let mut node_count = 0;

        for node_idx in network.graph.node_indices() {
            let node = &network.graph[node_idx];
            entropy_sum += node.belief_entropy();
            node_count += 1;
        }

        if node_count > 0 {
            let avg_entropy = entropy_sum / node_count as f64;
            1.0 / (1.0 + avg_entropy)
        } else {
            0.0
        }
    }
}

/// Fuzzy rule base for inference
#[derive(Debug, Clone)]
pub struct FuzzyRuleBase {
    rules: Vec<FuzzyRule>,
}

#[derive(Debug, Clone)]
pub struct FuzzyRule {
    antecedent: Vec<(String, String)>, // (variable, fuzzy_set)
    consequent: (String, String), // (variable, fuzzy_set)
    confidence: f64,
}

impl FuzzyRuleBase {
    pub fn new() -> Self {
        let mut rules = Vec::new();
        
        // Example rules for GPS trajectory analysis
        rules.push(FuzzyRule {
            antecedent: vec![("gps_confidence".to_string(), "high".to_string())],
            consequent: ("position_reliability".to_string(), "high".to_string()),
            confidence: 0.9,
        });

        rules.push(FuzzyRule {
            antecedent: vec![
                ("signal_strength".to_string(), "strong".to_string()),
                ("gps_confidence".to_string(), "medium".to_string())
            ],
            consequent: ("position_reliability".to_string(), "high".to_string()),
            confidence: 0.8,
        });

        rules.push(FuzzyRule {
            antecedent: vec![("gps_confidence".to_string(), "low".to_string())],
            consequent: ("position_reliability".to_string(), "low".to_string()),
            confidence: 0.85,
        });

        Self { rules }
    }

    /// Apply fuzzy rules to derive new evidence
    pub fn apply_rules(&self, evidence: &[FuzzyEvidence]) -> Vec<FuzzyEvidence> {
        let mut derived_evidence = Vec::new();
        
        for rule in &self.rules {
            let mut rule_activation = 1.0;
            let mut can_fire = true;
            
            // Check if all antecedents are satisfied
            for (var_name, fuzzy_set) in &rule.antecedent {
                let matching_evidence = evidence.iter()
                    .find(|e| e.variable_name == *var_name);
                
                if let Some(ev) = matching_evidence {
                    let membership = ev.fuzzy_confidence.get(fuzzy_set).unwrap_or(&0.0);
                    rule_activation = rule_activation.min(*membership);
                } else {
                    can_fire = false;
                    break;
                }
            }
            
            // Fire rule if possible
            if can_fire && rule_activation > 0.1 { // Minimum activation threshold
                let (consequent_var, consequent_set) = &rule.consequent;
                let confidence = rule_activation * rule.confidence;
                
                let new_evidence = FuzzyEvidence::new(
                    consequent_var.clone(),
                    confidence, // Use activation as crisp value
                    confidence,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                    "fuzzy_inference".to_string(),
                );
                
                derived_evidence.push(new_evidence);
            }
        }
        
        derived_evidence
    }
}

#[pymethods]
impl BayesianEvidenceNetwork {
    #[new]
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            objective_function: ObjectiveFunction::new("trajectory_optimization".to_string()),
            fuzzy_rule_base: FuzzyRuleBase::new(),
        }
    }

    /// Add a node to the network
    pub fn add_node(&mut self, node: BayesianNode) -> String {
        let node_id = node.id.clone();
        let node_idx = self.graph.add_node(node);
        self.node_indices.insert(node_id.clone(), node_idx);
        node_id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from_id: String, to_id: String, edge: BayesianEdge) -> bool {
        if let (Some(&from_idx), Some(&to_idx)) = 
            (self.node_indices.get(&from_id), self.node_indices.get(&to_id)) {
            self.graph.add_edge(from_idx, to_idx, edge);
            true
        } else {
            false
        }
    }

    /// Process GPS trajectory as fuzzy evidence
    pub fn process_gps_trajectory(&mut self, trajectory: Vec<GpsPoint>) -> f64 {
        // Create nodes for position variables if they don't exist
        self.ensure_position_nodes();
        
        // Convert GPS points to fuzzy evidence
        let mut all_evidence = Vec::new();
        for point in trajectory {
            let lat_evidence = FuzzyEvidence::from_gps_point(&point, "latitude".to_string());
            let lon_evidence = FuzzyEvidence::from_gps_point(&point, "longitude".to_string());
            all_evidence.push(lat_evidence);
            all_evidence.push(lon_evidence);
        }

        // Apply fuzzy rules to derive additional evidence
        let derived_evidence = self.fuzzy_rule_base.apply_rules(&all_evidence);
        all_evidence.extend(derived_evidence);

        // Distribute evidence to appropriate nodes
        self.distribute_evidence(all_evidence);

        // Perform belief propagation
        self.belief_propagation();

        // Evaluate objective function
        self.objective_function.evaluate(self)
    }

    fn ensure_position_nodes(&mut self) {
        let required_nodes = vec![
            ("latitude_node", "latent", "latitude"),
            ("longitude_node", "latent", "longitude"),
            ("position_reliability_node", "latent", "position_reliability"),
            ("trajectory_smoothness_node", "latent", "trajectory_smoothness"),
        ];

        for (node_id, node_type, var_name) in required_nodes {
            if !self.node_indices.contains_key(node_id) {
                let node = BayesianNode::new(
                    node_id.to_string(),
                    node_type.to_string(),
                    var_name.to_string(),
                );
                self.add_node(node);
            }
        }

        // Add edges between nodes
        let edges = vec![
            ("latitude_node", "position_reliability_node", "evidential"),
            ("longitude_node", "position_reliability_node", "evidential"),
            ("position_reliability_node", "trajectory_smoothness_node", "causal"),
        ];

        for (from, to, edge_type) in edges {
            let edge = BayesianEdge::new(from.to_string(), to.to_string(), edge_type.to_string());
            self.add_edge(from.to_string(), to.to_string(), edge);
        }
    }

    fn distribute_evidence(&mut self, evidence: Vec<FuzzyEvidence>) {
        for ev in evidence {
            // Find matching node
            for node_idx in self.graph.node_indices() {
                let mut node = &mut self.graph[node_idx];
                if node.variable_name == ev.variable_name {
                    node.add_evidence(ev.clone());
                    break;
                }
            }
        }
    }

    /// Perform belief propagation through the network
    pub fn belief_propagation(&mut self) {
        // Update all nodes with their evidence
        for node_idx in self.graph.node_indices() {
            let node = &mut self.graph[node_idx];
            node.update_belief();
        }

        // Message passing between connected nodes
        for edge_idx in self.graph.edge_indices() {
            let edge = &self.graph[edge_idx];
            let (from_idx, to_idx) = self.graph.edge_endpoints(edge_idx).unwrap();
            
            // Get parent belief
            let parent_belief = {
                let parent_node = &self.graph[from_idx];
                parent_node.current_belief.iter()
                    .map(|entry| (entry.key().clone(), *entry.value()))
                    .collect::<HashMap<String, f64>>()
            };

            // Propagate belief to child
            self.propagate_belief_to_child(to_idx, parent_belief, &edge.conditional_probability_table);
        }
    }

    fn propagate_belief_to_child(
        &mut self,
        child_idx: NodeIndex,
        parent_belief: HashMap<String, f64>,
        cpt: &HashMap<(String, String), f64>,
    ) {
        let child_node = &mut self.graph[child_idx];
        
        for parent_state in parent_belief.keys() {
            let parent_prob = parent_belief.get(parent_state).unwrap_or(&0.0);
            
            for entry in child_node.current_belief.iter_mut() {
                let child_state = entry.key();
                let conditional_prob = cpt.get(&(parent_state.clone(), child_state.clone())).unwrap_or(&0.5);
                let new_belief = *entry.value() + (parent_prob * conditional_prob);
                *entry.value_mut() = new_belief;
            }
        }
        
        child_node.normalize_beliefs();
    }

    /// Optimize the objective function using gradient-free methods
    pub fn optimize_objective(&mut self, max_iterations: usize) -> f64 {
        let mut best_score = self.objective_function.evaluate(self);
        let mut iteration = 0;
        
        while iteration < max_iterations {
            // Simulated annealing approach
            let temperature = 1.0 - (iteration as f64 / max_iterations as f64);
            
            // Perturb network state slightly
            self.perturb_network_state(temperature);
            
            // Perform belief propagation
            self.belief_propagation();
            
            // Evaluate new score
            let new_score = self.objective_function.evaluate(self);
            
            // Accept or reject based on simulated annealing criteria
            if new_score > best_score || 
               (temperature > 0.0 && (-(best_score - new_score) / temperature).exp() > rand::random::<f64>()) {
                best_score = new_score;
            } else {
                // Revert changes (simplified - in practice would need state backup)
                self.belief_propagation();
            }
            
            iteration += 1;
        }
        
        best_score
    }

    fn perturb_network_state(&mut self, temperature: f64) {
        // Add small random perturbations to node beliefs
        for node_idx in self.graph.node_indices() {
            let node = &mut self.graph[node_idx];
            
            for mut entry in node.current_belief.iter_mut() {
                let perturbation = (rand::random::<f64>() - 0.5) * temperature * 0.1;
                let new_value = (*entry.value() + perturbation).max(0.0).min(1.0);
                *entry.value_mut() = new_value;
            }
            
            node.normalize_beliefs();
        }
    }

    /// Get current network state as optimization result
    pub fn get_optimization_result(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        
        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];
            let (best_state, confidence) = node.map_state();
            result.insert(format!("{}_{}", node.variable_name, best_state), confidence);
            result.insert(format!("{}_entropy", node.variable_name), node.belief_entropy());
        }
        
        result.insert("objective_value".to_string(), self.objective_function.current_value);
        result
    }

    /// Get network statistics
    pub fn get_network_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("num_nodes".to_string(), self.graph.node_count() as f64);
        stats.insert("num_edges".to_string(), self.graph.edge_count() as f64);
        
        let total_entropy: f64 = self.graph.node_weights()
            .map(|node| node.belief_entropy())
            .sum();
        stats.insert("total_entropy".to_string(), total_entropy);
        stats.insert("avg_entropy".to_string(), total_entropy / self.graph.node_count() as f64);
        
        stats.insert("objective_value".to_string(), self.objective_function.current_value);
        
        stats
    }
}

/// High-level trajectory analysis using Bayesian Evidence Network
#[pyfunction]
pub fn analyze_trajectory_bayesian(
    trajectory: Vec<GpsPoint>,
    max_optimization_iterations: Option<usize>,
) -> HashMap<String, f64> {
    let mut network = BayesianEvidenceNetwork::new();
    
    // Process trajectory
    let initial_score = network.process_gps_trajectory(trajectory);
    
    // Optimize
    let iterations = max_optimization_iterations.unwrap_or(100);
    let final_score = network.optimize_objective(iterations);
    
    // Return results
    let mut result = network.get_optimization_result();
    result.insert("initial_objective".to_string(), initial_score);
    result.insert("final_objective".to_string(), final_score);
    result.insert("improvement".to_string(), final_score - initial_score);
    
    result
}

/// Batch analysis of multiple trajectories
#[pyfunction]
pub fn batch_analyze_trajectories_bayesian(
    trajectories: Vec<Vec<GpsPoint>>,
    max_optimization_iterations: Option<usize>,
) -> Vec<HashMap<String, f64>> {
    trajectories
        .into_par_iter()
        .map(|trajectory| analyze_trajectory_bayesian(trajectory, max_optimization_iterations))
        .collect()
}

/// Python module definition
#[pymodule]
fn sighthound_bayesian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FuzzySet>()?;
    m.add_class::<FuzzyEvidence>()?;
    m.add_class::<BayesianNode>()?;
    m.add_class::<BayesianEvidenceNetwork>()?;
    m.add_class::<ObjectiveFunction>()?;
    m.add_function(wrap_pyfunction!(analyze_trajectory_bayesian, m)?)?;
    m.add_function(wrap_pyfunction!(batch_analyze_trajectories_bayesian, m)?)?;
    Ok(())
} 