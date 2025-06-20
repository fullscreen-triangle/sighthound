use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use nalgebra::{DMatrix, DVector, Vector2, Vector3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, RwLock};
use dashmap::DashMap;
use sighthound_core::GpsPoint;
use sighthound_bayesian::{BayesianEvidenceNetwork, FuzzyEvidence, ObjectiveFunction};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Fuzzy T-norm and T-conorm operators for combining evidence
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FuzzyOperator {
    // T-norms (AND operations)
    Minimum,
    Product,
    LukasiewiczTNorm,
    DrasticTNorm,
    
    // T-conorms (OR operations)
    Maximum,
    ProbabilisticSum,
    LukasiewiczTConorm,
    DrasticTConorm,
    
    // Other operations
    AlgebraicSum,
    BoundedSum,
    Einstein,
}

impl FuzzyOperator {
    /// Apply T-norm operation
    pub fn apply_tnorm(&self, a: f64, b: f64) -> f64 {
        match self {
            FuzzyOperator::Minimum => a.min(b),
            FuzzyOperator::Product => a * b,
            FuzzyOperator::LukasiewiczTNorm => (a + b - 1.0).max(0.0),
            FuzzyOperator::DrasticTNorm => {
                if a == 1.0 { b } 
                else if b == 1.0 { a } 
                else { 0.0 }
            },
            _ => a.min(b), // Default to minimum
        }
    }

    /// Apply T-conorm operation
    pub fn apply_tconorm(&self, a: f64, b: f64) -> f64 {
        match self {
            FuzzyOperator::Maximum => a.max(b),
            FuzzyOperator::ProbabilisticSum => a + b - (a * b),
            FuzzyOperator::LukasiewiczTConorm => (a + b).min(1.0),
            FuzzyOperator::DrasticTConorm => {
                if a == 0.0 { b } 
                else if b == 0.0 { a } 
                else { 1.0 }
            },
            FuzzyOperator::AlgebraicSum => a + b - (a * b),
            FuzzyOperator::BoundedSum => (a + b).min(1.0),
            FuzzyOperator::Einstein => (a + b) / (1.0 + a * b),
            _ => a.max(b), // Default to maximum
        }
    }
}

/// Advanced fuzzy inference system
#[pyclass]
#[derive(Debug, Clone)]
pub struct FuzzyInferenceSystem {
    input_variables: HashMap<String, FuzzyVariable>,
    output_variables: HashMap<String, FuzzyVariable>,
    rule_base: Vec<FuzzyRule>,
    defuzzification_method: DefuzzificationMethod,
    tnorm_operator: FuzzyOperator,
    tconorm_operator: FuzzyOperator,
}

#[derive(Debug, Clone)]
pub struct FuzzyVariable {
    name: String,
    domain: (f64, f64),
    linguistic_terms: HashMap<String, LinguisticTerm>,
}

#[derive(Debug, Clone)]
pub struct LinguisticTerm {
    name: String,
    membership_function: MembershipFunction,
}

#[derive(Debug, Clone)]
pub enum MembershipFunction {
    Triangular { a: f64, b: f64, c: f64 },
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    Gaussian { mean: f64, sigma: f64 },
    Bell { a: f64, b: f64, c: f64 },
    Sigmoid { a: f64, c: f64 },
    ZShape { a: f64, b: f64 },
    SShape { a: f64, b: f64 },
}

impl MembershipFunction {
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            MembershipFunction::Triangular { a, b, c } => {
                if x <= *a || x >= *c {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else {
                    (c - x) / (c - b)
                }
            },
            MembershipFunction::Trapezoidal { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else if x <= *c {
                    1.0
                } else {
                    (d - x) / (d - c)
                }
            },
            MembershipFunction::Gaussian { mean, sigma } => {
                let exponent = -0.5 * ((x - mean) / sigma).powi(2);
                exponent.exp()
            },
            MembershipFunction::Bell { a, b, c } => {
                1.0 / (1.0 + ((x - c) / a).abs().powf(2.0 * b))
            },
            MembershipFunction::Sigmoid { a, c } => {
                1.0 / (1.0 + (-a * (x - c)).exp())
            },
            MembershipFunction::ZShape { a, b } => {
                if x <= *a {
                    1.0
                } else if x <= (a + b) / 2.0 {
                    1.0 - 2.0 * ((x - a) / (b - a)).powi(2)
                } else if x <= *b {
                    2.0 * ((x - b) / (b - a)).powi(2)
                } else {
                    0.0
                }
            },
            MembershipFunction::SShape { a, b } => {
                if x <= *a {
                    0.0
                } else if x <= (a + b) / 2.0 {
                    2.0 * ((x - a) / (b - a)).powi(2)
                } else if x <= *b {
                    1.0 - 2.0 * ((x - b) / (b - a)).powi(2)
                } else {
                    1.0
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuzzyRule {
    antecedent: Vec<FuzzyCondition>,
    consequent: Vec<FuzzyConclusion>,
    weight: f64,
    confidence: f64,
}

#[derive(Debug, Clone)]
pub struct FuzzyCondition {
    variable: String,
    term: String,
    negated: bool,
}

#[derive(Debug, Clone)]
pub struct FuzzyConclusion {
    variable: String,
    term: String,
    weight: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum DefuzzificationMethod {
    Centroid,           // Center of gravity
    Bisector,           // Bisector of area
    MeanOfMaximum,      // Mean of maximum
    SmallestOfMaximum,  // Smallest of maximum
    LargestOfMaximum,   // Largest of maximum
}

#[pymethods]
impl FuzzyInferenceSystem {
    #[new]
    pub fn new() -> Self {
        Self {
            input_variables: HashMap::new(),
            output_variables: HashMap::new(),
            rule_base: Vec::new(),
            defuzzification_method: DefuzzificationMethod::Centroid,
            tnorm_operator: FuzzyOperator::Minimum,
            tconorm_operator: FuzzyOperator::Maximum,
        }
    }

    /// Add input variable to the system
    pub fn add_input_variable(&mut self, name: String, domain_min: f64, domain_max: f64) {
        let variable = FuzzyVariable {
            name: name.clone(),
            domain: (domain_min, domain_max),
            linguistic_terms: HashMap::new(),
        };
        self.input_variables.insert(name, variable);
    }

    /// Add output variable to the system
    pub fn add_output_variable(&mut self, name: String, domain_min: f64, domain_max: f64) {
        let variable = FuzzyVariable {
            name: name.clone(),
            domain: (domain_min, domain_max),
            linguistic_terms: HashMap::new(),
        };
        self.output_variables.insert(name, variable);
    }

    /// Create GPS trajectory analysis fuzzy system
    #[staticmethod]
    pub fn create_gps_analysis_system() -> FuzzyInferenceSystem {
        let mut fis = FuzzyInferenceSystem::new();
        
        // Input variables
        fis.add_input_variable("gps_accuracy".to_string(), 0.0, 100.0);
        fis.add_input_variable("signal_strength".to_string(), -120.0, -30.0);
        fis.add_input_variable("speed_consistency".to_string(), 0.0, 1.0);
        fis.add_input_variable("temporal_gap".to_string(), 0.0, 300.0); // seconds
        
        // Output variables
        fis.add_output_variable("position_reliability".to_string(), 0.0, 1.0);
        fis.add_output_variable("trajectory_smoothness".to_string(), 0.0, 1.0);
        fis.add_output_variable("overall_quality".to_string(), 0.0, 1.0);
        
        // Add linguistic terms for GPS accuracy
        if let Some(accuracy_var) = fis.input_variables.get_mut("gps_accuracy") {
            accuracy_var.linguistic_terms.insert("excellent".to_string(), LinguisticTerm {
                name: "excellent".to_string(),
                membership_function: MembershipFunction::Trapezoidal { a: 0.0, b: 0.0, c: 5.0, d: 10.0 },
            });
            accuracy_var.linguistic_terms.insert("good".to_string(), LinguisticTerm {
                name: "good".to_string(),
                membership_function: MembershipFunction::Triangular { a: 5.0, b: 15.0, c: 25.0 },
            });
            accuracy_var.linguistic_terms.insert("fair".to_string(), LinguisticTerm {
                name: "fair".to_string(),
                membership_function: MembershipFunction::Triangular { a: 20.0, b: 40.0, c: 60.0 },
            });
            accuracy_var.linguistic_terms.insert("poor".to_string(), LinguisticTerm {
                name: "poor".to_string(),
                membership_function: MembershipFunction::Trapezoidal { a: 50.0, b: 80.0, c: 100.0, d: 100.0 },
            });
        }
        
        // Add rules
        fis.add_gps_analysis_rules();
        
        fis
    }

    fn add_gps_analysis_rules(&mut self) {
        // Rule 1: Excellent GPS accuracy + Good signal → High reliability
        self.rule_base.push(FuzzyRule {
            antecedent: vec![
                FuzzyCondition { variable: "gps_accuracy".to_string(), term: "excellent".to_string(), negated: false },
                FuzzyCondition { variable: "signal_strength".to_string(), term: "strong".to_string(), negated: false },
            ],
            consequent: vec![
                FuzzyConclusion { variable: "position_reliability".to_string(), term: "high".to_string(), weight: 1.0 },
                FuzzyConclusion { variable: "overall_quality".to_string(), term: "excellent".to_string(), weight: 0.8 },
            ],
            weight: 1.0,
            confidence: 0.95,
        });

        // Rule 2: Poor GPS accuracy → Low reliability
        self.rule_base.push(FuzzyRule {
            antecedent: vec![
                FuzzyCondition { variable: "gps_accuracy".to_string(), term: "poor".to_string(), negated: false },
            ],
            consequent: vec![
                FuzzyConclusion { variable: "position_reliability".to_string(), term: "low".to_string(), weight: 1.0 },
                FuzzyConclusion { variable: "overall_quality".to_string(), term: "poor".to_string(), weight: 0.9 },
            ],
            weight: 1.0,
            confidence: 0.9,
        });

        // Rule 3: High speed consistency + Small temporal gap → Good smoothness
        self.rule_base.push(FuzzyRule {
            antecedent: vec![
                FuzzyCondition { variable: "speed_consistency".to_string(), term: "high".to_string(), negated: false },
                FuzzyCondition { variable: "temporal_gap".to_string(), term: "small".to_string(), negated: false },
            ],
            consequent: vec![
                FuzzyConclusion { variable: "trajectory_smoothness".to_string(), term: "high".to_string(), weight: 1.0 },
            ],
            weight: 1.0,
            confidence: 0.85,
        });
    }

    /// Perform fuzzy inference on input values
    pub fn infer(&self, inputs: HashMap<String, f64>) -> HashMap<String, f64> {
        let mut output_aggregations: HashMap<String, Vec<f64>> = HashMap::new();
        
        // Initialize output aggregations
        for output_var in self.output_variables.keys() {
            output_aggregations.insert(output_var.clone(), Vec::new());
        }
        
        // Process each rule
        for rule in &self.rule_base {
            // Calculate rule activation (antecedent strength)
            let mut activation = 1.0;
            
            for condition in &rule.antecedent {
                if let Some(input_value) = inputs.get(&condition.variable) {
                    if let Some(input_var) = self.input_variables.get(&condition.variable) {
                        if let Some(term) = input_var.linguistic_terms.get(&condition.term) {
                            let membership = term.membership_function.evaluate(*input_value);
                            let final_membership = if condition.negated { 1.0 - membership } else { membership };
                            activation = self.tnorm_operator.apply_tnorm(activation, final_membership);
                        }
                    }
                }
            }
            
            // Apply rule weight and confidence
            activation *= rule.weight * rule.confidence;
            
            // Apply consequent (implication)
            for conclusion in &rule.consequent {
                if let Some(aggregation) = output_aggregations.get_mut(&conclusion.variable) {
                    let weighted_activation = activation * conclusion.weight;
                    aggregation.push(weighted_activation);
                }
            }
        }
        
        // Defuzzify outputs
        let mut results = HashMap::new();
        for (output_var, activations) in output_aggregations {
            if !activations.is_empty() {
                let defuzzified_value = self.defuzzify(&activations);
                results.insert(output_var, defuzzified_value);
            }
        }
        
        results
    }

    fn defuzzify(&self, activations: &[f64]) -> f64 {
        match self.defuzzification_method {
            DefuzzificationMethod::Centroid => {
                // Weighted average
                let sum: f64 = activations.iter().sum();
                let weighted_sum: f64 = activations.iter().enumerate()
                    .map(|(i, &activation)| (i as f64 + 1.0) * activation)
                    .sum();
                if sum > 0.0 { weighted_sum / sum } else { 0.0 }
            },
            DefuzzificationMethod::MeanOfMaximum => {
                // Mean of maximum values
                let max_activation = activations.iter().fold(0.0, |acc, &x| acc.max(x));
                let max_indices: Vec<usize> = activations.iter().enumerate()
                    .filter(|(_, &activation)| (activation - max_activation).abs() < 1e-6)
                    .map(|(i, _)| i)
                    .collect();
                
                if max_indices.is_empty() {
                    0.0
                } else {
                    max_indices.iter().sum::<usize>() as f64 / max_indices.len() as f64 / activations.len() as f64
                }
            },
            _ => {
                // Default to centroid
                let sum: f64 = activations.iter().sum();
                if sum > 0.0 { sum / activations.len() as f64 } else { 0.0 }
            }
        }
    }
}

/// Fuzzy optimization algorithm specifically for Bayesian Evidence Networks
#[pyclass]
#[derive(Debug, Clone)]
pub struct FuzzyBayesianOptimizer {
    population_size: usize,
    max_generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_size: usize,
    fuzzy_inference_system: FuzzyInferenceSystem,
}

#[pymethods]
impl FuzzyBayesianOptimizer {
    #[new]
    pub fn new(population_size: Option<usize>, max_generations: Option<usize>) -> Self {
        Self {
            population_size: population_size.unwrap_or(50),
            max_generations: max_generations.unwrap_or(100),
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_size: 5,
            fuzzy_inference_system: FuzzyInferenceSystem::create_gps_analysis_system(),
        }
    }

    /// Optimize Bayesian Evidence Network using fuzzy genetic algorithm
    pub fn optimize_bayesian_network(
        &mut self,
        network: &mut BayesianEvidenceNetwork,
        trajectory: Vec<GpsPoint>,
    ) -> HashMap<String, f64> {
        // Initialize population of network configurations
        let mut population = self.initialize_population(&trajectory);
        let mut best_solution = population[0].clone();
        let mut best_fitness = 0.0;
        
        for generation in 0..self.max_generations {
            // Evaluate fitness for each individual
            let fitness_scores: Vec<f64> = population.par_iter()
                .map(|individual| self.evaluate_fitness(network, individual, &trajectory))
                .collect();
            
            // Find best individual
            if let Some((best_idx, &fitness)) = fitness_scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_solution = population[best_idx].clone();
                }
            }
            
            // Selection, crossover, and mutation
            population = self.evolve_population(population, fitness_scores);
            
            // Apply fuzzy adaptation
            self.adapt_parameters(generation, best_fitness);
        }
        
        // Apply best solution to network
        self.apply_solution_to_network(network, &best_solution);
        
        // Return optimization results
        let mut results = HashMap::new();
        results.insert("best_fitness".to_string(), best_fitness);
        results.insert("generations".to_string(), self.max_generations as f64);
        results.insert("final_objective".to_string(), network.objective_function.current_value);
        
        results
    }

    fn initialize_population(&self, trajectory: &[GpsPoint]) -> Vec<NetworkConfiguration> {
        (0..self.population_size)
            .map(|_| NetworkConfiguration::random(trajectory))
            .collect()
    }

    fn evaluate_fitness(&self, network: &mut BayesianEvidenceNetwork, config: &NetworkConfiguration, trajectory: &[GpsPoint]) -> f64 {
        // Apply configuration to network
        self.apply_solution_to_network(network, config);
        
        // Process trajectory with current configuration
        let objective_value = network.process_gps_trajectory(trajectory.to_vec());
        
        // Use fuzzy inference to evaluate quality
        let mut fuzzy_inputs = HashMap::new();
        fuzzy_inputs.insert("position_reliability".to_string(), config.position_weight);
        fuzzy_inputs.insert("trajectory_smoothness".to_string(), config.smoothness_weight);
        fuzzy_inputs.insert("overall_quality".to_string(), objective_value);
        
        let fuzzy_outputs = self.fuzzy_inference_system.infer(fuzzy_inputs);
        
        // Combine objective function value with fuzzy assessment
        let fuzzy_quality = fuzzy_outputs.get("overall_quality").unwrap_or(&objective_value);
        0.7 * objective_value + 0.3 * fuzzy_quality
    }

    fn evolve_population(&mut self, mut population: Vec<NetworkConfiguration>, fitness_scores: Vec<f64>) -> Vec<NetworkConfiguration> {
        let mut new_population = Vec::new();
        
        // Elite selection
        let mut elite_indices: Vec<usize> = (0..population.len()).collect();
        elite_indices.sort_by(|&a, &b| fitness_scores[b].partial_cmp(&fitness_scores[a]).unwrap());
        
        for i in 0..self.elite_size.min(population.len()) {
            new_population.push(population[elite_indices[i]].clone());
        }
        
        // Generate rest of population through crossover and mutation
        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection(&population, &fitness_scores);
            let parent2 = self.tournament_selection(&population, &fitness_scores);
            
            let mut offspring = if rand::random::<f64>() < self.crossover_rate {
                self.crossover(&parent1, &parent2)
            } else {
                parent1.clone()
            };
            
            if rand::random::<f64>() < self.mutation_rate {
                self.mutate(&mut offspring);
            }
            
            new_population.push(offspring);
        }
        
        new_population
    }

    fn tournament_selection(&self, population: &[NetworkConfiguration], fitness_scores: &[f64]) -> NetworkConfiguration {
        let tournament_size = 3;
        let mut best_idx = rand::random::<usize>() % population.len();
        let mut best_fitness = fitness_scores[best_idx];
        
        for _ in 1..tournament_size {
            let idx = rand::random::<usize>() % population.len();
            if fitness_scores[idx] > best_fitness {
                best_idx = idx;
                best_fitness = fitness_scores[idx];
            }
        }
        
        population[best_idx].clone()
    }

    fn crossover(&self, parent1: &NetworkConfiguration, parent2: &NetworkConfiguration) -> NetworkConfiguration {
        NetworkConfiguration {
            position_weight: if rand::random::<bool>() { parent1.position_weight } else { parent2.position_weight },
            velocity_weight: if rand::random::<bool>() { parent1.velocity_weight } else { parent2.velocity_weight },
            smoothness_weight: if rand::random::<bool>() { parent1.smoothness_weight } else { parent2.smoothness_weight },
            confidence_threshold: if rand::random::<bool>() { parent1.confidence_threshold } else { parent2.confidence_threshold },
            temporal_decay: if rand::random::<bool>() { parent1.temporal_decay } else { parent2.temporal_decay },
            evidence_aggregation_method: if rand::random::<bool>() { parent1.evidence_aggregation_method } else { parent2.evidence_aggregation_method },
        }
    }

    fn mutate(&self, individual: &mut NetworkConfiguration) {
        let mutation_strength = 0.1;
        
        if rand::random::<f64>() < 0.2 {
            individual.position_weight += (rand::random::<f64>() - 0.5) * mutation_strength;
            individual.position_weight = individual.position_weight.max(0.0).min(1.0);
        }
        
        if rand::random::<f64>() < 0.2 {
            individual.velocity_weight += (rand::random::<f64>() - 0.5) * mutation_strength;
            individual.velocity_weight = individual.velocity_weight.max(0.0).min(1.0);
        }
        
        if rand::random::<f64>() < 0.2 {
            individual.smoothness_weight += (rand::random::<f64>() - 0.5) * mutation_strength;
            individual.smoothness_weight = individual.smoothness_weight.max(0.0).min(1.0);
        }
        
        if rand::random::<f64>() < 0.2 {
            individual.confidence_threshold += (rand::random::<f64>() - 0.5) * mutation_strength;
            individual.confidence_threshold = individual.confidence_threshold.max(0.0).min(1.0);
        }
        
        if rand::random::<f64>() < 0.2 {
            individual.temporal_decay += (rand::random::<f64>() - 0.5) * mutation_strength;
            individual.temporal_decay = individual.temporal_decay.max(0.0).min(1.0);
        }
    }

    fn adapt_parameters(&mut self, generation: usize, best_fitness: f64) {
        // Fuzzy adaptation of algorithm parameters
        let progress = generation as f64 / self.max_generations as f64;
        
        // Adaptive mutation rate
        self.mutation_rate = 0.2 * (1.0 - progress) + 0.05 * progress;
        
        // Adaptive crossover rate based on fitness improvement
        if best_fitness > 0.8 {
            self.crossover_rate = 0.6; // Exploit good solutions
        } else {
            self.crossover_rate = 0.9; // Explore more
        }
    }

    fn apply_solution_to_network(&self, network: &mut BayesianEvidenceNetwork, config: &NetworkConfiguration) {
        // This would apply the configuration to the network
        // Implementation depends on the specific network structure
        network.objective_function.current_value = config.position_weight * 0.5 + config.smoothness_weight * 0.5;
    }
}

/// Network configuration for optimization
#[derive(Debug, Clone)]
pub struct NetworkConfiguration {
    position_weight: f64,
    velocity_weight: f64,
    smoothness_weight: f64,
    confidence_threshold: f64,
    temporal_decay: f64,
    evidence_aggregation_method: usize, // Index into method enum
}

impl NetworkConfiguration {
    pub fn random(trajectory: &[GpsPoint]) -> Self {
        Self {
            position_weight: rand::random::<f64>(),
            velocity_weight: rand::random::<f64>(),
            smoothness_weight: rand::random::<f64>(),
            confidence_threshold: rand::random::<f64>() * 0.5 + 0.25, // 0.25-0.75
            temporal_decay: rand::random::<f64>() * 0.1 + 0.9, // 0.9-1.0
            evidence_aggregation_method: rand::random::<usize>() % 4,
        }
    }
}

/// Multi-objective fuzzy optimization for trajectory analysis
#[pyfunction]
pub fn optimize_trajectory_fuzzy_bayesian(
    trajectory: Vec<GpsPoint>,
    optimization_params: Option<HashMap<String, f64>>,
) -> HashMap<String, f64> {
    let mut network = BayesianEvidenceNetwork::new();
    let population_size = optimization_params.as_ref()
        .and_then(|p| p.get("population_size"))
        .map(|&x| x as usize)
        .unwrap_or(30);
    
    let max_generations = optimization_params.as_ref()
        .and_then(|p| p.get("max_generations"))
        .map(|&x| x as usize)
        .unwrap_or(50);
    
    let mut optimizer = FuzzyBayesianOptimizer::new(Some(population_size), Some(max_generations));
    
    let results = optimizer.optimize_bayesian_network(&mut network, trajectory);
    
    // Add network statistics
    let mut final_results = results;
    let network_stats = network.get_network_stats();
    final_results.extend(network_stats);
    
    final_results
}

/// Batch optimization of multiple trajectories
#[pyfunction]
pub fn batch_optimize_trajectories_fuzzy(
    trajectories: Vec<Vec<GpsPoint>>,
    optimization_params: Option<HashMap<String, f64>>,
) -> Vec<HashMap<String, f64>> {
    trajectories
        .into_par_iter()
        .map(|trajectory| optimize_trajectory_fuzzy_bayesian(trajectory, optimization_params.clone()))
        .collect()
}

/// Python module definition
#[pymodule]
fn sighthound_fuzzy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FuzzyInferenceSystem>()?;
    m.add_class::<FuzzyBayesianOptimizer>()?;
    m.add_function(wrap_pyfunction!(optimize_trajectory_fuzzy_bayesian, m)?)?;
    m.add_function(wrap_pyfunction!(batch_optimize_trajectories_fuzzy, m)?)?;
    Ok(())
} 