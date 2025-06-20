use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector, Vector2, Vector3, Point2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use rstar::{RTree, RTreeObject, AABB};
use kiddo::{KdTree, SquaredEuclidean};
use std::collections::HashMap;
use dashmap::DashMap;
use crossbeam::channel;
use sighthound_core::{GpsPoint, haversine_distance};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Cell tower data with signal strength
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CellTower {
    #[pyo3(get, set)]
    pub latitude: f64,
    #[pyo3(get, set)]
    pub longitude: f64,
    #[pyo3(get, set)]
    pub signal_strength: f64, // dBm
    #[pyo3(get, set)]
    pub cell_id: u64,
    #[pyo3(get, set)]
    pub timestamp: f64,
}

#[pymethods]
impl CellTower {
    #[new]
    pub fn new(latitude: f64, longitude: f64, signal_strength: f64, cell_id: u64, timestamp: f64) -> Self {
        Self {
            latitude,
            longitude,
            signal_strength,
            cell_id,
            timestamp,
        }
    }

    /// Calculate weight based on signal strength
    pub fn calculate_weight(&self) -> f64 {
        // Convert signal strength to weight (stronger signal = higher weight)
        let normalized_strength = (self.signal_strength + 120.0) / 50.0; // Normalize typical range [-120, -70] to [0, 1]
        normalized_strength.max(0.1).min(1.0) // Clamp to reasonable range
    }

    /// Estimate distance from signal strength (rough approximation)
    pub fn estimate_distance(&self) -> f64 {
        // Simplified path loss model: distance ∝ 10^((RSSI - A) / (10 * n))
        // where A is received power at 1m, n is path loss exponent
        let a = -30.0; // Typical value for 1m reference
        let n = 2.0;   // Free space path loss
        
        let distance = 10_f64.powf((self.signal_strength - a) / (-10.0 * n));
        distance.max(10.0).min(10000.0) // Clamp to reasonable range (10m to 10km)
    }
}

impl RTreeObject for CellTower {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.longitude, self.latitude])
    }
}

/// Configuration for triangulation algorithms
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangulationConfig {
    #[pyo3(get, set)]
    pub min_towers: usize,
    #[pyo3(get, set)]
    pub max_distance: f64, // meters
    #[pyo3(get, set)]
    pub confidence_threshold: f64,
    #[pyo3(get, set)]
    pub optimization_method: String,
    #[pyo3(get, set)]
    pub max_iterations: usize,
    #[pyo3(get, set)]
    pub convergence_threshold: f64,
    #[pyo3(get, set)]
    pub outlier_threshold: f64,
    #[pyo3(get, set)]
    pub parallel_processing: bool,
}

#[pymethods]
impl TriangulationConfig {
    #[new]
    pub fn new() -> Self {
        Self {
            min_towers: 3,
            max_distance: 10000.0, // 10km
            confidence_threshold: 0.5,
            optimization_method: "least_squares".to_string(),
            max_iterations: 100,
            convergence_threshold: 1e-6,
            outlier_threshold: 3.0,
            parallel_processing: true,
        }
    }
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance triangulation engine with spatial indexing
#[pyclass]
pub struct TriangulationEngine {
    towers: Vec<CellTower>,
    spatial_index: Option<RTree<CellTower>>,
    kd_tree: Option<KdTree<f64, u64, 2, 32, u32>>,
    config: TriangulationConfig,
    cache: DashMap<String, TriangulationResult>,
}

/// Result of triangulation with confidence metrics
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangulationResult {
    #[pyo3(get, set)]
    pub latitude: f64,
    #[pyo3(get, set)]
    pub longitude: f64,
    #[pyo3(get, set)]
    pub confidence: f64,
    #[pyo3(get, set)]
    pub uncertainty_radius: f64, // meters
    #[pyo3(get, set)]
    pub towers_used: usize,
    #[pyo3(get, set)]
    pub convergence_error: f64,
}

#[pymethods]
impl TriangulationResult {
    #[new]
    pub fn new(
        latitude: f64,
        longitude: f64,
        confidence: f64,
        uncertainty_radius: f64,
        towers_used: usize,
        convergence_error: f64,
    ) -> Self {
        Self {
            latitude,
            longitude,
            confidence,
            uncertainty_radius,
            towers_used,
            convergence_error,
        }
    }
}

#[pymethods]
impl TriangulationEngine {
    #[new]
    pub fn new(towers: Vec<CellTower>, config: Option<TriangulationConfig>) -> Self {
        let config = config.unwrap_or_default();
        
        Self {
            towers,
            spatial_index: None,
            kd_tree: None,
            config,
            cache: DashMap::new(),
        }
    }

    /// Build spatial indices for fast queries
    pub fn build_indices(&mut self) {
        // R-tree for spatial queries
        self.spatial_index = Some(RTree::bulk_load(self.towers.clone()));

        // K-d tree for nearest neighbor queries
        let mut kd_tree = KdTree::new();
        for (i, tower) in self.towers.iter().enumerate() {
            let point = [tower.longitude, tower.latitude];
            kd_tree.add(&point, i as u64).unwrap();
        }
        self.kd_tree = Some(kd_tree);
    }

    /// Triangulate position from nearby cell towers
    pub fn triangulate(&mut self, reference_point: GpsPoint) -> Option<TriangulationResult> {
        // Check cache first
        let cache_key = format!("{:.6}_{:.6}", reference_point.latitude, reference_point.longitude);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Some(cached.clone());
        }

        // Ensure indices are built
        if self.spatial_index.is_none() {
            self.build_indices();
        }

        // Find nearby towers
        let nearby_towers = self.find_nearby_towers(&reference_point)?;
        
        if nearby_towers.len() < self.config.min_towers {
            return None;
        }

        // Perform triangulation based on method
        let result = match self.config.optimization_method.as_str() {
            "weighted_centroid" => self.weighted_centroid_triangulation(&nearby_towers),
            "least_squares" => self.least_squares_triangulation(&nearby_towers),
            "maximum_likelihood" => self.maximum_likelihood_triangulation(&nearby_towers),
            _ => self.least_squares_triangulation(&nearby_towers),
        };

        // Cache the result
        if let Some(ref res) = result {
            self.cache.insert(cache_key, res.clone());
        }

        result
    }

    /// Batch triangulate multiple points in parallel
    pub fn batch_triangulate(&mut self, points: Vec<GpsPoint>) -> Vec<Option<TriangulationResult>> {
        if !self.config.parallel_processing {
            return points.into_iter().map(|p| self.triangulate(p)).collect();
        }

        // Ensure indices are built
        if self.spatial_index.is_none() {
            self.build_indices();
        }

        // Process in parallel chunks
        points
            .into_par_iter()
            .map(|point| {
                let nearby_towers = self.find_nearby_towers(&point)?;
                
                if nearby_towers.len() < self.config.min_towers {
                    return None;
                }

                match self.config.optimization_method.as_str() {
                    "weighted_centroid" => self.weighted_centroid_triangulation(&nearby_towers),
                    "least_squares" => self.least_squares_triangulation(&nearby_towers),
                    "maximum_likelihood" => self.maximum_likelihood_triangulation(&nearby_towers),
                    _ => self.least_squares_triangulation(&nearby_towers),
                }
            })
            .collect()
    }

    fn find_nearby_towers(&self, reference_point: &GpsPoint) -> Option<Vec<CellTower>> {
        let spatial_index = self.spatial_index.as_ref()?;
        
        // Search within max_distance radius
        let search_envelope = AABB::from_corners(
            [
                reference_point.longitude - self.config.max_distance / 111320.0,
                reference_point.latitude - self.config.max_distance / 111320.0,
            ],
            [
                reference_point.longitude + self.config.max_distance / 111320.0,
                reference_point.latitude + self.config.max_distance / 111320.0,
            ],
        );

        let mut nearby_towers: Vec<CellTower> = spatial_index
            .locate_in_envelope(&search_envelope)
            .filter(|tower| {
                let distance = haversine_distance(
                    reference_point.latitude,
                    reference_point.longitude,
                    tower.latitude,
                    tower.longitude,
                );
                distance <= self.config.max_distance
            })
            .cloned()
            .collect();

        // Sort by signal strength (stronger first)
        nearby_towers.sort_by(|a, b| b.signal_strength.partial_cmp(&a.signal_strength).unwrap());

        if nearby_towers.is_empty() {
            None
        } else {
            Some(nearby_towers)
        }
    }

    fn weighted_centroid_triangulation(&self, towers: &[CellTower]) -> Option<TriangulationResult> {
        let mut weighted_lat = 0.0;
        let mut weighted_lon = 0.0;
        let mut total_weight = 0.0;

        for tower in towers {
            let weight = tower.calculate_weight();
            weighted_lat += tower.latitude * weight;
            weighted_lon += tower.longitude * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return None;
        }

        let result_lat = weighted_lat / total_weight;
        let result_lon = weighted_lon / total_weight;

        // Calculate confidence based on tower distribution and signal strengths
        let confidence = self.calculate_confidence(towers, result_lat, result_lon);
        let uncertainty = self.calculate_uncertainty(towers, result_lat, result_lon);

        Some(TriangulationResult::new(
            result_lat,
            result_lon,
            confidence,
            uncertainty,
            towers.len(),
            0.0, // No iterative optimization for centroid method
        ))
    }

    fn least_squares_triangulation(&self, towers: &[CellTower]) -> Option<TriangulationResult> {
        if towers.len() < 3 {
            return None;
        }

        // Initial guess using weighted centroid
        let initial = self.weighted_centroid_triangulation(towers)?;
        let mut current_pos = Vector2::new(initial.longitude, initial.latitude);

        let mut convergence_error = f64::INFINITY;
        
        // Gauss-Newton optimization
        for iteration in 0..self.config.max_iterations {
            let mut jacobian = DMatrix::zeros(towers.len(), 2);
            let mut residuals = DVector::zeros(towers.len());

            for (i, tower) in towers.iter().enumerate() {
                let tower_pos = Vector2::new(tower.longitude, tower.latitude);
                let distance = haversine_distance(
                    current_pos.y,
                    current_pos.x,
                    tower.latitude,
                    tower.longitude,
                );
                
                let expected_distance = tower.estimate_distance();
                residuals[i] = distance - expected_distance;

                // Calculate Jacobian (partial derivatives)
                let diff = tower_pos - current_pos;
                let norm = diff.norm();
                if norm > 0.0 {
                    jacobian[(i, 0)] = -diff.x / norm * 111320.0; // Convert to meters
                    jacobian[(i, 1)] = -diff.y / norm * 111320.0;
                }
            }

            // Solve normal equations: (J^T * J) * Δx = -J^T * r
            let jt = jacobian.transpose();
            let normal_matrix = &jt * &jacobian;
            let rhs = -&jt * &residuals;

            if let Some(delta) = normal_matrix.lu().solve(&rhs) {
                current_pos += Vector2::new(delta[0] / 111320.0, delta[1] / 111320.0); // Convert back to degrees
                convergence_error = delta.norm();

                if convergence_error < self.config.convergence_threshold {
                    break;
                }
            } else {
                // Singular matrix, fallback to gradient descent
                let gradient = &jt * &residuals;
                let step_size = 1e-7;
                current_pos -= gradient.scale(step_size);
            }
        }

        let confidence = self.calculate_confidence(towers, current_pos.y, current_pos.x);
        let uncertainty = self.calculate_uncertainty(towers, current_pos.y, current_pos.x);

        Some(TriangulationResult::new(
            current_pos.y,
            current_pos.x,
            confidence,
            uncertainty,
            towers.len(),
            convergence_error,
        ))
    }

    fn maximum_likelihood_triangulation(&self, towers: &[CellTower]) -> Option<TriangulationResult> {
        // Start with least squares solution
        let mut result = self.least_squares_triangulation(towers)?;

        // Refine using maximum likelihood estimation
        // This would involve more sophisticated probability models
        // For now, we use the least squares result with adjusted confidence

        // Adjust confidence based on likelihood of measurements
        let likelihood_score = self.calculate_likelihood(towers, result.latitude, result.longitude);
        result.confidence *= likelihood_score;

        Some(result)
    }

    fn calculate_confidence(&self, towers: &[CellTower], lat: f64, lon: f64) -> f64 {
        if towers.is_empty() {
            return 0.0;
        }

        // Factors affecting confidence:
        // 1. Number of towers
        // 2. Signal strength distribution
        // 3. Geometric dilution of precision (GDOP)
        // 4. Measurement consistency

        let tower_factor = (towers.len() as f64 / 10.0).min(1.0);
        
        let avg_signal_strength: f64 = towers.iter().map(|t| t.signal_strength).sum::<f64>() / towers.len() as f64;
        let signal_factor = ((avg_signal_strength + 120.0) / 50.0).max(0.0).min(1.0);

        let gdop = self.calculate_gdop(towers, lat, lon);
        let gdop_factor = (1.0 / gdop).min(1.0);

        // Weighted combination
        0.4 * tower_factor + 0.3 * signal_factor + 0.3 * gdop_factor
    }

    fn calculate_uncertainty(&self, towers: &[CellTower], lat: f64, lon: f64) -> f64 {
        if towers.is_empty() {
            return 1000.0; // Large uncertainty if no towers
        }

        // Base uncertainty from measurement errors
        let mut measurement_errors: Vec<f64> = Vec::new();
        
        for tower in towers {
            let actual_distance = haversine_distance(lat, lon, tower.latitude, tower.longitude);
            let estimated_distance = tower.estimate_distance();
            measurement_errors.push((actual_distance - estimated_distance).abs());
        }

        let mean_error = measurement_errors.iter().sum::<f64>() / measurement_errors.len() as f64;
        let variance = measurement_errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() / measurement_errors.len() as f64;
        
        // Combine with GDOP
        let gdop = self.calculate_gdop(towers, lat, lon);
        mean_error * gdop + variance.sqrt()
    }

    fn calculate_gdop(&self, towers: &[CellTower], lat: f64, lon: f64) -> f64 {
        if towers.len() < 3 {
            return 10.0; // Poor GDOP for insufficient towers
        }

        // Calculate geometric dilution of precision
        let mut design_matrix = DMatrix::zeros(towers.len(), 2);
        
        for (i, tower) in towers.iter().enumerate() {
            let distance = haversine_distance(lat, lon, tower.latitude, tower.longitude);
            if distance > 0.0 {
                design_matrix[(i, 0)] = (tower.longitude - lon) / distance;
                design_matrix[(i, 1)] = (tower.latitude - lat) / distance;
            }
        }

        let gram_matrix = design_matrix.transpose() * design_matrix;
        if let Some(inverse) = gram_matrix.try_inverse() {
            inverse.trace().sqrt()
        } else {
            10.0 // Poor GDOP for singular matrix
        }
    }

    fn calculate_likelihood(&self, towers: &[CellTower], lat: f64, lon: f64) -> f64 {
        let mut log_likelihood = 0.0;
        
        for tower in towers {
            let predicted_distance = haversine_distance(lat, lon, tower.latitude, tower.longitude);
            let expected_distance = tower.estimate_distance();
            
            // Assume Gaussian measurement noise
            let sigma = 100.0; // Standard deviation in meters
            let error = predicted_distance - expected_distance;
            log_likelihood -= 0.5 * (error / sigma).powi(2);
        }

        (-log_likelihood / towers.len() as f64).exp().min(1.0)
    }
}

/// Parallel triangulation processor for large datasets
#[pyfunction]
pub fn batch_triangulate_parallel(
    points: Vec<GpsPoint>,
    towers: Vec<CellTower>,
    config: Option<TriangulationConfig>,
    num_workers: Option<usize>
) -> Vec<Option<TriangulationResult>> {
    let config = config.unwrap_or_default();
    let workers = num_workers.unwrap_or_else(|| rayon::current_num_threads());
    
    // Create thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();

    pool.install(|| {
        points
            .into_par_iter()
            .map(|point| {
                let mut engine = TriangulationEngine::new(towers.clone(), Some(config.clone()));
                engine.triangulate(point)
            })
            .collect()
    })
}

/// Advanced triangulation with outlier detection
#[pyfunction]
pub fn robust_triangulate(
    point: GpsPoint,
    towers: Vec<CellTower>,
    config: Option<TriangulationConfig>
) -> Option<TriangulationResult> {
    let config = config.unwrap_or_default();
    let mut engine = TriangulationEngine::new(towers, Some(config));
    
    // Perform initial triangulation
    let mut result = engine.triangulate(point)?;
    
    // Detect and remove outliers using RANSAC-like approach
    let towers = engine.towers.clone();
    let mut best_result = result.clone();
    let mut best_inliers = 0;
    
    // Try different subsets of towers
    for _ in 0..50 {
        if towers.len() < 4 {
            break;
        }
        
        // Randomly sample subset of towers
        let mut subset_indices: Vec<usize> = (0..towers.len()).collect();
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        subset_indices.shuffle(&mut thread_rng());
        subset_indices.truncate(towers.len().min(6));
        
        let subset: Vec<CellTower> = subset_indices.iter().map(|&i| towers[i]).collect();
        let mut subset_engine = TriangulationEngine::new(subset, Some(engine.config.clone()));
        
        if let Some(subset_result) = subset_engine.triangulate(point) {
            // Count inliers (towers with consistent measurements)
            let mut inliers = 0;
            for tower in &towers {
                let predicted_distance = haversine_distance(
                    subset_result.latitude,
                    subset_result.longitude,
                    tower.latitude,
                    tower.longitude,
                );
                let expected_distance = tower.estimate_distance();
                let error = (predicted_distance - expected_distance).abs();
                
                if error < engine.config.outlier_threshold * 100.0 { // 100m threshold
                    inliers += 1;
                }
            }
            
            if inliers > best_inliers {
                best_result = subset_result;
                best_inliers = inliers;
            }
        }
    }
    
    Some(best_result)
}

/// Python module definition
#[pymodule]
fn sighthound_triangulation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CellTower>()?;
    m.add_class::<TriangulationConfig>()?;
    m.add_class::<TriangulationEngine>()?;
    m.add_class::<TriangulationResult>()?;
    m.add_function(wrap_pyfunction!(batch_triangulate_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(robust_triangulate, m)?)?;
    Ok(())
} 