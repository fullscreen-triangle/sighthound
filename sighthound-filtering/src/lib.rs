use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{PyArray1, PyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use nalgebra::{DMatrix, DVector, Matrix4, Vector4, Matrix2, Vector2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::collections::VecDeque;
use sighthound_core::GpsPoint;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// High-performance Kalman Filter for GPS trajectory processing
#[pyclass]
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State vector [x, vx, y, vy]
    state: Vector4<f64>,
    /// State covariance matrix
    covariance: Matrix4<f64>,
    /// State transition matrix
    transition: Matrix4<f64>,
    /// Observation matrix
    observation: Matrix2<Vector4<f64>>,
    /// Process noise covariance
    process_noise: Matrix4<f64>,
    /// Measurement noise covariance
    measurement_noise: Matrix2<f64>,
    /// Time step
    dt: f64,
    /// Adaptive noise scaling factor
    adaptive_factor: f64,
    /// Innovation covariance history for adaptation
    innovation_history: VecDeque<f64>,
}

/// Configuration for Kalman Filter
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanConfig {
    #[pyo3(get, set)]
    pub process_noise: f64,
    #[pyo3(get, set)]
    pub measurement_noise: f64,
    #[pyo3(get, set)]
    pub initial_state_covariance: f64,
    #[pyo3(get, set)]
    pub dt: f64,
    #[pyo3(get, set)]
    pub adaptive_filter: bool,
    #[pyo3(get, set)]
    pub innovation_threshold: f64,
    #[pyo3(get, set)]
    pub max_velocity: f64,  // m/s
    #[pyo3(get, set)]
    pub confidence_weight: f64,
}

#[pymethods]
impl KalmanConfig {
    #[new]
    pub fn new() -> Self {
        Self {
            process_noise: 1e-3,
            measurement_noise: 1e-2,
            initial_state_covariance: 100.0,
            dt: 1.0,
            adaptive_filter: true,
            innovation_threshold: 5.0,
            max_velocity: 50.0, // 50 m/s = 180 km/h
            confidence_weight: 0.5,
        }
    }
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl KalmanFilter {
    #[new]
    pub fn new(config: Option<KalmanConfig>) -> Self {
        let config = config.unwrap_or_default();
        
        // State transition matrix F
        let mut transition = Matrix4::identity();
        transition[(0, 1)] = config.dt;
        transition[(2, 3)] = config.dt;

        // Observation matrix H [1 0 0 0; 0 0 1 0]
        let observation = Matrix2::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        );

        // Process noise matrix Q
        let dt2 = config.dt * config.dt;
        let dt3 = dt2 * config.dt;
        let dt4 = dt3 * config.dt;
        
        let q = config.process_noise;
        let process_noise = Matrix4::new(
            q * dt4 / 4.0, q * dt3 / 2.0, 0.0, 0.0,
            q * dt3 / 2.0, q * dt2, 0.0, 0.0,
            0.0, 0.0, q * dt4 / 4.0, q * dt3 / 2.0,
            0.0, 0.0, q * dt3 / 2.0, q * dt2
        );

        // Measurement noise matrix R
        let measurement_noise = Matrix2::identity() * config.measurement_noise;

        // Initial state covariance P
        let covariance = Matrix4::identity() * config.initial_state_covariance;

        Self {
            state: Vector4::zeros(),
            covariance,
            transition,
            observation,
            process_noise,
            measurement_noise,
            dt: config.dt,
            adaptive_factor: 1.0,
            innovation_history: VecDeque::with_capacity(50),
        }
    }

    /// Initialize filter with first GPS point
    pub fn initialize(&mut self, point: &GpsPoint) {
        self.state[0] = point.longitude;
        self.state[1] = 0.0; // Initial velocity
        self.state[2] = point.latitude;
        self.state[3] = 0.0; // Initial velocity
    }

    /// Predict step of Kalman filter
    pub fn predict(&mut self) {
        // x = F * x
        self.state = self.transition * self.state;
        
        // P = F * P * F^T + Q
        self.covariance = self.transition * self.covariance * self.transition.transpose() + 
                         self.process_noise * self.adaptive_factor;
    }

    /// Update step with GPS measurement
    pub fn update(&mut self, measurement: Vector2<f64>, confidence: Option<f64>) {
        // Adjust measurement noise based on confidence
        let mut r = self.measurement_noise;
        if let Some(conf) = confidence {
            let confidence_factor = (conf + 0.1).max(0.1); // Avoid division by zero
            r *= 1.0 / confidence_factor;
        }

        // Innovation: y = z - H * x
        let predicted_measurement = self.observation * self.state;
        let innovation = measurement - predicted_measurement;

        // Innovation covariance: S = H * P * H^T + R
        let innovation_covariance = self.observation * self.covariance * self.observation.transpose() + r;

        // Kalman gain: K = P * H^T * S^(-1)
        let kalman_gain = self.covariance * self.observation.transpose() * 
                         innovation_covariance.try_inverse().unwrap_or(Matrix2::identity());

        // Update state: x = x + K * y
        self.state += kalman_gain * innovation;

        // Update covariance: P = (I - K * H) * P
        let identity = Matrix4::identity();
        self.covariance = (identity - kalman_gain * self.observation) * self.covariance;

        // Adaptive filtering based on innovation
        self.update_adaptive_factor(&innovation, &innovation_covariance);

        // Velocity constraint
        self.constrain_velocity();
    }

    /// Process single GPS point
    pub fn process_point(&mut self, point: &GpsPoint) -> GpsPoint {
        if self.state.norm() == 0.0 {
            self.initialize(point);
            return *point;
        }

        self.predict();
        let measurement = Vector2::new(point.longitude, point.latitude);
        self.update(measurement, Some(point.confidence));

        GpsPoint::new(
            self.state[2], // latitude
            self.state[0], // longitude
            point.timestamp,
            point.confidence,
            point.altitude,
            Some((self.state[1].powi(2) + self.state[3].powi(2)).sqrt()), // speed
            point.heading
        )
    }

    /// Get current state as GPS point
    pub fn get_current_state(&self) -> (f64, f64, f64, f64) {
        (self.state[0], self.state[1], self.state[2], self.state[3])
    }

    /// Get position uncertainty (standard deviation in meters)
    pub fn get_position_uncertainty(&self) -> (f64, f64) {
        let pos_var_x = self.covariance[(0, 0)];
        let pos_var_y = self.covariance[(2, 2)];
        (pos_var_x.sqrt() * 111320.0, pos_var_y.sqrt() * 111320.0) // Convert to meters
    }

    fn update_adaptive_factor(&mut self, innovation: &Vector2<f64>, innovation_cov: &Matrix2<f64>) {
        // Mahalanobis distance for outlier detection
        let mahalanobis_dist = (innovation.transpose() * 
                               innovation_cov.try_inverse().unwrap_or(Matrix2::identity()) * 
                               innovation)[0].sqrt();

        self.innovation_history.push_back(mahalanobis_dist);
        if self.innovation_history.len() > 50 {
            self.innovation_history.pop_front();
        }

        // Calculate median of recent innovations
        let mut sorted_innovations: Vec<f64> = self.innovation_history.iter().cloned().collect();
        sorted_innovations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if !sorted_innovations.is_empty() {
            let median = sorted_innovations[sorted_innovations.len() / 2];
            
            // Adapt process noise based on innovation level
            if mahalanobis_dist > median * 2.0 {
                self.adaptive_factor = (self.adaptive_factor * 1.2).min(10.0);
            } else if mahalanobis_dist < median * 0.5 {
                self.adaptive_factor = (self.adaptive_factor * 0.9).max(0.1);
            }
        }
    }

    fn constrain_velocity(&mut self) {
        let max_vel = 50.0; // 50 m/s ≈ 180 km/h
        
        // Convert velocity to m/s (approximate)
        let vx_ms = self.state[1] * 111320.0; // longitude velocity to m/s
        let vy_ms = self.state[3] * 111320.0; // latitude velocity to m/s
        
        let speed = (vx_ms.powi(2) + vy_ms.powi(2)).sqrt();
        
        if speed > max_vel {
            let scale = max_vel / speed;
            self.state[1] *= scale;
            self.state[3] *= scale;
        }
    }
}

/// Batch process multiple trajectories in parallel
#[pyfunction]
pub fn batch_filter_trajectories(
    trajectories: Vec<Vec<GpsPoint>>,
    config: Option<KalmanConfig>
) -> Vec<Vec<GpsPoint>> {
    let config = config.unwrap_or_default();
    
    trajectories
        .into_par_iter()
        .map(|trajectory| {
            let mut filter = KalmanFilter::new(Some(config.clone()));
            trajectory
                .into_iter()
                .map(|point| filter.process_point(&point))
                .collect()
        })
        .collect()
}

/// Advanced Extended Kalman Filter for non-linear motion models
#[pyclass]
pub struct ExtendedKalmanFilter {
    filter: KalmanFilter,
    motion_model: MotionModel,
}

#[derive(Debug, Clone)]
enum MotionModel {
    ConstantVelocity,
    ConstantAcceleration,
    CoordinatedTurn,
}

#[pymethods]
impl ExtendedKalmanFilter {
    #[new]
    pub fn new(motion_model: Option<String>, config: Option<KalmanConfig>) -> Self {
        let model = match motion_model.as_deref() {
            Some("constant_acceleration") => MotionModel::ConstantAcceleration,
            Some("coordinated_turn") => MotionModel::CoordinatedTurn,
            _ => MotionModel::ConstantVelocity,
        };

        Self {
            filter: KalmanFilter::new(config),
            motion_model: model,
        }
    }

    pub fn process_point(&mut self, point: &GpsPoint) -> GpsPoint {
        // For now, delegate to regular Kalman filter
        // In practice, this would implement the non-linear prediction/update steps
        self.filter.process_point(point)
    }
}

/// Particle Filter for highly non-linear scenarios
#[pyclass]
pub struct ParticleFilter {
    particles: Vec<Particle>,
    num_particles: usize,
    resampling_threshold: f64,
}

#[derive(Debug, Clone)]
struct Particle {
    state: Vector4<f64>,
    weight: f64,
}

#[pymethods]
impl ParticleFilter {
    #[new]
    pub fn new(num_particles: Option<usize>) -> Self {
        let n = num_particles.unwrap_or(1000);
        let particles = (0..n)
            .map(|_| Particle {
                state: Vector4::zeros(),
                weight: 1.0 / n as f64,
            })
            .collect();

        Self {
            particles,
            num_particles: n,
            resampling_threshold: 0.5,
        }
    }

    pub fn process_point(&mut self, point: &GpsPoint) -> GpsPoint {
        // Predict step: propagate particles
        self.predict();
        
        // Update step: weight particles based on measurement
        self.update(point);
        
        // Resample if needed
        if self.effective_sample_size() < self.resampling_threshold * self.num_particles as f64 {
            self.resample();
        }

        // Return weighted average
        self.get_estimate(point)
    }

    fn predict(&mut self) {
        // Add process noise to each particle
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        for particle in &mut self.particles {
            let noise = Vector4::new(
                rng.gen::<f64>() * 0.001,
                rng.gen::<f64>() * 0.001,
                rng.gen::<f64>() * 0.001,
                rng.gen::<f64>() * 0.001,
            );
            particle.state += noise;
        }
    }

    fn update(&mut self, measurement: &GpsPoint) {
        let measurement_vec = Vector2::new(measurement.longitude, measurement.latitude);
        
        for particle in &mut self.particles {
            let predicted = Vector2::new(particle.state[0], particle.state[2]);
            let diff = measurement_vec - predicted;
            let likelihood = (-0.5 * diff.norm_squared() / 0.01).exp();
            particle.weight *= likelihood;
        }

        // Normalize weights
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total_weight > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= total_weight;
            }
        }
    }

    fn effective_sample_size(&self) -> f64 {
        let sum_squared_weights: f64 = self.particles.iter()
            .map(|p| p.weight.powi(2))
            .sum();
        
        if sum_squared_weights > 0.0 {
            1.0 / sum_squared_weights
        } else {
            0.0
        }
    }

    fn resample(&mut self) {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        // Systematic resampling
        let mut new_particles = Vec::with_capacity(self.num_particles);
        let step = 1.0 / self.num_particles as f64;
        let mut cumulative_weight = 0.0;
        let mut weight_iter = self.particles.iter();
        let mut current_particle = weight_iter.next().unwrap();
        
        for i in 0..self.num_particles {
            let target = (i as f64 + rng.gen::<f64>()) * step;
            
            while cumulative_weight + current_particle.weight < target {
                cumulative_weight += current_particle.weight;
                if let Some(next) = weight_iter.next() {
                    current_particle = next;
                } else {
                    break;
                }
            }
            
            new_particles.push(Particle {
                state: current_particle.state,
                weight: 1.0 / self.num_particles as f64,
            });
        }
        
        self.particles = new_particles;
    }

    fn get_estimate(&self, original: &GpsPoint) -> GpsPoint {
        let weighted_state: Vector4<f64> = self.particles.iter()
            .map(|p| p.state * p.weight)
            .sum();

        GpsPoint::new(
            weighted_state[2], // latitude
            weighted_state[0], // longitude
            original.timestamp,
            original.confidence,
            original.altitude,
            Some((weighted_state[1].powi(2) + weighted_state[3].powi(2)).sqrt()),
            original.heading
        )
    }
}

/// Python module definition
#[pymodule]
fn sighthound_filtering(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KalmanFilter>()?;
    m.add_class::<KalmanConfig>()?;
    m.add_class::<ExtendedKalmanFilter>()?;
    m.add_class::<ParticleFilter>()?;
    m.add_function(wrap_pyfunction!(batch_filter_trajectories, m)?)?;
    Ok(())
} 