use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::PyArray2;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use dashmap::DashMap;
use parking_lot::RwLock;
use anyhow::{Result, Context};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// High-performance GPS point with confidence scoring
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpsPoint {
    #[pyo3(get, set)]
    pub latitude: f64,
    #[pyo3(get, set)]
    pub longitude: f64,
    #[pyo3(get, set)]
    pub timestamp: f64,
    #[pyo3(get, set)]
    pub confidence: f64,
    #[pyo3(get, set)]
    pub altitude: Option<f64>,
    #[pyo3(get, set)]
    pub speed: Option<f64>,
    #[pyo3(get, set)]
    pub heading: Option<f64>,
}

#[pymethods]
impl GpsPoint {
    #[new]
    pub fn new(
        latitude: f64,
        longitude: f64,
        timestamp: f64,
        confidence: f64,
        altitude: Option<f64>,
        speed: Option<f64>,
        heading: Option<f64>,
    ) -> Self {
        Self {
            latitude,
            longitude,
            timestamp,
            confidence,
            altitude,
            speed,
            heading,
        }
    }

    /// Calculate Haversine distance to another point (meters)
    pub fn distance_to(&self, other: &GpsPoint) -> f64 {
        haversine_distance(self.latitude, self.longitude, other.latitude, other.longitude)
    }

    /// Calculate bearing to another point (degrees)
    pub fn bearing_to(&self, other: &GpsPoint) -> f64 {
        calculate_bearing(self.latitude, self.longitude, other.latitude, other.longitude)
    }

    /// Validate GPS coordinates
    pub fn is_valid(&self) -> bool {
        self.latitude >= -90.0 && self.latitude <= 90.0 &&
        self.longitude >= -180.0 && self.longitude <= 180.0 &&
        self.confidence >= 0.0 && self.confidence <= 1.0
    }
}

/// High-performance trajectory container with spatial indexing
#[pyclass]
pub struct Trajectory {
    points: Vec<GpsPoint>,
    spatial_index: RwLock<Option<rstar::RTree<SpatialPoint>>>,
    bounds: RwLock<Option<Bounds>>,
}

#[derive(Debug, Clone)]
struct SpatialPoint {
    point: GpsPoint,
    index: usize,
}

impl rstar::Point for SpatialPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        // This is a dummy implementation, real points are created differently
        Self {
            point: GpsPoint::new(generator(0), generator(1), 0.0, 1.0, None, None, None),
            index: 0,
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.point.latitude,
            1 => self.point.longitude,
            _ => panic!("Invalid dimension index"),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.point.latitude,
            1 => &mut self.point.longitude,
            _ => panic!("Invalid dimension index"),
        }
    }
}

#[derive(Debug, Clone)]
struct Bounds {
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,
}

#[pymethods]
impl Trajectory {
    #[new]
    pub fn new(points: Vec<GpsPoint>) -> Self {
        Self {
            points,
            spatial_index: RwLock::new(None),
            bounds: RwLock::new(None),
        }
    }

    /// Add a point to the trajectory
    pub fn add_point(&mut self, point: GpsPoint) {
        self.points.push(point);
        // Invalidate cached structures
        *self.spatial_index.write() = None;
        *self.bounds.write() = None;
    }

    /// Get points in the trajectory
    pub fn get_points(&self) -> Vec<GpsPoint> {
        self.points.clone()
    }

    /// Get trajectory length
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Calculate total distance (meters)
    pub fn total_distance(&self) -> f64 {
        self.points.windows(2)
            .par_iter()
            .map(|window| window[0].distance_to(&window[1]))
            .sum()
    }

    /// Build spatial index for fast queries
    pub fn build_spatial_index(&self) {
        let spatial_points: Vec<SpatialPoint> = self.points
            .iter()
            .enumerate()
            .map(|(i, point)| SpatialPoint {
                point: *point,
                index: i,
            })
            .collect();

        let tree = rstar::RTree::bulk_load(spatial_points);
        *self.spatial_index.write() = Some(tree);
    }

    /// Find nearest points within radius (meters)
    pub fn find_points_within_radius(&self, center: GpsPoint, radius_meters: f64) -> Vec<GpsPoint> {
        // Ensure spatial index exists
        if self.spatial_index.read().is_none() {
            drop(self.spatial_index.read());
            self.build_spatial_index();
        }

        let index = self.spatial_index.read();
        let tree = index.as_ref().unwrap();

        // Convert radius to approximate degrees
        let radius_degrees = radius_meters / 111320.0; // Rough conversion

        let search_point = SpatialPoint {
            point: center,
            index: 0,
        };

        tree.locate_within_distance(search_point, radius_degrees * radius_degrees)
            .map(|sp| sp.point)
            .collect()
    }

    /// Calculate trajectory bounds
    pub fn get_bounds(&self) -> (f64, f64, f64, f64) {
        if self.bounds.read().is_none() {
            drop(self.bounds.read());
            self.calculate_bounds();
        }

        let bounds = self.bounds.read();
        let b = bounds.as_ref().unwrap();
        (b.min_lat, b.max_lat, b.min_lon, b.max_lon)
    }

    fn calculate_bounds(&self) {
        if self.points.is_empty() {
            return;
        }

        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;

        for point in &self.points {
            min_lat = min_lat.min(point.latitude);
            max_lat = max_lat.max(point.latitude);
            min_lon = min_lon.min(point.longitude);
            max_lon = max_lon.max(point.longitude);
        }

        *self.bounds.write() = Some(Bounds {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        });
    }
}

/// Fast Haversine distance calculation (meters)
#[pyfunction]
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371000.0; // Earth radius in meters

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2) +
        lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);

    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    R * c
}

/// Calculate bearing between two points (degrees)
#[pyfunction]
pub fn calculate_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let y = delta_lon.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * delta_lon.cos();

    let bearing_rad = y.atan2(x);
    (bearing_rad.to_degrees() + 360.0) % 360.0
}

/// Batch process GPS points with parallel execution
#[pyfunction]
pub fn batch_haversine_distances(
    py: Python,
    lats1: &PyArray2<f64>,
    lons1: &PyArray2<f64>,
    lats2: &PyArray2<f64>,
    lons2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let lats1_array = unsafe { lats1.as_array() };
    let lons1_array = unsafe { lons1.as_array() };
    let lats2_array = unsafe { lats2.as_array() };
    let lons2_array = unsafe { lons2.as_array() };

    let distances: Array2<f64> = lats1_array
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(lons1_array.axis_iter(Axis(0)).into_par_iter())
        .zip(lats2_array.axis_iter(Axis(0)).into_par_iter())
        .zip(lons2_array.axis_iter(Axis(0)).into_par_iter())
        .map(|(((lat1_row, lon1_row), lat2_row), lon2_row)| {
            lat1_row
                .iter()
                .zip(lon1_row.iter())
                .zip(lat2_row.iter().zip(lon2_row.iter()))
                .map(|((&lat1, &lon1), (&lat2, &lon2))| {
                    haversine_distance(lat1, lon1, lat2, lon2)
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>()
        .into_iter()
        .flatten()
        .collect::<Vec<f64>>()
        .into_iter()
        .collect::<Array1<f64>>()
        .into_shape(lats1_array.dim())
        .unwrap();

    Ok(distances.to_pyarray(py).to_owned())
}

/// Memory-efficient trajectory smoothing using Douglas-Peucker algorithm
#[pyfunction]
pub fn simplify_trajectory(points: Vec<GpsPoint>, tolerance: f64) -> Vec<GpsPoint> {
    if points.len() <= 2 {
        return points;
    }

    douglas_peucker(&points, tolerance)
}

fn douglas_peucker(points: &[GpsPoint], tolerance: f64) -> Vec<GpsPoint> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut max_distance = 0.0;
    let mut max_index = 0;

    let first = &points[0];
    let last = &points[points.len() - 1];

    for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let distance = perpendicular_distance(point, first, last);
        if distance > max_distance {
            max_distance = distance;
            max_index = i;
        }
    }

    if max_distance > tolerance {
        let mut result1 = douglas_peucker(&points[0..=max_index], tolerance);
        let mut result2 = douglas_peucker(&points[max_index..], tolerance);
        
        result1.pop(); // Remove duplicate point
        result1.extend(result2);
        result1
    } else {
        vec![points[0], points[points.len() - 1]]
    }
}

fn perpendicular_distance(point: &GpsPoint, line_start: &GpsPoint, line_end: &GpsPoint) -> f64 {
    let A = line_end.latitude - line_start.latitude;
    let B = line_start.longitude - line_end.longitude;
    let C = line_end.longitude * line_start.latitude - line_start.longitude * line_end.latitude;

    let numerator = (A * point.longitude + B * point.latitude + C).abs();
    let denominator = (A * A + B * B).sqrt();

    if denominator == 0.0 {
        point.distance_to(line_start)
    } else {
        numerator / denominator * 111320.0 // Convert to meters
    }
}

/// Python module definition
#[pymodule]
fn sighthound_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GpsPoint>()?;
    m.add_class::<Trajectory>()?;
    m.add_function(wrap_pyfunction!(haversine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bearing, m)?)?;
    m.add_function(wrap_pyfunction!(batch_haversine_distances, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_trajectory, m)?)?;
    Ok(())
} 