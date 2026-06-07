/**
 * Cynegeticus Runtime Library
 * Core mathematical functions for S-entropy positioning
 */

import { Coord, Measurement, Position, Validation } from "./types";

// ============================================================================
// CONSTANTS
// ============================================================================

export const EARTH_RADIUS_KM = 6371.0;
export const DEG_TO_RAD = Math.PI / 180;
export const RAD_TO_DEG = 180 / Math.PI;

// ============================================================================
// GEOGRAPHIC CALCULATIONS
// ============================================================================

/**
 * Haversine formula: great-circle distance between two lat/lon points
 * Returns distance in kilometers
 */
export function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const dlat = (lat2 - lat1) * DEG_TO_RAD;
  const dlon = (lon2 - lon1) * DEG_TO_RAD;

  const a =
    Math.sin(dlat / 2) ** 2 +
    Math.cos(lat1 * DEG_TO_RAD) *
      Math.cos(lat2 * DEG_TO_RAD) *
      Math.sin(dlon / 2) ** 2;

  const c = 2 * Math.asin(Math.sqrt(a));
  return EARTH_RADIUS_KM * c;
}

/**
 * Calculate elevation angle from observer to satellite
 * Returns angle in degrees (positive = above horizon)
 */
export function elevationAngle(
  observerLat: number,
  observerLon: number,
  satelliteLat: number,
  satelliteLon: number,
  satelliteAltKm: number,
  observerAltKm: number = 0
): number {
  const gcDist = haversineDistance(
    observerLat,
    observerLon,
    satelliteLat,
    satelliteLon
  );

  const dalt = satelliteAltKm - observerAltKm;
  const elevation = Math.atan2(dalt, gcDist) * RAD_TO_DEG;

  return elevation;
}

/**
 * Compute bearing from point 1 to point 2
 * Returns bearing in degrees (0 = North, 90 = East)
 */
export function bearing(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const dlon = (lon2 - lon1) * DEG_TO_RAD;
  const y = Math.sin(dlon) * Math.cos(lat2 * DEG_TO_RAD);
  const x =
    Math.cos(lat1 * DEG_TO_RAD) * Math.sin(lat2 * DEG_TO_RAD) -
    Math.sin(lat1 * DEG_TO_RAD) *
      Math.cos(lat2 * DEG_TO_RAD) *
      Math.cos(dlon);

  const brg = (Math.atan2(y, x) * RAD_TO_DEG + 360) % 360;
  return brg;
}

/**
 * Calculate destination point given start point, bearing, and distance
 * Returns [lat, lon]
 */
export function destPoint(
  lat: number,
  lon: number,
  brg: number,
  distKm: number
): [number, number] {
  const latRad = lat * DEG_TO_RAD;
  const lonRad = lon * DEG_TO_RAD;
  const brgRad = brg * DEG_TO_RAD;
  const dRad = distKm / EARTH_RADIUS_KM;

  const latNew = Math.asin(
    Math.sin(latRad) * Math.cos(dRad) +
      Math.cos(latRad) * Math.sin(dRad) * Math.cos(brgRad)
  );

  const lonNew =
    lonRad +
    Math.atan2(
      Math.sin(brgRad) * Math.sin(dRad) * Math.cos(latRad),
      Math.cos(dRad) - Math.sin(latRad) * Math.sin(latNew)
    );

  return [latNew * RAD_TO_DEG, lonNew * RAD_TO_DEG];
}

// ============================================================================
// S-ENTROPY COORDINATE CALCULATIONS
// ============================================================================

/**
 * Compute S-entropy distance metric
 * Combines geographic, terrain, and infrastructure components
 */
export function sEntropyDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  // Geographic component (50% weight)
  const gcDist = haversineDistance(lat1, lon1, lat2, lon2);
  const maxDist = haversineDistance(-90, 0, 90, 180); // Half Earth circumference
  const geoComponent = gcDist / maxDist;

  // Terrain component (30% weight) - latitude-dependent roughness
  const terrain1 = Math.abs(lat1) / 90.0;
  const terrain2 = Math.abs(lat2) / 90.0;
  const terrainComponent = Math.abs(terrain1 - terrain2);

  // Infrastructure component (20% weight) - longitude-dependent density
  const infra1 = ((lon1 % 180) + 180) % 180 / 180.0;
  const infra2 = ((lon2 % 180) + 180) % 180 / 180.0;
  const infraComponent = Math.abs(infra1 - infra2);

  // Combine components
  const entropy = Math.sqrt(
    0.5 * geoComponent ** 2 +
      0.3 * terrainComponent ** 2 +
      0.2 * infraComponent ** 2
  );

  return Math.min(entropy, 1.0);
}

// ============================================================================
// COORDINATE TRANSFORMATIONS
// ============================================================================

/**
 * Convert geographic coordinates to S-entropy coordinates
 * This is a simplified forward mapping
 */
export function geoToSCoord(lat: number, lon: number): Coord {
  // Normalize latitude to [0, 1] (Sk = kinetic entropy ~terrain)
  const sk = (lat + 90) / 180; // -90 to +90 → 0 to 1

  // Normalize longitude to [0, 1] (St = temporal entropy ~velocity/time)
  const st = ((lon + 180) % 360) / 360; // -180 to +180 → 0 to 1

  // Energy component (Se) could be derived from altitude or temperature
  // For now, use a simple function
  const se = 0.5; // Default to middle of range

  return { sk, st, se };
}

/**
 * Convert S-entropy coordinates back to geographic coordinates
 * This is the inverse mapping (simplified)
 */
export function sCoordToGeo(coord: Coord): [number, number] {
  // Denormalize Sk to latitude
  const lat = coord.sk * 180 - 90;

  // Denormalize St to longitude
  const lon = coord.st * 360 - 180;

  return [lat, lon];
}

// ============================================================================
// POSITION RESOLUTION
// ============================================================================

/**
 * Resolve position from S-entropy coordinates
 * Returns [lat, lon, altitude]
 */
export function resolvePosition(coord: Coord, altitudeKm: number = 0): Position {
  const [lat, lon] = sCoordToGeo(coord);

  return {
    lat,
    lon,
    altitude: altitudeKm,
    timestamp: Date.now(),
  };
}

/**
 * Triangulate position from multiple satellites
 * Refines initial estimate using satellite measurements
 */
export function triangulatePosition(
  initialEstimate: [number, number],
  satellites: Array<{ lat: number; lon: number; altitude: number }>,
  elevationThreshold: number = 10
): Position {
  let refinedLat = initialEstimate[0];
  let refinedLon = initialEstimate[1];

  // Simple triangulation: weighted average of visible satellite directions
  let sumLat = 0;
  let sumLon = 0;
  let visibleCount = 0;

  for (const sat of satellites) {
    const elev = elevationAngle(
      refinedLat,
      refinedLon,
      sat.lat,
      sat.lon,
      sat.altitude
    );

    if (elev >= elevationThreshold) {
      // Weight by elevation angle (higher = more reliable)
      const weight = (elev / 90) ** 2; // Quadratic weighting

      sumLat += sat.lat * weight;
      sumLon += sat.lon * weight;
      visibleCount += weight;
    }
  }

  if (visibleCount > 0) {
    refinedLat = sumLat / visibleCount;
    refinedLon = sumLon / visibleCount;
  }

  return {
    lat: refinedLat,
    lon: refinedLon,
    altitude: 0,
    timestamp: Date.now(),
  };
}

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * Circular closure validation
 * Verify position round-trip consistency (geo → S-entropy → geo)
 */
export function validateCircularClosure(
  lat: number,
  lon: number,
  rmseThreshold: number
): Validation {
  // Forward: geo → S-entropy
  const sCoord = geoToSCoord(lat, lon);

  // Backward: S-entropy → geo
  const [latRecon, lonRecon] = sCoordToGeo(sCoord);

  // Calculate RMSE
  const latError = (lat - latRecon) ** 2;
  const lonError = (lon - lonRecon) ** 2;
  const rmse = Math.sqrt(latError + lonError); // In degrees

  const passed = rmse < rmseThreshold;

  return {
    type: "circular_closure",
    passed,
    value: rmse,
    threshold: rmseThreshold,
    details: `Round-trip RMSE: ${rmse.toFixed(6)}° (${(rmse * 111).toFixed(2)}km)`,
  };
}

/**
 * Accuracy validation against known position
 */
export function validateAgainstKnown(
  estimated: [number, number],
  known: [number, number]
): Validation {
  const distance = haversineDistance(
    estimated[0],
    estimated[1],
    known[0],
    known[1]
  );

  return {
    type: "position_vs_known",
    passed: distance < 1.0, // Within 1 km
    value: distance,
    threshold: 1.0,
    details: `Distance from known: ${distance.toFixed(3)}km`,
  };
}

// ============================================================================
// ATMOSPHERIC MODEL
// ============================================================================

/**
 * Simple atmospheric model: entropy calculation from modality measurements
 */
export function atmosphericEntropy(
  measurements: Measurement[],
  modality: string
): Coord {
  // Filter measurements by modality
  const relevant = measurements.filter((m) => m.modality === modality);

  if (relevant.length === 0) {
    return { sk: 0.5, st: 0.5, se: 0.5 };
  }

  // Calculate weighted average
  let sumSk = 0,
    sumSt = 0,
    sumSe = 0;

  for (const m of relevant) {
    // Normalize measurement value to [0, 1]
    const normalized = Math.min(m.value / 1000, 1.0); // Arbitrary scaling

    sumSk += normalized;
    sumSt += normalized * (m.timestamp % 1000) / 1000;
    sumSe += normalized;
  }

  const count = relevant.length;

  return {
    sk: Math.min(sumSk / count, 1.0),
    st: Math.min(sumSt / count, 1.0),
    se: Math.min(sumSe / count, 1.0),
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if coordinate is within bounds
 */
export function inBounds(
  coord: Coord,
  skBounds: [number, number],
  stBounds: [number, number],
  seBounds: [number, number]
): boolean {
  return (
    coord.sk >= skBounds[0] &&
    coord.sk <= skBounds[1] &&
    coord.st >= stBounds[0] &&
    coord.st <= stBounds[1] &&
    coord.se >= seBounds[0] &&
    coord.se <= seBounds[1]
  );
}

/**
 * Distance between two coordinates
 */
export function coordDistance(c1: Coord, c2: Coord): number {
  const dsk = (c1.sk - c2.sk) ** 2;
  const dst = (c1.st - c2.st) ** 2;
  const dse = (c1.se - c2.se) ** 2;

  return Math.sqrt(dsk + dst + dse);
}

/**
 * Linear interpolation between coordinates
 */
export function lerpCoord(
  c1: Coord,
  c2: Coord,
  t: number
): Coord {
  return {
    sk: c1.sk + (c2.sk - c1.sk) * t,
    st: c1.st + (c2.st - c1.st) * t,
    se: c1.se + (c2.se - c1.se) * t,
  };
}
