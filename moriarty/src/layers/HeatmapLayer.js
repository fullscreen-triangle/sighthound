import { HeatmapLayer } from "@deck.gl/layers";

/**
 * Create a Deck.gl heatmap layer
 * @param {Array<Object>} data - Array of points with {lat, lng, value}
 * @param {Object} options - Heatmap options
 * @returns {HeatmapLayer}
 */
export function createHeatmapLayer(data, options = {}) {
  const defaultOptions = {
    id: "heatmap-layer",
    data,
    getPosition: (d) => [d.lng, d.lat],
    getWeight: (d) => d.value || 1,
    radiusPixels: 50,
    intensity: 1,
    threshold: 0.05,
    colorRange: [
      [255, 0, 0],      // Red - high
      [255, 128, 0],    // Orange
      [255, 255, 0],    // Yellow
      [0, 255, 0],      // Green
      [0, 0, 255],      // Blue - low
    ],
    ...options,
  };

  return new HeatmapLayer(defaultOptions);
}

/**
 * Create a heatmap for signal strength / precision
 * @param {Array<Object>} towers - Array of towers with {lat, lng, signal_strength}
 * @returns {HeatmapLayer}
 */
export function createSignalStrengthHeatmap(towers) {
  return createHeatmapLayer(towers, {
    id: "signal-strength-heatmap",
    getWeight: (d) => d.signal_strength || 0.5,
    colorRange: [
      [255, 0, 0],      // Red - weak signal
      [255, 165, 0],    // Orange
      [255, 255, 0],    // Yellow
      [144, 238, 144],  // Light green
      [0, 128, 0],      // Green - strong signal
    ],
  });
}

/**
 * Create a heatmap for time synchronization precision
 * @param {Array<Object>} clockPoints - Array of points with {lat, lng, precision}
 * @returns {HeatmapLayer}
 */
export function createPrecisionHeatmap(clockPoints) {
  return createHeatmapLayer(clockPoints, {
    id: "precision-heatmap",
    getWeight: (d) => d.precision || 0.5, // Higher = better precision
    colorRange: [
      [255, 0, 0],      // Red - poor precision
      [255, 128, 0],
      [255, 255, 0],
      [144, 238, 144],
      [0, 128, 0],      // Green - excellent precision
    ],
  });
}

/**
 * Create a heatmap for air quality
 * @param {Array<Object>} aqPoints - Array of points with {lat, lng, aqi}
 * @returns {HeatmapLayer}
 */
export function createAirQualityHeatmap(aqPoints) {
  return createHeatmapLayer(aqPoints, {
    id: "air-quality-heatmap",
    getWeight: (d) => (d.aqi || 3) / 5, // Normalize AQI (1-5)
    colorRange: [
      [0, 128, 0],      // Green - good
      [255, 255, 0],    // Yellow - moderate
      [255, 165, 0],    // Orange - unhealthy for sensitive
      [255, 0, 0],      // Red - unhealthy
      [139, 0, 0],      // Dark red - hazardous
    ],
  });
}
