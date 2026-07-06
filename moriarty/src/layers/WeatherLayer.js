import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";

/**
 * Create wind vector field as arrows
 * @param {Array<Object>} windPoints - Array of {lat, lng, wind_speed, wind_direction}
 * @returns {LineLayer}
 */
export function createWindVectorLayer(windPoints) {
  // Convert wind direction/speed to line segments
  const windVectors = windPoints.flatMap((point) => {
    const { lat, lng, wind_speed, wind_direction } = point;
    // Normalize wind speed to line length (pixels)
    const length = Math.min(wind_speed / 10, 1); // cap at 1 degree per 10 m/s
    // Convert direction to radians (meteorological: 0° = from north)
    const rad = (wind_direction * Math.PI) / 180;
    // Calculate end point
    const endLng = lng + length * Math.sin(rad);
    const endLat = lat + length * Math.cos(rad);

    return [
      { sourcePosition: [lng, lat], targetPosition: [endLng, endLat], speed: wind_speed },
    ];
  });

  return new LineLayer({
    id: "wind-vector-layer",
    data: windVectors,
    getSourcePosition: (d) => d.sourcePosition,
    getTargetPosition: (d) => d.targetPosition,
    getColor: (d) => {
      // Color by wind speed
      if (d.speed < 5) return [0, 255, 0, 200]; // Light green
      if (d.speed < 10) return [255, 255, 0, 200]; // Yellow
      if (d.speed < 15) return [255, 165, 0, 200]; // Orange
      return [255, 0, 0, 200]; // Red
    },
    getWidth: (d) => Math.max(1, d.speed / 5),
  });
}

/**
 * Create pressure contour points visualization
 * @param {Array<Object>} pressurePoints - Array of {lat, lng, pressure, temperature}
 * @returns {ScatterplotLayer}
 */
export function createWeatherPointsLayer(pressurePoints) {
  return new ScatterplotLayer({
    id: "weather-points-layer",
    data: pressurePoints,
    getPosition: (d) => [d.lng, d.lat],
    getRadius: (d) => {
      // Radius based on temperature
      const temp = d.temperature || 20;
      return Math.abs(temp - 15) / 10 + 50; // Base 50 pixels, scale by temp deviation
    },
    getColor: (d) => {
      // Color by temperature
      const temp = d.temperature || 20;
      if (temp < 0) return [0, 0, 255, 200]; // Blue - cold
      if (temp < 10) return [100, 149, 237, 200]; // Cornflower blue
      if (temp < 20) return [144, 238, 144, 200]; // Light green
      if (temp < 30) return [255, 255, 0, 200]; // Yellow
      return [255, 0, 0, 200]; // Red - hot
    },
    radiusScale: 1,
    radiusMinPixels: 5,
    radiusMaxPixels: 30,
    pickable: true,
  });
}

/**
 * Create cloud coverage visualization
 * @param {Array<Object>} cloudPoints - Array of {lat, lng, cloud_cover (0-1)}
 * @returns {ScatterplotLayer}
 */
export function createCloudCoverageLayer(cloudPoints) {
  return new ScatterplotLayer({
    id: "cloud-coverage-layer",
    data: cloudPoints,
    getPosition: (d) => [d.lng, d.lat],
    getRadius: 100,
    getColor: (d) => {
      const coverage = d.cloud_cover || 0.5;
      // Grayscale: white (cloudy) to black (clear)
      const gray = Math.round(255 * coverage);
      return [gray, gray, gray, Math.round(200 * coverage)];
    },
    radiusScale: 1,
    radiusMinPixels: 20,
    radiusMaxPixels: 50,
  });
}

/**
 * Create precipitation intensity heatmap
 * @param {Array<Object>} precipPoints - Array of {lat, lng, precipitation}
 * @returns {ScatterplotLayer}
 */
export function createPrecipitationLayer(precipPoints) {
  return new ScatterplotLayer({
    id: "precipitation-layer",
    data: precipPoints,
    getPosition: (d) => [d.lng, d.lat],
    getRadius: (d) => {
      const precip = d.precipitation || 0;
      return Math.min(precip * 50, 200); // Scale precipitation to radius
    },
    getColor: (d) => {
      const precip = d.precipitation || 0;
      if (precip < 0.1) return [0, 0, 0, 0]; // Transparent - no rain
      if (precip < 2) return [65, 105, 225, 150]; // Blue - light rain
      if (precip < 5) return [30, 144, 255, 180]; // Dodger blue - moderate
      if (precip < 10) return [0, 0, 255, 200]; // Blue - heavy rain
      return [25, 0, 130, 220]; // Indigo - very heavy
    },
    radiusScale: 1,
    radiusMinPixels: 3,
    radiusMaxPixels: 50,
  });
}

/**
 * Convert weather data to visualization points
 * @param {Object} weatherData - OpenWeatherMap response
 * @returns {Array<Object>}
 */
export function weatherDataToPoints(weatherData) {
  const points = [];

  if (weatherData.current) {
    const { coord, main, wind, clouds, rain, snow } = weatherData.current;
    points.push({
      lat: coord.lat,
      lng: coord.lon,
      temperature: main.temp,
      pressure: main.pressure,
      wind_speed: wind.speed,
      wind_direction: wind.deg || 0,
      cloud_cover: clouds.all / 100,
      precipitation: (rain?.["1h"] || 0) + (snow?.["1h"] || 0),
    });
  }

  return points;
}
