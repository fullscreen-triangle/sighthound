import { PathLayer, ScatterplotLayer, LineLayer } from "@deck.gl/layers";

/**
 * Create a path layer for routing visualization
 * @param {Array<[number, number]>} path - Array of [lng, lat] coordinates
 * @param {Object} options - Path styling options
 * @returns {PathLayer}
 */
export function createPathLayer(path, options = {}) {
  const defaultOptions = {
    id: "route-path-layer",
    data: [{ path }],
    getPath: (d) => d.path,
    getColor: [0, 100, 255, 255],
    getWidth: 5,
    widthMinPixels: 2,
    widthMaxPixels: 20,
    ...options,
  };

  return new PathLayer(defaultOptions);
}

/**
 * Create visualization for FWDC algorithm path with signal timing
 * @param {Object} routeData - {path: [[lng, lat]], signals: [{lat, lng, cycle_time, current_phase}]}
 * @returns {Array<Layer>}
 */
export function createFWDCRoutingLayers(routeData) {
  const layers = [];

  if (routeData.path && routeData.path.length > 0) {
    // Main path layer
    layers.push(
      createPathLayer(routeData.path, {
        id: "fwdc-path",
        getColor: [76, 175, 80, 255],
        getWidth: 8,
      })
    );

    // Start point (green)
    const startPoint = routeData.path[0];
    layers.push(
      new ScatterplotLayer({
        id: "route-start",
        data: [{ position: startPoint, label: "Start" }],
        getPosition: (d) => d.position,
        getRadius: 200,
        getColor: [76, 175, 80, 255],
        radiusScale: 1,
        radiusMinPixels: 8,
        radiusMaxPixels: 15,
      })
    );

    // End point (red)
    const endPoint = routeData.path[routeData.path.length - 1];
    layers.push(
      new ScatterplotLayer({
        id: "route-end",
        data: [{ position: endPoint, label: "End" }],
        getPosition: (d) => d.position,
        getRadius: 200,
        getColor: [244, 67, 54, 255],
        radiusScale: 1,
        radiusMinPixels: 8,
        radiusMaxPixels: 15,
      })
    );
  }

  // Traffic signal visualization
  if (routeData.signals && routeData.signals.length > 0) {
    layers.push(
      new ScatterplotLayer({
        id: "traffic-signals",
        data: routeData.signals,
        getPosition: (d) => [d.lng, d.lat],
        getRadius: (d) => {
          // Larger radius for signals on the path
          return d.on_path ? 150 : 100;
        },
        getColor: (d) => {
          // Color by current phase
          if (d.current_phase === "green") return [76, 175, 80, 255]; // Green
          if (d.current_phase === "yellow") return [255, 193, 7, 255]; // Yellow
          return [244, 67, 54, 255]; // Red
        },
        radiusScale: 1,
        radiusMinPixels: 6,
        radiusMaxPixels: 12,
        pickable: true,
      })
    );
  }

  // Separation cost visualization (fuzzy intervals)
  if (routeData.separation_costs && routeData.separation_costs.length > 0) {
    layers.push(
      new LineLayer({
        id: "separation-costs",
        data: routeData.separation_costs,
        getSourcePosition: (d) => [d.lng, d.lat],
        getTargetPosition: (d) => [
          d.lng + 0.001 * (d.sigma_max - d.sigma_min) / 100,
          d.lat,
        ],
        getColor: (d) => {
          // Color by gap size
          const gap = d.sigma_max - d.sigma_min;
          if (gap < 10) return [76, 175, 80, 200]; // Green - well separated
          if (gap < 30) return [255, 193, 7, 200]; // Yellow - moderate
          return [244, 67, 54, 200]; // Red - overlapping
        },
        getWidth: (d) => Math.max(1, 10 - (d.sigma_max - d.sigma_min) / 5),
      })
    );
  }

  return layers;
}

/**
 * Create alternative route visualization
 * @param {Array<[number, number]>} altPath - Alternative path coordinates
 * @returns {PathLayer}
 */
export function createAlternativePathLayer(altPath) {
  return createPathLayer(altPath, {
    id: "alternative-path",
    getColor: [200, 200, 200, 150],
    getWidth: 3,
    dashArray: [4, 4],
  });
}

/**
 * Create isochrone visualization from a routing perspective
 * @param {Array<Object>} isochroneGeoJSON - GeoJSON from Mapbox Isochrone API
 * @returns {Layer}
 */
export function createRoutingIsochroneLayer(isochroneGeoJSON) {
  const { GeoJsonLayer } = require("@deck.gl/layers");

  return new GeoJsonLayer({
    id: "routing-isochrone",
    data: isochroneGeoJSON,
    stroked: true,
    filled: true,
    pickable: true,
    getFillColor: (feature) => {
      const minutes = feature.properties?.contour || 0;
      if (minutes <= 5) return [76, 175, 80, 60];
      if (minutes <= 10) return [33, 150, 243, 50];
      if (minutes <= 15) return [255, 152, 0, 40];
      return [244, 67, 54, 30];
    },
    getLineColor: [0, 0, 0, 100],
    getLineWidth: 1,
  });
}

/**
 * Convert route response to visualization data
 * @param {Object} routeResponse - Mapbox/TomTom directions response
 * @param {Object} signalData - Traffic signals with timing info
 * @returns {Object} Visualization ready data
 */
export function routeToVisualizationData(routeResponse, signalData = {}) {
  let path = [];

  // Extract coordinates from route response
  if (routeResponse.routes && routeResponse.routes[0]) {
    const geometry = routeResponse.routes[0].geometry;
    if (Array.isArray(geometry)) {
      path = geometry; // TomTom format
    } else if (geometry.coordinates) {
      path = geometry.coordinates; // Mapbox format
    }
  }

  return {
    path,
    signals: signalData.features || [],
    separation_costs: [],
  };
}
