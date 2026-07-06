import { GeoJsonLayer } from "@deck.gl/layers";

/**
 * Create a Deck.gl GeoJSON layer for isochrones
 * @param {GeoJSON} data - GeoJSON FeatureCollection from Mapbox Isochrone API
 * @param {Object} options - Styling options
 * @returns {GeoJsonLayer}
 */
export function createIsochroneLayer(data, options = {}) {
  const defaultOptions = {
    id: "isochrone-layer",
    data,
    stroked: true,
    filled: true,
    pickable: true,
    getFillColor: (feature) => {
      // Color based on contour value (minutes)
      const minutes = feature.properties?.contour || 0;
      if (minutes <= 5) return [76, 175, 80, 100]; // Green for 5 min
      if (minutes <= 10) return [33, 150, 243, 80]; // Blue for 10 min
      if (minutes <= 15) return [255, 152, 0, 60]; // Orange for 15 min
      return [244, 67, 54, 40]; // Red for 20+ min
    },
    getLineColor: [0, 0, 0, 255],
    getLineWidth: 2,
    getLineWidthMinPixels: 1,
    ...options,
  };

  return new GeoJsonLayer(defaultOptions);
}

/**
 * Create multiple isochrone layers with different walk speeds
 * @param {Object} isochroneData - { walking: GeoJSON, cycling: GeoJSON, driving: GeoJSON }
 * @returns {Array<GeoJsonLayer>}
 */
export function createMultiSpeedIsochroneLayers(isochroneData) {
  const layers = [];

  if (isochroneData.walking) {
    layers.push(
      createIsochroneLayer(isochroneData.walking, {
        id: "isochrone-walking",
        getFillColor: () => [76, 175, 80, 80],
      })
    );
  }

  if (isochroneData.cycling) {
    layers.push(
      createIsochroneLayer(isochroneData.cycling, {
        id: "isochrone-cycling",
        getFillColor: () => [33, 150, 243, 60],
      })
    );
  }

  if (isochroneData.driving) {
    layers.push(
      createIsochroneLayer(isochroneData.driving, {
        id: "isochrone-driving",
        getFillColor: () => [255, 152, 0, 40],
      })
    );
  }

  return layers;
}
