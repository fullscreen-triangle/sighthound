/**
 * Mapbox API wrapper
 * Handles isochrones, directions, matrix API calls
 */

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

/**
 * Get isochrone polygon(s) for a location
 * @param {number} lng - Longitude
 * @param {number} lat - Latitude
 * @param {number[]} contours - Minutes [5, 10, 15]
 * @param {string} profile - "walking" | "driving" | "cycling"
 * @returns {Promise<GeoJSON>}
 */
export async function getIsochrone(lng, lat, contours = [5, 10, 15], profile = "walking") {
  const contourParam = contours.join(",");
  const url = `https://api.mapbox.com/isochrone/v1/mapbox/${profile}/${lng},${lat}?contours_minutes=${contourParam}&polygons=true&denoise=1&access_token=${MAPBOX_TOKEN}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mapbox Isochrone API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get directions between two points
 * @param {[number, number]} from - [lng, lat]
 * @param {[number, number]} to - [lng, lat]
 * @param {string} profile - "walking" | "driving" | "cycling"
 * @returns {Promise<{routes: Array, waypoints: Array}>}
 */
export async function getDirections(from, to, profile = "walking") {
  const url = `https://api.mapbox.com/directions/v5/mapbox/${profile}/${from[0]},${from[1]};${to[0]},${to[1]}?steps=true&geometries=geojson&access_token=${MAPBOX_TOKEN}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mapbox Directions API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get travel time matrix between multiple points
 * @param {Array<[number, number]>} coordinates - Array of [lng, lat]
 * @param {string} profile - "walking" | "driving" | "cycling"
 * @returns {Promise<{durations: number[][], distances: number[][]}>}
 */
export async function getMatrix(coordinates, profile = "walking") {
  const coordStr = coordinates.map(([lng, lat]) => `${lng},${lat}`).join(";");
  const url = `https://api.mapbox.com/matrix/v1/mapbox/${profile}/${coordStr}?access_token=${MAPBOX_TOKEN}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mapbox Matrix API error: ${response.statusText}`);
  return response.json();
}

/**
 * Geocode a location name
 * @param {string} query - Location name (e.g., "Munich")
 * @returns {Promise<{features: Array}>}
 */
export async function geocode(query) {
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json?access_token=${MAPBOX_TOKEN}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mapbox Geocoding API error: ${response.statusText}`);
  return response.json();
}

/**
 * Reverse geocode coordinates to get location name
 * @param {number} lng - Longitude
 * @param {number} lat - Latitude
 * @returns {Promise<{features: Array}>}
 */
export async function reverseGeocode(lng, lat) {
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${lng},${lat}.json?access_token=${MAPBOX_TOKEN}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mapbox Reverse Geocoding API error: ${response.statusText}`);
  return response.json();
}
