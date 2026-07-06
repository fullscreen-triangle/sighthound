/**
 * TomTom API wrapper
 * Handles routing, matrix, traffic
 */

const API_KEY = process.env.TOMTOM_API_KEY;

/**
 * Get route between two points
 * @param {[number, number]} from - [lat, lng]
 * @param {[number, number]} to - [lat, lng]
 * @param {string} vehicleType - "car" | "pedestrian" | "bicycle"
 * @returns {Promise<{routes: Array}>}
 */
export async function getRoute(from, to, vehicleType = "pedestrian") {
  const url = `https://api.tomtom.com/routing/1/calculateRoute/${from[0]},${from[1]}:${to[0]},${to[1]}/json?key=${API_KEY}&vehicleType=${vehicleType}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`TomTom Routing API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get travel time matrix between multiple points
 * @param {Array<[number, number]>} origins - Array of [lat, lng]
 * @param {Array<[number, number]>} destinations - Array of [lat, lng]
 * @returns {Promise<{matrix: Array}>}
 */
export async function getMatrix(origins, destinations) {
  const body = {
    origins: origins.map(([lat, lng]) => ({ point: { latitude: lat, longitude: lng } })),
    destinations: destinations.map(([lat, lng]) => ({ point: { latitude: lat, longitude: lng } })),
  };

  const url = `https://api.tomtom.com/routing/1/matrix/json?key=${API_KEY}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) throw new Error(`TomTom Matrix API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get traffic flow for a location
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @param {number} radius - Radius in meters
 * @returns {Promise<{flowSegmentData: Array}>}
 */
export async function getTrafficFlow(lat, lng, radius = 100) {
  const url = `https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=${lat},${lng}&radius=${radius}&key=${API_KEY}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`TomTom Traffic Flow API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get traffic incidents for a location
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @param {number} radius - Radius in meters
 * @returns {Promise<{incidents: Array}>}
 */
export async function getTrafficIncidents(lat, lng, radius = 1000) {
  const url = `https://api.tomtom.com/traffic/services/5/incidentDetails?point=${lat},${lng}&radius=${radius}&key=${API_KEY}&format=json`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`TomTom Traffic Incidents API error: ${response.statusText}`);
  return response.json();
}
