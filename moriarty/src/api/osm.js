/**
 * OpenStreetMap Overpass API wrapper
 * Queries infrastructure: traffic signals, towers, POIs
 */

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

/**
 * Query OSM for traffic signals in a bounding box
 * @param {Array<number>} bbox - [south, west, north, east] in degrees
 * @returns {Promise<{elements: Array}>}
 */
export async function getTrafficSignals(bbox) {
  const [south, west, north, east] = bbox;
  const query = `
    [bbox:${south},${west},${north},${east}];
    (
      node["highway"="traffic_signals"];
      way["highway"="traffic_signals"];
    );
    out geom;
  `;

  return queryOverpass(query);
}

/**
 * Query OSM for cell towers in a bounding box
 * @param {Array<number>} bbox - [south, west, north, east] in degrees
 * @returns {Promise<{elements: Array}>}
 */
export async function getCellTowers(bbox) {
  const [south, west, north, east] = bbox;
  const query = `
    [bbox:${south},${west},${north},${east}];
    (
      node["man_made"="mast"]["tower:type"="communication"];
      node["man_made"="tower"]["tower:type"="communication"];
    );
    out center;
  `;

  return queryOverpass(query);
}

/**
 * Query OSM for amenities (restaurants, shops, transit stops)
 * @param {Array<number>} bbox - [south, west, north, east] in degrees
 * @param {string} amenity - OSM amenity tag (e.g., "restaurant", "cafe")
 * @returns {Promise<{elements: Array}>}
 */
export async function getAmenities(bbox, amenity = "restaurant") {
  const [south, west, north, east] = bbox;
  const query = `
    [bbox:${south},${west},${north},${east}];
    (
      node["amenity"="${amenity}"];
      way["amenity"="${amenity}"];
    );
    out center;
  `;

  return queryOverpass(query);
}

/**
 * Query OSM for public transit stops
 * @param {Array<number>} bbox - [south, west, north, east] in degrees
 * @returns {Promise<{elements: Array}>}
 */
export async function getTransitStops(bbox) {
  const [south, west, north, east] = bbox;
  const query = `
    [bbox:${south},${west},${north},${east}];
    (
      node["public_transport"="stop_position"];
      node["amenity"="bus_station"];
      node["amenity"="taxi"];
    );
    out center;
  `;

  return queryOverpass(query);
}

/**
 * Query OSM for building footprints
 * @param {Array<number>} bbox - [south, west, north, east] in degrees
 * @returns {Promise<{elements: Array}>}
 */
export async function getBuildings(bbox) {
  const [south, west, north, east] = bbox;
  const query = `
    [bbox:${south},${west},${north},${east}];
    way["building"];
    out geom;
  `;

  return queryOverpass(query);
}

/**
 * Generic Overpass query
 * @param {string} query - Overpass QL query
 * @returns {Promise<{elements: Array}>}
 */
export async function queryOverpass(query) {
  const response = await fetch(OVERPASS_URL, {
    method: "POST",
    body: query,
  });

  if (!response.ok) throw new Error(`Overpass API error: ${response.statusText}`);
  return response.json();
}

/**
 * Convert Overpass elements to GeoJSON
 * @param {Array} elements - OSM elements from Overpass
 * @returns {GeoJSON} FeatureCollection
 */
export function toGeoJSON(elements) {
  const features = elements
    .filter((el) => {
      // Only include elements with coordinates
      if (el.type === "node") return el.lat && el.lon;
      if (el.type === "way") return el.geometry && el.geometry.length > 0;
      return false;
    })
    .map((el) => {
      if (el.type === "node") {
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [el.lon, el.lat] },
          properties: el.tags || {},
        };
      }
      if (el.type === "way") {
        const coords = el.geometry.map((p) => [p.lon, p.lat]);
        return {
          type: "Feature",
          geometry: { type: "LineString", coordinates: coords },
          properties: el.tags || {},
        };
      }
      return null;
    })
    .filter((f) => f !== null);

  return {
    type: "FeatureCollection",
    features,
  };
}
