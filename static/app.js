// S-Entropy Positioning Web Tool - Common Application Logic

console.log('%c S-Entropy Positioning Framework ', 'background: #0066ff; color: white; font-size: 14px; padding: 5px 10px;');
console.log('Interactive demonstrations of:');
console.log('1. Shortest Path Algorithm (SEBD)');
console.log('2. Reachable Region Calculator');
console.log('3. Satellite Visibility Analyzer');

// Utility functions
const Utils = {
    /**
     * Format distance in kilometers
     */
    formatDistance: (km) => {
        if (km < 1) {
            return `${(km * 1000).toFixed(0)}m`;
        }
        return `${km.toFixed(2)}km`;
    },

    /**
     * Format coordinates
     */
    formatCoords: (lat, lon) => {
        const latDir = lat >= 0 ? 'N' : 'S';
        const lonDir = lon >= 0 ? 'E' : 'W';
        return `${Math.abs(lat).toFixed(2)}°${latDir}, ${Math.abs(lon).toFixed(2)}°${lonDir}`;
    },

    /**
     * Get color based on value (for elevation, distance, etc.)
     */
    getColor: (value, max = 100) => {
        const ratio = Math.min(value / max, 1);
        if (ratio < 0.33) return '#ff0000';  // Red
        if (ratio < 0.66) return '#ff9900';  // Orange
        return '#00ff00';                    // Green
    },

    /**
     * Calculate bearing between two points
     */
    calculateBearing: (lat1, lon1, lat2, lon2) => {
        const toRad = Math.PI / 180;
        const dLon = (lon2 - lon1) * toRad;
        const y = Math.sin(dLon) * Math.cos(lat2 * toRad);
        const x = Math.cos(lat1 * toRad) * Math.sin(lat2 * toRad) -
                  Math.sin(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.cos(dLon);
        const bearing = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
        return bearing.toFixed(0);
    }
};

// Event tracking
const Analytics = {
    events: [],

    log: (event, data = {}) => {
        const entry = {
            timestamp: new Date().toISOString(),
            event,
            ...data
        };
        Analytics.events.push(entry);
        console.log(`[${event}]`, data);
    },

    getSession: () => {
        return Analytics.events;
    }
};

// API helper
const API = {
    /**
     * Make POST request to API endpoint
     */
    post: async (endpoint, data) => {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            Analytics.log(endpoint, { success: true, dataSize: JSON.stringify(result).length });
            return result;

        } catch (error) {
            Analytics.log(endpoint, { success: false, error: error.message });
            throw error;
        }
    },

    /**
     * Make GET request to API endpoint
     */
    get: async (endpoint) => {
        try {
            const response = await fetch(endpoint);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            Analytics.log(endpoint, { success: true, dataSize: JSON.stringify(result).length });
            return result;

        } catch (error) {
            Analytics.log(endpoint, { success: false, error: error.message });
            throw error;
        }
    }
};

// Map helper for Leaflet
const MapHelper = {
    /**
     * Create a Leaflet map with base layer
     */
    createMap: (containerId, initialLat = 20, initialLon = 0, zoom = 3) => {
        const map = L.map(containerId).setView([initialLat, initialLon], zoom);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }).addTo(map);
        return map;
    },

    /**
     * Add circle marker to map
     */
    addMarker: (map, lat, lon, options = {}) => {
        const defaultOptions = {
            radius: 8,
            fillColor: '#0066ff',
            color: '#000',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        };
        return L.circleMarker([lat, lon], { ...defaultOptions, ...options }).addTo(map);
    },

    /**
     * Add polyline to map
     */
    addPolyline: (map, points, options = {}) => {
        const defaultOptions = {
            color: '#0066ff',
            weight: 3,
            opacity: 0.7
        };
        return L.polyline(points, { ...defaultOptions, ...options }).addTo(map);
    },

    /**
     * Add polygon to map
     */
    addPolygon: (map, points, options = {}) => {
        const defaultOptions = {
            color: '#ff9900',
            weight: 2,
            opacity: 0.7,
            fillColor: '#ff9900',
            fillOpacity: 0.2
        };
        return L.polygon(points, { ...defaultOptions, ...options }).addTo(map);
    },

    /**
     * Fit bounds to contain all markers
     */
    fitBounds: (map, points) => {
        const bounds = L.latLngBounds(points);
        map.fitBounds(bounds, { padding: [50, 50] });
    }
};

// Store utility
const Storage = {
    /**
     * Save to localStorage
     */
    save: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.warn('localStorage save failed:', e);
        }
    },

    /**
     * Load from localStorage
     */
    load: (key, defaultValue = null) => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.warn('localStorage load failed:', e);
            return defaultValue;
        }
    },

    /**
     * Remove from localStorage
     */
    remove: (key) => {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('localStorage remove failed:', e);
        }
    }
};

// Export utilities to global scope
window.Utils = Utils;
window.Analytics = Analytics;
window.API = API;
window.MapHelper = MapHelper;
window.Storage = Storage;

console.log('Utilities loaded: Utils, Analytics, API, MapHelper, Storage');
