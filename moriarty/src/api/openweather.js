/**
 * OpenWeatherMap API wrapper
 * Handles current weather, one-call API, air quality
 */

const API_KEY = process.env.OPENWEATHERMAP_API_KEY;

/**
 * Get current weather for coordinates
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {Promise<{current, hourly, daily}>}
 */
export async function getCurrentWeather(lat, lng) {
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=${API_KEY}&units=metric`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`OpenWeather API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get one-call weather (current, hourly, daily, alerts)
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {Promise<{current, hourly, daily, alerts}>}
 */
export async function getOneCallWeather(lat, lng) {
  const url = `https://api.openweathermap.org/data/2.5/onecall?lat=${lat}&lon=${lng}&appid=${API_KEY}&units=metric`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`OpenWeather One Call API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get air quality data for coordinates
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {Promise<{list: Array}>}
 */
export async function getAirQuality(lat, lng) {
  const url = `https://api.openweathermap.org/data/2.5/air_pollution?lat=${lat}&lon=${lng}&appid=${API_KEY}`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`OpenWeather Air Quality API error: ${response.statusText}`);
  return response.json();
}

/**
 * Get forecast for coordinates
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {Promise<{list: Array}>}
 */
export async function getForecast(lat, lng) {
  const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lng}&appid=${API_KEY}&units=metric`;

  const response = await fetch(url);
  if (!response.ok) throw new Error(`OpenWeather Forecast API error: ${response.statusText}`);
  return response.json();
}
