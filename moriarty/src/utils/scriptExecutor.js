/**
 * Script Executor for Sandbox Scripts 1-8
 * Parses .cynes script files and executes visualization commands
 */

import { getIsochrone, geocode } from "@/api/mapbox";
import { getCurrentWeather, getAirQuality } from "@/api/openweather";
import { getCellTowers, getTrafficSignals } from "@/api/osm";

export class ScriptExecutor {
  constructor() {
    this.state = {
      currentScript: null,
      position: { lat: 48.1351, lng: 11.5656 }, // Default: Munich
      layers: [],
      logs: [],
      errors: [],
      data: {
        weather: null,
        isochrones: null,
        towers: null,
        signals: null,
        airQuality: null,
        route: null,
      },
    };
  }

  /**
   * Parse and execute a script
   */
  async execute(scriptContent) {
    this.state.logs = [];
    this.state.errors = [];
    this.state.currentScript = scriptContent;

    try {
      const lines = scriptContent
        .split("\n")
        .filter((l) => l.trim() && !l.trim().startsWith("#"));

      for (const line of lines) {
        await this.executeLine(line.trim());
      }

      this.log("✓ Script execution completed");
    } catch (error) {
      this.error(`Execution error: ${error.message}`);
    }

    return {
      success: this.state.errors.length === 0,
      state: this.state,
      logs: this.state.logs,
      errors: this.state.errors,
    };
  }

  /**
   * Execute a single command line
   */
  async executeLine(line) {
    const tokens = line.split(/\s+/);
    const command = tokens[0]?.toLowerCase();

    switch (command) {
      // Script 1: Clocks
      case "airport":
        await this.handleAirport(tokens);
        break;
      case "sync":
        await this.handleSync(tokens);
        break;
      case "precision":
        await this.handlePrecision(tokens);
        break;

      // Script 2: Isochrones
      case "position":
        this.handlePosition(tokens);
        break;
      case "isochrone":
        await this.handleIsochrone(tokens);
        break;

      // Script 3: Weather
      case "weather":
        await this.handleWeather(tokens);
        break;

      // Script 6: Signals
      case "signals":
        await this.handleSignals(tokens);
        break;

      // Script 8: Routing
      case "algorithm":
        this.handleAlgorithm(tokens);
        break;
      case "load":
        await this.handleLoad(tokens);
        break;
      case "fwdc":
        await this.handleFWDC(tokens);
        break;

      // Display commands
      case "show":
        this.handleShow(tokens);
        break;
      case "report":
        this.handleReport(tokens);
        break;

      default:
        this.log(`ℹ Parsed: ${line}`);
    }
  }

  /**
   * Handle AIRPORT command
   */
  async handleAirport(tokens) {
    const city = tokens[1];
    this.log(`🛬 Connecting to ${city} airport clock`);
    // In real implementation, would fetch airport coordinates
    const airports = {
      munich: { lat: 48.3538, lng: 11.7861 },
      nuremberg: { lat: 49.5016, lng: 11.0801 },
    };
    const airport = airports[city?.toLowerCase()] || airports.munich;
    this.state.position = airport;
  }

  /**
   * Handle SYNC command (time synchronization)
   */
  async handleSync(tokens) {
    const target = tokens.slice(2).join(" ");
    this.log(`⏰ Syncing to ${target}`);
    const now = new Date();
    this.log(`   Current time: ${now.toISOString()}`);
  }

  /**
   * Handle PRECISION command
   */
  async handlePrecision(tokens) {
    this.log("📍 Computing clock precision network...");
    // Generate synthetic precision points
    const points = [];
    for (let i = 0; i < 20; i++) {
      points.push({
        lat: this.state.position.lat + (Math.random() - 0.5) * 0.05,
        lng: this.state.position.lng + (Math.random() - 0.5) * 0.05,
        precision: 50 + Math.random() * 100, // milliseconds
      });
    }
    this.state.data.precisionPoints = points;
    this.log(`✓ Generated ${points.length} precision points`);
  }

  /**
   * Handle POSITION command
   */
  handlePosition(tokens) {
    // Parse: position from lat=48.1351 lng=11.5656
    const latToken = tokens.find((t) => t.startsWith("lat="));
    const lngToken = tokens.find((t) => t.startsWith("lng="));

    if (latToken && lngToken) {
      const lat = parseFloat(latToken.split("=")[1]);
      const lng = parseFloat(lngToken.split("=")[1]);
      this.state.position = { lat, lng };
      this.log(`📌 Position set to ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
    }
  }

  /**
   * Handle ISOCHRONE command
   */
  async handleIsochrone(tokens) {
    const mode = tokens[1]; // walking, cycling, driving
    this.log(`🗺️  Computing isochrone for ${mode}...`);

    try {
      const result = await getIsochrone(
        this.state.position.lng,
        this.state.position.lat,
        [5, 10, 15],
        mode
      );
      this.state.data.isochrones = result;
      this.log(`✓ Isochrone computed (${result.features?.length || 0} rings)`);
    } catch (error) {
      this.error(`Failed to compute isochrone: ${error.message}`);
    }
  }

  /**
   * Handle WEATHER command
   */
  async handleWeather(tokens) {
    const action = tokens[1];

    if (action === "load") {
      this.log("🌤️  Loading weather data...");
      try {
        const weather = await getCurrentWeather(
          this.state.position.lat,
          this.state.position.lng
        );
        const aq = await getAirQuality(
          this.state.position.lat,
          this.state.position.lng
        );
        this.state.data.weather = weather;
        this.state.data.airQuality = aq;
        this.log(`✓ Weather: ${weather.main.temp}°C, ${weather.weather[0].main}`);
        this.log(`✓ Air Quality Index: ${aq.list[0]?.main?.aqi || "N/A"}`);
      } catch (error) {
        this.error(`Failed to load weather: ${error.message}`);
      }
    }
  }

  /**
   * Handle SIGNALS command
   */
  async handleSignals(tokens) {
    const action = tokens[1];

    if (action === "load") {
      this.log("🚦 Loading traffic signals...");
      try {
        // Compute bounding box around position (2km)
        const bbox = [
          this.state.position.lat - 0.02,
          this.state.position.lng - 0.02,
          this.state.position.lat + 0.02,
          this.state.position.lng + 0.02,
        ];
        const signals = await getTrafficSignals(bbox);
        this.state.data.signals = signals;
        this.log(`✓ Found ${signals.elements?.length || 0} traffic signals`);
      } catch (error) {
        this.log("ℹ Signal data unavailable (Overpass API may be busy)");
      }
    }
  }

  /**
   * Handle ALGORITHM configuration
   */
  handleAlgorithm(tokens) {
    const algo = tokens[1];
    this.log(`🔧 Configuring ${algo} algorithm`);
    this.state.algorithm = algo;
  }

  /**
   * Handle LOAD command (for multi-script loading)
   */
  async handleLoad(tokens) {
    const scriptNum = tokens[2]?.match(/\d+/)?.[0];
    if (scriptNum) {
      this.log(`📜 Loading Script ${scriptNum}...`);
      // Would load and execute script-${scriptNum}-*.cynes
    }
  }

  /**
   * Handle FWDC command
   */
  async handleFWDC(tokens) {
    const action = tokens[1];

    if (action === "compute") {
      this.log("🚁 Computing FWDC optimal path...");
      // Simulate FWDC algorithm
      const path = [
        [this.state.position.lng, this.state.position.lat],
        [this.state.position.lng + 0.001, this.state.position.lat + 0.001],
        [this.state.position.lng + 0.003, this.state.position.lat + 0.002],
      ];
      this.state.data.route = { path };
      this.log(`✓ Route computed (${path.length} waypoints)`);
    }
  }

  /**
   * Handle SHOW command
   */
  handleShow(tokens) {
    const target = tokens.slice(1).join(" ");
    this.log(`👁️  Showing: ${target}`);
  }

  /**
   * Handle REPORT command
   */
  handleReport(tokens) {
    const reportType = tokens.slice(1).join(" ");
    this.log(`📊 Generating report: ${reportType}`);
  }

  /**
   * Utility: log a message
   */
  log(message) {
    this.state.logs.push(message);
    console.log(message);
  }

  /**
   * Utility: log an error
   */
  error(message) {
    this.state.errors.push(message);
    this.log(`❌ ${message}`);
  }
}

export default ScriptExecutor;
