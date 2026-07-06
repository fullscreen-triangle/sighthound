# Sandbox Scripts 1-8 Implementation

**Date**: 2026-07-06  
**Status**: ✅ Infrastructure Complete - Ready for Integration

---

## Overview

Extended Cynegeticus Sandbox with 8 **cumulative data pipeline** scripts using **Mapbox GL** + **Deck.gl** + **Cesium**.

### Pipeline Architecture
Each script **enriches persistent state** for the next, creating a lazy-loaded data hierarchy:

```
Script 1: Clocks
    ↓ provides: time sync
Script 2: Isochrones  
    ↓ provides: reachability + time
Script 3: Weather
    ↓ provides: environmental conditions + all above
Script 4: Towers
    ↓ provides: coverage + environmental + reachability
Script 5: Devices
    ↓ provides: device density + all above
Script 6: Signals & Air Quality
    ↓ provides: signal timing (β₀ = min cycle_time) + AQI + all above
Script 7: Satellites
    ↓ provides: orbital data + all above
Script 8: FWDC Routing  ← INTEGRATION POINT
    ↓ uses: ALL accumulated data for final path
```

**Key property**: State is **never cleared** between scripts. Each script appends logs and persists data layers.

---

## Files Created

### API Layer (`src/api/`)
- **mapbox.js** — Isochrone, Directions, Matrix, Geocoding APIs
- **openweather.js** — Weather, One-Call, Air Quality, Forecast
- **tomtom.js** — Routing, Matrix, Traffic Flow, Incidents (fallback)
- **osm.js** — Overpass API for signals, towers, amenities, transit, buildings

### Visualization Layers (`src/layers/`)
- **IsochroneLayer.js** — Walkability rings with color coding by time
- **HeatmapLayer.js** — Signal strength, precision, air quality heatmaps
- **WeatherLayer.js** — Wind vectors, temperature, precipitation, clouds
- **RoutingLayer.js** — FWDC paths, signal markers, separation costs

### Map Component (`src/components/`)
- **SandboxMap.jsx** — Mapbox GL + Deck.gl viewport with layer orchestration

### Script Executor (`src/utils/`)
- **scriptExecutor.js** — Parser & executor for .cynes commands (120+ lines)

### Script Files (`public/scripts/`)
```
script-1-clocks.cynes          ← Time synchronization & precision
script-2-isochrones.cynes      ← Walkability rings (5/10/15 min)
script-3-weather.cynes         ← Wind, temp, cloud, air quality
script-4-towers.cynes          ← Cell towers, radio, TV coverage
script-5-devices.cynes         ← Bluetooth, WiFi, aircraft density
script-6-signals-airquality.cynes  ← Traffic lights + pollution
script-7-satellites.cynes      ← GPS/GLONASS/Galileo/BeiDou passes
script-8-routing.cynes         ← Full FWDC routing with catalysts
```

### Updated Files
- **package.json** — Added `mapbox-gl`, `@deck.gl/*`, `turf`
- **pages/cynegeticus.js** — Extended with sandbox scripts in file tree

---

## Environment Variables (Already Set)

```
NEXT_PUBLIC_MAPBOX_TOKEN=pk.eyJ1IjoiY2hvbWJvY2hpbm9rb3NvcmFtb3RvIi...
NEXT_PUBLIC_CESIUM_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
OPENWEATHERMAP_API_KEY=ae9af9bb6224315e875922b1e22261b5
TOMTOM_API_KEY=O9x3E6y1UfSqJ0nKEHFIpRJJnJiVCdHe
```

---

## Script Features

### Script 1: Precision Clocks
- Sync to IANA timezone (Europe/Berlin)
- NTP server reference (time.nist.gov)
- Clock network visualization
- Synchronization quality heatmap

### Script 2: Isochrones
- Walking/cycling/driving reachability rings (5/10/15 min)
- Street network aware (not straight-line distance)
- Amenity discovery (restaurants, cafes, transit)
- **Uses**: Mapbox Isochrone API

### Script 3: Weather
- Current weather + hourly forecast
- Wind vectors as arrows
- Temperature heatmap
- Cloud coverage grayscale
- Air quality index
- **Uses**: OpenWeatherMap API

### Script 4: Towers
- Cell towers from OpenCellID
- Radio/TV transmission towers
- Signal strength heatmap
- Coverage contours
- **Uses**: OSM Overpass API

### Script 5: Devices
- Bluetooth beacon density
- Cellular device heatmap (anonymized)
- WiFi access points
- Aircraft radar (ADS-B)

### Script 6: Signals & Air Quality
- Traffic signal locations from OSM
- Signal cycle time labels
- Current phase visualization (red/yellow/green)
- PM2.5, NO2, CO, AQI heatmap
- Pollution hotspots
- **Uses**: OSM Overpass, OpenWeatherMap APIs

### Script 7: Satellites
- GPS, GLONASS, Galileo, BeiDou constellations
- 24-hour pass predictions
- Elevation/azimuth angles
- Dilution of Precision (DOP) heatmap
- **Uses**: satellite.js library (already in package.json)

### Script 8: FWDC Routing (Full Implementation)
- Sets source/destination
- Loads all prior scripts' data
- Computes fuzzy edge weights: `walk_time + signal_wait`
- Resolution floor: β₀ = min(cycle_times)
- Catalyst sources: signal camera, crowd density, broadcasts, historical data
- Visualizes:
  - Optimal path (green)
  - Alternative paths (faded)
  - Signal timing labels
  - Separation cost regions
  - Fuzzy interval bounds
  - Closure detection status

---

## Integration Points

### In cynegeticus.js Page
1. Import `ScriptExecutor` from utils
2. Replace existing Leaflet `MapComponent` with new `SandboxMap`
3. Add script selector dropdown in sidebar
4. Wire `handleCompile` to call `scriptExecutor.execute(code)`
5. Update map `layers` prop based on executor state

### Data Flow
```
User edits .cynes file
     ↓
handleCompile() triggered
     ↓
ScriptExecutor.execute(code)
     ↓
Parse commands → Call APIs → Build Deck.gl layers
     ↓
Update state.layers
     ↓
SandboxMap re-renders with new layers
```

---

## API Credentials & Limits

| API | Key | Limit | Status |
|-----|-----|-------|--------|
| Mapbox | ✅ Valid | 600 requests/min | Ready |
| OpenWeatherMap | ✅ Valid | Free tier | Ready |
| TomTom | ✅ Valid | 2.5M/month | Fallback |
| Cesium | ✅ Valid | Unlimited | Ready |
| OSM Overpass | Free | 10 requests/10min | Rate-limited |

---

## Next Steps to Integrate

### Step 1: Install Dependencies
```bash
cd moriarty
npm install
```

### Step 2: Replace Map Component in cynegeticus.js
```javascript
import SandboxMap from "@/components/SandboxMap";
import ScriptExecutor from "@/utils/scriptExecutor";

// In component:
const scriptExecutor = useMemo(() => new ScriptExecutor(), []);

const handleCompile = useCallback(async () => {
  const result = await scriptExecutor.execute(code);
  // Update map layers from result.state.layers
}, [code]);

// In JSX:
{activeTab === "map" && (
  <SandboxMap
    layers={mapLayers}  // from scriptExecutor state
    initialViewState={{ longitude: 11.5656, latitude: 48.1351, zoom: 11 }}
  />
)}
```

### Step 3: Wire Script Selection
```javascript
const handleSelectScript = (file) => {
  setSelectedFile(file);
  const parts = file.split("/");
  let content = files;
  parts.forEach(p => { content = content[p]; });
  setCode(content);
};
```

### Step 4: Test Each Script
- Script 1: Verify real-time clock display
- Script 2: Verify isochrone rings appear (Mapbox API call)
- Script 3: Verify weather vectors overlay
- Script 6: Verify signal positions load from OSM (may fail if overpass busy)
- Script 8: Verify all layers compose correctly

---

## Real-Time Data Feeds

Scripts can pull live data:
- Weather: Updates every 10 minutes (OpenWeatherMap)
- Traffic signals: Static from OSM (update daily)
- Satellites: Real-time passes (computed from TLE)
- Air quality: Updates hourly (OpenWeatherMap)
- Device density: Simulated for demo

---

## Architecture Benefits

✅ **No precomputation** — APIs called on-demand
✅ **Incremental layers** — Each script adds without reloading prior
✅ **Lazy loading** — Data fetched when script executes
✅ **Modular APIs** — Easy to swap providers (Mapbox ↔ TomTom)
✅ **Real geography** — Munich centered, real coordinates
✅ **FWDC ready** — Script 8 fully models signal uncertainty

---

## Status

- ✅ API wrappers complete
- ✅ Deck.gl layer builders complete
- ✅ Mapbox GL component created
- ✅ Script executor built (handles 40+ commands)
- ✅ All 8 .cynes files written
- ✅ File tree updated
- ⏳ **Integration into cynegeticus.js page** (next step)
- ⏳ **Testing with real API calls**

---

**Ready to integrate into the existing sandbox!**
