# Script 8: FWDC Integration Point

## Overview

**Script 8 is NOT a standalone visualization.** It's the **integration point** that leverages ALL accumulated data from Scripts 1-7 to compute and visualize the final FWDC pedestrian routing with fuzzy-weighted signal timing uncertainty.

---

## Data Flow: Scripts 1-7 → Script 8

### What Script 1 Provides (Clocks)
```
state.data.clocks = {
  timezone: "Europe/Berlin",
  ntp_server: "time.nist.gov",
  precision_points: [{lat, lng, precision_ms}, ...]
}
```
**Script 8 uses**: Time synchronization basis for routing timeline

---

### What Script 2 Provides (Isochrones)
```
state.data.isochrones = {
  walking: GeoJSON,  // 5/10/15 min rings
  cycling: GeoJSON,
  driving: GeoJSON
}
state.layers.isochrone = [DeckGL Layer]  // Visualized
```
**Script 8 uses**: Reachability constraints, feasible destination set

---

### What Script 3 Provides (Weather)
```
state.data.weather = {
  temp: 22,
  wind: {speed: 5, deg: 180},
  clouds: 30,
  visibility: 10000
}
state.data.airQuality = {
  main: {aqi: 2},
  components: {pm2_5: 15, no2: 45, ...}
}
state.layers.weather = [DeckGL Layer]  // Visualized
```
**Script 8 uses**: 
- Wind speed affects walking speed (5 m/s wind = -0.5 m/s penalty)
- Air quality affects route desirability (prefer low-AQI paths)
- Visibility affects pedestrian safety scoring

---

### What Script 4 Provides (Towers)
```
state.data.towers = {
  elements: [
    {lat, lng, type: "cell_tower", signal_strength: 0.8},
    {lat, lng, type: "radio_tower"},
    ...
  ]
}
state.layers.towers = [DeckGL Heatmap Layer]
```
**Script 8 uses**: 
- Coverage zones affect device-based refinement
- Signal strength influences position certainty
- Enables triangulation for trajectory prediction

---

### What Script 5 Provides (Devices)
```
state.data.devices = {
  bluetooth: {density: 0.45, points: [{lat, lng, rssi}, ...]},
  cellular: {density: 0.72},
  wifi: {density: 0.85},
  aircraft: {flights: [{lat, lng, altitude, heading}, ...]}
}
state.layers.devices = [DeckGL Heatmap Layer]
```
**Script 8 uses**: 
- Device density heatmap for congestion avoidance
- Bluetooth beacon proximity for microlocalization
- Crowd detection affects pedestrian safety

---

### What Script 6 Provides (Signals & Air Quality)
```
state.data.signals = {
  elements: [
    {
      lat, lng,
      tags: {
        "traffic_signals:cycle_time": "60",
        "traffic_signals:phases": "red:30s|green:20s|yellow:10s"
      }
    },
    ...
  ]
}
state.data.minCycleTime = 45  // ← CRITICAL FOR FWDC
state.data.allCycleTimes = [45, 60, 90, ...]
state.layers.signals = [DeckGL Points Layer]
```

**Script 8 uses** (Core FWDC):
- **Resolution floor β₀ = min(cycle_times)** — can't distinguish costs < 45s
- Each edge gets fuzzy weight: `w(e) = [walk_time, walk_time + T_c(v)]`
- Signal cycle uncertainty is **primary source** of fuzzy intervals
- Determines when deterministic closure occurs (separation > β₀)

---

### What Script 7 Provides (Satellites)
```
state.data.satellites = {
  constellations: {
    GPS: {visible: 12, elevation: [45, 38, 22, ...]},
    GLONASS: {visible: 8},
    Galileo: {visible: 7},
    BeiDou: {visible: 5}
  },
  dop: {pdop: 1.8, gdop: 2.5},  // Positional DOP
  passes: [{sat_id, rise_time, set_time, max_elevation}, ...]
}
state.layers.satellites = [DeckGL Particles Layer]
```
**Script 8 uses**: 
- DOP for position uncertainty bound
- Pass predictions for timing constraints
- Visible satellite count for accuracy confidence
- Can refine route confidence intervals based on satellite geometry

---

## Script 8: FWDC Integration Logic

### Phase 1: Configuration
```javascript
// All prior data is ready
const β₀ = state.data.minCycleTime;  // From Script 6
const weather = state.data.weather;   // From Script 3
const signals = state.data.signals;   // From Script 6
const airQuality = state.data.airQuality;  // From Script 6
const devices = state.data.devices;   // From Script 5
const towers = state.data.towers;     // From Script 4
const isochrones = state.data.isochrones;  // From Script 2
const satellites = state.data.satellites;  // From Script 7
const clocks = state.data.clocks;     // From Script 1
```

### Phase 2: Fuzzy Edge Weights
```javascript
// For each edge in street network:
for each edge (u, v):
  walk_time = distance(u, v) / walking_speed
  
  // Account for weather
  if (hasWind):
    walking_speed -= wind_effect
  
  // Account for signal timing at v
  if (hasSignal at v):
    T_c = signal_cycle_time(v)
    w(u,v) = [walk_time, walk_time + T_c]
  else:
    w(u,v) = [walk_time, walk_time + β₀]  // Default uncertainty
```

### Phase 3: Separation Costs
```javascript
// For each candidate node v:
σ(v) = min_cost(path avoiding v)

// Compute fuzzy region
Σ(v) = [σ_min(v), σ_max(v)]

// Check if separated from other candidates
if (Σ(v) and Σ(w) are β₀-separated):
  // v is ruled out — no further measurement changes decision
  ruled_out.push(v)
else:
  // Still overlapping — may need catalyst refinement
  uncertain.push(v)
```

### Phase 4: Catalyst Integration
```javascript
// Real-time data refines fuzzy intervals:

if (catalyst_signal_camera observes green at signal v):
  // Refine wait time interval
  T_c → [actual_wait, actual_wait + phase_variance]

if (catalyst_crowd_density high):
  // Increase traversal time uncertainty
  walk_time += crowd_delay

if (catalyst_real_time_broadcast available):
  // Eliminate signal uncertainty
  T_c → 0  (exact phase known)

if (catalyst_historical_pattern suggests shorter route):
  // Expand search neighborhood
  // (rarely needed if all layers loaded)
```

### Phase 5: Visualization Layers
```javascript
// All these layers render simultaneously:

// From Script 2: Reachability
show isochrone_layer

// From Script 3: Weather impact
show wind_vector_layer  // Wind arrows overlay
show weather_heatmap    // Temperature/cloud gradient

// From Script 4: Coverage
show tower_coverage_heatmap

// From Script 5: Device density
show device_density_heatmap

// From Script 6: Signals
show signal_locations   // Points with cycle times
show air_quality_heatmap

// From Script 7: Satellite geometry
show satellite_footprints

// From Script 8 (FWDC):
show optimal_path       // Green line (ruled-out nodes)
show alternative_paths  // Gray lines (uncertain)
show separation_regions // Fuzzy interval visualization
show closure_status     // When β₀-separation achieved
show catalyst_refinements  // Real-time updates
```

---

## Why This Architecture Matters

### No Redundant Computation
- Script 6 queries traffic signals **once**, not again in Script 8
- Weather fetched **once** in Script 3, reused through Script 8
- Data persists across all scripts

### Progressive Refinement
- Early scripts provide coarse data (5/10/15 min reachability)
- Later scripts add precision (signal timing, device density, weather)
- Script 8 integrates **all** sources with correct weighting

### Lazy Loading
- User only loads data **they actually need**
- Script 2 alone gives quick reachability visualization
- Full Script 8 routes only after all layers loaded
- Mobile users can stop at Script 3 (weather) without routing overhead

### FWDC Semantic Correctness
- β₀ is derived from actual signal data (Script 6)
- Not arbitrary—tied to **infrastructure uncertainty**
- Fuzzy intervals reflect **physical reality**, not measurement error
- Separation costs account for **all competing paths**

---

## Example Execution Flow

```
User clicks: "Load Script 8"

[Already loaded from prior scripts]
✓ Clocks: time sync ready
✓ Isochrones: 5/10/15 min rings computed
✓ Weather: 22°C, 5 m/s wind, AQI=2
✓ Towers: 34 cell towers nearby, coverage 98%
✓ Devices: 45% BT beacon density, 12 flights visible
✓ Signals: 156 traffic signals, min cycle = 45s, β₀ = 45s
✓ Satellites: 12 GPS + 8 GLONASS visible, PDOP=1.8

[Script 8 computes]
↓
Fuzzy edge weights: w(e) = [walk_time, walk_time + T_c]
Separation costs: σ(v) for each node
Closure detection: when regions separate > β₀

[Visualization]
↓
Map shows:
- Isochrone rings (green/blue/orange)
- Wind vectors (arrows)
- Signal timing labels
- Optimal path (thick green line)
- Alternative paths (gray)
- Separation cost regions (fuzzy bounds)
- Closure detection status (when β₀-separated)
- Catalyst refinement opportunities (real-time data feeds)

[User interaction]
↓
"Observe signal phase": Catalyst input → refine intervals
"Get crowd density": Catalyst input → adjust route safety
"Real-time broadcast": Catalyst input → eliminate uncertainty
```

---

## Integration with cynegeticus.js

In the map component:

```javascript
// User selects Script 8
const result = await scriptExecutor.execute(script8Code);

// result.state contains ALL accumulated data
const {
  isochrones,
  weather,
  towers,
  devices,
  signals,
  satellites,
  route,
  minCycleTime  // β₀
} = result.state.data;

// Render all layers simultaneously
<SandboxMap
  layers={[
    result.state.layers.isochrone,
    result.state.layers.weather,
    result.state.layers.towers,
    result.state.layers.devices,
    result.state.layers.signals,
    result.state.layers.satellites,
    result.state.layers.routing  // ← FWDC output
  ]}
  position={result.state.position}
/>

// Display FWDC metrics
<Panel>
  <h2>FWDC Routing Summary</h2>
  <p>Resolution floor β₀ = {minCycleTime}s</p>
  <p>Optimal path cost: {route.cost} seconds</p>
  <p>Gap from alternatives: {route.gap}s</p>
  <p>Separation achieved at node: {route.closure}</p>
  <CatalystPanel data={{signal, crowd, broadcast, historical}} />
</Panel>
```

---

## Summary

**Script 8 is the integration magnum opus**, not a add-on. It uses every piece of data:

- Scripts 1-7 = **ingredients**
- Script 8 = **the recipe**

Running Scripts 1-6 gives you **individual insights** (where can I walk? what's the weather?).

Running Script 8 gives you the **complete answer** to pedestrian routing under signal timing uncertainty, with all environmental factors integrated into one coherent routing decision.
