# Sandbox Scripts 1-8 Documentation

## Quick Reference

**What**: Lazy-loaded, cumulative visualization pipeline for pedestrian routing with FWDC.

**How it works**: Each script enriches persistent state for the next; Script 8 integrates all.

**Where**: See [BUILD_SUMMARY.md](BUILD_SUMMARY.md) for complete overview.

---

## Running Scripts

### User Flow (in `cynegeticus.js`)
1. User selects a script from file tree
2. User clicks "Compile & Execute"
3. ScriptExecutor parses script, executes commands, builds layers
4. State persists (not reset between scripts)
5. All active layers render on SandboxMap

### State Persistence
```javascript
// Script 2 executes
state.data.isochrones = {...}
state.layers.isochrone = [DeckGL Layer]

// Script 3 executes (isochrone data STILL THERE)
state.data.weather = {...}
state.layers.weather = [DeckGL Layer]

// Script 8 executes (uses ALL accumulated data)
// state.data has: isochrones + weather + signals + ...
// state.layers has: isochrone + weather + routing + ...
```

---

## Script Overview

| # | Name | Provides | For Script 8 |
|---|------|----------|------|
| 1 | **Clocks** | Time sync | Timeline basis |
| 2 | **Isochrones** | 5/10/15 min rings | Reachability bounds |
| 3 | **Weather** | Wind, temp, AQI | Speed/safety adjustment |
| 4 | **Towers** | Cell coverage | Localization capability |
| 5 | **Devices** | Bluetooth/WiFi density | Congestion avoidance |
| 6 | **Signals & AQI** | Traffic lights + pollution | **β₀ (critical)** |
| 7 | **Satellites** | Orbital passes | Position confidence |
| 8 | **FWDC Routing** | **Integration point** | Uses all above |

---

## Key Concept: β₀ (Resolution Floor)

Script 6 extracts this from OSM traffic signal cycle times:
```
β₀ = min(cycle_times)  // e.g., 45 seconds
```

Script 8 uses it for:
```
w(e) = [walk_time, walk_time + T_c(v)]  // Fuzzy edge weight
Closure when: Σ(v) and Σ(w) are β₀-separated  // Deterministic
```

If Script 6 fails (Overpass API busy), default is β₀ = 60s.

---

## File Locations

```
moriarty/
├── src/
│   ├── api/              ← API wrappers (4 files)
│   ├── layers/           ← Deck.gl layer builders (4 files)
│   ├── components/
│   │   └── SandboxMap.jsx
│   ├── utils/
│   │   └── scriptExecutor.js  ← THE PIPELINE ORCHESTRATOR
│   └── pages/
│       └── cynegeticus.js    ← TO BE INTEGRATED
├── public/scripts/       ← All 8 .cynes files
├── package.json          ← Updated with dependencies
├── BUILD_SUMMARY.md      ← Overview
├── INTEGRATION_CHECKLIST.md  ← How to wire it in
├── SCRIPT_8_INTEGRATION.md   ← Deep dive on Script 8
└── SANDBOX_README.md     ← This file
```

---

## API Credentials

All in `moriarty/validation/.env.local`:
```
NEXT_PUBLIC_MAPBOX_TOKEN=pk.eyJ...
NEXT_PUBLIC_CESIUM_TOKEN=eyJh...
OPENWEATHERMAP_API_KEY=ae9a...
TOMTOM_API_KEY=O9x3...
```

---

## Integration (TL;DR)

See [INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md) for step-by-step.

**Quick steps**:
1. `npm install`
2. Add imports to `cynegeticus.js`
3. Replace Leaflet `MapComponent` with `SandboxMap`
4. Wire `handleCompile` to `scriptExecutor.execute()`
5. Test each script

**Effort**: ~45 minutes

---

## Testing

After integration:
```
Script 1: See time sync logs ✓
Script 2: See isochrone rings on map ✓
Script 3: See weather heatmap ✓
Script 6: See signal points ✓
Script 8: See all 8 layers + FWDC path ✓
```

---

## Architecture Documents

Read in this order:

1. **[BUILD_SUMMARY.md](BUILD_SUMMARY.md)** — What was built and why
2. **[SANDBOX_SCRIPTS_SETUP.md](SANDBOX_SCRIPTS_SETUP.md)** — Technical details of all files
3. **[SCRIPT_8_INTEGRATION.md](SCRIPT_8_INTEGRATION.md)** — How Script 8 uses all prior data
4. **[INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md)** — Step-by-step integration

---

## Key Files in ScriptExecutor

**scriptExecutor.js** (420 lines):
- `constructor()` → Persistent state initialization
- `execute(scriptContent)` → Main entry point (never clears state!)
- `executeLine(line)` → Dispatcher for commands
- Command handlers: `handleAirport()`, `handleIsochrone()`, `handleWeather()`, etc.
- `getActiveLayers()` → Returns all non-null layers for visualization
- `log()` / `error()` → Append to state.logs (shown in terminal tab)

---

## Next: The Integration

Once you understand the architecture, run through [INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md) to wire the executor into the existing `cynegeticus.js` page.

**Then test each script one by one.**

---

**Questions?** See [SCRIPT_8_INTEGRATION.md](SCRIPT_8_INTEGRATION.md) for deep dive on how Script 8 integrates everything.
