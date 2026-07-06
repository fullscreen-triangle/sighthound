# Integration Checklist: Scripts 1-8 → Cynegeticus Sandbox

**What's been built**: Complete Mapbox GL + Deck.gl infrastructure with **cumulative data pipeline** where each script enriches state for the next, culminating in Script 8 (FWDC) integrating all layers.

**Architecture**: Script 8 is NOT a standalone script—it's an **integration point** that uses all data from Scripts 1-7 for final FWDC routing.

**What's needed**: Wire it into the existing cynegeticus.js page.

---

## Pre-Integration Checks

- [x] All API wrappers created (`src/api/`)
- [x] All Deck.gl layer builders created (`src/layers/`)
- [x] SandboxMap component created (`src/components/SandboxMap.jsx`)
- [x] ScriptExecutor built (`src/utils/scriptExecutor.js`)
- [x] All 8 .cynes scripts written (`public/scripts/`)
- [x] package.json updated with dependencies
- [x] cynegeticus.js updated with script content in file tree
- [ ] npm install (dependencies need installation)
- [ ] Import statements added to cynegeticus.js
- [ ] Map component swapped from Leaflet to SandboxMap
- [ ] Script executor wired to handleCompile()
- [ ] Layer state management connected

---

## Step-by-Step Integration

### 1. Install Dependencies
```bash
cd moriarty
npm install
# This will add: mapbox-gl, @deck.gl/*, turf
# Takes ~3-5 minutes
```

### 2. Update Imports in cynegeticus.js
Add these at the top of the file:

```javascript
import ScriptExecutor from "@/utils/scriptExecutor";
import SandboxMap from "@/components/SandboxMap";
import { createIsochroneLayer } from "@/layers/IsochroneLayer";
import { createHeatmapLayer } from "@/layers/HeatmapLayer";
import { createWindVectorLayer } from "@/layers/WeatherLayer";
import { createFWDCRoutingLayers } from "@/layers/RoutingLayer";
```

### 3. Add Script Executor Initialization
In the component function (after `const [selectedFile, ...]`):

```javascript
const scriptExecutor = useMemo(() => new ScriptExecutor(), []);
const [mapLayers, setMapLayers] = useState([]);
```

### 4. Update handleCompile Function
Replace the existing `executeCynegeticus` call with:

```javascript
const handleCompile = useCallback(async () => {
  const result = await scriptExecutor.execute(code);
  setResults({
    success: result.success,
    position: result.state.position,
    logs: result.logs,
    errors: result.errors,
  });

  // Build Deck.gl layers from script executor state
  const newLayers = [];

  // Add isochrone layer if computed
  if (result.state.data.isochrones) {
    newLayers.push(createIsochroneLayer(result.state.data.isochrones));
  }

  // Add weather heatmap if loaded
  if (result.state.data.weather) {
    const weatherPoints = [
      {
        lat: result.state.position.lat,
        lng: result.state.position.lng,
        value: result.state.data.weather.main.temp / 50, // Normalize
      },
    ];
    newLayers.push(createHeatmapLayer(weatherPoints));
  }

  // Add wind vectors if weather data
  if (result.state.data.weather?.wind) {
    const windPoints = [
      {
        lat: result.state.position.lat,
        lng: result.state.position.lng,
        wind_speed: result.state.data.weather.wind.speed,
        wind_direction: result.state.data.weather.wind.deg || 0,
      },
    ];
    newLayers.push(createWindVectorLayer(windPoints));
  }

  // Add FWDC routing if computed
  if (result.state.data.route) {
    const routingLayers = createFWDCRoutingLayers({
      path: result.state.data.route.path,
      signals: [],
    });
    newLayers.push(...routingLayers);
  }

  setMapLayers(newLayers);
}, [code]);
```

### 5. Replace Map Component in Output Section
Find this section (around line 731):

```javascript
{activeTab === "map" && (
  <div className="h-full w-full">
    {results.position ? (
      <MapComponent position={results.position} satellites={results.satellites} provider={mapProvider} />
    ) : (
      <div className="text-gray-500 flex items-center justify-center h-full">
        // Compile to view position on map
      </div>
    )}
  </div>
)}
```

Replace with:

```javascript
{activeTab === "map" && (
  <div className="h-full w-full">
    {mapLoaded ? (
      <SandboxMap
        layers={mapLayers}
        initialViewState={{
          longitude: results.position?.lon || 11.5656,
          latitude: results.position?.lat || 48.1351,
          zoom: 11,
          pitch: 0,
          bearing: 0,
        }}
      />
    ) : (
      <div className="text-gray-500 flex items-center justify-center h-full">
        // Compile a script to load map
      </div>
    )}
  </div>
)}
```

### 6. Add State for Map
Add near the other state declarations:

```javascript
const [mapLoaded, setMapLoaded] = useState(false);

// Update when map renders:
const handleCompile = useCallback(async () => {
  // ... existing code ...
  setMapLoaded(mapLayers.length > 0);
}, [code, mapLayers.length]);
```

---

## Testing Checklist

After integration, test each script:

### Script 1: Precision Clocks
- [ ] Execute script
- [ ] Verify logs show: "🛬 Connecting to Munich airport clock"
- [ ] Verify current time displayed
- [ ] Verify precision points generated in state

### Script 2: Isochrones (Primary)
- [ ] Execute script
- [ ] Verify logs show: "✓ Isochrone computed"
- [ ] Verify colored rings appear on map (green/blue/orange)
- [ ] Check zoom to Munich coordinates
- [ ] Verify GeoJSON features loaded correctly

### Script 3: Weather
- [ ] Execute script
- [ ] Verify logs show current temperature
- [ ] Verify air quality index loaded
- [ ] Check wind vectors visible on map
- [ ] Verify heatmap renders

### Script 6: Signals & Air Quality
- [ ] Execute script
- [ ] Verify logs show signal count (may show "unavailable" if Overpass busy)
- [ ] Verify air quality heatmap renders
- [ ] Check signal colors (red/yellow/green)

### Script 8: Full Routing (Integration Test)
- [ ] Execute script
- [ ] Verify all prior scripts' data loaded
- [ ] Verify path computed (green line)
- [ ] Verify multiple layers render together
- [ ] Check separation cost visualization (if implemented)

---

## API Call Trace

Watch browser console network tab to verify:

| Script | API Call | Expected | Status |
|--------|----------|----------|--------|
| 1 | (none) | Time synced | ✅ Local |
| 2 | POST Mapbox Isochrone | 3 GeoJSON rings | ✅ API call |
| 3 | GET OpenWeatherMap | Temp, wind, AQI | ✅ API call |
| 4 | POST Overpass | Cell towers (may timeout) | ⚠️ Rate-limited |
| 6 | POST Overpass + OpenWeatherMap | Signals + AQI | ⚠️ Rate-limited |
| 8 | Combined | All layers + route | ✅ Full integration |

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "mapbox-gl not found" | npm install not run | Run `npm install` |
| "SandboxMap undefined" | Import path wrong | Check import in cynegeticus.js |
| Blank map | No layers rendered | Check console for API errors |
| "Overpass busy" | Rate limit hit | Retry after 10 minutes |
| Isochrone fails | Mapbox API error | Verify NEXT_PUBLIC_MAPBOX_TOKEN in .env |
| Weather data null | OpenWeatherMap error | Verify OPENWEATHERMAP_API_KEY |

---

## After Integration Success

Once all scripts work:

1. **Script 4 & 5** (Towers, Devices):
   - Wire to OSM Overpass API (towers already in osm.js)
   - Add ScatterplotLayer for tower locations
   - Generate synthetic device density heatmap

2. **Script 7** (Satellites):
   - Integrate satellite.js for pass predictions
   - Create 3D visualization (may use Cesium for this)
   - Display DOP heatmap

3. **Script 8 Enhancements**:
   - Implement actual FWDC algorithm
   - Visualize fuzzy intervals and separation costs
   - Show catalyst refinement in real-time
   - Display closure detection status

---

## Files Modified/Created

```
Created:
✅ src/api/mapbox.js
✅ src/api/openweather.js
✅ src/api/tomtom.js
✅ src/api/osm.js
✅ src/layers/IsochroneLayer.js
✅ src/layers/HeatmapLayer.js
✅ src/layers/WeatherLayer.js
✅ src/layers/RoutingLayer.js
✅ src/components/SandboxMap.jsx
✅ src/utils/scriptExecutor.js
✅ public/scripts/script-*.cynes (8 files)
✅ SANDBOX_SCRIPTS_SETUP.md (this file)

Modified:
✅ package.json (added dependencies)
✅ src/pages/cynegeticus.js (added script content to file tree)

Need modification:
⏳ src/pages/cynegeticus.js (imports, component swap, executor wiring)
```

---

## Estimated Integration Time

- **Install dependencies**: 5 minutes
- **Add imports**: 2 minutes
- **Update handleCompile**: 10 minutes
- **Replace map component**: 5 minutes
- **Test all scripts**: 10-15 minutes
- **Debug/troubleshoot**: 10 minutes (if needed)

**Total**: ~45 minutes to full integration + testing

---

**Next Action**: Follow steps 1-6 above in order. Each step builds on the prior one.
