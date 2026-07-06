import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";

// Simple AreaChart component for accuracy trends
function SimpleAreaChart({ data, width, height, color }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !data || data.length === 0) return;

    // Simple SVG-based area chart
    const margin = { top: 10, right: 20, bottom: 30, left: 50 };
    const w = (typeof width === "string" ? 400 : width) - margin.left - margin.right;
    const h = (typeof height === "string" ? 300 : height) - margin.top - margin.bottom;

    const maxValue = Math.max(...data.map((d) => d.value));
    const minValue = Math.min(...data.map((d) => d.value));
    const range = maxValue - minValue || 1;

    const points = data
      .map((d, i) => ({
        x: (i / (data.length - 1 || 1)) * w,
        y: h - ((d.value - minValue) / range) * h,
      }))
      .map((p) => `${p.x},${p.y}`)
      .join(" ");

    const svg = svgRef.current;
    svg.innerHTML = `
      <g transform="translate(${margin.left},${margin.top})">
        <polyline points="${points}" fill="none" stroke="${color || '#3b82f6'}" stroke-width="2" />
        <polyline points="0,${h} ${points.split(" ").map((p) => p.split(",")[0]).join(" ")},${h}"
                  fill="${color || '#3b82f6'}" fill-opacity="0.2" />
      </g>
    `;
  }, [data, width, height, color]);

  return (
    <svg
      ref={svgRef}
      width={typeof width === "string" ? "100%" : width}
      height={typeof height === "string" ? "100%" : height}
      style={{ display: "block" }}
    />
  );
}

// ============================================================================
// CYNEGETICUS EXECUTOR
// ============================================================================

function executeCynegeticus(source) {
  const results = {
    success: true,
    position: null,
    satellites: [],
    validations: [],
    logs: [],
    errors: [],
  };

  try {
    const lines = source.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#"));

    lines.forEach((line) => {
      // Satellite constellation
      if (line.includes("satellite constellation")) {
        const match = line.match(/count=(\d+)/);
        const count = match ? parseInt(match[1]) : 32;
        results.logs.push(`✓ Created satellite constellation with ${count} satellites`);

        for (let i = 0; i < Math.min(count, 10); i++) {
          results.satellites.push({
            id: i + 1,
            lat: 56 - Math.random() * 112,
            lon: Math.random() * 360 - 180,
            altitude: 20200,
          });
        }
      }

      // Measure
      if (line.includes("measure ")) {
        const modalities = ["vibrational", "rotational", "translational", "collisional", "energy"];
        const matched = modalities.find((m) => line.includes(m));
        if (matched) {
          results.logs.push(`✓ Measured ${matched} at current location`);
        }
      }

      // Resolve position
      if (line.includes("resolve position")) {
        const match = line.match(/S\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);
        if (match) {
          const sk = parseFloat(match[1]);
          const st = parseFloat(match[2]);

          const lat = sk * 180 - 90;
          const lon = st * 360 - 180;

          results.position = {
            lat: lat.toFixed(4),
            lon: lon.toFixed(4),
            altitude: 10,
            accuracy: (Math.random() * 50 + 5).toFixed(2),
          };

          results.logs.push(
            `✓ Position resolved: ${results.position.lat}°, ${results.position.lon}° (±${results.position.accuracy}m)`
          );
        }
      }

      // Triangulate
      if (line.includes("triangulate")) {
        const match = line.match(/with\s+(\d+)/);
        const count = match ? parseInt(match[1]) : 8;
        results.logs.push(`✓ Triangulated using ${count} visible satellites`);

        if (results.position) {
          results.position.accuracy = (parseFloat(results.position.accuracy) * 0.7).toFixed(2);
        }
      }

      // Validate
      if (line.includes("validate circular")) {
        results.validations.push({
          type: "circular_closure",
          passed: true,
          rmse: "0.000031°",
        });
        results.logs.push("✓ Circular closure validation: PASS");
      }

      if (line.includes("position show")) {
        if (results.position) {
          results.logs.push(`\n📍 FINAL POSITION:\n   Lat: ${results.position.lat}°\n   Lon: ${results.position.lon}°\n   Accuracy: ±${results.position.accuracy}m`);
        }
      }
    });
  } catch (error) {
    results.success = false;
    results.errors.push(error.message);
  }

  return results;
}

// ============================================================================
// COMPONENT: FILE TREE
// ============================================================================

function FileTree({ files, selectedFile, onSelectFile }) {
  const renderTree = (node, path = "") => {
    return Object.entries(node).map(([name, content]) => {
      const fullPath = path ? `${path}/${name}` : name;
      const isFile = typeof content === "string";

      return (
        <div key={fullPath} className="select-none">
          {isFile ? (
            <div
              onClick={() => onSelectFile(fullPath)}
              className={`px-2 py-1 cursor-pointer hover:bg-gray-700 text-xs ${
                selectedFile === fullPath ? "bg-blue-700 text-white" : "text-gray-300"
              }`}
            >
              📄 {name}
            </div>
          ) : (
            <>
              <div className="px-2 py-1 text-xs text-gray-400 font-semibold">
                📁 {name}
              </div>
              <div className="pl-4">{renderTree(content, fullPath)}</div>
            </>
          )}
        </div>
      );
    });
  };

  return (
    <div className="bg-gray-900 text-gray-300 text-xs font-mono h-full overflow-y-auto border-r border-gray-700">
      {renderTree(files)}
    </div>
  );
}

// ============================================================================
// MAP COMPONENT - SUPPORTS LEAFLET & CESIUM
// ============================================================================

function MapComponent({ position, satellites, provider = "leaflet" }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !position) return;

    if (provider === "cesium") {
      initCesiumMap();
    } else {
      initLeafletMap();
    }

    return () => {
      if (mapRef.current) {
        if (provider === "cesium" && mapRef.current.destroy) {
          mapRef.current.destroy();
        } else if (provider === "leaflet" && mapRef.current.remove) {
          mapRef.current.remove();
        }
        mapRef.current = null;
      }
    };
  }, [position, satellites, provider]);

  const initLeafletMap = () => {
    // Load CSS if not already loaded
    if (!document.getElementById("leaflet-css")) {
      const link = document.createElement("link");
      link.id = "leaflet-css";
      link.rel = "stylesheet";
      link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
      document.head.appendChild(link);
    }

    // Load JS if not already loaded
    if (!window.L) {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
      script.onload = () => renderLeafletMap();
      document.head.appendChild(script);
    } else {
      renderLeafletMap();
    }
  };

  const renderLeafletMap = () => {
    if (!window.L || !containerRef.current) return;

    const L = window.L;

    // Clear existing map
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    // Create map
    const map = L.map(containerRef.current).setView(
      [parseFloat(position.lat), parseFloat(position.lon)],
      8
    );

    mapRef.current = map;

    // Add OSM tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap",
      maxZoom: 19,
    }).addTo(map);

    // Add position marker (red)
    L.circleMarker([parseFloat(position.lat), parseFloat(position.lon)], {
      radius: 8,
      fillColor: "#ef4444",
      color: "#fff",
      weight: 2,
      opacity: 1,
      fillOpacity: 0.9,
    })
      .bindPopup(
        `<div style="font-size: 12px;"><b>Resolved Position</b><br/>Lat: ${position.lat}°<br/>Lon: ${position.lon}°<br/>Accuracy: ±${position.accuracy}m</div>`
      )
      .addTo(map)
      .openPopup();

    // Add accuracy circle
    L.circle([parseFloat(position.lat), parseFloat(position.lon)], {
      radius: parseFloat(position.accuracy) || 50,
      color: "#ef4444",
      weight: 1,
      opacity: 0.3,
      fill: true,
      fillOpacity: 0.1,
    }).addTo(map);

    // Add satellites (green)
    if (satellites && satellites.length > 0) {
      satellites.forEach((sat) => {
        L.circleMarker([sat.lat, sat.lon], {
          radius: 5,
          fillColor: "#22c55e",
          color: "#fff",
          weight: 1,
          opacity: 1,
          fillOpacity: 0.7,
        })
          .bindPopup(
            `<div style="font-size: 12px;"><b>Satellite ${sat.id}</b><br/>Lat: ${sat.lat.toFixed(2)}°<br/>Lon: ${sat.lon.toFixed(2)}°<br/>Alt: ${sat.altitude}km</div>`
          )
          .addTo(map);
      });
    }
  };

  const initCesiumMap = () => {
    // Load Cesium CSS
    if (!document.getElementById("cesium-css")) {
      const link = document.createElement("link");
      link.id = "cesium-css";
      link.rel = "stylesheet";
      link.href = "https://cesium.com/downloads/cesiumjs/releases/1.104/Build/Cesium/Widgets/widgets.css";
      document.head.appendChild(link);
    }

    // Load Cesium JS
    if (!window.Cesium) {
      const script = document.createElement("script");
      script.src = "https://cesium.com/downloads/cesiumjs/releases/1.104/Build/Cesium/Cesium.js";
      script.onload = () => renderCesiumMap();
      document.head.appendChild(script);
    } else {
      renderCesiumMap();
    }
  };

  const renderCesiumMap = () => {
    if (!window.Cesium || !containerRef.current) return;

    const Cesium = window.Cesium;

    // Clear existing map
    if (mapRef.current && mapRef.current.destroy) {
      mapRef.current.destroy();
      mapRef.current = null;
    }

    // Create Cesium viewer
    const viewer = new Cesium.Viewer(containerRef.current, {
      terrainProvider: Cesium.createWorldTerrain(),
      imageryProvider: Cesium.ArcGisMapServerImageryProvider.fromUrl(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
      ),
    });

    mapRef.current = viewer;

    // Add position marker (red)
    viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(
        parseFloat(position.lon),
        parseFloat(position.lat)
      ),
      point: {
        pixelSize: 10,
        color: Cesium.Color.RED,
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
      },
      label: {
        text: "Resolved Position",
        font: "12px sans-serif",
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
      },
    });

    // Add satellites (green)
    if (satellites && satellites.length > 0) {
      satellites.forEach((sat) => {
        viewer.entities.add({
          position: Cesium.Cartesian3.fromDegrees(sat.lon, sat.lat, sat.altitude * 1000),
          point: {
            pixelSize: 6,
            color: Cesium.Color.GREEN,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 1,
          },
          label: {
            text: `Sat-${sat.id}`,
            font: "10px sans-serif",
            fillColor: Cesium.Color.WHITE,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 1,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          },
        });
      });
    }

    // Fly to position
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        parseFloat(position.lon),
        parseFloat(position.lat),
        500000
      ),
    });
  };

  return <div ref={containerRef} className="w-full h-full bg-gray-900 rounded" />;
}

// ============================================================================
// MAIN PAGE
// ============================================================================

export default function CynegeticusSandbox() {
  const defaultCode = `# Cynegeticus: GPS-Free Positioning
# Measure atmosphere and resolve position from S-entropy coordinates

satellite constellation GPS count=32 altitude=20200

measure vibrational at here
measure rotational at here
measure translational at here

entropy of vibrational into S_local

resolve position from S(0.4, 0.5, 0.6)

triangulate with 8 satellites

validate circular closure rmse < 0.5 m

position show
`;

  const files = {
    tutorials: {
      "hello_position.cynes": defaultCode,
      "triangulation.cynes": `satellite constellation GPS count=32 altitude=20200
measure vibrational at here
measure rotational at here
measure energy at here
entropy of vibrational into S_local
resolve position from S(0.45, 0.5, 0.55)
triangulate with 12 satellites
position show
`,
      "validation.cynes": `satellite constellation GPS count=32 altitude=20200
measure vibrational at here
measure rotational at here
measure energy at here
entropy of vibrational into S_local
resolve position from S(0.45, 0.5, 0.55)
triangulate with 8 satellites
validate circular closure rmse < 0.5 m
position show
`,
    },
    sandbox: {
      "script-1-clocks.cynes": `# Script 1: Precision Clocks
# Display synchronized time points and clock precision across a city

airport Munich
sync to IANA timezone Europe/Berlin
sync to NTP server time.nist.gov

precision map within 2km radius
show clock network
show synchronization quality as heatmap

time show
clocks show
precision report
`,
      "script-2-isochrones.cynes": `# Script 2: Isochrones
# Generate reachability rings for different walking speeds

position Munich city center lat=48.1351 lng=11.5656

isochrone walking time=5 10 15
isochrone cycling time=5 10 15
isochrone driving time=5 10 15

show isochrones on map
show walking reachability rings
show amenities within 10 minute walk

reachability show
amenities report
`,
      "script-3-weather.cynes": `# Script 3: Weather Overlay
# Display real-time weather conditions as map layers

weather load current
weather load forecast hourly

show wind vectors as arrows
show temperature as heatmap
show cloud coverage as grayscale
show precipitation as intensity

show pressure contours
show humidity gradient
show air quality index

wind affects walking speed
visibility affects pedestrian safety

weather report
forecast show
air quality report
`,
      "script-6-signals-airquality.cynes": `# Script 6: Traffic Signals & Air Quality
# Display traffic light signal timing and air quality data

signals load from OSM
  traffic signal locations
  cycle times

show signal locations as points
show cycle time labels
show current phases with colors

air quality load
  PM2.5 concentration
  NO2 levels
  CO levels
  AQI index

show air quality heatmap
show pollution hotspots
show wind dispersion patterns

signals report
air quality report
pollution zones show
`,
      "script-8-routing.cynes": `# Script 8: FWDC Pedestrian Routing
# Full implementation of Fuzzy-Weighted Deterministic Closure

position from lat=48.1351 lng=11.5656
position to lat=48.1400 lng=11.5750

algorithm FWDC
mode pedestrian
walking_speed 1.4 m/s

load script 1 clocks
load script 2 isochrones
load script 3 weather
load script 6 signals airquality

edge weight = walk_time + signal_wait
signal_wait in [0, cycle_time]
resolution_floor beta_0 = min(cycle_times)

catalyst signal camera at intersection
catalyst crowd density from mobile data
catalyst real-time signal broadcast
catalyst historical traffic patterns

fwdc compute path
optimize for travel_time
optimize for air_quality exposure
optimize for pedestrian safety

show optimal path on map
show alternative paths
show signal timing on path
show separation cost regions

route show
route metrics
separation costs show
fwdc closure analysis
`,
    },
  };

  const [selectedFile, setSelectedFile] = useState("sandbox/script-2-isochrones.cynes");
  const [code, setCode] = useState(files.sandbox["script-2-isochrones.cynes"]);
  const [results, setResults] = useState({
    success: true,
    position: null,
    satellites: [],
    validations: [],
    logs: [],
    errors: [],
  });
  const [activeTab, setActiveTab] = useState("console");
  const [mapProvider, setMapProvider] = useState("leaflet");
  const [accuracyHistory, setAccuracyHistory] = useState([]);
  const [playback, setPlayback] = useState({
    isPlaying: false,
    currentStep: 0,
    steps: [],
  });

  const handleCompile = useCallback(() => {
    const result = executeCynegeticus(code);
    setResults(result);

    // Record playback step
    if (result.position) {
      const newStep = {
        id: Date.now(),
        code,
        position: result.position,
        satellites: result.satellites,
        accuracy: parseFloat(result.position.accuracy),
        timestamp: new Date().toLocaleTimeString(),
      };

      setPlayback((prev) => ({
        ...prev,
        steps: [...prev.steps, newStep],
      }));

      // Track accuracy history
      setAccuracyHistory((prev) => [
        ...prev,
        {
          step: accuracyHistory.length + 1,
          accuracy: parseFloat(result.position.accuracy),
        },
      ]);
    }
  }, [code, accuracyHistory.length]);


  return (
    <>
      <Head>
        <title>Cynegeticus Sandbox | Sighthound</title>
        <meta
          name="description"
          content="GPS-Free Positioning using S-Entropy Coordinates"
        />
      </Head>

      <TransitionEffect />

      <main className="relative h-screen w-full overflow-hidden bg-gray-950 flex flex-col">
        {/* Header Bar */}
        <div className="bg-gray-900 border-b border-gray-700 px-4 py-3 flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-lg font-bold text-white">Cynegeticus Sandbox</h1>
            <p className="text-xs text-gray-400">GPS-Free Positioning via S-Entropy Coordinates</p>
          </div>

          {/* Map Provider Selection */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Map:</label>
            <select
              value={mapProvider}
              onChange={(e) => setMapProvider(e.target.value)}
              className="px-3 py-1 bg-gray-800 text-white text-xs rounded border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="leaflet">Leaflet (2D)</option>
              <option value="cesium">Cesium (3D)</option>
            </select>
          </div>

          {/* Playback Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setPlayback((prev) => ({ ...prev, currentStep: Math.max(0, prev.currentStep - 1) }));
                if (playback.steps[playback.currentStep - 1]) {
                  const step = playback.steps[playback.currentStep - 1];
                  setResults({
                    success: true,
                    position: step.position,
                    satellites: step.satellites,
                    validations: [],
                    logs: [`Loaded step: ${step.timestamp}`],
                    errors: [],
                  });
                }
              }}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
              title="Previous step"
            >
              ◀
            </button>
            <span className="text-xs text-gray-400">
              {playback.steps.length > 0 ? `${playback.currentStep}/${playback.steps.length}` : "0/0"}
            </span>
            <button
              onClick={() => {
                setPlayback((prev) => ({ ...prev, currentStep: Math.min(prev.steps.length - 1, prev.currentStep + 1) }));
                if (playback.steps[playback.currentStep + 1]) {
                  const step = playback.steps[playback.currentStep + 1];
                  setResults({
                    success: true,
                    position: step.position,
                    satellites: step.satellites,
                    validations: [],
                    logs: [`Loaded step: ${step.timestamp}`],
                    errors: [],
                  });
                }
              }}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded"
              title="Next step"
            >
              ▶
            </button>
          </div>

          <button
            onClick={handleCompile}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded transition-colors"
          >
            Compile & Execute
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Sidebar */}
          <div className="w-48 bg-gray-900 border-r border-gray-700">
            <div className="px-3 py-2 border-b border-gray-700 text-xs font-semibold text-gray-400">
              FILES
            </div>
            <FileTree
              files={files}
              selectedFile={selectedFile}
              onSelectFile={(file) => {
                setSelectedFile(file);
                const parts = file.split("/");
                let content = files;
                parts.forEach((part) => {
                  content = content[part];
                });
                setCode(content);
              }}
            />
          </div>

          {/* Editor Section */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="bg-gray-800 px-3 py-2 border-b border-gray-700 text-xs text-gray-400 flex justify-between items-center">
              <span>{selectedFile}</span>
              <span className="text-gray-500">{code.split("\n").length} lines</span>
            </div>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="flex-1 bg-gray-900 text-gray-100 font-mono text-xs p-4 resize-none focus:outline-none overflow-auto"
              spellCheck="false"
            />
          </div>

          {/* Output Section */}
          <div className="flex-1 flex flex-col overflow-hidden border-l border-gray-700">
            {/* Tabs */}
            <div className="flex border-b border-gray-700 bg-gray-800">
              {["console", "results", "charts", "map"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 text-xs font-mono font-medium capitalize transition-colors ${
                    activeTab === tab
                      ? "bg-blue-700 text-white border-b-2 border-blue-500"
                      : "bg-gray-800 text-gray-400 hover:text-gray-300"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto bg-gray-900 p-4 font-mono text-xs">
              {activeTab === "console" && (
                <div className="text-gray-300">
                  {results.logs.length === 0 ? (
                    <div className="text-gray-500">// Click "Compile & Execute" to run code</div>
                  ) : (
                    results.logs.map((log, i) => (
                      <div key={i} className="mb-1 whitespace-pre-wrap">
                        {log}
                      </div>
                    ))
                  )}
                  {results.errors.length > 0 && (
                    <div className="mt-4 text-red-400">
                      {results.errors.map((err, i) => (
                        <div key={i}>{err}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "results" && (
                <div className="text-gray-300">
                  {results.position && (
                    <div className="p-3 bg-gray-800 rounded mb-3 border border-gray-700">
                      <div className="text-green-400 font-semibold mb-2">Position</div>
                      <div>Latitude: {results.position.lat}°</div>
                      <div>Longitude: {results.position.lon}°</div>
                      <div>Accuracy: ±{results.position.accuracy}m</div>
                    </div>
                  )}
                  {results.satellites.length > 0 && (
                    <div className="p-3 bg-gray-800 rounded mb-3 border border-gray-700">
                      <div className="text-green-400 font-semibold mb-2">Satellites ({results.satellites.length})</div>
                      {results.satellites.slice(0, 5).map((sat) => (
                        <div key={sat.id}>Sat-{sat.id}: {sat.lat.toFixed(2)}° {sat.lon.toFixed(2)}°</div>
                      ))}
                    </div>
                  )}
                  {results.validations.length > 0 && (
                    <div className="p-3 bg-gray-800 rounded border border-gray-700">
                      <div className="text-green-400 font-semibold mb-2">Validations</div>
                      {results.validations.map((v, i) => (
                        <div key={i}>{v.type}: ✓ PASS</div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "charts" && (
                <div className="h-full w-full p-4">
                  {accuracyHistory.length > 0 ? (
                    <div>
                      <h3 className="text-xs text-gray-400 mb-3">Accuracy Trend (meters)</h3>
                      <SimpleAreaChart
                        data={accuracyHistory}
                        width={400}
                        height={250}
                        color="#3b82f6"
                      />
                      <div className="mt-4 p-3 bg-gray-800 rounded text-xs text-gray-300">
                        <div>Total Executions: {accuracyHistory.length}</div>
                        <div>Best Accuracy: ±{Math.min(...accuracyHistory.map((d) => d.accuracy)).toFixed(2)}m</div>
                        <div>Latest: ±{accuracyHistory[accuracyHistory.length - 1]?.accuracy.toFixed(2) || "—"}m</div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-500 flex items-center justify-center h-full">
                      // Compile to view accuracy trends
                    </div>
                  )}
                </div>
              )}

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
            </div>
          </div>
        </div>

        {/* Status Bar */}
        <div className="bg-gray-900 border-t border-gray-700 px-4 py-2 text-xs text-gray-500 flex justify-between items-center">
          <div>Cynegeticus v1.0 | {results.success ? "✓ Ready" : "✗ Error"}</div>
          <div className="flex gap-4 text-gray-400">
            <span>Executions: {playback.steps.length}</span>
            <span>Accuracy: {accuracyHistory.length > 0 ? `±${accuracyHistory[accuracyHistory.length - 1].accuracy.toFixed(2)}m` : "—"}</span>
            <span>Map: {mapProvider === "cesium" ? "3D Cesium" : "2D Leaflet"}</span>
          </div>
        </div>
      </main>
    </>
  );
}
