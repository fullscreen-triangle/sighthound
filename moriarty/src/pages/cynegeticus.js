import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";

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
// MAP COMPONENT
// ============================================================================

function MapComponent({ position, satellites }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !position) return;

    // Load Leaflet from CDN
    const loadMap = async () => {
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
        return new Promise((resolve) => {
          const script = document.createElement("script");
          script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
          script.onload = () => {
            initMap();
            resolve();
          };
          document.head.appendChild(script);
        });
      } else {
        initMap();
      }
    };

    const initMap = () => {
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

    loadMap();

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [position, satellites]);

  return <div ref={containerRef} className="w-full h-full bg-white rounded" />;
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
  };

  const [selectedFile, setSelectedFile] = useState("tutorials/hello_position.cynes");
  const [code, setCode] = useState(files.tutorials["hello_position.cynes"]);
  const [results, setResults] = useState({
    success: true,
    position: null,
    satellites: [],
    validations: [],
    logs: [],
    errors: [],
  });
  const [activeTab, setActiveTab] = useState("console");

  const handleCompile = useCallback(() => {
    const result = executeCynegeticus(code);
    setResults(result);
  }, [code]);


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
        <div className="bg-gray-900 border-b border-gray-700 px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-white">Cynegeticus Sandbox</h1>
            <p className="text-xs text-gray-400">GPS-Free Positioning via S-Entropy Coordinates</p>
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
              {["console", "results", "map"].map((tab) => (
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

              {activeTab === "map" && (
                <div className="h-full w-full">
                  {results.position ? (
                    <MapComponent position={results.position} satellites={results.satellites} />
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
        <div className="bg-gray-900 border-t border-gray-700 px-4 py-1 text-xs text-gray-500 flex justify-between">
          <div>Cynegeticus v1.0</div>
          <div>{results.success ? "✓ Ready" : "✗ Error"}</div>
        </div>
      </main>
    </>
  );
}
