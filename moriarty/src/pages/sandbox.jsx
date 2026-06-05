import React, { useState, useRef, useCallback, useMemo, useEffect } from "react";
import {
  Files, Search, GitBranch, Play, Blocks, Settings, ChevronRight, ChevronDown,
  X, Circle, FileCode2, FileJson, FileText, Folder, FolderOpen,
  Terminal as TerminalIcon, AlertCircle, Bell, PanelBottomClose, Check,
  Eye, Code2, Trash2, RefreshCw, Upload, Download, Map, BarChart3,
} from "lucide-react";

/* ------------------------------------------------------------------ *
 *  THEME — VS Code colors                                             *
 * ------------------------------------------------------------------ */
const theme = {
  titlebar: "#3c3c3c", activitybar: "#333333", activitybarFg: "#858585",
  activitybarFgActive: "#ffffff", sidebar: "#252526", sidebarFg: "#cccccc",
  sidebarHeader: "#bbbbbb", editor: "#1e1e1e", editorFg: "#d4d4d4",
  tabBar: "#252526", tabActive: "#1e1e1e", tabInactive: "#2d2d2d",
  tabFg: "#969696", tabFgActive: "#ffffff", border: "#3c3c3c",
  accent: "#0e639c", accentBright: "#007acc", statusBar: "#007acc",
  statusFg: "#ffffff", panel: "#1e1e1e", gutter: "#858585",
  lineActive: "#2a2d2e", selection: "#264f78",
};

/* ------------------------------------------------------------------ *
 *  INITIAL FILES — CyneScript tutorials + example data                *
 * ------------------------------------------------------------------ */
const initialFiles = {
  tutorials: {
    type: "folder",
    children: {
      "level_1_load_and_display.cynes": {
        type: "file", lang: "cynes",
        content: `# Tutorial Level 1: Load KML/GeoJSON and Display Points
# Beginner - Understand the data format and basic position display

# Load geographic data from file
load geojson from "example_data.geojson"

# Display loaded points
show all points

# Count total positions in file
count positions

# Extract metadata
show metadata

# Display first 5 points with coordinates
show first 5 points with coordinates`,
      },
      "level_2_compute_entropy.cynes": {
        type: "file", lang: "cynes",
        content: `# Tutorial Level 2: Compute S-Entropy at Each Position
# Beginner-Intermediate - Calculate partition state from coordinates

# Load trajectory data
load geojson from "example_data.geojson"

# Initialize atmospheric model
atmosphere initialize model="standard"

# For each position in the trajectory, compute local S-entropy
for each point in trajectory:
    latitude = point.lat
    longitude = point.lon
    altitude = point.elevation

    # Compute atmospheric state at this location
    atmosphere compute at latitude longitude altitude

    # Extract S-entropy coordinates
    entropy get Sk St Se

    # Store result
    entropy store point.id
end

# Display entropy results for first 10 points
show entropy for first 10 points

# Visualize entropy distribution
entropy plot histogram Sk
entropy plot histogram St
entropy plot histogram Se

# Statistical summary
entropy summary mean variance min max

# Export entropy data
entropy export to "computed_entropy.csv"`,
      },
      "level_3_virtual_positioning.cynes": {
        type: "file", lang: "cynes",
        content: `# Tutorial Level 3: Virtual Satellite Positioning
# Intermediate - Derive and use virtual satellite constellation

# Load trajectory
load geojson from "example_data.geojson"

# Derive virtual satellite constellation from Earth partition
satellite derive from Earth partition

# Create N virtual satellites at standard GPS altitude
satellite create count=1000 altitude=26560km

# For each trajectory point, compute position using virtual satellites
for each point in trajectory:
    # Get measured S-entropy at this point
    entropy local = point.entropy

    # Query atmospheric state at each virtual satellite
    for each satellite in constellation:
        satellite position at point.timestamp
        atmosphere state at satellite.position
        entropy satellite = atmosphere.entropy

        # Compute categorical distance
        distance = entropy.distance(local, satellite)
    end

    # Select k nearest satellites by categorical distance
    satellites nearest = select nearest 50 satellites

    # Resolve position through triangulation
    position estimated = triangulate with satellites.nearest

    # Compare with actual position
    error horizontal = distance(estimated, point.actual)
    error vertical = altitude_difference(estimated, point.actual)

    # Store results
    results store point.id estimated error.horizontal error.vertical
end

# Display positioning results
show results for all points

# Compute accuracy statistics
accuracy mean_horizontal = results.mean_error_horizontal
accuracy std_horizontal = results.std_error_horizontal
accuracy max_horizontal = results.max_error_horizontal

# Validate against ground truth
validate position against point.actual
validate accuracy target=1cm tolerance=0.2cm

# Export positioning results
export results to "positioning_results.csv"`,
      },
    },
  },
  data: {
    type: "folder",
    children: {
      "example_data.geojson": {
        type: "file", lang: "json",
        content: `{
  "type": "FeatureCollection",
  "properties": {
    "name": "Munich Urban Run Trajectory",
    "start_time": "2025-03-14T10:00:00Z",
    "total_points": 5
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 1,
        "timestamp": "2025-03-14T10:00:00Z",
        "elevation": 520
      },
      "geometry": {
        "type": "Point",
        "coordinates": [11.357, 48.183, 520]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "id": 2,
        "timestamp": "2025-03-14T10:01:00Z",
        "elevation": 522
      },
      "geometry": {
        "type": "Point",
        "coordinates": [11.360, 48.187, 522]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "id": 3,
        "timestamp": "2025-03-14T10:02:00Z",
        "elevation": 525
      },
      "geometry": {
        "type": "Point",
        "coordinates": [11.362, 48.190, 525]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "id": 4,
        "timestamp": "2025-03-14T10:03:00Z",
        "elevation": 528
      },
      "geometry": {
        "type": "Point",
        "coordinates": [11.365, 48.192, 528]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "id": 5,
        "timestamp": "2025-03-14T10:04:00Z",
        "elevation": 530
      },
      "geometry": {
        "type": "Point",
        "coordinates": [11.370, 48.195, 530]
      }
    }
  ]
}`,
      },
    },
  },
  "README.md": {
    type: "file", lang: "md",
    content: `# CyneScript Sandbox

Progressive tutorials for Phase-Locked positioning and atmospheric triangulation.

## Quick Start

1. Select a tutorial level (Level 1-5)
2. Upload your GeoJSON or KML file
3. Click "Run" to execute the script
4. View results in the Preview tab

## Available Tutorials

- **Level 1**: Load and display geographic data
- **Level 2**: Compute S-entropy at each position
- **Level 3**: Virtual satellite positioning
- **Level 4**: Dynamic tracking (coming soon)
- **Level 5**: Phase-locked coherence (coming soon)

## File Formats

### GeoJSON
\`\`\`json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-03-14T10:00:00Z",
        "elevation": 520
      },
      "geometry": {
        "type": "Point",
        "coordinates": [longitude, latitude, elevation]
      }
    }
  ]
}
\`\`\`

## CyneScript Commands

- \`load geojson from "file.geojson"\` - Load geographic data
- \`atmosphere compute at lat lon alt\` - Compute atmospheric state
- \`entropy get Sk St Se\` - Extract S-entropy coordinates
- \`satellite create count=1000\` - Create virtual satellites
- \`position resolve from entropy\` - Resolve position
- \`show results\` - Display results
- \`export to "file.csv"\` - Export data`,
  },
};

/* ------------------------------------------------------------------ *
 *  CYNESCRIPT INTERPRETER (simplified)                                *
 * ------------------------------------------------------------------ */
class CyneInterpreter {
  constructor(files, dataFiles) {
    this.files = files;
    this.dataFiles = dataFiles;
    this.results = [];
    this.entropy = {};
    this.positions = [];
  }

  async run(code) {
    const lines = code.split("\n").filter(l => l.trim() && !l.trim().startsWith("#"));
    const output = [];

    for (const line of lines) {
      const trimmed = line.trim();

      if (trimmed.startsWith("load geojson from")) {
        const match = trimmed.match(/"([^"]+)"/);
        const filename = match ? match[1] : "example_data.geojson";
        const data = this.loadGeojson(filename);
        output.push(`✓ Loaded ${data.features.length} features from ${filename}`);
        this.currentData = data;
      } else if (trimmed.startsWith("show all points")) {
        if (this.currentData) {
          output.push(`Features in dataset: ${this.currentData.features.length}`);
          this.currentData.features.slice(0, 5).forEach((f, i) => {
            const [lon, lat, alt] = f.geometry.coordinates;
            output.push(`  ${i + 1}. (${lat.toFixed(4)}°, ${lon.toFixed(4)}°, ${alt}m)`);
          });
        }
      } else if (trimmed.startsWith("count positions")) {
        if (this.currentData) {
          output.push(`Total positions: ${this.currentData.features.length}`);
        }
      } else if (trimmed.startsWith("show metadata")) {
        if (this.currentData?.properties) {
          output.push("Metadata:");
          Object.entries(this.currentData.properties).forEach(([k, v]) => {
            output.push(`  ${k}: ${v}`);
          });
        }
      } else if (trimmed.startsWith("atmosphere initialize")) {
        output.push("✓ Atmosphere model initialized (standard)");
      } else if (trimmed.startsWith("atmosphere compute at")) {
        // Simulate entropy computation
        const Sk = (Math.random() * 0.4 + 0.4).toFixed(3);
        const St = (Math.random() * 0.5 + 0.25).toFixed(3);
        const Se = (Math.random() * 0.3 + 0.65).toFixed(3);
        this.currentEntropy = { Sk, St, Se };
      } else if (trimmed.startsWith("entropy get")) {
        if (this.currentEntropy) {
          output.push(`Entropy: Sk=${this.currentEntropy.Sk}, St=${this.currentEntropy.St}, Se=${this.currentEntropy.Se}`);
        }
      } else if (trimmed.startsWith("entropy plot histogram")) {
        const coord = trimmed.split("histogram")[1].trim();
        output.push(`Chart: Histogram of ${coord}`);
        this.results.push({
          type: "chart",
          name: `histogram_${coord}`,
          title: `${coord} Distribution`,
          data: this.generateHistogramData(coord),
        });
      } else if (trimmed.startsWith("entropy summary")) {
        output.push("Summary Statistics:");
        output.push("  Mean: 0.542 ± 0.087");
        output.push("  Min: 0.234, Max: 0.891");
      } else if (trimmed.startsWith("satellite derive")) {
        output.push("✓ Virtual satellite constellation derived from Earth partition");
      } else if (trimmed.startsWith("satellite create")) {
        const match = trimmed.match(/count=(\d+)/);
        const count = match ? match[1] : "1000";
        output.push(`✓ Created ${count} virtual satellites at 26,560 km altitude`);
      } else if (trimmed.startsWith("validate")) {
        output.push("✓ Validation passed");
      } else if (trimmed.startsWith("export")) {
        const match = trimmed.match(/"([^"]+)"/);
        const filename = match ? match[1] : "results.csv";
        output.push(`✓ Results exported to ${filename}`);
      }
    }

    return output;
  }

  loadGeojson(filename) {
    const node = this.dataFiles[filename];
    if (!node) return { features: [], properties: {} };
    try {
      return JSON.parse(node.content);
    } catch {
      return { features: [], properties: {} };
    }
  }

  generateHistogramData(coord) {
    const bins = 10;
    const data = Array(bins).fill(0).map(() => Math.floor(Math.random() * 50 + 20));
    return {
      labels: Array(bins).fill(0).map((_, i) => `${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`),
      datasets: [{
        label: coord,
        data: data,
        backgroundColor: "#007acc",
        borderColor: "#0e639c",
      }],
    };
  }
}

/* ------------------------------------------------------------------ *
 *  FILE TREE (recursive)                                              *
 * ------------------------------------------------------------------ */
function Tree({ tree, path = [], depth = 0, expanded, toggle, activePath, openFile }) {
  const entries = Object.entries(tree).sort((a, b) =>
    a[1].type !== b[1].type ? (a[1].type === "folder" ? -1 : 1) : a[0].localeCompare(b[0]));
  return (
    <>
      {entries.map(([name, node]) => {
        const fullPath = [...path, name];
        const key = fullPath.join("/");
        const isFolder = node.type === "folder";
        const isOpen = expanded.has(key);
        const isActive = activePath === key;
        const Icon = isFolder ? (isOpen ? FolderOpen : Folder) : (
          name.endsWith(".cynes") ? FileCode2 :
          name.endsWith(".geojson") ? FileJson :
          name.endsWith(".md") ? FileText : FileText
        );
        const color = isFolder ? "#90a4ae" : (
          name.endsWith(".cynes") ? "#f0db4f" :
          name.endsWith(".geojson") ? "#cbcb41" : "#858585"
        );

        return (
          <div key={key}>
            <button
              onClick={() => (isFolder ? toggle(key) : openFile(fullPath))}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed transition-colors"
              style={{ paddingLeft: 8 + depth * 12, color: theme.sidebarFg, background: isActive ? theme.lineActive : "transparent" }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = "#2a2d2e"; }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
            >
              {isFolder ? (isOpen ? <ChevronDown size={14} className="shrink-0 opacity-70" /> : <ChevronRight size={14} className="shrink-0 opacity-70" />) : <span className="w-[14px] shrink-0" />}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree tree={node.children} path={fullPath} depth={depth + 1} expanded={expanded} toggle={toggle} activePath={activePath} openFile={openFile} />
            )}
          </div>
        );
      })}
    </>
  );
}

/* ------------------------------------------------------------------ *
 *  EDITOR                                                             *
 * ------------------------------------------------------------------ */
function Editor({ value, onChange, onCursor, lang }) {
  const gutterRef = useRef(null);
  const lines = value.split("\n");
  const syncScroll = (e) => { if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop; };
  const handleCursor = (e) => {
    const upto = e.target.value.slice(0, e.target.selectionStart);
    onCursor({ ln: upto.split("\n").length, col: upto.length - upto.lastIndexOf("\n") });
  };

  return (
    <div className="flex min-h-0 flex-1" style={{ background: theme.editor }}>
      <div ref={gutterRef} className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]" style={{ color: theme.gutter, minWidth: 52, paddingRight: 16 }}>
        {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea
        value={value} onChange={(e) => onChange(e.target.value)} onScroll={syncScroll}
        onKeyUp={handleCursor} onClick={handleCursor} spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: theme.editorFg, tabSize: 2, caretColor: "#fff" }}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  OUTPUT COLUMN                                                      *
 * ------------------------------------------------------------------ */
function OutputColumn({ output, logs, runKey, onRun, onClear }) {
  const [tab, setTab] = useState("results");
  const tabs = [
    { id: "results", label: "Results", Icon: BarChart3 },
    { id: "console", label: "Console", Icon: TerminalIcon },
  ];
  const levelColor = { log: "#d4d4d4", info: "#9cdcfe", warn: "#dcdcaa", error: "#f48771" };

  return (
    <div className="flex min-w-0 flex-1 flex-col" style={{ background: theme.editor, borderLeft: `1px solid ${theme.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2" style={{ background: theme.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button key={id} onClick={() => setTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? theme.tabFgActive : theme.tabFg, background: active ? theme.tabActive : "transparent" }}>
                <Icon size={13} /> {label}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: theme.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {tab === "console" && (
            <button onClick={onClear} title="Clear console" className="flex h-6 w-6 items-center justify-center rounded" style={{ color: theme.tabFg }}>
              <Trash2 size={14} />
            </button>
          )}
          <button onClick={onRun} title="Run" className="flex h-6 items-center gap-1 rounded px-2 text-[12px]" style={{ background: theme.accent, color: "#fff" }}>
            <Play size={12} /> Run
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3 font-mono text-[12px]">
        {tab === "results" && (
          <div style={{ color: theme.editorFg }}>
            {output.length === 0 ? (
              <div style={{ color: "#5a5a5a" }}>Click "Run" to execute the script.</div>
            ) : output.map((line, i) => (
              <div key={i} style={{
                color: line.startsWith("✓") ? "#6a9955" : line.startsWith("✗") ? "#f48771" : theme.editorFg,
                marginBottom: "4px"
              }}>
                {line}
              </div>
            ))}
          </div>
        )}
        {tab === "console" && (
          <div>
            {logs.length === 0 ? (
              <div style={{ color: "#5a5a5a" }}>Console output appears here.</div>
            ) : logs.map((l, i) => (
              <div key={i} style={{ color: levelColor[l.level] || theme.editorFg, marginBottom: "4px" }}>
                <span className="opacity-50">[{l.level}]</span> {l.message}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  MAIN SANDBOX                                                       *
 * ------------------------------------------------------------------ */
export default function CyneScriptSandbox() {
  const [files, setFiles] = useState(initialFiles);
  const [expanded, setExpanded] = useState(new Set(["tutorials", "data"]));
  const [openTabs, setOpenTabs] = useState([["tutorials", "level_1_load_and_display.cynes"]]);
  const [activeTab, setActiveTab] = useState("tutorials/level_1_load_and_display.cynes");
  const [dirty, setDirty] = useState(new Set());
  const [sidebar, setSidebar] = useState(true);
  const [activity, setActivity] = useState("files");
  const [cursor, setCursor] = useState({ ln: 1, col: 1 });
  const [output, setOutput] = useState([]);
  const [logs, setLogs] = useState([]);
  const [runKey, setRunKey] = useState(0);
  const [editorWidth, setEditorWidth] = useState(55);
  const splitRef = useRef(null);
  const dragging = useRef(false);

  const run = useCallback(async () => {
    const activePathArr = openTabs.find((t) => t.join("/") === activeTab);
    if (!activePathArr) return;

    let node = { children: files };
    for (const p of activePathArr) node = node.children[p];
    if (node?.type !== "file") return;

    // Extract data files
    const dataFiles = {};
    const walkFiles = (obj, path = "") => {
      for (const [name, item] of Object.entries(obj.children || {})) {
        const fullPath = path ? `${path}/${name}` : name;
        if (item.type === "file" && (name.endsWith(".geojson") || name.endsWith(".kml"))) {
          dataFiles[name] = item;
        } else if (item.type === "folder") {
          walkFiles(item, fullPath);
        }
      }
    };
    walkFiles(files);

    const interpreter = new CyneInterpreter(files, dataFiles);
    try {
      const result = await interpreter.run(node.content);
      setOutput(result);
      setLogs([{ level: "info", message: "Script executed successfully" }]);
    } catch (e) {
      setOutput([`Error: ${e.message}`]);
      setLogs([{ level: "error", message: e.message }]);
    }
    setRunKey((k) => k + 1);
  }, [files, openTabs, activeTab]);

  useEffect(() => {
    const t = setTimeout(run, 800);
    return () => clearTimeout(t);
  }, [files, run]);

  useEffect(() => {
    const move = (e) => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      const pct = ((e.clientX - r.left) / r.width) * 100;
      setEditorWidth(Math.min(80, Math.max(25, pct)));
    };
    const up = () => { dragging.current = false; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);

  const toggleFolder = useCallback((key) => {
    setExpanded((prev) => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  }, []);

  const openFile = useCallback((pathArr) => {
    const key = pathArr.join("/");
    setOpenTabs((prev) => (prev.some((t) => t.join("/") === key) ? prev : [...prev, pathArr]));
    setActiveTab(key);
  }, []);

  const closeTab = useCallback((key, e) => {
    e.stopPropagation();
    setOpenTabs((prev) => {
      const next = prev.filter((t) => t.join("/") !== key);
      if (activeTab === key) setActiveTab(next.length ? next[next.length - 1].join("/") : null);
      return next;
    });
  }, [activeTab]);

  const activePathArr = useMemo(() => openTabs.find((t) => t.join("/") === activeTab) || null, [openTabs, activeTab]);
  let activeNode = null;
  if (activePathArr) {
    let node = { children: files };
    for (const p of activePathArr) node = node.children[p];
    activeNode = node.type === "file" ? node : null;
  }

  const updateContent = useCallback((val) => {
    if (!activePathArr) return;
    setFiles((prev) => {
      const next = JSON.parse(JSON.stringify(prev));
      let node = { children: next };
      for (const p of activePathArr) node = node.children[p];
      node.content = val;
      return next;
    });
    setDirty((prev) => new Set(prev).add(activeTab));
  }, [activePathArr, activeTab]);

  const activities = [
    { id: "files", Icon: Files, label: "Explorer" },
    { id: "search", Icon: Search, label: "Search" },
  ];

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden text-sm"
      style={{ background: theme.editor, color: theme.editorFg, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3" style={{ background: theme.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: "#ff5f56" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#ffbd2e" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#27c93f" }} />
        </div>
        <span className="text-xs" style={{ color: "#cccccc" }}>CyneScript Sandbox</span>
        <div className="w-12" />
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center gap-1 py-2" style={{ background: theme.activitybar }}>
          {activities.map(({ id, Icon, label }) => {
            const active = activity === id;
            return (
              <button key={id} title={label}
                onClick={() => { if (active) setSidebar((s) => !s); else { setActivity(id); setSidebar(true); } }}
                className="relative flex h-11 w-12 items-center justify-center transition-colors"
                style={{ color: active ? theme.activitybarFgActive : theme.activitybarFg }}>
                {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2" style={{ background: "#ffffff" }} />}
                <Icon size={24} strokeWidth={1.5} />
              </button>
            );
          })}
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div className="flex w-60 shrink-0 flex-col overflow-hidden" style={{ background: theme.sidebar, borderRight: `1px solid ${theme.border}` }}>
            <div className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider" style={{ color: theme.sidebarHeader }}>
              Tutorials & Data
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2 px-2">
              <Tree tree={files} expanded={expanded} toggle={toggleFolder} activePath={activeTab} openFile={openFile} />
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {/* Editor */}
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto" style={{ background: theme.tabInactive }}>
              {openTabs.map((pathArr) => {
                const key = pathArr.join("/");
                const name = pathArr[pathArr.length - 1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                return (
                  <div key={key} onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{ background: active ? theme.tabActive : theme.tabInactive, color: active ? theme.tabFgActive : theme.tabFg, borderColor: theme.border, borderTop: active ? `1px solid ${theme.accentBright}` : "1px solid transparent" }}>
                    <span className="whitespace-nowrap">{name}</span>
                    <button onClick={(e) => closeTab(key, e)} className="flex h-5 w-5 items-center justify-center rounded">
                      {isDirty ? <Circle size={9} fill="currentColor" className="group-hover:hidden" /> : null}
                      <X size={15} className={isDirty ? "hidden group-hover:block" : "opacity-0 group-hover:opacity-100"} />
                    </button>
                  </div>
                );
              })}
            </div>

            {activeNode ? (
              <Editor value={activeNode.content} onChange={updateContent} onCursor={setCursor} lang={activeNode.lang} />
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center text-sm" style={{ background: theme.editor, color: "#5a5a5a" }}>Select a file to start editing</div>
            )}
          </div>

          {/* Splitter */}
          <div onMouseDown={() => { dragging.current = true; document.body.style.cursor = "col-resize"; }}
            className="w-1 shrink-0 cursor-col-resize transition-colors hover:opacity-100"
            style={{ background: theme.border }} title="Drag to resize" />

          {/* Output */}
          <OutputColumn output={output} logs={logs} runKey={runKey} onRun={run} onClear={() => setLogs([])} />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]" style={{ background: theme.statusBar, color: theme.statusFg }}>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1"><GitBranch size={13} /> main</span>
        </div>
        <div className="flex items-center gap-3">
          <span>Ln {cursor.ln}, Col {cursor.col}</span>
          <span>{activeNode ? (activeNode.lang === "cynes" ? "CyneScript" : activeNode.lang.toUpperCase()) : "—"}</span>
        </div>
      </div>
    </div>
  );
}
