# CyneScript Browser Sandbox Architecture

A three-column interactive sandbox for learning and demonstrating Phase-Locked positioning through progressive CyneScript tutorials.

## User Interface Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CyneScript Sandbox - Tutorial: Level 3 (Virtual Positioning)           │
├──────────────────┬─────────────────────┬──────────────────────────────┤
│                  │                     │                              │
│  FILES           │  CODE               │  OUTPUT                      │
│  (Left Column)   │  (Middle Column)    │  (Right Column)              │
│                  │                     │                              │
│  ┌─────────────┐ │ ┌─────────────────┐ │  ┌──────────────────────┐   │
│  │ Example     │ │ │# Load trajectory│ │  │  Results:            │   │
│  │ Data        │ │ │load geojson     │ │  │  ─────────────────   │   │
│  │ ─────────── │ │ │from "data.geo"  │ │  │  Total points: 25    │   │
│  │ ✓ munich_   │ │ │                 │ │  │  Mean error: 1.2 cm  │   │
│  │   run.gj    │ │ │# Create sat     │ │  │  Std dev: 0.8 cm     │   │
│  │ ✓ two_      │ │ │satellite derive │ │  │  Max error: 3.5 cm   │   │
│  │   device_   │ │ │from Earth       │ │  │                      │   │
│  │   sync.gj   │ │ │partition        │ │  │  [Progress: 100%]    │   │
│  │             │ │ │                 │ │  │                      │   │
│  │ Upload File │ │ │[RUN] [SAVE]     │ │  │ [EXPORT] [VISUALIZE] │   │
│  │ [+ Choose]  │ │ │                 │ │  │                      │   │
│  └─────────────┘ │ └─────────────────┘ │  └──────────────────────┘   │
│                  │                     │                              │
│  Script Library  │  Syntax Highlight   │  Real-time Visualization    │
│  ─────────────── │  Type Checking      │  Data Export                │
│  [L1] Load &     │  Error Display      │  Result Logging             │
│  Display         │                     │                              │
│  [L2] Entropy    │  Dimensions:        │  Charts:                     │
│  Compute         │  ✓ Verified         │  • Accuracy histogram       │
│  [L3] Virtual    │  ⚠ Warning: unit    │  • Error timeline           │
│  Positioning     │    mismatch         │  • Position trajectory      │
│  [L4] Dynamic    │  ✗ Error: invalid   │                              │
│  Tracking        │    entropy range    │                              │
│  [L5] Phase-     │                     │                              │
│  Lock            │                     │                              │
│                  │                     │                              │
└──────────────────┴─────────────────────┴──────────────────────────────┘
```

## Architecture Components

### 1. File Manager (Left Column)

**Responsibilities:**
- Display available example data files
- Handle file upload (KML, GeoJSON)
- Show script library (Level 1-5)
- Manage file metadata

**Implementation:**
```javascript
class FileManager {
  constructor() {
    this.files = {};           // Loaded files
    this.scripts = {           // Available scripts
      level1: { name: 'Load and Display', url: '...' },
      level2: { name: 'Entropy Compute', url: '...' },
      // ...
    };
  }
  
  uploadFile(file) {
    if (file.type === 'application/json') {
      return this.parseGeoJSON(file);
    } else if (file.type === 'application/vnd.google-earth.kml+xml') {
      return this.parseKML(file);
    }
  }
  
  parseGeoJSON(file) {
    const data = JSON.parse(file.text());
    return {
      type: 'geojson',
      features: data.features,
      metadata: data.properties
    };
  }
}
```

### 2. Code Editor (Middle Column)

**Responsibilities:**
- Display CyneScript code with syntax highlighting
- Perform dimensional type checking
- Show error/warning messages
- Execute compiled code
- Support script editing and saving

**Implementation:**
```javascript
class CodeEditor {
  constructor() {
    this.editor = createMonacoEditor({
      language: 'cynescript',
      theme: 'vs-light'
    });
  }
  
  loadScript(scriptName) {
    fetch(`/tutorials/${scriptName}.cynes`)
      .then(r => r.text())
      .then(code => this.editor.setValue(code));
  }
  
  checkDimensions(code) {
    // Dimensional type checking
    const lines = code.split('\n');
    const errors = [];
    const warnings = [];
    
    // Parse declarations and check dimensional consistency
    lines.forEach((line, idx) => {
      if (line.match(/entropy get \w+/)) {
        // Check if entropy is in [0,1]
      }
      if (line.match(/satellite create count=(\d+)/)) {
        // Check if count is positive integer
      }
    });
    
    return { errors, warnings };
  }
  
  async execute(code) {
    const ast = this.parse(code);
    const runtime = new CyneRuntime();
    return runtime.evaluate(ast);
  }
}
```

### 3. Output Panel (Right Column)

**Responsibilities:**
- Display execution results
- Show progress bars and logging
- Render visualizations (charts, maps)
- Provide export options
- Stream live results during execution

**Implementation:**
```javascript
class OutputPanel {
  constructor() {
    this.results = {};
    this.charts = [];
    this.logs = [];
  }
  
  displayResults(data) {
    const html = this.formatResults(data);
    document.getElementById('output').innerHTML = html;
  }
  
  renderChart(chartName, chartType, data) {
    const chart = new Chart(document.getElementById(`chart-${chartName}`), {
      type: chartType,  // 'bar', 'line', 'scatter', 'histogram'
      data: data
    });
    this.charts.push(chart);
  }
  
  streamLog(message, level = 'info') {
    const timestamp = new Date().toISOString();
    this.logs.push({ timestamp, level, message });
    this.displayLog(message, level);
  }
  
  exportResults(format) {
    if (format === 'csv') {
      return this.toCSV(this.results);
    } else if (format === 'json') {
      return JSON.stringify(this.results, null, 2);
    }
  }
}
```

## Execution Flow

### Step 1: Parse CyneScript

```
User Code (CyneScript)
        ↓
    [Lexer] → Tokens
        ↓
    [Parser] → Abstract Syntax Tree (AST)
        ↓
    [Type Checker] → Validated AST
        ↓
    [Compiler] → Executable IR
```

### Step 2: Load Data

```
User File (GeoJSON/KML)
        ↓
    [Parser] → Feature Collection
        ↓
    [Validator] → {features, metadata}
        ↓
    [Memory] → Available to runtime
```

### Step 3: Execute

```
Validated AST + Data
        ↓
    [Runtime Interpreter]
        ↓
    [Atmosphere Model] ←┐
    [Satellite Calc]    ├→ Computations
    [Triangulation]     ←┘
        ↓
    [Results] → Output Panel
```

## CyneScript Language Features

### Keywords

```
Data:           load, show, count, export
Atmosphere:     atmosphere, measure, entropy, compute
Satellites:     satellite, derive, create, position
Position:       position, resolve, triangulate, accuracy
Tracking:       tracker, predict, update, state
Coherence:      coherence, establish, log, validate
Control:        for, if, end, break, continue
```

### Type System

```
Scalars:        number, string, timestamp, bool
Coordinates:    (lat, lon, alt) ∈ ℝ³
Entropy:        (Sk, St, Se) ∈ [0,1]³
Position:       (x, y, z) ∈ ℝ³
Velocity:       (vx, vy, vz) ∈ ℝ³ (m/s)
Distance:       value ∈ ℝ (meters)
```

### Dimensional Checking

```cynes
load geojson from "data.geo"        # ✓ Valid: string path
entropy get Sk                      # ✓ Valid: scalar in [0,1]
distance = 1.2 cm                   # ✓ Valid: unit specified
error = position - actual           # ✓ Valid: dimension matching
position = "hello"                  # ✗ Error: type mismatch
```

## Browser APIs Used

### File Handling
- `File API` for user uploads
- `FileReader API` for file parsing
- `Fetch API` for loading example files

### Visualization
- `Chart.js` for histograms, timelines, scatter plots
- `Leaflet.js` for map visualization
- `D3.js` for complex visualizations (optional)

### Code Editing
- `Monaco Editor` for syntax highlighting
- `Monaco Language Server` for CyneScript language support

### Real-Time Execution
- `Web Workers` for background computation (large datasets)
- `async/await` for non-blocking operations
- `EventEmitter` for progress updates

## Example: Level 3 Execution

```javascript
// 1. User selects Level 3 script
fileManager.loadScript('level_3_virtual_positioning');

// 2. User uploads data
const file = document.getElementById('file-input').files[0];
const data = fileManager.uploadFile(file);

// 3. User clicks Run
const code = codeEditor.editor.getValue();
const { errors, warnings } = codeEditor.checkDimensions(code);

if (errors.length === 0) {
  // 4. Execute script
  const runtime = new CyneRuntime(data);
  runtime.on('progress', (msg) => outputPanel.streamLog(msg));
  
  const results = await runtime.execute(code);
  
  // 5. Display results
  outputPanel.displayResults(results);
  outputPanel.renderChart('accuracy', 'histogram', results.accuracy_dist);
  outputPanel.renderChart('error', 'line', results.error_timeline);
  
  // 6. Export options
  outputPanel.enableExport(['csv', 'json', 'png']);
}
```

## Data Flow: Level 5 Phase-Lock Example

```
device_a_trajectory.geojson  device_b_trajectory.geojson
        ↓                                    ↓
    [Parse Features]                [Parse Features]
        ↓                                    ↓
    [Extract Points & Timestamps] (sync to 50ms tolerance)
        ↓                                    ↓
    [Compute Entropy at Each Point]        ↓
        ↓                          ─────────┴────────
        ├─ [Coherence Establishment]
        │  - Measure entropy difference
        │  - Validate convergence
        │  - Establish phase-lock at Omega_i
        │
        ├─ [Independent Positioning]
        │  - Triangulate position separately
        │  - Compute error vs ground truth
        │
        ├─ [Coherent Positioning]
        │  - Resolve positions in phase-lock
        │  - Measure observation simultaneity
        │  - Validate synchronization lag
        │
        └─ [Comparison & Validation]
           - Compute improvement factor
           - Display coherent vs independent
           - Export detailed results

Output:
├─ Convergence rate: 96% ✓
├─ Independent error: 1.2 cm
├─ Coherent error: 0.6 cm (2.0× improvement)
└─ Phase-lock: SUCCESS ✓
```

## Performance Considerations

### Large Datasets (>10,000 points)

```javascript
// Use Web Worker for background processing
const worker = new Worker('/workers/positioning-worker.js');

worker.postMessage({
  command: 'process',
  data: largeTrajectory,
  script: compiledAST
});

worker.onmessage = (event) => {
  const results = event.data;
  outputPanel.displayResults(results);
};
```

### Real-Time Updates

```javascript
// Stream results as they're computed
runtime.on('checkpoint', (checkpoint) => {
  outputPanel.streamLog(
    `Processed point ${checkpoint.index}/${checkpoint.total}`
  );
  
  // Update visualization incrementally
  if (checkpoint.index % 10 === 0) {
    outputPanel.renderChart('current_error', checkpoint.latest_error);
  }
});
```

## Testing Strategy

### Unit Tests
```javascript
// Test lexer
assert(lexer.tokenize('load geojson') === ['LOAD', 'GEOJSON']);

// Test parser
const ast = parser.parse(tokenStream);
assert(ast.type === 'Program');

// Test runtime
const result = runtime.evaluate(ast, testData);
assert(result.accuracy < 0.05);  // < 5 cm error
```

### Integration Tests
```javascript
// Load Level 1 script, test with example data
const level1 = await fetchScript('level_1_load_and_display');
const ast = parser.parse(lexer.tokenize(level1));
const results = runtime.evaluate(ast, exampleData);
assert(results.point_count === 25);
```

## Deployment

### Development
```bash
npm run dev
# → Runs on localhost:3000
# → Hot reload enabled
# → Full source maps
```

### Production
```bash
npm run build
npm run serve
# → Minified assets
# → Service worker for offline support
# → CDN-friendly structure
```

### Chromebook Compatibility
- No special installation required
- Works in any modern browser (Chrome, Firefox, Safari)
- Local file uploads (no server transfer required)
- WebGL for 3D visualization (if needed)

## Future Extensions

1. **Collaborative Mode**: Multiple users editing/running same script
2. **Real Device Integration**: Connect actual phones/smartwatches via WebSocket
3. **Advanced Visualization**: 3D trajectory rendering, live map updates
4. **Hypothesis Testing**: Add statistical validation framework
5. **Custom Atmosphere Models**: Allow users to define custom models
6. **Physics Simulation**: Integrate real atmospheric data from weather APIs
