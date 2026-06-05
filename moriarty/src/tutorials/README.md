# CyneScript Tutorial Progression

A progressive tutorial system for learning Phase-Locked positioning through executable scripts. Each level builds on the previous, introducing concepts and complexity incrementally.

## Overview

The tutorials teach:
1. **Data loading and visualization** (Level 1)
2. **S-entropy computation** (Level 2)
3. **Virtual satellite positioning** (Level 3)
4. **Dynamic tracking** (Level 4)
5. **Phase-Locked coherence** (Level 5)

## Structure

```
tutorials/
├── level_1_load_and_display.cynes          # Load KML/GeoJSON files
├── level_2_compute_entropy.cynes           # Calculate S-entropy at positions
├── level_3_virtual_positioning.cynes       # Virtual satellite triangulation
├── level_4_dynamic_tracking.cynes          # Real-time tracking with state estimation
├── level_5_phase_locked.cynes              # Phase-lock coherence between devices
├── example_data_munich_run.geojson         # Sample trajectory data
├── example_data_two_device_sync.geojson    # Two simultaneous trajectories
└── README.md                               # This file
```

## Quick Start

### 1. Upload Your Data

Supported formats:
- **GeoJSON**: Feature collections with Point geometries and timestamps
- **KML**: Google Earth format with placemarks and timestamps

Example GeoJSON structure:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "timestamp": "2025-03-14T10:00:00Z",
        "elevation": 520,
        "id": 1
      },
      "geometry": {
        "type": "Point",
        "coordinates": [11.357, 48.183, 520]
      }
    }
  ]
}
```

### 2. Select Tutorial Level

Click **Load Script** and choose a level:

- **Level 1**: New to the framework? Start here.
- **Level 2**: Understand S-entropy representation
- **Level 3**: Learn virtual satellite positioning
- **Level 4**: Follow real-time tracking
- **Level 5**: Explore phase-lock synchronization

### 3. Execute and Explore

1. Upload your GeoJSON/KML file
2. Click **Run** to execute the script
3. View **Results** in the Output column
4. Export data or visualizations

## Level-by-Level Guide

### Level 1: Load and Display

**What you'll learn:**
- Load geographic data (KML/GeoJSON)
- Display point collections
- Extract metadata
- Count and inspect data

**Example usage:**
```cynes
load geojson from "my_trajectory.geojson"
show first 5 points with coordinates
count positions
```

**Expected output:**
```
Loaded 25 positions from my_trajectory.geojson
Metadata:
  - Start: 2025-03-14T10:00:00Z
  - End: 2025-03-14T10:24:00Z
  - Sampling rate: 1 Hz

First 5 points:
  1. (11.357°, 48.183°, 520 m)
  2. (11.358°, 48.185°, 522 m)
  3. (11.360°, 48.187°, 525 m)
  ...
```

### Level 2: Compute S-Entropy

**What you'll learn:**
- Understand atmospheric partition state
- Compute S-entropy coordinates (Sk, St, Se)
- Analyze entropy distribution
- Compare entropy across positions

**Key concept:**
S-entropy encodes atmospheric thermodynamic state:
- **Sk**: Kinetic/vibrational entropy (air density, composition)
- **St**: Temporal/velocity entropy (wind, temperature)
- **Se**: Energy entropy (pressure, altitude)

**Example usage:**
```cynes
load geojson from "trajectory.geojson"
atmosphere initialize model="standard"
for each point in trajectory:
    atmosphere compute at point.lat point.lon point.altitude
    entropy get Sk St Se
    entropy store point.id
end
entropy plot histogram Sk
```

**Expected output:**
```
Computed entropy for 25 positions:
  Mean Sk: 0.587 ± 0.045
  Mean St: 0.419 ± 0.078
  Mean Se: 0.937 ± 0.032
  
Entropy variation along trajectory: Visible in histograms
```

### Level 3: Virtual Satellite Positioning

**What you'll learn:**
- Create virtual satellite constellation
- Compute categorical distance
- Triangulate position from S-entropy
- Validate against ground truth
- Measure positioning accuracy

**Key concept:**
Virtual satellites are derived from Earth's gravitational partition structure. Position is determined by finding the nearest satellites in S-entropy space.

**Example usage:**
```cynes
load geojson from "trajectory.geojson"
satellite derive from Earth partition
satellite create count=1000
for each point in trajectory:
    entropy local = point.entropy
    satellites nearest = select nearest 50 satellites
    position estimated = triangulate with satellites.nearest
    error = distance(estimated, point.actual)
    results store point.id estimated error
end
show results mean_error std_error
```

**Expected output:**
```
Positioning Results:
  Total points: 25
  Mean error: 1.2 cm (192× better than GPS)
  Std dev: 0.8 cm
  Max error: 3.5 cm
  
Accuracy achieved: 1cm ✓
```

### Level 4: Dynamic Tracking

**What you'll learn:**
- Process time-series trajectory data
- Estimate velocity from position changes
- Use state estimation (Kalman-like filtering)
- Detect anomalies in tracking
- Analyze tracking quality over time

**Key concept:**
Dynamic tracking filters noisy position measurements to produce smooth, accurate state estimates including position and velocity.

**Example usage:**
```cynes
load geojson from "trajectory.geojson" with_timestamps=true
tracker initialize state=(x, y, z, vx, vy, vz)
for each point in trajectory:
    entropy local = atmosphere.compute_at(point)
    position = triangulate_position(entropy)
    tracker predict timestamp
    tracker update position
    state estimated = tracker.get_state()
    tracking store epoch state.estimated velocity
end
tracking analyze rmse horizontal vertical
show velocity profile
```

**Expected output:**
```
Tracking Performance:
  RMSE horizontal: 0.9 cm
  RMSE vertical: 1.4 cm
  
Velocity Analysis:
  Mean velocity: 4.2 m/s (running pace)
  Max velocity: 6.1 m/s
  
High-frequency content detected at 3.7 Hz (footfall cadence)
```

### Level 5: Phase-Locked Coherence

**What you'll learn:**
- Establish phase-lock between two devices
- Validate coherence convergence
- Measure observation simultaneity
- Compare coherent vs independent positioning
- Demonstrate improved accuracy through coherence

**Key concept:**
Phase-locked devices share a synchronized observation of position changes. When two devices are coherent, they observe state transitions instantaneously without transmission delay, improving positioning accuracy.

**Example usage:**
```cynes
load geojson from "device_a_trajectory.geojson" as device_a
load geojson from "device_b_trajectory.geojson" as device_b

coherence establish device_a device_b
for epoch in 0 to 100:
    entropy_a = atmosphere.compute_at(device_a[epoch])
    entropy_b = atmosphere.compute_at(device_b[epoch])
    entropy_delta = entropy.distance(entropy_a, entropy_b)
    coherence log epoch entropy_delta
end

for epoch in 100 to end:
    position_a_coherent = coherence.resolve(entropy_a)
    position_b_coherent = coherence.resolve(entropy_b)
    coherent store epoch position_a_coherent position_b_coherent
end

show comparison coherent vs independent
show improvement_factor
```

**Expected output:**
```
Phase-Lock Convergence:
  Convergence rate: 96% (95% target ✓)
  Epochs to convergence: 12
  
Coherent vs Independent Positioning:
  Device A - Independent: 1.2 cm error
  Device A - Coherent: 0.6 cm error (2.0× improvement)
  Device B - Independent: 1.3 cm error
  Device B - Coherent: 0.65 cm error (2.0× improvement)
  
Phase-lock validation: SUCCESS ✓
```

## Data Format Reference

### GeoJSON Requirements

```json
{
  "type": "FeatureCollection",
  "properties": {
    "name": "Trajectory name",
    "start_time": "ISO 8601 timestamp",
    "end_time": "ISO 8601 timestamp",
    "total_points": 25
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
        "coordinates": [longitude, latitude, elevation]
      }
    }
  ]
}
```

**Required fields:**
- `features[].properties.timestamp`: ISO 8601 format for time-series analysis
- `features[].geometry.coordinates`: [lon, lat, elevation]
- `features[].properties.id`: Unique identifier

**Optional fields:**
- `features[].properties.elevation`: Height above sea level (meters)
- `features[].properties.gps_accuracy`: GPS horizontal accuracy (meters)
- `features[].properties.actual`: Ground truth position for validation

### KML Format

Standard Google Earth KML with `<Placemark>` elements:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Trajectory</name>
    <Placemark>
      <name>Point 1</name>
      <TimeStamp>
        <when>2025-03-14T10:00:00Z</when>
      </TimeStamp>
      <Point>
        <coordinates>11.357,48.183,520</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>
```

## Example Scenarios

### Scenario 1: Validate Outdoor Positioning Accuracy

1. Collect GPS trajectory (Level 1)
2. Compute entropy distribution (Level 2)
3. Run virtual satellite positioning (Level 3)
4. Compare to known GPS ground truth
5. Validate 1cm accuracy target

**Files needed:**
- `example_data_munich_run.geojson` (provided)

**Steps:**
```
1. Click "Load Example Data" → munich_run
2. Select Level 3 script
3. Click "Run"
4. View accuracy statistics
5. Export results to CSV
```

### Scenario 2: Demonstrate Phase-Lock Synchronization

1. Load two simultaneous device trajectories (Level 1)
2. Establish phase-lock coherence (Level 5)
3. Measure coherence convergence
4. Show 2× accuracy improvement from synchronization

**Files needed:**
- Two GeoJSON files with synchronized timestamps
- Example: `example_data_two_device_sync.geojson` (coming soon)

### Scenario 3: High-Frequency Tracking

1. Load trajectory with >100 Hz sampling
2. Compute tracking state estimates (Level 4)
3. Analyze velocity and acceleration
4. Detect periodic motion (footfalls, rotation)

**Files needed:**
- High-frequency trajectory (>100 points/second)

## Browser Sandbox Features

### File Column (Left)
- Upload KML/GeoJSON files
- Browse example data
- Create new scripts
- Edit existing scripts

### Code Column (Middle)
- Syntax highlighting for CyneScript
- Dimensional type checking (real-time)
- Error/warning display
- Auto-complete for keywords

### Output Column (Right)
- Results and statistics
- Real-time progress
- Visualization (plots, maps)
- Export options (CSV, JSON, PNG)

## Advanced Usage

### Custom Atmospheric Model

```cynes
atmosphere initialize model="custom"
atmosphere set model.temperature_lapse=-6.5 K/km
atmosphere set model.humidity_gradient=0.004 1/km
```

### High-Density Satellite Constellation

For sub-centimeter accuracy:
```cynes
satellite create count=100000 altitude=26560km
```

### Parallel Processing

For large trajectories (>10,000 points):
```cynes
configuration set parallel_workers=8
for parallel each point in trajectory:
    ...
end
```

## Troubleshooting

### "Failed to load geojson"
- Ensure coordinates are [longitude, latitude, elevation]
- Check timestamp format is ISO 8601
- Validate JSON syntax

### "Convergence failed"
- Increase iteration count
- Check input entropy values are in [0, 1]
- Verify atmospheric model parameters

### "Positions outside Earth bounds"
- Check coordinate order (lon, lat not lat, lon)
- Validate elevation values

## Next Steps

After completing Level 5, you're ready for:
- **Implementation**: Deploy actual phase-locked devices
- **Research**: Extend framework with new protocols
- **Optimization**: Tune positioning algorithms for your environment
- **Integration**: Combine with other sensing modalities

## Support

For questions or issues:
1. Check Level 1-5 tutorials for similar examples
2. Review example data formats
3. Examine error messages for hints
4. Consult CyneScript language reference
