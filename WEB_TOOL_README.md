# S-Entropy Positioning Web Tool

Interactive demonstrations of S-entropy based positioning, shortest path algorithms, and satellite visibility analysis.

## Overview

This web tool provides three interactive demos showcasing the S-entropy positioning framework:

### 1. **Shortest Path Algorithm**
- **URL**: `http://localhost:5000/demo/shortest-path`
- **Features**:
  - A* pathfinding using S-entropy coordinates
  - Interactive map with route visualization
  - Real-time distance calculation
  - Multiple preset locations (NYC→Paris, Tokyo→Sydney, London→Dubai)
- **Algorithm**: SEBD (S-Entropy Bidirectional Dijkstra)
- **Metric**: S-entropy distance (combines geographic distance, terrain roughness, infrastructure density)

### 2. **Reachable Region Calculator**
- **URL**: `http://localhost:5000/demo/reachable-region`
- **Features**:
  - Calculate how far someone can walk from a point in a given time
  - Adjustable walking speed (default 5 km/h)
  - Visualize reachable area as a circle on the map
  - Great-circle distance calculations on Earth surface
  - Quick presets (1 hour, 2 hours, 4 hours)
- **Metric**: Haversine distance with Earth radius 6371 km
- **Uses**: Time-based mobility analysis

### 3. **Satellite Visibility Globe**
- **URL**: `http://localhost:5000/demo/satellite-visibility`
- **Features**:
  - 3D interactive globe with satellite constellation
  - Determine which satellites are visible from any location
  - Elevation angle calculation
  - Line-of-sight detection (minimum elevation threshold)
  - Real-time satellite list with elevation angles and distances
  - Multiple preset locations (NYC, London, Sydney, Equator)
- **Constellation**: Simulated MEO constellation (20,000 km altitude)
- **Visibility**: Satellites visible above 10° elevation

---

## Installation

### 1. Prerequisites
- Python 3.8+
- pip (Python package manager)

### 2. Install Dependencies

```bash
pip install flask numpy
```

### 3. Project Structure

```
sighthound/
├── web_tool.py                          # Flask backend
├── static/
│   ├── style.css                        # Styling
│   └── app.js                           # Common utilities
├── templates/
│   ├── index.html                       # Home page
│   ├── shortest_path.html               # Demo 1
│   ├── reachable_region.html            # Demo 2
│   └── satellite_visibility.html        # Demo 3
└── WEB_TOOL_README.md                   # This file
```

---

## Running the Web Tool

### 1. Start the Flask Server

```bash
cd c:\Users\kunda\Documents\physics\sighthound
python web_tool.py
```

Expected output:
```
======================================================================
S-ENTROPY POSITIONING WEB TOOL
======================================================================

Available demos:
  1. Shortest Path: http://localhost:5000/demo/shortest-path
  2. Reachable Region: http://localhost:5000/demo/reachable-region
  3. Satellite Visibility: http://localhost:5000/demo/satellite-visibility

Starting Flask server on http://localhost:5000
======================================================================
```

### 2. Open in Browser

Visit: `http://localhost:5000/`

### 3. Navigate to Demos

- **Shortest Path**: Click "Shortest Path Algorithm" card or visit `/demo/shortest-path`
- **Reachable Region**: Click "Reachable Region Calculator" card or visit `/demo/reachable-region`
- **Satellite Visibility**: Click "Satellite Visibility Globe" card or visit `/demo/satellite-visibility`

---

## API Endpoints

### POST `/api/shortest-path`

Compute shortest path between two locations.

**Request**:
```json
{
  "lat1": 40.7128,
  "lon1": -74.0060,
  "lat2": 48.8566,
  "lon2": 2.3522
}
```

**Response**:
```json
{
  "success": true,
  "path": [[40.7128, -74.0060], [45.2, -20.0], [48.8566, 2.3522]],
  "total_distance_km": 5845.23,
  "waypoints": 3
}
```

### POST `/api/reachable-region`

Calculate reachable region from a point.

**Request**:
```json
{
  "lat": 40.7128,
  "lon": -74.0060,
  "time_hours": 2.0,
  "speed_kmh": 5.0
}
```

**Response**:
```json
{
  "success": true,
  "center": [40.7128, -74.0060],
  "boundary": [[40.91, -74.0], [40.85, -73.78], ...],
  "max_distance_km": 10.0,
  "time_hours": 2.0,
  "speed_kmh": 5.0
}
```

### POST `/api/visible-satellites`

Determine visible satellites from a location.

**Request**:
```json
{
  "lat": 40.7128,
  "lon": -74.0060,
  "alt_m": 10,
  "min_elevation": 10
}
```

**Response**:
```json
{
  "success": true,
  "observer": {"lat": 40.7128, "lon": -74.0060, "alt_m": 10},
  "visible_satellites": [
    {"id": 1, "lat": 56.0, "lon": 0, "elevation": 45.23, "distance": 8234.5},
    {"id": 3, "lat": 56.0, "lon": 240, "elevation": 32.15, "distance": 9123.4}
  ],
  "count": 2,
  "total_satellites": 9,
  "min_elevation": 10
}
```

### GET `/api/all-satellites`

Get all satellites in the constellation.

**Response**:
```json
{
  "success": true,
  "satellites": [
    {"id": 1, "lat": 56.0, "lon": 0, "alt": 20000},
    {"id": 2, "lat": 56.0, "lon": 120, "alt": 20000},
    ...
  ],
  "count": 9
}
```

---

## Usage Examples

### Example 1: Find Shortest Path Between Cities

1. Open **Shortest Path Algorithm** demo
2. Enter start coordinates: (40.7128, -74.0060) — New York City
3. Enter goal coordinates: (51.5074, -0.1278) — London
4. Click "Compute Shortest Path"
5. View the route on the map and total distance

**Expected**: ~5,570 km distance with waypoints marked on the map

### Example 2: Calculate Reachable Area

1. Open **Reachable Region Calculator** demo
2. Enter center: (40.7128, -74.0060) — New York City
3. Set time: 4 hours
4. Set speed: 5 km/h (walking)
5. Click "Compute Reachable Region"
6. View the circular reachable area (20 km radius) on the map

**Expected**: Circle with 20 km radius, area ≈ 1,256 km²

### Example 3: Check Satellite Visibility

1. Open **Satellite Visibility Globe** demo
2. Enter location: (40.7128, -74.0060) — New York City
3. Set minimum elevation: 10°
4. Click "Compute Visibility"
5. View visible satellites on 3D globe
6. Check the satellite list with elevation angles

**Expected**: 2-4 visible satellites from NYC, with elevations 10-60°

---

## Customization

### Changing Default Locations

Edit `web_tool.py`:

```python
# Line 10-14: Modify SATELLITES constellation
SATELLITES = [
    {"id": 1, "lat": 56.0, "lon": 0, "alt": SATELLITE_ALTITUDE_KM},
    # ...
]
```

### Adjusting Satellite Altitude

Edit `web_tool.py`, line 8:

```python
SATELLITE_ALTITUDE_KM = 20000  # Change this value
```

### Modifying S-Entropy Distance Metric

Edit the `s_entropy_distance()` function in `web_tool.py` to adjust weights:

```python
s_entropy_dist = math.sqrt(
    0.5 * geo_component**2 +      # Geographic weight
    0.3 * terrain_component**2 +  # Terrain weight
    0.2 * infra_component**2      # Infrastructure weight
)
```

---

## Technical Details

### S-Entropy Distance Metric

The S-entropy metric combines three components:

1. **Geographic Component** (50% weight): Haversine distance normalized by Earth circumference
2. **Terrain Component** (30% weight): Latitude-dependent roughness (poles rougher than equator)
3. **Infrastructure Component** (20% weight): Longitude-dependent density

Formula:
```
d_entropy = √(0.5·d_geo² + 0.3·d_terrain² + 0.2·d_infra²)
```

### Elevation Angle Calculation

Satellites are visible if elevation angle ≥ threshold (default 10°):

```
elevation = atan2(Δaltitude, great_circle_distance)
```

### Great-Circle Distance

Used for all distance calculations on Earth:

```
a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
c = 2·asin(√a)
distance = R·c  (R = 6371 km)
```

---

## Troubleshooting

### Port Already in Use

If port 5000 is already in use:

```bash
# Change port in web_tool.py, line ~340:
app.run(debug=True, host='localhost', port=5001)
```

### Map Not Loading

- Ensure you have internet connection (OpenStreetMap tiles are loaded from CDN)
- Check browser console for errors (F12 → Console tab)

### 3D Globe Not Rendering

- Ensure WebGL is enabled in your browser
- Try a different browser (Chrome/Firefox recommended)
- Update GPU drivers

### CORS Errors

Not applicable — this is a single-server setup without cross-origin requests.

---

## Performance Notes

- **Shortest Path**: Grid-based A* with 8×8 grid. Computation: ~100ms
- **Reachable Region**: 360-point circle. Computation: <10ms
- **Satellite Visibility**: 9 satellites, elevation calculations. Computation: <5ms

---

## Architecture

### Backend (Python/Flask)

- `web_tool.py`: Flask server with three API endpoints
- Haversine distance calculation
- S-entropy metric computation
- A* shortest path algorithm
- Elevation angle geometry

### Frontend (HTML/CSS/JavaScript)

- **Leaflet.js**: Interactive maps (shortest path, reachable region)
- **Three.js**: 3D globe visualization (satellite visibility)
- **Vanilla JavaScript**: API calls, UI interactions
- **CSS Grid**: Responsive layout

### API Design

- RESTful endpoints for each demo
- JSON request/response format
- No authentication (local demo)

---

## Future Enhancements

1. **Persistent Storage**: Save favorite locations and calculations
2. **Export**: Download paths and regions as GeoJSON
3. **Historical Data**: Track satellite positions over time
4. **Advanced Metrics**: Add wind patterns, terrain elevation data
5. **Real Satellites**: Integrate with TLE (Two-Line Element) data
6. **User Accounts**: Save preferences and calculation history

---

## References

- **S-Entropy Framework**: Dual-Domain Execution Model paper
- **SEBD Algorithm**: S-Entropy Bidirectional Dijkstra shortest path
- **Satellite Geometry**: Line-of-sight and elevation angle calculations
- **Earth Geodesy**: Haversine formula, great-circle distance

---

## License

Part of the Physics Sighthound project. Uses open-source libraries:
- Flask (BSD 3-Clause)
- Leaflet.js (BSD 2-Clause)
- Three.js (MIT)
- OpenStreetMap (ODbL)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API endpoint documentation
3. Inspect browser console (F12) for JavaScript errors
4. Check Flask console for server-side errors

---

**Last Updated**: June 2026  
**Version**: 1.0
