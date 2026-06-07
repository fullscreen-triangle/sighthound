# S-Entropy Positioning Tool - Quick Start

## 100% Frontend Application

This is a pure client-side web application. **No backend required.** All computations happen in the browser.

---

## Installation (Choose One)

### Option 1: Python HTTP Server (Simplest)

```bash
cd c:\Users\kunda\Documents\physics\sighthound
python -m http.server 8000
```

Then open: `http://localhost:8000/templates/`

### Option 2: Node.js HTTP Server

```bash
npm install
npm run serve
```

Then open: `http://localhost:8000/templates/`

### Option 3: Direct Browser (No Server)

Simply open in your browser:
```
file:///C:/Users/kunda/Documents/physics/sighthound/templates/index.html
```

(Some features may be limited with file:// protocol)

---

## Directory Structure

```
sighthound/
├── package.json                          # npm config (corrected versions)
├── templates/
│   ├── index.html                        # Home page
│   ├── shortest_path.html                # Demo 1
│   ├── reachable_region.html             # Demo 2
│   └── satellite_visibility.html         # Demo 3
├── static/
│   ├── style.css                         # Styling
│   └── app.js                            # Utilities
├── public/
│   └── soyuz_apollo.glb                  # 3D Soyuz-Apollo model
├── QUICKSTART.md                         # This file
├── FRONTEND_TOOL_README.md              # Full documentation
└── WEB_TOOL_README.md                   # Old docs (archived)
```

---

## Three Interactive Demos

### 1. **Shortest Path Algorithm** 📍
- Find optimal routes using S-entropy coordinates
- Interactive Leaflet map
- Real-time distance calculation
- **Launch**: `shortest_path.html`

### 2. **Reachable Region Calculator** 🚶
- Calculate reachable area from a point in time
- Great-circle distance on Earth
- Time-based mobility analysis
- **Launch**: `reachable_region.html`

### 3. **Satellite Visibility Globe** 🛰️
- 3D interactive globe with Three.js
- Animated Soyuz-Apollo docking model
- Elevation angle calculations
- **Launch**: `satellite_visibility.html`

---

## Frontend Stack

- **HTML5** + **CSS3** + **Vanilla JavaScript**
- **Leaflet.js** - Interactive maps
- **Three.js** - 3D globe visualization
- **OpenStreetMap** - Free tile layer
- **No frameworks** - Pure frontend

---

## Package.json - Corrected Versions

```json
{
  "dependencies": {
    "three": "^r128",
    "leaflet": "^1.9.4"
  },
  "devDependencies": {
    "vite": "^5.0.8",
    "@vitejs/plugin-basic-ssl": "^1.0.1"
  }
}
```

**Note**: Libraries are loaded from CDN, npm packages are optional for development only.

---

## Core Algorithms (100% JavaScript)

✅ **Haversine Distance** - Great-circle distance on Earth  
✅ **S-Entropy Metric** - Geographic + terrain + infrastructure components  
✅ **Elevation Angle** - Line-of-sight to satellites  
✅ **Reachable Region** - Geodesic circles on sphere  

---

## Usage Examples

### Shortest Path: NYC → Paris
1. Open `shortest_path.html`
2. Start: `40.7128°N, 74.0060°W`
3. Goal: `48.8566°N, 2.3522°E`
4. Click "Compute Shortest Path"
5. **Result**: ~5,570 km with interactive map

### Reachable Region: 4-Hour Walk from NYC
1. Open `reachable_region.html`
2. Center: `40.7128°N, 74.0060°W`
3. Time: `4` hours, Speed: `5` km/h
4. Click "Compute Reachable Region"
5. **Result**: 20 km radius circle, ~1,256 km² area

### Satellites from NYC
1. Open `satellite_visibility.html`
2. Location: `40.7128°N, 74.0060°W`
3. Min Elevation: `10°`
4. Click "Compute Visibility"
5. **Result**: 2-4 visible satellites on 3D globe

---

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome  | ✅ | Recommended |
| Firefox | ✅ | Full support |
| Safari  | ✅ | Full support |
| Edge    | ✅ | Full support |
| IE 11   | ❌ | No WebGL |

---

## Troubleshooting

**Port 8000 already in use?**
```bash
python -m http.server 8001
# Then visit http://localhost:8001/templates/
```

**3D Globe not rendering?**
- Check browser console (F12)
- Ensure WebGL is enabled
- Try a different browser

**Models not loading?**
- Ensure `public/soyuz_apollo.glb` exists
- App falls back to sphere markers if model missing

---

## No Backend Needed!

This tool is **100% frontend**:
- ✅ No Python Flask
- ✅ No API calls
- ✅ No database
- ✅ All computation in browser
- ✅ Works offline (after first load)

---

## For Development

Install dependencies and build:
```bash
npm install
npm run dev      # Hot reload dev server
npm run build    # Production build
npm run preview  # Preview production
```

---

## Full Documentation

See `FRONTEND_TOOL_README.md` for detailed documentation covering:
- Core algorithms
- API design
- Customization
- Future enhancements

---

**Version**: 2.0 (Frontend-Only)  
**Last Updated**: June 2026
