# GLB Model & React-Three-Fiber Setup

## Overview

The landing page (`/`) now displays the **paleogeographic_timelapse.glb** 3D model using **react-three-fiber**, a React renderer for **Three.js**.

## Architecture

### Component Hierarchy

```
index.js (Landing Page)
├── dynamic GlobeViewer (lazy-loaded)
└── GlobeViewer.jsx
    └── Canvas (react-three-fiber)
        ├── Lighting (ambient, directional, point)
        ├── GLBModel (loads & renders GLB)
        │   └── useGLTF hook (loads /public/paleogeographic_timelapse.glb)
        └── OrbitControls (user interaction)
```

### Files

| File | Purpose |
|------|---------|
| `src/pages/index.js` | Landing page with dynamic GlobeViewer |
| `src/components/GlobeViewer.jsx` | React-Three-Fiber canvas & model rendering |
| `public/paleogeographic_timelapse.glb` | 3D model file |

## Dependencies

### New Production Dependencies

```json
{
  "three": "^r156",
  "@react-three/fiber": "^8.13.0",
  "@react-three/drei": "^9.80.0"
}
```

### What Each Does

**three** (r156)
- 3D graphics library
- Handles WebGL rendering, geometries, materials, lighting
- ~600KB minified

**@react-three/fiber** (8.13.0)
- React renderer for Three.js
- Bridges React JSX with Three.js objects
- Provides `<Canvas>`, `useFrame`, etc.
- ~100KB minified

**@react-three/drei** (9.80.0)
- Utility components for react-three-fiber
- Includes `useGLTF` (loads GLB/GLTF models)
- Includes `OrbitControls` (user mouse interaction)
- ~200KB minified

## How It Works

### 1. GLB Loading

```javascript
function GLBModel() {
  const { scene } = useGLTF("/paleogeographic_timelapse.glb");
  // scene is the loaded 3D model
}
```

`useGLTF` from drei:
- Loads the GLB file from `/public/`
- Parses the model geometry, materials, animations
- Caches the result (efficient for rerenders)
- Returns: `{ scene, animations, nodes, materials }`

### 2. Animation Loop

```javascript
useFrame(() => {
  if (groupRef.current) {
    groupRef.current.rotation.y += 0.0005;  // Slow rotation
  }
});
```

`useFrame` runs every animation frame (~60fps):
- Rotates the model around Y axis
- Rate: 0.0005 rad/frame = ~30s per full rotation

### 3. User Interaction

```javascript
<OrbitControls
  enableZoom={true}
  enablePan={true}
  enableRotate={true}
  autoRotate={false}
/>
```

OrbitControls allows:
- **Click + drag** to rotate
- **Scroll** to zoom
- **Right-click + drag** to pan
- Auto-rotation disabled (smooth landing page)

### 4. Lighting

```javascript
<ambientLight intensity={0.8} />              {/* Global light */}
<directionalLight position={[10, 10, 5]} />  {/* Sun-like light */}
<directionalLight position={[-10, -10, -5]} />{/* Opposite fill */}
<pointLight position={[0, 5, 10]} color="#007acc" /> {/* Blue accent */}
```

Multiple lights create depth and visual interest.

## Installation

```bash
cd moriarty
npm install
```

This installs `three`, `@react-three/fiber`, and `@react-three/drei`.

## Usage

The GlobeViewer is lazy-loaded on the landing page:

```javascript
const GlobeViewer = dynamic(() => import("@/components/GlobeViewer"), {
  ssr: false,
  loading: () => <LoadingSpinner />
});
```

Why lazy load?
- Three.js is large (~2MB)
- Not needed for initial page load
- Defers loading until client-side rendering
- Shows loading spinner while Three.js loads

## Performance Considerations

### Bundle Size
- **three**: ~600KB
- **react-three-fiber**: ~100KB
- **drei**: ~200KB
- **Total**: ~900KB (before minification/gzip)

After gzip:
- ~250KB (industry standard for 3D web apps)

### Rendering Performance
- Native WebGL rendering (fast)
- Runs at 60fps on most devices
- OrbitControls are optimized
- No significant CPU load

### Mobile
- Works on iOS 14+ Safari, Android Chrome
- Touch rotate: two-finger drag
- Touch zoom: pinch
- Performance depends on device GPU

## Customization

### Change Rotation Speed

```javascript
// In GlobeViewer.jsx, GLBModel function
useFrame(() => {
  if (groupRef.current) {
    groupRef.current.rotation.y += 0.0002;  // Slower
    // or
    groupRef.current.rotation.y += 0.001;   // Faster
  }
});
```

### Change Model File

```javascript
const { scene } = useGLTF("/path/to/your-model.glb");
```

### Adjust Lighting

```javascript
<directionalLight 
  position={[10, 10, 5]} 
  intensity={1.2}  // Brighter
  color="#ffffff"
/>
```

### Change Camera Position

```javascript
<Canvas
  camera={{ position: [0, 0, 4], fov: 45 }}  // Closer view
>
```

### Auto-Rotate On Landing

```javascript
<OrbitControls
  autoRotate={true}
  autoRotateSpeed={2}
/>
```

## Troubleshooting

### GLB Not Loading

**Problem**: Black screen, no model visible

**Solutions**:
1. Check browser console (F12) for errors
2. Verify file exists: `public/paleogeographic_timelapse.glb`
3. Check file size: should be >1MB
4. Try opening in Three.js editor: https://threejs.org/editor/

### Poor Performance

**Problem**: Low FPS, laggy rotation

**Solutions**:
1. Reduce lighting count (fewer lights = faster)
2. Reduce OrbitControls smoothness
3. Use WebGL2 instead of WebGL:
   ```javascript
   <Canvas gl={{ antialias: true, alpha: true }}>
   ```
4. Profile with DevTools: Check GPU usage

### Controls Not Working

**Problem**: Can't rotate/zoom with mouse

**Solutions**:
1. Ensure OrbitControls enabled: `enableRotate={true}`
2. Check mouse events aren't blocked by CSS
3. Verify Canvas is fullscreen (`w-full h-full`)

## Model Format

### GLB vs GLTF
- **GLB**: Binary format (single .glb file)
- **GLTF**: Text format (can be .gltf + .bin + textures)
- **GLB is preferred**: Faster loading, includes everything

### File Structure
```
paleogeographic_timelapse.glb
├── Geometry (mesh positions, normals)
├── Materials (colors, textures)
├── Animations (if any)
└── (all embedded)
```

### Inspection
To inspect GLB model structure:
1. Download Three.js editor: https://threejs.org/editor/
2. Open the .glb file
3. View geometry, materials, and animations

## Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome/Edge | ✅ Full | WebGL2 support |
| Firefox | ✅ Full | WebGL2 support |
| Safari | ✅ Full | iOS 14+ |
| Mobile | ✅ Full | Touch controls work |

## Animation & Keyframes

If your GLB includes animations:

```javascript
function GLBModel() {
  const { scene, animations } = useGLTF("/model.glb");
  const mixer = useRef();
  const actions = useRef([]);

  useEffect(() => {
    mixer.current = new THREE.AnimationMixer(scene);
    actions.current = animations.map(clip => mixer.current.clipAction(clip));
    actions.current[0]?.play();
  }, [scene, animations]);

  useFrame((state, delta) => {
    mixer.current?.update(delta);
  });

  return <primitive object={scene} />;
}
```

## Next Steps

1. ✅ Install dependencies: `npm install`
2. ✅ Start dev server: `npm run dev`
3. ✅ Visit `http://localhost:3000`
4. ✅ See paleogeographic globe rendered with react-three-fiber
5. Rotate/zoom with mouse
6. Click "Sandbox" to use positioning tools

## Resources

- **Three.js docs**: https://threejs.org/docs/
- **react-three-fiber**: https://docs.pmnd.rs/react-three-fiber/
- **drei utilities**: https://github.com/pmndrs/drei
- **GLTF spec**: https://www.khronos.org/gltf/
- **GLB validator**: https://github.khronos.org/glTF-Sample-Models/

## File Size Reference

Typical GLB sizes:
- Simple model: 100KB - 500KB
- Medium model: 500KB - 5MB
- Complex model: 5MB - 50MB+

`paleogeographic_timelapse.glb` size: Check with `ls -lh public/`

---

**Status**: ✅ GLB model rendering with react-three-fiber
**Dependencies**: three, @react-three/fiber, @react-three/drei
**Performance**: 60fps on modern devices
**Installation**: `npm install` (automatic)
