# Sighthound Project Setup

## Project Configuration

This is a **Next.js 13+ project** using:
- **JavaScript** with JSConfig (not TypeScript, but TypeScript support is available)
- **React 18.2**
- **Tailwind CSS** for styling
- **Next.js** as the framework

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `next` | ^13.2.1 | React framework |
| `react` | 18.2.0 | UI library |
| `react-dom` | 18.2.0 | DOM rendering |
| `lucide-react` | ^0.263.1 | Icon library (used in Sandbox) |
| `leaflet` | ^1.9.4 | Map visualization (used in MapVisualization) |
| `react-globe.gl` | ^2.32.0 | 3D globe visualization |
| `satellite.js` | ^1.53.0 | Satellite position calculations |
| `framer-motion` | ^10.0.1 | Animation library |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tailwindcss` | ^3.2.7 | CSS utility framework |
| `autoprefixer` | ^10.4.13 | CSS prefixing |
| `postcss` | ^8.4.21 | CSS processing |
| `eslint` | ^8.35.0 | Code linting |
| `eslint-config-next` | ^13.2.1 | Next.js ESLint config |
| `eslint-plugin-react` | ^7.32.0 | React linting rules |
| `@types/react` | ^18.2.0 | React TypeScript types |
| `@types/node` | ^18.16.0 | Node TypeScript types |
| `@types/leaflet` | ^1.9.8 | Leaflet TypeScript types |

## Installation

### 1. Install Dependencies

```bash
cd moriarty
npm install
```

This will install all packages from `package.json`.

### 2. Create Environment Variables

```bash
cp .env.local.example .env.local
```

Edit `.env.local` if needed (optional for development).

### 3. Verify Installation

```bash
# Check Next.js version
npx next --version

# Check Node version
node --version

# List installed packages
npm list
```

## Development

### Start Development Server

```bash
npm run dev
```

This starts Next.js on `http://localhost:3000`:
- **Landing page** (`/`): Animated 3D globe
- **Sandbox** (`/sandbox`): CyneScript IDE

### Build for Production

```bash
npm run build
npm start
```

### Linting

```bash
npm run lint
```

## Project Structure

```
moriarty/
├── src/
│   ├── pages/
│   │   ├── index.js          ← Landing page with 3D globe
│   │   ├── sandbox.jsx       ← CyneScript IDE
│   │   └── ...
│   ├── components/
│   │   ├── MapVisualization.jsx  ← Map component for sandbox
│   │   └── ...
│   ├── tutorials/            ← CyneScript tutorial files
│   │   ├── level_1_*.cynes
│   │   ├── example_data.geojson
│   │   └── ...
│   └── public/
│       └── ...
├── package.json              ← Dependencies (updated with lucide-react, leaflet)
├── tsconfig.json            ← TypeScript configuration (NEW)
├── jsconfig.json            ← JavaScript path aliases
└── .env.local.example       ← Environment variable template (NEW)
```

## Configuration Files

### tsconfig.json (NEW)
Provides TypeScript support even though the project uses JavaScript. This enables:
- Better IDE intellisense
- Type checking with `tsc --noEmit`
- Future TypeScript migration path

### jsconfig.json (EXISTING)
JavaScript path aliases - allows `@/` imports:
```javascript
import Component from "@/components/MyComponent";
```

## Troubleshooting

### Missing Dependencies

If you get errors about missing modules (e.g., "Cannot find module 'lucide-react'"):

```bash
npm install lucide-react leaflet @types/leaflet
```

### Map Not Showing

The `MapVisualization` component loads Leaflet dynamically from CDN. Ensure:
- Internet connection is available
- Browser allows CDN scripts
- Check browser console for errors

### Globe Not Rendering

The landing page uses WebGL. Ensure:
- Browser supports WebGL (check with `http://get.webgl.org/`)
- Hardware acceleration is enabled
- Try a different browser if it fails

### Next.js Errors

Clear build cache and reinstall:

```bash
rm -rf .next node_modules
npm install
npm run dev
```

## Scripts Reference

```bash
npm run dev        # Start development server (port 3000)
npm run build      # Build for production
npm start          # Start production server
npm run lint       # Run ESLint
```

## Browser Requirements

- **Modern browser** supporting:
  - ES2020+ JavaScript
  - WebGL (for 3D globe)
  - Leaflet maps (JavaScript)
  - React 18+

Tested on:
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+

## IDE Setup

### VS Code
Install recommended extensions:
- ES7+ React/Redux/React-Native snippets
- ESLint
- Prettier
- Tailwind CSS IntelliSense

### IntelliJ / WebStorm
Built-in support for Next.js, React, and Tailwind CSS.

## Performance Notes

- **Landing page**: ~200KB initial load (globe WebGL)
- **Sandbox**: ~150KB (lazy-loaded MapVisualization component)
- **Map**: ~500KB (Leaflet + tiles from CDN)

## Next Steps

1. ✅ Install dependencies: `npm install`
2. ✅ Start dev server: `npm run dev`
3. ✅ Visit http://localhost:3000
4. ✅ Click "Sandbox" to run CyneScript tutorials

## Support

- **Next.js docs**: https://nextjs.org/docs
- **React docs**: https://react.dev
- **Leaflet docs**: https://leafletjs.com
- **Lucide icons**: https://lucide.dev

---

**Last updated**: 2025-03-14
**Node version required**: 16.x or higher
**npm version required**: 7.x or higher
