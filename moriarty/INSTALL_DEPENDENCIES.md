# Install All Dependencies

## Quick Start

```bash
cd moriarty
npm install
npm run dev
```

Visit `http://localhost:3000` to see the landing page with animated globe.

## Step-by-Step Installation

### 1. Prerequisites

Ensure you have:
- **Node.js**: 16.x or higher
- **npm**: 7.x or higher

Check versions:
```bash
node --version
npm --version
```

### 2. Install Dependencies

```bash
# Navigate to project
cd moriarty

# Install all dependencies from package.json
npm install
```

This will install:
- ✅ Next.js 13
- ✅ React 18
- ✅ Tailwind CSS
- ✅ Lucide React (icons)
- ✅ Leaflet (maps)
- ✅ Framer Motion (animations)
- ✅ TypeScript types (for better IDE support)

### 3. Verify Installation

```bash
# Check if all packages are installed
npm list

# Output should show:
# moriarty@0.1.0 /path/to/moriarty
# ├── lucide-react@0.263.1
# ├── leaflet@1.9.4
# ├── next@13.2.1
# ├── react@18.2.0
# └── ... (other packages)
```

### 4. Environment Setup

```bash
# Copy environment template
cp .env.local.example .env.local

# Edit if needed (optional)
# nano .env.local  or  code .env.local
```

### 5. Start Development Server

```bash
npm run dev
```

You should see:
```
> developer-portfolio@0.1.0 dev
> next dev

   ▲ Next.js 13.2.1
   - Local:        http://localhost:3000
   - Environments: .env.local

✓ Ready in 1234ms
```

### 6. Verify Everything Works

Open your browser and visit:
- **http://localhost:3000** → Should see animated 3D globe with "Sandbox" button
- **http://localhost:3000/sandbox** → Should see CyneScript IDE with code editor and map output

## Troubleshooting Installation

### Issue: "Cannot find module 'lucide-react'"

**Solution**: Run npm install again
```bash
npm install lucide-react leaflet @types/leaflet
```

### Issue: "npm ERR! code ERESOLVE"

**Solution**: Use npm 7+ with legacy peer deps
```bash
npm install --legacy-peer-deps
```

### Issue: Port 3000 already in use

**Solution**: Run on different port
```bash
npm run dev -- -p 3001
```

### Issue: "Module not found" errors after npm install

**Solution**: Clear cache and reinstall
```bash
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## What's Installed

### Production Dependencies
```
lucide-react (0.263.1)      - Icon library for UI
leaflet (1.9.4)             - Interactive maps
react-globe.gl (2.32.0)     - 3D globe visualization
satellite.js (1.53.0)       - Satellite calculations
framer-motion (10.0.1)      - Animation library
next (13.2.1)               - Framework
react (18.2.0)              - UI library
react-dom (18.2.0)          - DOM rendering
```

### Development Dependencies
```
@types/leaflet (1.9.8)      - Leaflet TypeScript types
@types/react (18.2.0)       - React TypeScript types
@types/node (18.16.0)       - Node TypeScript types
tailwindcss (3.2.7)         - CSS framework
postcss (8.4.21)            - CSS processing
autoprefixer (10.4.13)      - CSS vendor prefixes
eslint (8.35.0)             - Code linting
eslint-plugin-react (7.32.0) - React linting
```

## File Structure After Installation

```
moriarty/
├── node_modules/          ← All installed packages (created by npm install)
├── src/
│   ├── pages/
│   │   ├── index.js       ← Landing page (animated globe)
│   │   └── sandbox.jsx    ← CyneScript IDE
│   ├── components/
│   │   └── MapVisualization.jsx  ← Map component
│   └── tutorials/
│       ├── level_*.cynes  ← Tutorial scripts
│       └── example_data.geojson
├── package.json           ← Dependencies list (UPDATED)
├── package-lock.json      ← Locked versions
├── tsconfig.json          ← TypeScript config (NEW)
├── jsconfig.json          ← JavaScript path aliases
├── next.config.js         ← Next.js config
├── .env.local             ← Environment vars (NEW from template)
└── .env.local.example     ← Template (NEW)
```

## Verify Installation Commands

Run these to verify everything is set up correctly:

```bash
# 1. Check Node version
node --version
# Should output: v16.x.x or higher

# 2. Check npm version
npm --version
# Should output: 7.x.x or higher

# 3. List installed packages
npm list --depth=0
# Should show: lucide-react, leaflet, react, next, etc.

# 4. Check Next.js
npx next --version
# Should output: 13.2.1 or similar

# 5. Verify builds
npm run build
# Should complete without errors

# 6. Run linter
npm run lint
# Should show no major errors (warnings are OK)
```

## Common Commands After Installation

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linter
npm run lint

# Add new package
npm install package-name

# Remove package
npm uninstall package-name

# Update packages
npm update
```

## Performance Checklist

After installation, verify:
- ✅ `npm install` completes without errors
- ✅ `npm run dev` starts without errors
- ✅ Landing page loads with animated globe
- ✅ Sandbox page loads IDE with code editor
- ✅ Map visualization component loads (may show blank until script runs)
- ✅ `npm run build` completes successfully

## Next Steps

After successful installation:

1. **Explore the landing page**
   ```bash
   npm run dev
   # Visit http://localhost:3000
   ```

2. **Try the Sandbox**
   - Click "Sandbox" button
   - Load example GeoJSON data
   - Run CyneScript tutorials
   - See results on interactive map

3. **Customize**
   - Edit `.env.local` for configuration
   - Modify `src/pages/index.js` for landing page
   - Edit `src/pages/sandbox.jsx` for IDE
   - Add new tutorial scripts to `src/tutorials/`

## Support

If installation fails:

1. Check Node/npm versions are correct
2. Delete `node_modules` and `package-lock.json`, reinstall
3. Check internet connection (CDN resources needed)
4. Check disk space (node_modules can be ~500MB)
5. Try with `--legacy-peer-deps` if peer dependency conflicts

---

**Installation complete!** 🎉 Your project is ready to develop.
