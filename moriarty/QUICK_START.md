# Quick Start Checklist

## ✅ Pre-Installation (5 minutes)

- [ ] Ensure Node.js 16+ is installed: `node --version`
- [ ] Ensure npm 7+ is installed: `npm --version`
- [ ] Navigate to project: `cd moriarty`
- [ ] Review updated `package.json` (includes lucide-react, leaflet)
- [ ] Read `DEPENDENCIES_UPDATE.md` for what changed

## ✅ Installation (2 minutes)

```bash
cd moriarty
npm install
```

Wait for installation to complete (this may take 1-3 minutes).

**Progress indicators:**
- ✓ `added XXX packages`
- ✓ `up to date in XXs`
- ✓ No `ERR!` messages

If you see errors, see **Troubleshooting** section below.

## ✅ Post-Installation (1 minute)

```bash
# Verify installation
npm list --depth=0

# Should output packages like:
# lucide-react@0.263.1
# leaflet@1.9.4
# react@18.2.0
# next@13.2.1
```

## ✅ Start Development (1 minute)

```bash
npm run dev
```

You should see:
```
   ▲ Next.js 13.2.1
   - Local: http://localhost:3000
✓ Ready in XXXms
```

## ✅ Verify Everything Works (2 minutes)

Open your browser:

### Landing Page
- [ ] Visit `http://localhost:3000`
- [ ] See animated paleogeographic globe (GLB model)
- [ ] See "Sighthound" title
- [ ] See "Sandbox" button
- [ ] Can rotate/zoom globe with mouse

### Sandbox IDE
- [ ] Click "Sandbox" button
- [ ] See code editor with CyneScript syntax
- [ ] See file explorer on left
- [ ] See "Map", "Results", "Console" tabs on right
- [ ] See example tutorial loaded (level_1_load_and_display.cynes)

### Map Visualization
- [ ] Click "Run" button
- [ ] Switch to "Map" tab
- [ ] Should see map loading (may take a few seconds)
- [ ] Should see 5 points plotted on map
- [ ] Should see connecting lines (trajectory)

## ✅ Total Setup Time: ~10 minutes

```
Pre-check      2 min
npm install    3 min
Start dev      1 min
Verification   2 min
Documentation  2 min
─────────────
Total         ~10 min
```

## 🛠️ Troubleshooting

### "npm: command not found"
- Install Node.js from https://nodejs.org
- Restart terminal after install
- Verify: `npm --version`

### "Cannot find module 'lucide-react'"
```bash
npm install lucide-react leaflet
```

### "Port 3000 already in use"
```bash
npm run dev -- -p 3001
```

### "npm ERR! code ERESOLVE"
```bash
npm install --legacy-peer-deps
```

### "Module not found: @/components/MapVisualization"
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Map not showing in Sandbox
- Ensure you're on the "Map" tab (not "Results")
- Click "Run" button to execute script
- Wait 2-3 seconds for Leaflet to load from CDN
- Check browser console (F12) for errors

## 📚 Documentation

| File | Purpose |
|------|---------|
| `package.json` | Dependencies list (UPDATED) |
| `tsconfig.json` | TypeScript config (NEW) |
| `SETUP.md` | Complete setup guide |
| `INSTALL_DEPENDENCIES.md` | Step-by-step installation |
| `DEPENDENCIES_UPDATE.md` | What changed & why |
| `QUICK_START.md` | This file |

## 🚀 You're Done!

Your project is now fully set up with:

✅ All dependencies installed
✅ TypeScript support configured
✅ Landing page with animated globe
✅ Sandbox IDE with map visualization
✅ Example tutorials loaded

### Next Steps

1. **Explore the UI**
   - Edit code in the Sandbox
   - Upload your own GeoJSON files
   - Run CyneScript tutorials

2. **Customize**
   - Edit `src/pages/index.js` for landing page
   - Edit `src/pages/sandbox.jsx` for IDE
   - Add new tutorials to `src/tutorials/`

3. **Build for Production**
   ```bash
   npm run build
   npm start
   ```

4. **Deploy**
   - Deploy to Vercel (easiest for Next.js)
   - Or your own hosting

## 💡 Common Tasks

```bash
# Start development
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linter
npm run lint

# Add a new package
npm install package-name

# Update packages
npm update

# Remove old node_modules
rm -rf node_modules package-lock.json
npm install
```

## 🎯 Verification Commands

Copy and paste to verify setup:

```bash
# Check Node
node --version

# Check npm
npm --version

# Check Next.js
npx next --version

# Check packages
npm list lucide-react leaflet react

# Build check
npm run build

# Linter check
npm run lint
```

All should complete without major errors.

---

**Status**: ✅ Ready to develop
**Setup Complete**: When `npm run dev` shows "Ready in XXXms"
**Next**: Visit http://localhost:3000 and click "Sandbox"
