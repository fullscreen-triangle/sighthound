# Publication Figure Panels

Generated from circular validation results on 2026-02-27

All figures are 300 DPI, publication-quality PNG images.

## Cynegeticus Positioning Paper (4 panels)

### Panel 1: GPS Trajectories and S-Entropy Spatial Distribution
**File**: `cynegeticus_panel_1.png`

- **Chart 1**: GPS trajectories from both watches (blue/red lines, green start marker)
- **Chart 2**: S_k spatial distribution (viridis colormap heatmap)
- **Chart 3**: **3D trajectory** in (lon, lat, S_k) space
- **Chart 4**: Circular closure visualization (original vs reconstructed with error vectors)

**Key insight**: Shows the bijective mapping between GPS position and S-entropy state.

---

### Panel 2: S-Entropy Coordinate Analysis
**File**: `cynegeticus_panel_2.png`

- **Chart 1**: S_k evolution along trajectory
- **Chart 2**: S_t evolution along trajectory
- **Chart 3**: S_e evolution along trajectory
- **Chart 4**: **3D scatter** of all points in (S_k, S_t, S_e) partition space

**Key insight**: Demonstrates how S-entropy coordinates encode trajectory information.

---

### Panel 3: Position Reconstruction Validation
**File**: `cynegeticus_panel_3.png`

- **Chart 1**: Original vs reconstructed latitude (scatter with diagonal)
- **Chart 2**: Original vs reconstructed longitude (scatter with diagonal)
- **Chart 3**: **3D trajectory comparison** (blue=original, red=reconstructed)
- **Chart 4**: Position error distribution histogram (mean: 0.22 m)

**Key insight**: Validates the S-entropy inverse mapping with sub-meter accuracy.

---

### Panel 4: Virtual Satellite Concept
**File**: `cynegeticus_panel_4.png`

- **Chart 1**: Earth view with Munich measurement location
- **Chart 2**: Virtual satellite constellation (8 satellites at GPS orbit)
- **Chart 3**: **3D measurement geometry** (Earth + satellites + Munich)
- **Chart 4**: Position uncertainty ellipses along trajectory

**Key insight**: Visualizes the virtual satellite constellation derived from partition structure.

---

## Ober Weather Prediction Paper (4 panels)

### Panel 1: Weather Forecast Evolution
**File**: `ober_panel_1.png`

- **Chart 1**: Temperature evolution (forecast vs actual)
- **Chart 2**: Pressure evolution (forecast vs actual)
- **Chart 3**: Humidity evolution (forecast vs actual)
- **Chart 4**: **3D weather state trajectory** in (T, P, H) space

**Key insight**: Shows 10-day weather evolution through partition dynamics.

---

### Panel 2: Partition Dynamics Evolution
**File**: `ober_panel_2.png`

- **Chart 1**: S_k compositional partition evolution
- **Chart 2**: S_t temporal partition evolution
- **Chart 3**: S_e energy partition evolution
- **Chart 4**: **3D partition trajectory** in (S_k, S_t, S_e) space with day colormap

**Key insight**: Demonstrates deterministic evolution in partition space with start/end markers.

---

### Panel 3: Forecast Validation
**File**: `ober_panel_3.png`

- **Chart 1**: Temperature forecast vs actual (scatter with diagonal)
- **Chart 2**: Pressure forecast vs actual (scatter with diagonal)
- **Chart 3**: Wind speed forecast vs actual (scatter with diagonal)
- **Chart 4**: **3D error distribution** in (temp_error, pressure_error, wind_error) space

**Key insight**: Validates forecast accuracy with correlation analysis.

---

### Panel 4: Skill Metrics and Performance
**File**: `ober_panel_4.png`

- **Chart 1**: Cumulative RMSE evolution over forecast period
- **Chart 2**: Forecast vs actual correlation (scatter with r-value)
- **Chart 3**: Skill score evolution (green=skillful, red=unskillful)
- **Chart 4**: **3D performance space** showing forecast vs actual in (T, P, H) with error connections

**Key insight**: Quantifies forecast skill degradation and maintains positive skill throughout 10-day period.

---

## Technical Specifications

- **Resolution**: 300 DPI (publication quality)
- **Size**: 20" × 5" (4-panel row format)
- **Format**: PNG with transparency
- **Color schemes**:
  - Viridis/Plasma for continuous data
  - Blue/Red for comparisons
  - Green for positive, Red for negative
- **3D views**: Optimized angles for maximum clarity
- **Minimal text**: Axis labels and titles only, no legends where obvious

## Usage in Papers

### Cynegeticus Paper
- Use Panel 1 in Results section (GPS trajectory validation)
- Use Panel 2 in Methods section (S-entropy extraction)
- Use Panel 3 in Results section (circular closure validation)
- Use Panel 4 in Methods section (virtual satellite concept)

### Ober Paper
- Use Panel 1 in Results section (weather forecast results)
- Use Panel 2 in Methods section (partition dynamics theory)
- Use Panel 3 in Results section (forecast validation)
- Use Panel 4 in Results section (skill assessment)

All figures demonstrate **trajectory completion** principles with minimal text and maximum visual impact.