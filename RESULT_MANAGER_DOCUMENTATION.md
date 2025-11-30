# Result Manager Documentation

## Overview

`result_manager.py` is a Python-based interactive visualization tool built with Panel and Bokeh for viewing and comparing geodetic model results. It provides a web-based interface for displaying GPS velocity vectors, fault segments, triangular dislocation elements (TDEs), and slip rate data on a map with Web Mercator projection.

The application supports loading data from two separate folders simultaneously, enabling side-by-side comparison of model results.

---

## Input Files

The application reads CSV files from user-selected directories. Each folder must contain the following files:

### 1. `model_station.csv`

Contains GPS station data with observed and modeled velocities.

**Required Columns:**

| Column Name | Description |
|-------------|-------------|
| `lon` | Station longitude (degrees) |
| `lat` | Station latitude (degrees) |
| `name` | Station name identifier |
| `east_vel` | Observed east velocity component (mm/yr) |
| `north_vel` | Observed north velocity component (mm/yr) |
| `model_east_vel` | Total modeled east velocity component (mm/yr) |
| `model_north_vel` | Total modeled north velocity component (mm/yr) |
| `model_east_vel_residual` | Residual east velocity (observed - modeled) (mm/yr) |
| `model_north_vel_residual` | Residual north velocity (observed - modeled) (mm/yr) |
| `model_east_vel_rotation` | East velocity due to block rotation (mm/yr) |
| `model_north_vel_rotation` | North velocity due to block rotation (mm/yr) |
| `model_east_elastic_segment` | East velocity from elastic segment deformation (mm/yr) |
| `model_north_elastic_segment` | North velocity from elastic segment deformation (mm/yr) |
| `model_east_vel_tde` | East velocity from triangular dislocation elements (mm/yr) |
| `model_north_vel_tde` | North velocity from triangular dislocation elements (mm/yr) |
| `model_east_vel_block_strain_rate` | East velocity from block strain rate (mm/yr) |
| `model_north_vel_block_strain_rate` | North velocity from block strain rate (mm/yr) |
| `model_east_vel_mogi` | East velocity from Mogi sources (mm/yr) |
| `model_north_vel_mogi` | North velocity from Mogi sources (mm/yr) |

### 2. `model_segment.csv`

Contains fault segment geometry and slip rate data.

**Required Columns:**

| Column Name | Description |
|-------------|-------------|
| `lon1` | Longitude of segment endpoint 1 (degrees) |
| `lat1` | Latitude of segment endpoint 1 (degrees) |
| `lon2` | Longitude of segment endpoint 2 (degrees) |
| `lat2` | Latitude of segment endpoint 2 (degrees) |
| `name` | Segment name identifier |
| `dip` | Fault dip angle (degrees, 0-90) |
| `locking_depth` | Fault locking depth (km) |
| `model_strike_slip_rate` | Modeled strike-slip rate (mm/yr) |
| `model_dip_slip_rate` | Modeled dip-slip rate (mm/yr) |
| `model_tensile_slip_rate` | Modeled tensile slip rate (mm/yr) |

### 3. `model_meshes.csv`

Contains triangular dislocation element (TDE) mesh geometry and slip rates.

**Required Columns:**

| Column Name | Description |
|-------------|-------------|
| `lon1` | Longitude of triangle vertex 1 (degrees) |
| `lat1` | Latitude of triangle vertex 1 (degrees) |
| `dep1` | Depth of triangle vertex 1 (km) |
| `lon2` | Longitude of triangle vertex 2 (degrees) |
| `lat2` | Latitude of triangle vertex 2 (degrees) |
| `dep2` | Depth of triangle vertex 2 (km) |
| `lon3` | Longitude of triangle vertex 3 (degrees) |
| `lat3` | Latitude of triangle vertex 3 (degrees) |
| `dep3` | Depth of triangle vertex 3 (km) |
| `mesh_idx` | Mesh index identifier (integer) |
| `strike_slip_rate` | Strike-slip rate on the TDE (mm/yr) |
| `dip_slip_rate` | Dip-slip rate on the TDE (mm/yr) |

### 4. `GSHHS_c_L1_0_360.npz` (Optional - Global Coastlines)

A NumPy compressed file containing coastline data for map display.

**Required Arrays:**
- `lon`: Coastline longitude coordinates
- `lat`: Coastline latitude coordinates

### 5. `mapbox_token.py` (Optional)

Contains a Mapbox access token for enhanced map styling.

```python
mapbox_access_token = "your_token_here"
```

---

## Arrow Types and Their Data Sources

All velocity arrows are drawn from station locations to computed endpoint positions. The arrows use a base `VELOCITY_SCALE = 1000` constant for visual scaling.

### Arrow Type Reference Table

| Arrow Type | UI Label | Color (Folder 1) | Color (Folder 2) | Source CSV | East Velocity Column | North Velocity Column |
|------------|----------|------------------|------------------|------------|---------------------|----------------------|
| Observed | `obs` | Blue (0,0,256) | Blue (0,0,205) | `model_station.csv` | `east_vel` | `north_vel` |
| Modeled | `mod` | Red (256,0,0) | Red (205,0,0) | `model_station.csv` | `model_east_vel` | `model_north_vel` |
| Residual | `res` | Magenta (256,0,256) | Magenta (205,0,205) | `model_station.csv` | `model_east_vel_residual` | `model_north_vel_residual` |
| Rotation | `rot` | Green (0,256,0) | Green (0,205,0) | `model_station.csv` | `model_east_vel_rotation` | `model_north_vel_rotation` |
| Segment Elastic | `seg` | Cyan (0,256,256) | Cyan (0,205,205) | `model_station.csv` | `model_east_elastic_segment` | `model_north_elastic_segment` |
| TDE | `tri` | Orange (256,166,0) | Orange (205,133,0) | `model_station.csv` | `model_east_vel_tde` | `model_north_vel_tde` |
| Strain | `str` | Teal (0,128,128) | Teal (0,102,102) | `model_station.csv` | `model_east_vel_block_strain_rate` | `model_north_vel_block_strain_rate` |
| Mogi | `mog` | Gray (128,128,128) | Gray (102,102,102) | `model_station.csv` | `model_east_vel_mogi` | `model_north_vel_mogi` |

### Arrow Endpoint Calculation

For each arrow type, the endpoint coordinates are calculated as:

```
x_end = station_x + VELOCITY_SCALE × slider_value × east_velocity
y_end = station_y + VELOCITY_SCALE × slider_value × north_velocity
```

Where:
- `station_x`, `station_y` are the Web Mercator projected station coordinates
- `VELOCITY_SCALE = 1000` is the base scaling constant
- `slider_value` is the current velocity scale slider value (0-50, default 1)
- `east_velocity`, `north_velocity` are the velocity components in mm/yr

---

## Velocity Scale Slider

### Overview

The velocity scale slider controls the visual length of all velocity arrows and the size of residual magnitude scatter points.

### Parameters

| Parameter | Value |
|-----------|-------|
| Minimum | 0 |
| Maximum | 50 |
| Default | 1 |
| Step | 1 |
| Width | 200 pixels |

### How It Works

The slider value acts as a **multiplier** applied to the base `VELOCITY_SCALE` constant (1000).

**Arrow Length Formula:**
```
arrow_endpoint = station_position + (VELOCITY_SCALE × slider_value × velocity_component)
```

**Residual Magnitude Size Formula:**
```
scatter_size = (VELOCITY_SCALE / 2500) × slider_value × residual_magnitude
```

### Practical Effects

| Slider Value | Arrow Multiplier | Effect |
|--------------|------------------|--------|
| 0 | ×0 | Arrows collapse to points |
| 1 | ×1000 | Default display (1 mm/yr → 1000 map units) |
| 10 | ×10000 | Arrows 10× longer than default |
| 50 | ×50000 | Maximum arrow length |

### Implementation Details

The slider uses a JavaScript callback (`velocity_scaler_callback`) that:
1. Reads the current slider value
2. Recalculates all arrow endpoint positions for both folders
3. Recalculates residual magnitude sizes
4. Updates the ColumnDataSource objects to trigger re-rendering

---

## Slip Rate Visualization

### Segment Slip Rates (UI: `slip` checkbox + `ss`/`ds` radio buttons)

**Source File:** `model_segment.csv`

**Columns Used:**
| Display Option | Column Name | Description |
|----------------|-------------|-------------|
| `ss` (Strike-Slip) | `model_strike_slip_rate` | Strike-slip rate (mm/yr) |
| `ds` (Dip-Slip) | `model_dip_slip_rate` - `model_tensile_slip_rate` | Net dip-slip rate (mm/yr) |

**Visualization:**
- Segments are drawn as colored lines using the `RdBu` (Red-Blue) diverging colormap
- Color range: -100 to +100 mm/yr
- Positive values (right-lateral/thrust): Red
- Negative values (left-lateral/normal): Blue

**How Dip-Slip is Computed:**
```python
dsrate = model_dip_slip_rate - model_tensile_slip_rate
```

The tensile slip rate is subtracted from the dip-slip rate to isolate the shear component of dip-slip motion.

---

## TDE (Triangular Dislocation Element) Visualization

### TDE Slip Rates (UI: `tde` checkbox + `ss`/`ds` radio buttons)

**Source File:** `model_meshes.csv`

**Columns Used:**
| Display Option | Column Name | Description |
|----------------|-------------|-------------|
| `ss` (Strike-Slip) | `strike_slip_rate` | Strike-slip rate on the TDE (mm/yr) |
| `ds` (Dip-Slip) | `dip_slip_rate` | Dip-slip rate on the TDE (mm/yr) |

**Visualization:**
- TDEs are drawn as filled triangular patches
- Uses the same `RdBu` colormap as segments
- Color range: -100 to +100 mm/yr
- Mesh perimeter edges are drawn as lines:
  - **Black lines**: Horizontal/shallow dipping mesh boundaries
  - **Red lines**: Steeply dipping (>75°) projected mesh boundaries

### Mesh Processing

The code performs several processing steps on mesh data:

1. **Longitude Wrapping**: Negative longitudes are converted to 0-360 range
2. **Dip Calculation**: Element dip is calculated from triangle normal vectors
3. **Steep Mesh Projection**: Meshes with average dip >75° are projected to the surface along the dip direction for visibility
4. **Area-Based Ordering**: Meshes are drawn in order of decreasing area (largest first, smallest on top) to ensure smaller meshes are visible

### Mesh Geometry Columns

| Column | Description |
|--------|-------------|
| `lon1`, `lat1`, `dep1` | Vertex 1 coordinates |
| `lon2`, `lat2`, `dep2` | Vertex 2 coordinates |
| `lon3`, `lat3`, `dep3` | Vertex 3 coordinates |
| `mesh_idx` | Mesh identifier for grouping elements |

---

## Scatter Plot Types

### 1. Station Locations (UI: `locs` checkbox)

**Source File:** `model_station.csv`

**Columns Used:**
| Column | Purpose |
|--------|---------|
| `lon` | Station longitude |
| `lat` | Station latitude |
| `name` | Station name (shown in hover tooltip) |

**Visualization:**
- Small black circles (size 2.7 for folder 1, size 1 for folder 2)
- Hover tooltip shows station name

### 2. Residual Magnitude (UI: `res mag` checkbox)

**Source File:** `model_station.csv`

**Columns Used:**
| Column | Purpose |
|--------|---------|
| `lon` | Station longitude |
| `lat` | Station latitude |
| `model_east_vel_residual` | East residual component |
| `model_north_vel_residual` | North residual component |

**Computed Value:**
```python
residual_magnitude = sqrt(model_east_vel_residual² + model_north_vel_residual²)
```

**Visualization:**
- Scatter points at station locations
- Size: `(VELOCITY_SCALE / 2500) × slider_value × residual_magnitude`
- Color: Viridis colormap, range 0-5 mm/yr

### 3. Residual Comparison (UI: `res compare` checkbox)

**Source Files:** `model_station.csv` from both loaded folders

**Columns Used:**
| Column | Purpose |
|--------|---------|
| `lon` | Station longitude |
| `lat` | Station latitude |
| `model_east_vel_residual` | East residual component |
| `model_north_vel_residual` | North residual component |

**Computed Values:**
```python
res_mag_1 = sqrt(east_residual_1² + north_residual_1²)
res_mag_2 = sqrt(east_residual_2² + north_residual_2²)
res_mag_diff = res_mag_2 - res_mag_1
```

**Visualization:**

*Common Stations (present in both folders):*
- Size: `(VELOCITY_SCALE / 2500) × slider_value × |res_mag_diff|`
- Color: `RdBu` diverging colormap, range -5 to +5 mm/yr
  - **Red**: Folder 2 has larger residuals (folder 2 is worse)
  - **Blue**: Folder 2 has smaller residuals (folder 2 is better)

*Unique Stations (present in only one folder):*
- Black "X" markers, size 15
- Indicates stations that cannot be compared

---

## Fault Projection Visualization (UI: `fault proj` checkbox)

**Source File:** `model_segment.csv`

**Columns Used:**
| Column | Purpose |
|--------|---------|
| `lon1`, `lat1` | Segment endpoint 1 |
| `lon2`, `lat2` | Segment endpoint 2 |
| `dip` | Fault dip angle (degrees) |
| `locking_depth` | Fault locking depth (km) |
| `name` | Segment name |

**What It Shows:**
Surface projection polygons of non-vertical fault planes, calculated from:
- Top edge: segment trace on surface
- Bottom edge: computed from dip angle and locking depth

**Calculation:**
The bottom edge is offset from the top edge in the dip direction by:
```
horizontal_distance = locking_depth / tan(dip)
```

**Visualization:**
- Folder 1: Light blue fill, blue outline
- Folder 2: Light green fill, dashed green outline

**Note:** Vertical faults (dip = 90°) do not generate projection polygons.

---

## Color Scales

### Slip Rate Colorbar
- **Palette:** `RdBu` (Red-Blue diverging, 11 colors)
- **Range:** -100 to +100 mm/yr
- **Label:** "Slip rate (mm/yr)"

### Residual Magnitude Colorbar
- **Palette:** Viridis (10 colors)
- **Range:** 0 to 5 mm/yr
- **Label:** "Resid. mag. (mm/yr)"

### Residual Difference Colorbar
- **Palette:** `RdBu` (Red-Blue diverging, 11 colors)
- **Range:** -5 to +5 mm/yr
- **Label:** "Resid. diff. (mm/yr)"

---

## Coordinate Transformations

All geographic coordinates are transformed to Web Mercator (EPSG:3857) for display:

```python
x = 6378137.0 × lon_radians
y = 6378137.0 × ln(tan(π/4 + lat_radians/2))
```

Longitudes are wrapped to 0-360 range for meshes to handle dateline crossings.

---

## UI Layout Summary

| UI Element | Description |
|------------|-------------|
| **load** button | Opens file dialog to select data folder |
| **locs** | Toggle station location markers |
| **obs** | Toggle observed velocity arrows |
| **mod** | Toggle modeled velocity arrows |
| **res** | Toggle residual velocity arrows |
| **rot** | Toggle rotation velocity arrows |
| **seg** | Toggle segment elastic velocity arrows |
| **tri** | Toggle TDE velocity arrows |
| **str** | Toggle strain rate velocity arrows |
| **mog** | Toggle Mogi source velocity arrows |
| **res mag** | Toggle residual magnitude scatter plot |
| **slip** + ss/ds | Toggle segment slip rate coloring |
| **tde** + ss/ds | Toggle TDE slip rate patches |
| **fault proj** | Toggle fault surface projection polygons |
| **res compare** | Toggle residual comparison (requires 2 folders) |
| **vel scale** | Slider to adjust arrow/scatter sizes (0-50) |
